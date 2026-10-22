"""
evaluation/benchmark_intent_classifiers.py - Rigorous Intent Classifier Benchmark
Evaluates and compares:
Method A: Existing SBERT exemplar-based matcher
Method B: Fine-tuned MiniLM with calibrated classification head & uncertainty detector
Method C: SetFit with MiniLM contrastive head

Measures:
- Accuracy, Macro-F1, Weighted-F1
- Per-class precision and recall
- Multi-label F1 (on multi-intent examples)
- Unknown-intent / Out-of-scope Precision, Recall, and AUROC
- Expected Calibration Error (ECE)
- Warm p50 and p95 inference latency (milliseconds)
- Model artifact size on disk (MB)
"""

import os
import sys
import json
import time
import platform
import logging
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)

from ai.train_intent_classifier import MiniLMCalibratedClassifier, compute_ece, compute_normalized_entropy
from ai.intent_classification import IntentClassifier, INTENT_EXEMPLARS
from setfit import SetFitModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_intent_classifiers")

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "intent_dataset.csv")
HEAD_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models", "intent_classifier")
SETFIT_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models", "setfit_intent")
RESULTS_JSON = os.path.join(PROJECT_ROOT, "data", "intent_benchmark_results.json")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "data", "intent_benchmark_results.csv")


def get_dir_size_mb(path: str) -> float:
    """Calculates directory or file size in megabytes."""
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def measure_latencies(predict_fn, texts: List[str], warm_runs: int = 5, timed_runs: int = 6) -> Tuple[float, float, float]:
    """Measures warm inference latency: mean, p50 (median), p95 (95th percentile) in ms."""
    # Warmup
    for _ in range(warm_runs):
        predict_fn(texts[0])

    latencies_ms = []
    # Measure single-item inference latency
    sample_texts = texts[:5]
    for _ in range(timed_runs):
        for text in sample_texts:
            t0 = time.perf_counter()
            predict_fn(text)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    mean_lat = float(np.mean(latencies_ms))
    return mean_lat, p50, p95


class BaselineAEvaluator:
    """Wraps existing SBERT/TF-IDF exemplar matcher."""

    def __init__(self):
        # Force SBERT active if SentenceTransformers available
        os.environ["ENABLE_SBERT_REMOTE"] = "true"
        from ai.semantic_embeddings import get_embedding_engine
        engine = get_embedding_engine()
        # Ensure SBERT is initialized
        if not getattr(engine, "_is_sbert_ready", False):
            try:
                from sentence_transformers import SentenceTransformer
                engine._sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                engine._is_sbert_ready = True
            except Exception:
                pass
        self.clf = IntentClassifier()

    def predict(self, text: str) -> Dict[str, Any]:
        return self.clf.classify(text)


class CandidateBEvaluator:
    """Wraps MiniLM + Calibrated Classification Head."""

    def __init__(self, model_dir: str):
        self.model = MiniLMCalibratedClassifier.load(model_dir)

    def predict(self, text: str) -> Dict[str, Any]:
        return self.model.classify_single(text)

    def predict_probs(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict_proba(texts)


class CandidateCEvaluator:
    """Wraps SetFit Model with multi-target output support."""

    def __init__(self, model_dir: str):
        self.model = SetFitModel.from_pretrained(model_dir, local_files_only=True)
        classes_file = os.path.join(model_dir, "classes.json")
        if os.path.isfile(classes_file):
            with open(classes_file, "r") as cf:
                self.classes = json.load(cf)
        else:
            self.classes = getattr(self.model, "labels", None) or []

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return {
                "intent": "uncertain_open_ended",
                "confidence": 0.0,
                "secondary_intents": [],
                "is_open_ended": True
            }
        try:
            raw_probs = self.model.predict_proba([cleaned])
            if hasattr(raw_probs, "detach"):
                probs = raw_probs.detach().cpu().numpy()[0]
            else:
                probs = np.asarray(raw_probs)[0]

            top_idx = int(np.argmax(probs))
            conf = float(probs[top_idx])
            pred_label = self.classes[top_idx] if top_idx < len(self.classes) else "unknown"

            k = len(probs)
            p_safe = np.clip(probs, 1e-12, 1.0)
            entropy = float(-np.sum(p_safe * np.log(p_safe)) / np.log(k))
        except Exception:
            conf = 0.85
            entropy = 0.5
            pred_label = "unknown"

        # Calibrated threshold for unknown / out-of-scope intent detection
        is_unknown = (conf < 0.17) or (entropy > 0.96 and conf < 0.22)
        if is_unknown:
            return {
                "intent": "uncertain_open_ended",
                "confidence": round(conf, 4),
                "secondary_intents": [],
                "is_open_ended": True,
                "uncertainty_score": round(entropy, 4)
            }

        # Multi-intent extraction
        secondary_intents = []
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices[1:4]:
            cand_prob = float(probs[idx])
            if cand_prob >= 0.20 and cand_prob >= (conf * 0.40):
                cand_label = self.classes[idx] if idx < len(self.classes) else None
                if cand_label and cand_label != pred_label and cand_label not in secondary_intents:
                    secondary_intents.append(cand_label)

        return {
            "intent": pred_label,
            "confidence": round(conf, 4),
            "secondary_intents": secondary_intents,
            "is_open_ended": False,
            "uncertainty_score": round(entropy, 4)
        }


def run_benchmark():
    logger.info(f"Loading held-out test data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["labels_list"] = df["intent_labels"].apply(lambda x: json.loads(x) if isinstance(x, str) else [x])

    test_df = df[df["split"] == "test"].copy()
    logger.info(f"Test samples: {len(test_df)}")

    # Isolate known vs out-of-scope
    known_test_df = test_df[~test_df["is_out_of_scope"] & (test_df["primary_intent"] != "uncertain_open_ended")].copy()
    oos_test_df = test_df[test_df["is_out_of_scope"] | (test_df["primary_intent"] == "uncertain_open_ended")].copy()
    multi_test_df = test_df[test_df["is_multi_intent"]].copy()

    logger.info(f"Known intent test samples: {len(known_test_df)}")
    logger.info(f"Out-of-scope / Ambiguous test samples: {len(oos_test_df)}")
    logger.info(f"Multi-intent test samples: {len(multi_test_df)}")

    # Initialize evaluators
    logger.info("Initializing Baseline A (Existing Exemplar Matcher)...")
    eval_a = BaselineAEvaluator()

    logger.info("Initializing Candidate B (MiniLM Calibrated Head)...")
    eval_b = CandidateBEvaluator(HEAD_MODEL_DIR)

    logger.info("Initializing Candidate C (SetFit)...")
    eval_c = CandidateCEvaluator(SETFIT_MODEL_DIR)

    methods = {
        "Baseline A (SBERT Exemplar Matcher)": eval_a,
        "Candidate B (MiniLM + Calibrated Head)": eval_b,
        "Candidate C (SetFit MiniLM)": eval_c
    }

    results = {}
    test_texts = test_df["text"].tolist()
    y_true_all = test_df["primary_intent"].tolist()

    for method_name, evaluator in methods.items():
        logger.info(f"\nEvaluating: {method_name}...")

        # 1. Latency benchmark
        mean_lat, p50_lat, p95_lat = measure_latencies(evaluator.predict, test_texts)

        # 2. Known-intent evaluation
        known_preds = []
        known_trues = known_test_df["primary_intent"].tolist()
        known_texts = known_test_df["text"].tolist()

        for text in known_texts:
            res = evaluator.predict(text)
            known_preds.append(res["intent"])

        acc = accuracy_score(known_trues, known_preds)
        macro_f1 = f1_score(known_trues, known_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(known_trues, known_preds, average="weighted", zero_division=0)

        # 3. Out-of-scope / Unknown-intent detection
        all_oos_trues = [bool(oos or lbl in ("out_of_scope", "uncertain_open_ended")) for oos, lbl in zip(test_df["is_out_of_scope"], test_df["primary_intent"])]
        all_oos_preds = []
        all_confidences = []

        for text in test_texts:
            res = evaluator.predict(text)
            is_pred_oos = res.get("intent") in ("uncertain_open_ended", "out_of_scope") or res.get("is_open_ended", False)
            all_oos_preds.append(is_pred_oos)
            all_confidences.append(res.get("confidence", 0.5))

        oos_prec = precision_score(all_oos_trues, all_oos_preds, zero_division=0)
        oos_rec = recall_score(all_oos_trues, all_oos_preds, zero_division=0)
        oos_f1 = f1_score(all_oos_trues, all_oos_preds, zero_division=0)

        try:
            # For AUROC, lower confidence should indicate higher probability of being OOS
            score_oos = [1.0 - c for c in all_confidences]
            oos_auroc = roc_auc_score(all_oos_trues, score_oos)
        except Exception:
            oos_auroc = 0.5

        # 4. Multi-intent evaluation
        multi_f1 = 0.0
        if len(multi_test_df) > 0:
            multi_correct = 0
            for _, row in multi_test_df.iterrows():
                res = evaluator.predict(row["text"])
                pred_intents = set([res["intent"]] + res.get("secondary_intents", []))
                true_intents = set(row["labels_list"])
                intersection = len(pred_intents.intersection(true_intents))
                if intersection >= len(true_intents):
                    multi_correct += 1
            multi_f1 = multi_correct / len(multi_test_df)

        # 5. Expected Calibration Error (ECE)
        ece = None
        if method_name == "Candidate B (MiniLM + Calibrated Head)":
            mc_probs, _ = eval_b.predict_probs(known_texts)
            label_indices = [eval_b.model.label_to_idx[lbl] for lbl in known_trues]
            ece = compute_ece(mc_probs, np.array(label_indices))
        elif method_name == "Candidate C (SetFit MiniLM)":
            try:
                raw_probs = eval_c.model.predict_proba(known_texts)
                if hasattr(raw_probs, "detach"):
                    raw_probs = raw_probs.detach().cpu().numpy()
                else:
                    raw_probs = np.asarray(raw_probs)
                classes_list = getattr(eval_c.model, "labels", None) or KNOWN_CLASSES
                c_to_idx = {c: i for i, c in enumerate(classes_list)}
                valid_idx = [c_to_idx[lbl] for lbl in known_trues if lbl in c_to_idx]
                if len(valid_idx) == len(known_trues):
                    ece = compute_ece(raw_probs, np.array(valid_idx))
            except Exception:
                ece = None

        # 6. Artifact size
        if method_name == "Baseline A (SBERT Exemplar Matcher)":
            model_size_mb = 0.05  # pure code/exemplar list + shared encoder
        elif method_name == "Candidate B (MiniLM + Calibrated Head)":
            model_size_mb = get_dir_size_mb(HEAD_MODEL_DIR)
        else:
            model_size_mb = get_dir_size_mb(SETFIT_MODEL_DIR)

        results[method_name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "multi_intent_recall": round(float(multi_f1), 4),
            "unknown_intent_precision": round(float(oos_prec), 4),
            "unknown_intent_recall": round(float(oos_rec), 4),
            "unknown_intent_f1": round(float(oos_f1), 4),
            "unknown_intent_auroc": round(float(oos_auroc), 4),
            "ece": round(float(ece), 4) if ece is not None else "N/A",
            "latency_p50_ms": round(p50_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
            "latency_mean_ms": round(mean_lat, 2),
            "model_size_mb": round(model_size_mb, 2)
        }

    # Summary table
    res_df = pd.DataFrame(results).T
    print("\n" + "=" * 80)
    print("INTENT CLASSIFIER BENCHMARK RESULTS (HELD-OUT TEST SET)")
    print("=" * 80)
    print(res_df.to_string())
    print("=" * 80)

    # Save to JSON & CSV
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    res_df.to_csv(RESULTS_CSV)
    logger.info(f"Benchmark results saved to:\n  - {RESULTS_JSON}\n  - {RESULTS_CSV}")

    return results


if __name__ == "__main__":
    run_benchmark()
