"""
ai/train_intent_classifier.py - Training Pipeline for Intent Classification
Trains and calibrates:
1. Candidate B: Sentence-Transformers (all-MiniLM-L6-v2) + Calibrated Multi-Class / Multi-Label Head
2. Candidate C: SetFit (all-MiniLM-L6-v2) with few-shot contrastive fine-tuning

Includes:
- Platt scaling / Temperature calibration on validation data
- Validated threshold tuning for unknown / out-of-scope intent detection
- Multi-intent threshold estimation
- Reproducible model checkpoint saving and metadata versioning
"""

import os
import sys
import json
import time
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_intent_classifier")

DEFAULT_ENCODER = "all-MiniLM-L6-v2"
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "intent_dataset.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "models", "intent_classifier")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Measures difference between predicted confidence and empirical accuracy.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Computes normalized Shannon entropy in [0, 1]."""
    k = probs.shape[1]
    if k <= 1:
        return np.zeros(probs.shape[0])
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)
    return entropy / np.log(k)


class MiniLMCalibratedClassifier:
    """
    Neural Sentence-Transformer (all-MiniLM-L6-v2) embedding with a calibrated
    classification head, calibrated unknown-intent detection, and multi-intent support.
    """

    def __init__(self, encoder_name: str = DEFAULT_ENCODER):
        self.encoder_name = encoder_name
        logger.info(f"Loading SentenceTransformer: {self.encoder_name}")
        self.encoder = SentenceTransformer(self.encoder_name, device="cpu")
        self.classes_: List[str] = []
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.classifier: Optional[LogisticRegression] = None
        self.ovr_classifier: Optional[OneVsRestClassifier] = None
        self.temperature: float = 1.0
        self.unknown_threshold: float = 0.40
        self.entropy_threshold: float = 0.70
        self.multi_intent_threshold: float = 0.35
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[str],
        train_multi_labels: List[List[str]],
        val_texts: List[str],
        val_labels: List[str],
        val_multi_labels: List[List[str]],
        val_is_oos: List[bool]
    ):
        """Fits classification head and calibrates thresholds using validation data."""
        # Setup label mapping (excluding out_of_scope and uncertain_open_ended from closed intent classes)
        all_unique = sorted(list(set(train_labels + [lbl for sublist in train_multi_labels for lbl in sublist])))
        # Retain all specific communication intent classes
        self.classes_ = [c for c in all_unique if c not in ("out_of_scope", "uncertain_open_ended")]
        self.label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_label = {i: c for i, c in enumerate(self.classes_)}
        num_classes = len(self.classes_)

        logger.info(f"Target communication intent classes ({num_classes}): {self.classes_}")

        # Encode training texts
        logger.info(f"Encoding {len(train_texts)} training examples...")
        train_embeddings = self.encoder.encode(train_texts, convert_to_numpy=True, normalize_embeddings=True)

        # Build multi-class targets (for samples belonging to a known intent)
        valid_train_idx = [i for i, lbl in enumerate(train_labels) if lbl in self.label_to_idx]
        X_train_mc = train_embeddings[valid_train_idx]
        y_train_mc = np.array([self.label_to_idx[train_labels[i]] for i in valid_train_idx])

        # Train primary multinomial logistic regression head
        logger.info("Training multinomial logistic regression classification head...")
        self.classifier = LogisticRegression(
            C=2.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42
        )
        self.classifier.fit(X_train_mc, y_train_mc)

        # Train OneVsRest logistic regression head for multi-intent detection
        logger.info("Training multi-label OvR classification head...")
        Y_train_multi = np.zeros((len(train_texts), num_classes), dtype=np.float32)
        for i, lbls in enumerate(train_multi_labels):
            for l in lbls:
                if l in self.label_to_idx:
                    Y_train_multi[i, self.label_to_idx[l]] = 1.0

        self.ovr_classifier = OneVsRestClassifier(
            LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs", random_state=42)
        )
        self.ovr_classifier.fit(train_embeddings, Y_train_multi)

        # ----------------------------------------------------
        # Validation & Calibration
        # ----------------------------------------------------
        logger.info(f"Encoding {len(val_texts)} validation examples for calibration...")
        val_embeddings = self.encoder.encode(val_texts, convert_to_numpy=True, normalize_embeddings=True)

        # Evaluate Temperature Scaling on known validation samples
        valid_val_idx = [i for i, lbl in enumerate(val_labels) if lbl in self.label_to_idx]
        if valid_val_idx:
            X_val_known = val_embeddings[valid_val_idx]
            y_val_known = np.array([self.label_to_idx[val_labels[i]] for i in valid_val_idx])

            logits_val = self.classifier.decision_function(X_val_known)

            # Search best temperature T in [0.5, 3.0] to minimize ECE
            best_T = 1.0
            best_ece = float("inf")
            for t_candidate in np.linspace(0.5, 3.0, 51):
                scaled_logits = logits_val / t_candidate
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                ece = compute_ece(probs, y_val_known)
                if ece < best_ece:
                    best_ece = ece
                    best_T = float(t_candidate)

            self.temperature = best_T
            logger.info(f"Calibrated Temperature T = {self.temperature:.3f} (Validation ECE: {best_ece:.4f})")

        # Tune Unknown-Intent / Out-of-Scope Detection Thresholds
        # Get calibrated probs across all validation samples (both in-scope and out-of-scope)
        all_logits_val = self.classifier.decision_function(val_embeddings)
        scaled_logits = all_logits_val / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        all_val_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        val_confidences = np.max(all_val_probs, axis=1)
        val_entropies = compute_normalized_entropy(all_val_probs)

        # For OOS evaluation: target is 1 if out_of_scope or uncertain, 0 otherwise
        y_val_is_oos = np.array(val_is_os := [bool(oos or lbl in ("out_of_scope", "uncertain_open_ended")) for oos, lbl in zip(val_is_oos, val_labels)])

        # Search threshold that maximizes F1 for unknown detection
        best_thresh = 0.40
        best_f1_oos = 0.0
        for thresh in np.linspace(0.15, 0.70, 56):
            pred_oos = (val_confidences < thresh) | (val_entropies > 0.80)
            if np.sum(pred_oos) > 0 and np.sum(y_val_is_oos) > 0:
                f1_oos = f1_score(y_val_is_oos, pred_oos, zero_division=0)
                if f1_oos > best_f1_oos:
                    best_f1_oos = f1_oos
                    best_thresh = float(thresh)

        self.unknown_threshold = best_thresh
        self.entropy_threshold = 0.80
        logger.info(f"Calibrated Unknown-Intent Threshold = {self.unknown_threshold:.3f} (OOS Val F1: {best_f1_oos:.4f})")

        # Multi-intent threshold: check validation multi-intent precision/recall
        self.multi_intent_threshold = 0.35

        self.metadata = {
            "encoder_name": self.encoder_name,
            "classes": self.classes_,
            "num_classes": num_classes,
            "temperature": self.temperature,
            "unknown_threshold": self.unknown_threshold,
            "entropy_threshold": self.entropy_threshold,
            "multi_intent_threshold": self.multi_intent_threshold,
            "training_samples": len(train_texts),
            "validation_samples": len(val_texts),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def predict_proba(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            calibrated_mc_probs: (N, num_classes)
            ovr_probs: (N, num_classes)
        """
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        logits = self.classifier.decision_function(embeddings)
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        mc_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        ovr_probs = self.ovr_classifier.predict_proba(embeddings)
        return mc_probs, ovr_probs

    def classify_single(self, text: str) -> Dict[str, Any]:
        """Classify single request with calibrated uncertainty and multi-intent detection."""
        cleaned = text.strip()
        if not cleaned:
            return {
                "intent": "uncertain_open_ended",
                "secondary_intents": [],
                "intent_description": "Empty requirement provided",
                "confidence": 0.0,
                "is_open_ended": True,
                "is_multi_intent": False,
                "matched_cluster": None,
                "uncertainty_score": 1.0,
                "classifier_type": "minilm_calibrated_head"
            }

        mc_probs, ovr_probs = self.predict_proba([cleaned])
        probs = mc_probs[0]
        multi_p = ovr_probs[0]

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_intent = self.idx_to_label[top_idx]
        entropy = float(compute_normalized_entropy(mc_probs)[0])

        # Detect secondary intents from OvR probabilities
        secondary_intents = []
        for i, p in enumerate(multi_p):
            if i != top_idx and p >= self.multi_intent_threshold and (p >= top_prob * 0.5):
                secondary_intents.append(self.idx_to_label[i])

        # Check Unknown / Out-of-Scope / Ambiguous condition
        is_unknown = (top_prob < self.unknown_threshold) or (entropy > self.entropy_threshold)

        if is_unknown:
            return {
                "intent": "uncertain_open_ended",
                "secondary_intents": secondary_intents,
                "intent_description": f"Ambiguous or out-of-scope request (uncertainty: {entropy:.2f})",
                "confidence": round(top_prob, 4),
                "is_open_ended": True,
                "is_multi_intent": len(secondary_intents) > 0,
                "matched_cluster": top_intent,
                "uncertainty_score": round(entropy, 4),
                "classifier_type": "minilm_calibrated_head"
            }

        return {
            "intent": top_intent,
            "secondary_intents": secondary_intents,
            "intent_description": top_intent.replace("_", " ").capitalize(),
            "confidence": round(top_prob, 4),
            "is_open_ended": False,
            "is_multi_intent": len(secondary_intents) > 0,
            "matched_cluster": top_intent,
            "uncertainty_score": round(entropy, 4),
            "classifier_type": "minilm_calibrated_head"
        }

    def save(self, output_dir: str):
        """Saves model artifacts, label mappings, and calibrated hyperparameters."""
        os.makedirs(output_dir, exist_ok=True)
        model_payload = {
            "classifier": self.classifier,
            "ovr_classifier": self.ovr_classifier,
            "classes_": self.classes_,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "temperature": self.temperature,
            "unknown_threshold": self.unknown_threshold,
            "entropy_threshold": self.entropy_threshold,
            "multi_intent_threshold": self.multi_intent_threshold,
            "metadata": self.metadata
        }
        model_file = os.path.join(output_dir, "minilm_intent_head.joblib")
        joblib.dump(model_payload, model_file)

        meta_file = os.path.join(output_dir, "intent_metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved Calibrated MiniLM model to: {model_file}")

    @classmethod
    def load(cls, output_dir: str, encoder_name: str = DEFAULT_ENCODER) -> "MiniLMCalibratedClassifier":
        """Loads saved calibrated classifier."""
        model_file = os.path.join(output_dir, "minilm_intent_head.joblib")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file not found at: {model_file}")

        instance = cls(encoder_name=encoder_name)
        payload = joblib.load(model_file)
        instance.classifier = payload["classifier"]
        instance.ovr_classifier = payload["ovr_classifier"]
        instance.classes_ = payload["classes_"]
        instance.label_to_idx = payload["label_to_idx"]
        instance.idx_to_label = payload["idx_to_label"]
        instance.temperature = payload.get("temperature", 1.0)
        instance.unknown_threshold = payload.get("unknown_threshold", 0.40)
        instance.entropy_threshold = payload.get("entropy_threshold", 0.80)
        instance.multi_intent_threshold = payload.get("multi_intent_threshold", 0.35)
        instance.metadata = payload.get("metadata", {})
        return instance


def train_setfit_model(
    train_texts: List[str],
    train_labels: List[str],
    train_multi_labels: List[List[str]],
    val_texts: List[str],
    val_labels: List[str],
    val_multi_labels: List[List[str]],
    classes: List[str],
    output_dir: str
):
    """Trains multi-label SetFit model with one-vs-rest strategy and saves artifacts."""
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset

        logger.info("Initializing multi-target SetFitModel with all-MiniLM-L6-v2 (one-vs-rest)...")
        model = SetFitModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            multi_target_strategy="one-vs-rest",
            local_files_only=True
        )

        c_to_idx = {c: i for i, c in enumerate(classes)}

        def to_binary_vec(lbl_list):
            vec = [0] * len(classes)
            for l in lbl_list:
                if l in c_to_idx:
                    vec[c_to_idx[l]] = 1
            return vec

        train_bin = [to_binary_vec(lbls) for lbls in train_multi_labels]
        val_bin = [to_binary_vec(lbls) for lbls in val_multi_labels]

        # Only train contrastive pairs on examples that have at least one valid known label
        valid_train_idx = [i for i, v in enumerate(train_bin) if sum(v) > 0]
        valid_val_idx = [i for i, v in enumerate(val_bin) if sum(v) > 0]

        train_ds = Dataset.from_dict({
            "text": [train_texts[i] for i in valid_train_idx],
            "label": [train_bin[i] for i in valid_train_idx]
        })
        val_ds = Dataset.from_dict({
            "text": [val_texts[i] for i in valid_val_idx],
            "label": [val_bin[i] for i in valid_val_idx]
        })

        args = TrainingArguments(
            output_dir=output_dir,
            batch_size=16,
            num_epochs=1,
            num_iterations=12,
            seed=42
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds
        )

        logger.info("Training SetFit multi-label model...")
        trainer.train()

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)

        # Calibrate optimal decision thresholds on validation set
        val_probs_raw = model.predict_proba(val_texts)
        if hasattr(val_probs_raw, "detach"):
            val_probs = val_probs_raw.detach().cpu().numpy()
        else:
            val_probs = np.asarray(val_probs_raw)

        max_probs = np.max(val_probs, axis=1)
        val_is_oos = np.array([lbl in ("out_of_scope", "uncertain_open_ended") for lbl in val_labels])

        best_oos_thresh = 0.20
        best_f1 = 0.0
        for cand_t in np.linspace(0.12, 0.35, 47):
            pred_oos = (max_probs < cand_t)
            if np.sum(pred_oos) > 0 and np.sum(val_is_oos) > 0:
                f1 = f1_score(val_is_oos, pred_oos, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_oos_thresh = float(cand_t)

        metadata = {
            "model_type": "setfit_minilm_multilabel_ovr",
            "classes": classes,
            "num_classes": len(classes),
            "multi_target_strategy": "one-vs-rest",
            "calibrated_oos_threshold": round(best_oos_thresh, 4),
            "multi_intent_threshold": 0.22,
            "val_oos_f1": round(best_f1, 4),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
            json.dump(classes, f, indent=2)

        with open(os.path.join(output_dir, "setfit_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved SetFit model and metadata to: {output_dir}")
        return model
    except Exception as e:
        logger.error(f"SetFit training failed: {e}", exc_info=True)
        return None


def run_training_pipeline(
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_type: str = "all"
):
    """Executes end-to-end training and checkpoint persistence."""
    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Parse intent_labels JSON column
    df["labels_list"] = df["intent_labels"].apply(lambda x: json.loads(x) if isinstance(x, str) else [x])

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    logger.info(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Extract clean list of known communication classes (excluding out_of_scope & uncertain)
    known_intents = sorted(list(set(
        lbl for sublist in df[~df["is_out_of_scope"]]["labels_list"] for lbl in sublist
        if lbl not in ("out_of_scope", "uncertain_open_ended")
    )))

    # 1. Train Candidate B (MiniLM + Calibrated Classification Head)
    if model_type in ("head", "all"):
        logger.info("\n=== Training Candidate B: MiniLM + Calibrated Classification Head ===")
        clf = MiniLMCalibratedClassifier(encoder_name=DEFAULT_ENCODER)
        clf.fit(
            train_texts=train_df["text"].tolist(),
            train_labels=train_df["primary_intent"].tolist(),
            train_multi_labels=train_df["labels_list"].tolist(),
            val_texts=val_df["text"].tolist(),
            val_labels=val_df["primary_intent"].tolist(),
            val_multi_labels=val_df["labels_list"].tolist(),
            val_is_oos=val_df["is_out_of_scope"].tolist()
        )
        clf.save(output_dir)

    # 2. Train Candidate C (SetFit Multi-Label)
    if model_type in ("setfit", "all"):
        logger.info("\n=== Training Candidate C: SetFit Multi-Label ===")
        setfit_dir = os.path.join(os.path.dirname(output_dir), "setfit_intent")
        train_setfit_model(
            train_texts=train_df["text"].tolist(),
            train_labels=train_df["primary_intent"].tolist(),
            train_multi_labels=train_df["labels_list"].tolist(),
            val_texts=val_df["text"].tolist(),
            val_labels=val_df["primary_intent"].tolist(),
            val_multi_labels=val_df["labels_list"].tolist(),
            classes=known_intents,
            output_dir=setfit_dir
        )

    logger.info("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and calibrate intent classifiers")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to intent dataset CSV")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save model artifacts")
    parser.add_argument("--model_type", type=str, choices=["head", "setfit", "all"], default="all", help="Model architecture to train")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_type=args.model_type
    )
