"""
evaluation/experiments.py - Experiment Runner and Coordinator
Executes benchmark evaluations, collects per-example records, computes aggregates,
and coordinates CSV export for reproducible research.
"""

import os
import sys

# Ensure project root is on sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.schemas.evaluation_schemas import (
    EvaluationExample, SingleResultRecord, MethodMetrics,
    ExperimentRunRequest, ExperimentRunResponse
)
from evaluation.dataset_loader import DatasetLoader
from evaluation.baselines import BaselineEvaluator
from evaluation.metrics import ResearchMetricsCalculator
from evaluation.export_results import ResultExporter
from backend.services.storage_service import StorageService


class ExperimentRunner:
    """Manages execution of research evaluations across baseline architectures."""

    def __init__(self):
        self.evaluator = BaselineEvaluator()

    def run_experiment(
        self,
        request: ExperimentRunRequest,
        db: Optional[Session] = None
    ) -> ExperimentRunResponse:
        """
        Runs the specified experiment configuration over the evaluation dataset.
        """
        start_time = time.time()
        experiment_id = str(uuid.uuid4())

        # 1. Load dataset examples
        examples = DatasetLoader.load_dataset(
            split=request.split_filter,
            max_samples=request.max_examples
        )

        all_results: List[SingleResultRecord] = []
        method_records_map: Dict[str, List[SingleResultRecord]] = {
            m: [] for m in request.methods_to_run
        }

        # 2. Iterate through examples for each selected method
        for ex in examples:
            for method in request.methods_to_run:
                if method == "baseline_a":
                    out = self.evaluator.run_baseline_a(ex)
                elif method == "baseline_b":
                    out = self.evaluator.run_baseline_b(ex)
                elif method in ("proposed", "proposed_system"):
                    out = self.evaluator.run_proposed(ex)
                else:
                    continue

                report = out["report"]
                rec = SingleResultRecord(
                    example_id=ex.id,
                    domain=ex.domain,
                    method=method,
                    subject=out["subject"],
                    body=out["body"],
                    intent_preserved=report.intent_preserved,
                    completeness_score=report.completeness_score,
                    semantic_similarity=report.semantic_similarity,
                    tone_suitable=report.tone_suitable,
                    unsupported_count=len(report.unsupported_details),
                    validation_status=report.overall_status.value,
                    latency_ms=out["latency_ms"],
                    missing_items=report.missing_details,
                    unsupported_items=report.unsupported_details
                )

                all_results.append(rec)
                method_records_map[method].append(rec)

                # Persist to DB if session provided
                if db:
                    StorageService.save_experiment_result(
                        db=db,
                        experiment_id=experiment_id,
                        example_id=ex.id,
                        method=method,
                        subject=out["subject"],
                        body=out["body"],
                        intent_score=1.0 if report.intent_preserved else 0.0,
                        completeness_score=report.completeness_score,
                        semantic_similarity=report.semantic_similarity,
                        tone_score=1.0 if report.tone_suitable else 0.0,
                        unsupported_count=len(report.unsupported_details),
                        latency_ms=out["latency_ms"],
                        status=report.overall_status.value
                    )

        # 3. Compute aggregate metrics
        aggregates: Dict[str, MethodMetrics] = {}
        for method, records in method_records_map.items():
            aggregates[method] = ResearchMetricsCalculator.aggregate_method_results(method, records)

        # 4. Export results to CSV
        csv_filename = ResultExporter.export_to_csv(all_results, experiment_id=experiment_id)

        # Save experiment master record in DB
        if db:
            StorageService.save_experiment(
                db=db,
                experiment_name=request.experiment_name,
                description=f"Methods: {', '.join(request.methods_to_run)}, Filter: {request.split_filter}",
                baseline_type=",".join(request.methods_to_run),
                dataset_name="synthetic_examples",
                sample_count=len(examples),
                config_params={"methods": request.methods_to_run, "max": request.max_examples}
            )

        elapsed_total = round(time.time() - start_time, 2)

        return ExperimentRunResponse(
            experiment_id=experiment_id,
            experiment_name=request.experiment_name,
            total_examples_evaluated=len(examples),
            methods_evaluated=request.methods_to_run,
            aggregate_metrics=aggregates,
            detailed_results=all_results,
            runtime_seconds=elapsed_total,
            export_csv_filename=csv_filename
        )


if __name__ == "__main__":
    import argparse
    from backend.database.connection import get_db_context

    parser = argparse.ArgumentParser(description="Run Email Assistant Evaluation Benchmarks")
    parser.add_argument("--name", type=str, default="CLI Benchmark Run", help="Experiment name")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("--split", type=str, default="all", help="Dataset split filter: all, test, dev, train")
    parser.add_argument(
        "--methods",
        type=str,
        default="baseline_a,baseline_b,proposed_system",
        help="Comma-separated list of methods to run"
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    req = ExperimentRunRequest(
        experiment_name=args.name,
        dataset_name="synthetic_examples",
        split_filter=args.split,
        methods_to_run=methods,
        max_examples=args.limit
    )

    print(f"=== Starting Experiment: {args.name} ===")
    print(f"Methods: {methods} | Split: {args.split} | Max Examples: {args.limit}")

    with get_db_context() as db:
        runner = ExperimentRunner()
        response = runner.run_experiment(req, db=db)

    print(f"\nCompleted in {response.runtime_seconds}s. Total Examples: {response.total_examples_evaluated}")
    print(f"Results CSV exported to: {response.export_csv_filename}")
    print("\n--- Aggregate Metrics ---")
    for method, metrics in response.aggregate_metrics.items():
        print(f"\n[{method.upper()}] (n={metrics.sample_count})")
        print(f"  Intent Preservation Rate:  {metrics.intent_preservation_rate:.3f}")
        print(f"  Mean Completeness:         {metrics.mean_completeness:.3f}")
        print(f"  Mean Semantic Similarity:  {metrics.mean_semantic_similarity:.3f}")
        print(f"  Tone Suitability Rate:     {metrics.tone_suitability_rate:.3f}")
        print(f"  Unsupported Info Rate:     {metrics.unsupported_information_rate:.3f}")
        print(f"  Pass Rate:                 {metrics.pass_rate:.3f}")
        print(f"  Mean Latency:              {metrics.mean_latency_ms:.1f}ms")

