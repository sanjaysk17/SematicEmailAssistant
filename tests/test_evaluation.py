"""
tests/test_evaluation.py - Tests for Dataset Loader, Metrics, and Benchmark Runner
Verifies CSV dataset loading, metric formulas, and experiment runner output.
"""

import os
import pytest
from evaluation.dataset_loader import DatasetLoader
from evaluation.experiments import ExperimentRunner
from evaluation.metrics import ResearchMetricsCalculator
from backend.schemas.evaluation_schemas import ExperimentRunRequest, SingleResultRecord


def test_dataset_loader():
    examples = DatasetLoader.load_dataset(split="all")
    assert len(examples) >= 10
    assert any(ex.domain == "academic" for ex in examples)
    assert any(ex.expected_intent == "uncertain_open_ended" for ex in examples)


def test_metrics_calculator():
    records = [
        SingleResultRecord(
            example_id="EX-1",
            domain="work",
            method="proposed",
            subject="Sub 1",
            body="Body 1",
            intent_preserved=True,
            completeness_score=1.0,
            semantic_similarity=0.75,
            tone_suitable=True,
            unsupported_count=0,
            validation_status="PASS",
            latency_ms=50.0
        ),
        SingleResultRecord(
            example_id="EX-2",
            domain="work",
            method="proposed",
            subject="Sub 2",
            body="Body 2",
            intent_preserved=False,
            completeness_score=0.5,
            semantic_similarity=0.60,
            tone_suitable=True,
            unsupported_count=1,
            validation_status="NEEDS_REVIEW",
            latency_ms=60.0
        )
    ]
    metrics = ResearchMetricsCalculator.aggregate_method_results("proposed", records)
    assert metrics.sample_count == 2
    assert metrics.intent_preservation_rate == 0.5
    assert metrics.mean_completeness == 0.75
    assert metrics.unsupported_information_rate == 0.5
    assert metrics.pass_rate == 0.5


def test_experiment_runner_execution():
    runner = ExperimentRunner()
    req = ExperimentRunRequest(
        experiment_name="Unit Test Experiment",
        methods_to_run=["baseline_a", "proposed"],
        max_examples=2,
        split_filter="test"
    )
    res = runner.run_experiment(req)
    assert res.total_examples_evaluated == 2
    assert "baseline_a" in res.aggregate_metrics
    assert "proposed" in res.aggregate_metrics
    assert os.path.exists(res.export_csv_filename)
