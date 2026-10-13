"""
backend/schemas/evaluation_schemas.py - Evaluation and Experiment Schemas
Schemas for running research experiments, tracking per-example results,
and calculating aggregate metrics.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    id: str
    split: str = "test"
    domain: str
    user_requirement: str
    expected_intent: str
    expected_tone: str
    recipient: Optional[str] = None
    topic: Optional[str] = None
    required_action: Optional[str] = None
    reason: Optional[str] = None
    important_facts: Optional[str] = None
    date_deadline_duration: Optional[str] = None
    constraints: Optional[str] = None
    expected_key_information: Optional[str] = None
    is_synthetic: bool = True


class SingleResultRecord(BaseModel):
    example_id: str
    domain: str
    method: str  # baseline_a, baseline_b, proposed
    subject: str
    body: str
    intent_preserved: bool
    completeness_score: float
    semantic_similarity: float
    tone_suitable: bool
    unsupported_count: int
    validation_status: str
    latency_ms: float
    missing_items: List[str] = Field(default_factory=list)
    unsupported_items: List[str] = Field(default_factory=list)


class MethodMetrics(BaseModel):
    method: str
    sample_count: int
    intent_preservation_rate: float
    mean_completeness: float
    mean_semantic_similarity: float
    tone_suitability_rate: float
    unsupported_information_rate: float
    mean_latency_ms: float
    pass_rate: float


class ExperimentRunRequest(BaseModel):
    experiment_name: str = "Baseline vs Proposed Comparison"
    methods_to_run: List[str] = Field(
        default=["baseline_a", "baseline_b", "proposed"],
        description="List of systems: baseline_a, baseline_b, proposed"
    )
    max_examples: Optional[int] = Field(default=None, description="Limit examples to run (None for all)")
    split_filter: Optional[str] = Field(default="test", description="Filter dataset by split (test, train, all)")


class ExperimentRunResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    total_examples_evaluated: int
    methods_evaluated: List[str]
    aggregate_metrics: Dict[str, MethodMetrics]
    detailed_results: List[SingleResultRecord]
    runtime_seconds: float
    export_csv_filename: Optional[str] = None
