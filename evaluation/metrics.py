"""
evaluation/metrics.py - Research Evaluation Metrics Module
Defines mathematical formulas, denominators, and aggregation routines
for all benchmark performance metrics.
"""

from typing import List, Dict, Any
import numpy as np
from backend.schemas.evaluation_schemas import SingleResultRecord, MethodMetrics


class ResearchMetricsCalculator:
    """
    Computes rigorous statistical metrics across benchmark runs:
    - Intent Preservation Rate = sum(intent_preserved) / N
    - Mean Completeness = sum(completeness_score) / N
    - Mean Semantic Similarity = sum(cosine_similarity) / N
    - Tone Suitability Rate = sum(tone_suitable) / N
    - Unsupported Information Rate = sum(unsupported_count > 0) / N
    - Pass Rate = sum(status == 'PASS') / N
    - Mean Latency = sum(latency_ms) / N
    """

    METRIC_FORMULAS = {
        "intent_preservation_rate": "Fraction of emails preserving the intended communication purpose (Denominator: total examples N).",
        "mean_completeness": "Average completeness score across examples (Range [0.0, 1.0]; Denominator: total examples N).",
        "mean_semantic_similarity": "Average cosine embedding similarity between requirement and email (Denominator: total examples N).",
        "tone_suitability_rate": "Fraction of emails adhering to requested tone and formality constraints (Denominator: total examples N).",
        "unsupported_information_rate": "Fraction of emails containing at least one unprompted claim or promise (Denominator: total examples N).",
        "pass_rate": "Fraction of generated emails achieving full validation PASS (Denominator: total examples N)."
    }

    @classmethod
    def aggregate_method_results(cls, method: str, records: List[SingleResultRecord]) -> MethodMetrics:
        """Calculates aggregate metrics for a specific method across all evaluated samples."""
        n = len(records)
        if n == 0:
            return MethodMetrics(
                method=method,
                sample_count=0,
                intent_preservation_rate=0.0,
                mean_completeness=0.0,
                mean_semantic_similarity=0.0,
                tone_suitability_rate=0.0,
                unsupported_information_rate=0.0,
                mean_latency_ms=0.0,
                pass_rate=0.0
            )

        intent_count = sum(1 for r in records if r.intent_preserved)
        completeness_sum = sum(r.completeness_score for r in records)
        similarity_sum = sum(r.semantic_similarity for r in records)
        tone_count = sum(1 for r in records if r.tone_suitable)
        unsupported_with_flaws = sum(1 for r in records if r.unsupported_count > 0)
        pass_count = sum(1 for r in records if r.validation_status == "PASS")
        latency_sum = sum(r.latency_ms for r in records)

        return MethodMetrics(
            method=method,
            sample_count=n,
            intent_preservation_rate=round(intent_count / n, 4),
            mean_completeness=round(completeness_sum / n, 4),
            mean_semantic_similarity=round(similarity_sum / n, 4),
            tone_suitability_rate=round(tone_count / n, 4),
            unsupported_information_rate=round(unsupported_with_flaws / n, 4),
            mean_latency_ms=round(latency_sum / n, 2),
            pass_rate=round(pass_count / n, 4)
        )
