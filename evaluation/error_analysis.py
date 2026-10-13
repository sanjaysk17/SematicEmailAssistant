"""
evaluation/error_analysis.py - Error Category Classifier and Diagnostic Utilities
Categorizes failures across omission, hallucination, tone divergence, and intent drift.
"""

from typing import List, Dict, Any
from backend.schemas.evaluation_schemas import SingleResultRecord


class ErrorAnalyzer:
    """Categorizes validation failures into actionable diagnostic taxonomies."""

    @classmethod
    def categorize_errors(cls, results: List[SingleResultRecord]) -> Dict[str, Any]:
        """
        Groups results into error buckets:
        - intent_failures
        - omission_failures (missing required details)
        - hallucination_failures (unsupported additions)
        - tone_divergence_failures
        - syntax_formatting_failures
        """
        summary = {
            "total_evaluated": len(results),
            "intent_failures": [],
            "omission_failures": [],
            "hallucination_failures": [],
            "tone_divergence_failures": [],
            "pass_count": 0
        }

        for r in results:
            if r.validation_status == "PASS":
                summary["pass_count"] += 1
                continue

            if not r.intent_preserved:
                summary["intent_failures"].append({
                    "example_id": r.example_id,
                    "method": r.method,
                    "subject": r.subject
                })

            if r.missing_items:
                summary["omission_failures"].append({
                    "example_id": r.example_id,
                    "method": r.method,
                    "missing": r.missing_items
                })

            if r.unsupported_count > 0:
                summary["hallucination_failures"].append({
                    "example_id": r.example_id,
                    "method": r.method,
                    "unsupported": r.unsupported_items
                })

            if not r.tone_suitable:
                summary["tone_divergence_failures"].append({
                    "example_id": r.example_id,
                    "method": r.method
                })

        return summary
