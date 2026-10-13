"""
evaluation/export_results.py - CSV and Dataframe Exporter
Exports detailed per-example results and aggregate metrics to disk.
"""

import os
import pandas as pd
from typing import List, Optional
from backend.schemas.evaluation_schemas import SingleResultRecord


class ResultExporter:
    """Exports benchmark evaluation runs to CSV files."""

    @classmethod
    def export_to_csv(
        cls,
        results: List[SingleResultRecord],
        output_dir: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> str:
        """
        Serializes results list to a timestamped CSV file and returns the path.
        """
        directory = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )
        os.makedirs(directory, exist_ok=True)

        exp_suffix = f"_{experiment_id[:8]}" if experiment_id else ""
        filename = f"experiment_results{exp_suffix}.csv"
        filepath = os.path.join(directory, filename)

        rows = []
        for r in results:
            rows.append({
                "example_id": r.example_id,
                "domain": r.domain,
                "method": r.method,
                "subject": r.subject,
                "intent_preserved": r.intent_preserved,
                "completeness_score": r.completeness_score,
                "semantic_similarity": r.semantic_similarity,
                "tone_suitable": r.tone_suitable,
                "unsupported_count": r.unsupported_count,
                "validation_status": r.validation_status,
                "latency_ms": r.latency_ms,
                "missing_items": "; ".join(r.missing_items),
                "unsupported_items": "; ".join(r.unsupported_items)
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return filepath
