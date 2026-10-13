"""
evaluation/dataset_loader.py - Dataset Loader and Validator
Loads, filters, and validates experimental benchmark datasets from CSV or JSON.
"""

import os
import pandas as pd
from typing import List, Optional
from backend.schemas.evaluation_schemas import EvaluationExample


class DatasetLoader:
    """Loads and validates evaluation datasets with split filtering."""

    DEFAULT_DATASET_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "synthetic_examples.csv"
    )

    @classmethod
    def load_dataset(
        cls,
        filepath: Optional[str] = None,
        split: Optional[str] = "test",
        max_samples: Optional[int] = None
    ) -> List[EvaluationExample]:
        """
        Loads CSV dataset into validated EvaluationExample models.
        """
        path = filepath or cls.DEFAULT_DATASET_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

        df = pd.read_csv(path)

        # Split filter
        if split and split != "all" and "split" in df.columns:
            df = df[df["split"] == split]

        if max_samples and max_samples > 0:
            df = df.head(max_samples)

        examples: List[EvaluationExample] = []
        for _, row in df.iterrows():
            example = EvaluationExample(
                id=str(row.get("id")),
                split=str(row.get("split", "test")),
                domain=str(row.get("domain", "general")),
                user_requirement=str(row.get("user_requirement")),
                expected_intent=str(row.get("expected_intent")),
                expected_tone=str(row.get("expected_tone")),
                recipient=str(row.get("recipient")) if pd.notna(row.get("recipient")) else None,
                topic=str(row.get("topic")) if pd.notna(row.get("topic")) else None,
                required_action=str(row.get("required_action")) if pd.notna(row.get("required_action")) else None,
                reason=str(row.get("reason")) if pd.notna(row.get("reason")) else None,
                important_facts=str(row.get("important_facts")) if pd.notna(row.get("important_facts")) else None,
                date_deadline_duration=str(row.get("date_deadline_duration")) if pd.notna(row.get("date_deadline_duration")) else None,
                constraints=str(row.get("constraints")) if pd.notna(row.get("constraints")) else None,
                expected_key_information=str(row.get("expected_key_information")) if pd.notna(row.get("expected_key_information")) else None,
                is_synthetic=bool(row.get("is_synthetic", True))
            )
            examples.append(example)

        return examples
