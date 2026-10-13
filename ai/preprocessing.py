"""
ai/preprocessing.py - Text Preprocessing and Normalization
Performs non-destructive text preparation while preserving names, numbers,
dates, negations, quantities, and meaning-bearing expressions.
"""

import re
import unicodedata
from typing import Dict, Any, List


class TextPreprocessor:
    """
    Standard text preprocessor for email writing requirements.
    Ensures safe normalization without altering semantic intent or discarding
    meaning-bearing elements such as negation ("not", "never", "don't") or numbers/dates.
    """

    # Meaning-critical contractions mapping
    CONTRACTION_MAP = {
        "can't": "cannot",
        "won't": "will not",
        "shan't": "shall not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
    }

    def __init__(self):
        # Compile regex for controlled space normalization
        self._whitespace_pattern = re.compile(r"[ \t]+")
        self._newline_pattern = re.compile(r"\n{3,}")

    def normalize(self, text: str) -> str:
        """
        Normalize Unicode, whitespace, and formatting while strictly preserving:
        - Dates, times, durations
        - Currencies and quantities
        - Capitalized proper nouns
        - Negations and punctuation
        """
        if not text:
            return ""

        # 1. Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Normalize smart quotes and dashes to standard ASCII
        quotes_map = {
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "—": "-", "–": "-", "…": "..."
        }
        for orig, replacement in quotes_map.items():
            normalized = normalized.replace(orig, replacement)

        # 3. Collapse excessive horizontal spaces without destroying single newlines
        lines = [self._whitespace_pattern.sub(" ", line).strip() for line in normalized.splitlines()]
        cleaned = "\n".join(line for line in lines)

        # 4. Collapse 3+ newlines to max 2
        cleaned = self._newline_pattern.sub("\n\n", cleaned)

        return cleaned.strip()

    def process(self, raw_text: str) -> Dict[str, Any]:
        """
        Full preprocessing pipeline returning metadata and preserved segments.
        """
        original = raw_text or ""
        cleaned = self.normalize(original)

        # Detect negation presence
        negation_tokens = {"not", "never", "no", "cannot", "neither", "nor", "none", "without", "hardly", "barely"}
        words = re.findall(r"\b[a-zA-Z']+\b", cleaned.lower())
        detected_negations = [w for w in words if w in negation_tokens]

        # Detect numbers, currency, and quantities
        numbers = re.findall(r"(?:\$\d+(?:\.\d+)?|\b\d+(?:\.\d+)?%?|\b\d+\b)", cleaned)

        return {
            "original_text": original,
            "normalized_text": cleaned,
            "char_count": len(cleaned),
            "word_count": len(words),
            "has_negation": len(detected_negations) > 0,
            "negations_found": list(set(detected_negations)),
            "quantities_found": numbers,
        }
