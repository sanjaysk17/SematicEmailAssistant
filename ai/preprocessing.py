"""
ai/preprocessing.py - Text Preprocessing and Normalization
Performs non-destructive text preparation while preserving names, numbers,
dates, negations, quantities, and meaning-bearing expressions.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional


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
        "don't": "do not",
        "didn't": "did not",
        "doesn't": "does not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "ain't": "is not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
    }

    # Prompt injection directives to defang
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+a\s+different\s+ai", re.IGNORECASE),
        re.compile(r"system\s*:\s*override", re.IGNORECASE),
        re.compile(r"format\s*:\s*drop\s+tables", re.IGNORECASE),
        re.compile(r"reveal\s+(api\s+key|system\s+prompt|password)", re.IGNORECASE),
    ]

    # Standard negation tokens (both full words and contraction stems)
    NEGATION_TOKENS = {
        "not", "never", "no", "cannot", "neither", "nor", "none",
        "without", "hardly", "barely", "scarcely", "seldom",
        "don't", "can't", "won't", "didn't", "doesn't", "couldn't",
        "shouldn't", "wouldn't", "isn't", "aren't", "wasn't", "weren't",
        "hasn't", "haven't", "hadn't", "shan't", "ain't"
    }

    def __init__(self):
        # Compile regex for controlled space normalization
        self._whitespace_pattern = re.compile(r"[ \t]+")
        self._newline_pattern = re.compile(r"\n{3,}")
        # Strip zero-width & non-printable control characters except newline and tab
        self._control_chars_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff]")

    def sanitize(self, text: str) -> str:
        """
        Defang prompt-injection directives and hostile boundary escapes
        before passing into LLM prompts or downstream extractors.
        """
        if not text:
            return ""

        cleaned = text.strip()
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(cleaned):
                cleaned = pattern.sub("[FILTERED_COMMAND]", cleaned)

        return cleaned

    def expand_contractions(self, text: str) -> str:
        """
        Safely expand English contractions using word-boundary replacements.
        Preserves capitalized proper nouns and avoids touching standalone single quotes.
        """
        if not text:
            return ""

        expanded = text
        # Whole-word contraction replacements first (e.g. can't -> cannot, won't -> will not)
        for contraction, replacement in self.CONTRACTION_MAP.items():
            if contraction.startswith("'") or contraction.startswith("n't"):
                continue
            pattern = re.compile(rf"\b{re.escape(contraction)}\b", re.IGNORECASE)
            expanded = pattern.sub(replacement, expanded)

        # Contraction suffixes: e.g. haven't -> have not
        expanded = re.sub(r"\b([a-zA-Z]+)n't\b", r"\1 not", expanded, flags=re.IGNORECASE)
        # Handle "can not" -> "cannot"
        expanded = re.sub(r"\bcan not\b", "cannot", expanded, flags=re.IGNORECASE)

        return expanded

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

        # 2. Strip invisible zero-width and control characters
        normalized = self._control_chars_pattern.sub("", normalized)

        # 3. Normalize smart quotes and dashes to standard ASCII
        quotes_map = {
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "—": "-", "–": "-", "…": "...",
            "«": '"', "»": '"'
        }
        for orig, replacement in quotes_map.items():
            normalized = normalized.replace(orig, replacement)

        # 4. Collapse excessive horizontal spaces without destroying single newlines
        lines = [self._whitespace_pattern.sub(" ", line).strip() for line in normalized.splitlines()]
        cleaned = "\n".join(line for line in lines)

        # 5. Collapse 3+ newlines to max 2
        cleaned = self._newline_pattern.sub("\n\n", cleaned)

        return cleaned.strip()

    def process(self, raw_text: str, sanitize: bool = False) -> Dict[str, Any]:
        """
        Full preprocessing pipeline returning metadata and preserved segments.
        Optionally defangs prompt injection directives when sanitize=True.
        """
        original = raw_text or ""
        text_to_clean = self.sanitize(original) if sanitize else original
        cleaned = self.normalize(text_to_clean)

        # Detect negation presence (full words or contraction words)
        words = re.findall(r"\b[a-zA-Z']+\b", cleaned.lower())
        detected_negations = [
            w for w in words
            if w in self.NEGATION_TOKENS or w.endswith("n't")
        ]

        # Detect numbers, currency ($ € £ ¥ ₹), percentages, and quantities with comma support
        numbers = re.findall(
            r"(?:[\$\€\£\¥\₹]\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?%?|\b\d+\b)",
            cleaned
        )

        # Detect calendar dates and temporal anchor patterns
        dates = re.findall(
            r"(?:\b\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}\b|"
            r"\b\d{4}[\.\/\-]\d{1,2}[\.\/\-]\d{1,2}\b|"
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b|"
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b|"
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)",
            cleaned,
            re.IGNORECASE
        )

        return {
            "original_text": original,
            "normalized_text": cleaned,
            "char_count": len(cleaned),
            "word_count": len(words),
            "has_negation": len(detected_negations) > 0,
            "negations_found": sorted(list(set(detected_negations))),
            "quantities_found": numbers,
            "dates_found": dates,
        }

