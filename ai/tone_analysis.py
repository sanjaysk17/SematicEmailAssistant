"""
ai/tone_analysis.py - Tone, Politeness, and Formality Analysis
Distinguishes between politeness, assertiveness, urgency, and formality.
Analyzes requested tone from requirement and evaluates tone suitability of generated emails.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class ToneAnalyzer:
    """
    Analyzes tone across multidimensional linguistic markers:
    formality, politeness, assertiveness, urgency, and sentiment.
    """

    POLITENESS_MARKERS = [
        "please", "kindly", "thank you", "thanks", "appreciate", "grateful",
        "would you be able", "could you please", "if possible", "respectfully",
        "sincerely", "warm regards", "best regards", "apologies", "pardon"
    ]

    FORMALITY_MARKERS = [
        "furthermore", "consequently", "regarding", "respectfully", "inquire",
        "assistance", "schedule", "commence", "opportunity", "reimbursement",
        "investigation", "sincerely", "dear", "deliberation"
    ]

    INFORMALITY_MARKERS = [
        "hey", "hi guys", "gonna", "wanna", "stuff", "weird", "asap",
        "lol", "btw", "cool", "super", "yep", "nope"
    ]

    ASSERTIVENESS_MARKERS = [
        "require", "must", "insist", "expect", "necessary", "promptly",
        "unacceptable", "resolve", "rectify", "firmly"
    ]

    URGENCY_MARKERS = [
        "urgent", "urgently", "asap", "immediate", "immediately", "critical",
        "time-sensitive", "without delay", "promptly"
    ]

    def extract_requested_tone(self, text: str, user_hint: Optional[str] = None) -> str:
        """
        Determines the intended tone requested by user.
        Prioritizes explicit tone hints or explicit mentions in requirement text.
        """
        if user_hint and user_hint.strip():
            return user_hint.strip()

        lower = text.lower()
        if "firm but polite" in lower or "firm and polite" in lower:
            return "firm and polite"
        if "polite" in lower and "courteous" in lower:
            return "polite and courteous"
        if "polite" in lower:
            return "polite and professional"
        if "courteous" in lower:
            return "courteous and urgent"
        if "appreciative" in lower:
            return "appreciative and professional"
        if "apologize" in lower or "apology" in lower:
            return "apologetic and professional"
        if "assertive" in lower or "firm" in lower:
            return "firm and professional"
        if "casual" in lower or "informal" in lower:
            return "friendly and casual"
        if "formal" in lower:
            return "formal and respectful"

        # Default professional register
        return "polite and professional"

    def analyze_text_tone(self, text: str) -> Dict[str, Any]:
        """
        Computes tone metrics for a given text snippet (subject + body).
        """
        if not text:
            return {
                "primary_tone": "neutral",
                "formality_score": 0.5,
                "politeness_score": 0.5,
                "urgency_score": 0.0,
                "assertiveness_score": 0.0,
                "detected_markers": []
            }

        lower = text.lower()
        words = re.findall(r"\b[a-zA-Z']+\b", lower)
        word_count = max(1, len(words))

        # Count markers
        polite_hits = [m for m in self.POLITENESS_MARKERS if m in lower]
        formal_hits = [m for m in self.FORMALITY_MARKERS if m in lower]
        informal_hits = [m for m in self.INFORMALITY_MARKERS if m in lower]
        assertive_hits = [m for m in self.ASSERTIVENESS_MARKERS if m in lower]
        urgent_hits = [m for m in self.URGENCY_MARKERS if m in lower]

        # Normalized scores [0.0, 1.0]
        politeness_score = min(1.0, len(polite_hits) / 3.0)
        formality_score = min(1.0, (len(formal_hits) + 2) / (len(informal_hits) * 3 + 4))
        urgency_score = min(1.0, len(urgent_hits) / 2.0)
        assertiveness_score = min(1.0, len(assertive_hits) / 2.0)

        # Classify primary tone label
        if politeness_score >= 0.6 and formality_score >= 0.6:
            primary_tone = "polite and professional"
        elif assertiveness_score >= 0.5 and politeness_score >= 0.4:
            primary_tone = "firm and polite"
        elif "apolog" in lower or "regret" in lower:
            primary_tone = "apologetic and professional"
        elif "appreciat" in lower or "thank" in lower:
            primary_tone = "appreciative and professional"
        elif urgency_score >= 0.5:
            primary_tone = "courteous and urgent"
        elif formality_score < 0.4:
            primary_tone = "casual and conversational"
        else:
            primary_tone = "professional"

        return {
            "primary_tone": primary_tone,
            "formality_score": round(formality_score, 2),
            "politeness_score": round(politeness_score, 2),
            "urgency_score": round(urgency_score, 2),
            "assertiveness_score": round(assertiveness_score, 2),
            "detected_markers": polite_hits + formal_hits + assertive_hits
        }

    def evaluate_tone_suitability(self, requested_tone: str, generated_email_text: str) -> Tuple[bool, str, str]:
        """
        Determines whether the generated email honors the requested tone.
        Returns (is_suitable, detected_tone, explanation).
        """
        analysis = self.analyze_text_tone(generated_email_text)
        detected_tone = analysis["primary_tone"]
        req_lower = requested_tone.lower()
        det_lower = detected_tone.lower()

        is_suitable = True
        explanation = f"Generated email matches requested '{requested_tone}' style."

        # Check for politeness violation
        if ("polite" in req_lower or "courteous" in req_lower) and analysis["politeness_score"] < 0.2:
            is_suitable = False
            explanation = "Requested polite tone, but email lacks courtesy expressions ('please', 'thank you', 'appreciate')."

        # Check for excessive informality in professional requests
        if ("professional" in req_lower or "formal" in req_lower) and analysis["formality_score"] < 0.35:
            is_suitable = False
            explanation = "Requested professional tone, but email contains overly casual or colloquial expressions."

        # Check for missing apologetic markers
        if "apologetic" in req_lower and "apolog" not in generated_email_text.lower() and "sorry" not in generated_email_text.lower():
            is_suitable = False
            explanation = "Requested apologetic tone, but email omits explicit apology phrasing."

        return is_suitable, detected_tone, explanation
