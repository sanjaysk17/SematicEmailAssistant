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

    def extract_requested_tone(self, text: str, user_hint: Optional[str] = None, intent: Optional[str] = None) -> str:
        """
        Determines the intended tone requested by user.
        Prioritizes:
        1. Explicit user tone hint (if not 'default' or 'auto-detect').
        2. Explicit tone keywords/instructions in requirement text (e.g. casual, friendly, warm, informal, polite, formal, urgent).
        3. Intent-based classification when user does not explicitly request a tone (e.g. informal_invitation_or_catchup -> friendly and casual).
        4. Default professional register.
        """
        if user_hint and user_hint.strip() and user_hint.strip().lower() not in ("default", "auto-detect", "auto", "none"):
            return user_hint.strip()

        lower = text.lower()

        # 1. Explicit user tone directives in text
        # Friendly / Casual / Warm / Informal directives
        if re.search(r"\b(casual|friendly|informal|warm|relaxed|colloquial|chatty)\b", lower):
            return "friendly and casual"

        # Firm / Assertive directives
        if "firm but polite" in lower or "firm and polite" in lower:
            return "firm and polite"
        if re.search(r"\b(assertive|firm|resolute|boundary)\b", lower):
            if "polite" in lower or "courteous" in lower:
                return "firm and polite"
            return "firm and professional"

        # Polite / Courteous directives
        if "polite and courteous" in lower or ("polite" in lower and "courteous" in lower):
            return "polite and courteous"
        if "courteous and urgent" in lower or ("courteous" in lower and "urgent" in lower):
            return "courteous and urgent"

        # Urgent / Critical directives
        if re.search(r"\b(urgent|urgently|time-sensitive|critical|emergency|asap)\b", lower):
            if "polite" in lower or "courteous" in lower:
                return "courteous and urgent"
            return "urgent and professional"

        # Apologetic directives
        if re.search(r"\b(apologetic|apology|apologize|apologies|regretful|sorry)\b", lower):
            return "apologetic and professional"

        # Appreciative / Gratitude directives
        if re.search(r"\b(appreciative|appreciation|grateful|gratitude|thankful)\b", lower):
            return "appreciative and professional"

        # Formal / Respectful directives
        if re.search(r"\b(strictly\s+formal|formal|diplomatic|respectful|official)\b", lower):
            return "formal and respectful"

        # Explicit polite directive
        if re.search(r"\b(polite|courteous)\b", lower):
            return "polite and professional"

        # 2. When user did NOT explicitly specify a tone in the requirement,
        # correctly classify tone based on communication intent / domain
        if intent:
            intent_lower = intent.lower()
            if intent_lower in ("informal_invitation_or_catchup", "invitation_catchup", "social_catchup", "peer_invitation"):
                return "friendly and casual"
            if intent_lower in ("billing_dispute", "complaint_or_issue", "escalation"):
                return "firm and polite"
            if intent_lower in ("urgent_escalation", "incident_alert"):
                return "urgent and professional"
            if intent_lower in ("apology", "service_outage_notice"):
                return "apologetic and professional"
            if intent_lower in ("gratitude", "thank_you", "appreciation"):
                return "appreciative and professional"
            if intent_lower in ("formal_business_letter", "recommendation_request", "legal_notice"):
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
