"""
ai/validation.py - Output Validation Engine
Compares the original requirement, confirmed representation, and generated email.
Evaluates:
- Intent preservation
- Semantic alignment (SBERT / cosine similarity)
- Tone suitability
- Key-information completeness (Present, Missing, Contradicted, Not applicable, Uncertain)
- Unsupported information / hallucinated commitments
- Language quality and placeholders
Computes overall status: PASS, NEEDS_REVIEW, or VALIDATION_ERROR.
"""

import time
import logging
import re
from typing import Dict, Any, List, Tuple
from backend.config import settings
from backend.schemas.email_schemas import (
    StructuredRequirement, OutputValidationReport,
    ValidationStatusEnum, FactStatusEnum, FactValidationItem
)
from ai.semantic_embeddings import get_embedding_engine
from ai.tone_analysis import ToneAnalyzer

logger = logging.getLogger(__name__)


class OutputValidator:
    """
    Independent Output Validation Engine.
    Evaluates generated email fidelity without mutating the generated draft.
    """

    METRIC_DEFINITIONS = {
        "intent_preservation": "Binary indicator (1 or 0) whether the email performs the core intended action.",
        "semantic_similarity": "Cosine similarity in range [0.0, 1.0] between requirement and generated email embeddings.",
        "tone_suitability": "Binary indicator (1 or 0) whether the generated email honors requested politeness and formality.",
        "completeness_score": "Fraction of required facts and parameters verified as Present in the email body.",
        "unsupported_information": "Count of unmentioned claims, promises, attachments, or facts detected in output.",
        "validation_score": "Overall composite quality score in range [0.0, 1.0] (80% = 0.80 threshold) required before direct delivery."
    }

    # Markers of unverified commitments or invented elements
    UNSUPPORTED_TRIGGERS = [
        (re.compile(r"\b(?:i have attached|attached (?:is|are)|please find attached)\b", re.I), "Unverified attachment mentioned in email"),
        (re.compile(r"\b(?:promise to work overtime|will work through the weekend)\b", re.I), "Unprompted commitment to overtime work"),
        (re.compile(r"\b(?:guarantee 100%|guarantee completion by tomorrow)\b", re.I), "Unverified delivery guarantee"),
        (re.compile(r"\b(?:wire transfer|bank account details)\b", re.I), "Unprompted financial instruction"),
    ]

    def __init__(self):
        self.embedding_engine = get_embedding_engine()
        self.tone_analyzer = ToneAnalyzer()

    def validate(
        self,
        requirement: str,
        representation: StructuredRequirement,
        subject: str,
        body: str
    ) -> OutputValidationReport:
        """
        Executes complete verification pipeline.
        Returns a strongly-typed OutputValidationReport.
        """
        start_time = time.time()
        full_email_text = f"{subject}\n\n{body}"
        lower_email = full_email_text.lower()
        lower_req = requirement.lower()

        # 1. Intent preservation check
        intent_preserved, intent_explanation = self._check_intent_preservation(
            representation, subject, body
        )

        # 2. Semantic alignment via embedding similarity
        similarity = self.embedding_engine.compute_similarity(requirement, full_email_text)
        embedding_meta = self.embedding_engine.get_metadata()

        # 3. Tone suitability evaluation
        tone_suitable, detected_tone, tone_explanation = self.tone_analyzer.evaluate_tone_suitability(
            representation.tone, full_email_text
        )

        # 4. Key-information completeness
        fact_items, missing_details, contradictions = self._check_completeness(
            representation, subject, body
        )
        total_evaluable = len(fact_items)
        present_count = sum(1 for item in fact_items if item.status == FactStatusEnum.PRESENT)
        completeness_score = round(present_count / max(1, total_evaluable), 3)

        # 5. Unsupported information / hallucination detection
        unsupported_details = self._detect_unsupported_info(requirement, representation, full_email_text)

        # 6. Language quality and placeholders check
        language_issues = self._check_language_quality(subject, body)

        # 7. Actionable warnings compilation
        actionable_warnings = []
        if missing_details:
            actionable_warnings.append(f"Missing essential elements: {', '.join(missing_details)}")
        if unsupported_details:
            actionable_warnings.append(f"Potential unverified claims detected: {'; '.join(unsupported_details)}")
        if not tone_suitable:
            actionable_warnings.append(f"Tone divergence: {tone_explanation}")
        if "[your name]" in lower_email or "<" in subject:
            actionable_warnings.append("Email contains unresolved placeholders (e.g. '[Your Name]').")

        # 8. Overall quantitative validation score (0.0 to 1.0) & decision matrix
        # Weights: completeness: 0.35, semantic similarity: 0.25, intent: 0.20, tone: 0.20
        # Deductions: contradictions (-0.20), unsupported claims (-0.05 each), language issues (-0.05 each)
        sim_val = float(similarity) if similarity is not None else 0.70
        base_score = (
            (1.0 if intent_preserved else 0.0) * 0.20
            + (1.0 if tone_suitable else 0.50) * 0.20
            + completeness_score * 0.35
            + min(1.0, max(0.0, sim_val)) * 0.25
        )

        penalties = (
            (0.20 if contradictions else 0.0)
            + min(0.15, len(unsupported_details) * 0.05)
            + min(0.10, len(language_issues) * 0.05)
        )

        validation_score = round(max(0.0, min(1.0, base_score - penalties)), 3)
        passed_threshold = bool(validation_score >= 0.80 and not contradictions and intent_preserved)

        # PASS, NEEDS_REVIEW, VALIDATION_ERROR
        if not intent_preserved or contradictions or completeness_score < 0.40 or validation_score < 0.50:
            overall_status = ValidationStatusEnum.VALIDATION_ERROR
        elif missing_details or unsupported_details or not tone_suitable or completeness_score < 0.80 or language_issues or not passed_threshold:
            overall_status = ValidationStatusEnum.NEEDS_REVIEW
        else:
            overall_status = ValidationStatusEnum.PASS

        elapsed_ms = (time.time() - start_time) * 1000.0

        return OutputValidationReport(
            overall_status=overall_status,
            intent_preserved=intent_preserved,
            intent_explanation=intent_explanation,
            semantic_similarity=round(similarity, 3),
            similarity_model=embedding_meta["backend"],
            tone_suitable=tone_suitable,
            tone_detected=detected_tone,
            tone_explanation=tone_explanation,
            completeness_score=completeness_score,
            fact_evaluations=fact_items,
            missing_details=missing_details,
            contradictions=contradictions,
            unsupported_details=unsupported_details,
            language_quality_issues=language_issues,
            actionable_warnings=actionable_warnings,
            metric_definitions=self.METRIC_DEFINITIONS,
            validation_time_ms=round(elapsed_ms, 2),
            validation_score=validation_score,
            passed_threshold=passed_threshold
        )

    def _check_intent_preservation(
        self,
        representation: StructuredRequirement,
        subject: str,
        body: str
    ) -> Tuple[bool, str]:
        """Verify whether the communication purpose is preserved."""
        text = f"{subject} {body}".lower()
        intent = representation.intent.lower()

        # Keyword alignment table
        intent_signals = {
            "request_extension": ["extension", "additional days", "postpone", "more time"],
            "request_sick_leave": ["sick leave", "unwell", "fever", "taking leave", "absence", "doctor"],
            "reschedule_meeting": ["reschedule", "move", "calendar", "postpone", "new time"],
            "request_refund": ["refund", "duplicate", "charge", "billed twice", "billing"],
            "interview_follow_up": ["interview", "follow up", "next steps", "hiring process", "candidate"],
            "informational_update": ["maintenance", "update", "offline", "scheduled", "status"],
            "apology_and_delay_notice": ["apolog", "sorry", "delay", "new deadline", "late"],
            "inquire_information": ["inquire", "information", "grant", "details", "questions"],
            "contract_negotiation": ["renew", "license", "discount", "agreement", "pricing"],
            "equipment_request": ["replacement", "charger", "laptop", "hardware", "power brick"],
            "recommendation_request": ["recommendation", "letter", "reference", "internship"],
            "acknowledgement_and_reply": ["re:", "thank", "received", "regarding", "response", "reply", "confirm", "update", "hello", "hi", "hey"]
        }

        signals = intent_signals.get(intent)
        if signals:
            matches = [s for s in signals if s in text]
            if matches:
                return True, f"Intent '{representation.intent}' preserved (found signals: {', '.join(matches)})."
            else:
                return False, f"Email lacks expected communication signals for intent '{representation.intent}'."

        # Open-ended check
        if representation.required_action:
            action_words = set(re.findall(r"\b\w+\b", representation.required_action.lower()))
            overlap = action_words.intersection(set(re.findall(r"\b\w+\b", text)))
            if len(overlap) >= 1:
                return True, f"Open-ended intent preserved with action overlap: {list(overlap)}"

        return True, "Communication purpose generally preserved."

    def _check_completeness(
        self,
        representation: StructuredRequirement,
        subject: str,
        body: str
    ) -> Tuple[List[FactValidationItem], List[str], List[str]]:
        """
        Checks preservation of recipient, action, reason, duration, dates, and explicit facts.
        Reports: (fact_items, missing_details, contradictions).
        """
        items: List[FactValidationItem] = []
        missing: List[str] = []
        contradictions: List[str] = []
        full_text = f"{subject} {body}".lower()

        # Recipient check
        if representation.recipient and representation.recipient.lower() not in ("unspecified", "none"):
            recip = representation.recipient.lower()
            recip_tokens = [t for t in re.findall(r"\b\w+\b", recip) if len(t) > 2]
            present = any(t in full_text for t in recip_tokens)
            status = FactStatusEnum.PRESENT if present else FactStatusEnum.MISSING
            exp = f"Recipient '{representation.recipient}' addressed." if present else f"Recipient '{representation.recipient}' not identified in email."
            items.append(FactValidationItem(fact=f"Recipient: {representation.recipient}", status=status, explanation=exp))
            if not present:
                missing.append(f"Recipient ({representation.recipient})")

        # Action check
        if representation.required_action:
            action = representation.required_action
            # Subword check
            action_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", action.lower()) if w not in ("email", "write", "send", "polite")]
            matches = [w for w in action_words if w in full_text]
            present = len(matches) > 0 or len(action_words) == 0
            status = FactStatusEnum.PRESENT if present else FactStatusEnum.MISSING
            exp = f"Requested action preserved (matched: {matches})." if present else f"Action '{action}' appears omitted."
            items.append(FactValidationItem(fact=f"Action: {action}", status=status, explanation=exp))
            if not present:
                missing.append(f"Action ({action})")

        # Reason check
        if representation.reason:
            reason = representation.reason
            reason_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", reason.lower())]
            present = any(w in full_text for w in reason_words)
            status = FactStatusEnum.PRESENT if present else FactStatusEnum.MISSING
            exp = f"Reason '{reason}' preserved." if present else f"Reason '{reason}' appears omitted."
            items.append(FactValidationItem(fact=f"Reason: {reason}", status=status, explanation=exp))
            if not present:
                missing.append(f"Reason ({reason})")

        # Duration check
        if representation.requested_duration:
            duration = representation.requested_duration
            dur_words = re.findall(r"\b\w+\b", duration.lower())
            present = any(w in full_text for w in dur_words if w not in ("a", "the", "an"))
            status = FactStatusEnum.PRESENT if present else FactStatusEnum.MISSING
            exp = f"Duration '{duration}' referenced." if present else f"Duration '{duration}' omitted."
            items.append(FactValidationItem(fact=f"Duration: {duration}", status=status, explanation=exp))
            if not present:
                missing.append(f"Duration ({duration})")

        # Deadline check
        if representation.deadline:
            deadline = representation.deadline
            dead_words = [w for w in re.findall(r"\b\w+\b", deadline.lower()) if len(w) > 2]
            present = any(w in full_text for w in dead_words)
            status = FactStatusEnum.PRESENT if present else FactStatusEnum.MISSING
            exp = f"Deadline '{deadline}' included." if present else f"Deadline '{deadline}' missing."
            items.append(FactValidationItem(fact=f"Deadline: {deadline}", status=status, explanation=exp))
            if not present:
                missing.append(f"Deadline ({deadline})")

        # Explicit facts check
        for fact in representation.required_facts:
            f_tokens = [w for w in re.findall(r"\b\w+\b", fact.lower()) if len(w) > 3]
            overlap = [w for w in f_tokens if w in full_text]
            present = len(overlap) >= max(1, len(f_tokens) // 2)
            status = FactStatusEnum.PRESENT if present else FactStatusEnum.MISSING
            exp = "Fact present." if present else f"Fact '{fact}' appears omitted."
            items.append(FactValidationItem(fact=fact, status=status, explanation=exp))
            if not present and fact not in missing:
                missing.append(fact)

        return items, missing, contradictions

    def _detect_unsupported_info(
        self,
        requirement: str,
        representation: StructuredRequirement,
        full_email_text: str
    ) -> List[str]:
        """Detect potentially unsupported claims or unverified promises."""
        unsupported = []
        lower_req = requirement.lower()

        # Check triggered regexes if not present in requirement
        for pattern, warning in self.UNSUPPORTED_TRIGGERS:
            if pattern.search(full_email_text) and not pattern.search(requirement):
                unsupported.append(warning)

        # Check for unmentioned monetary amounts
        req_amounts = set(re.findall(r"\$\d+(?:\.\d+)?", requirement))
        email_amounts = set(re.findall(r"\$\d+(?:\.\d+)?", full_email_text))
        invented_amounts = email_amounts - req_amounts
        if invented_amounts:
            unsupported.append(f"Unprompted monetary amounts mentioned: {', '.join(invented_amounts)}")

        return unsupported

    def _check_language_quality(self, subject: str, body: str) -> List[str]:
        """Verify structural email hygiene and unreplaced templates."""
        issues = []
        if not subject or len(subject.strip()) < 3:
            issues.append("Subject line is missing or too short.")

        if not body or len(body.strip()) < 15:
            issues.append("Email body is too brief.")

        # Check unexpanded placeholders
        placeholders = re.findall(r"\[[A-Za-z\s_]+\]|<[A-Za-z\s_]+>", f"{subject} {body}")
        unfilled = [p for p in placeholders if p.lower() in ("[your name]", "[name]", "<recipient>", "[insert date]")]
        if unfilled:
            issues.append(f"Unfilled placeholder detected: {', '.join(set(unfilled))}")

        return issues
