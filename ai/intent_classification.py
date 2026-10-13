"""
ai/intent_classification.py - Open-Ended and Semantic Intent Interpreter
Interprets communication goals without forcing all requests into a narrow closed taxonomy.
Supports open-ended intent descriptions and explicit uncertain/other outcomes.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from ai.semantic_embeddings import get_embedding_engine

logger = logging.getLogger(__name__)


# Exemplar anchors for major communication categories
INTENT_EXEMPLARS = {
    "request_extension": [
        "request an extension on project deadline",
        "ask for more time to submit assignment",
        "request additional days due to delay",
        "postpone due date for submission"
    ],
    "request_sick_leave": [
        "request sick leave due to illness",
        "inform manager of taking time off sick",
        "call in sick for work today",
        "medical leave notice to supervisor"
    ],
    "reschedule_meeting": [
        "reschedule upcoming team meeting",
        "move calendar appointment to another day and time",
        "propose a new time for conference call",
        "postpone scheduled discussion"
    ],
    "request_refund": [
        "ask for a refund for duplicate billing charge",
        "request reimbursement for erroneous invoice",
        "dispute payment and request money back",
        "billing dispute for overcharge"
    ],
    "interview_follow_up": [
        "follow up after job interview with recruiter",
        "inquire regarding hiring status after interview",
        "check on interview decision and next steps",
        "send thank you and status check after interview"
    ],
    "informational_update": [
        "notify team or management about scheduled maintenance",
        "provide status update on system or project milestone",
        "inform stakeholders of scheduled downtime",
        "deliver progress report to department head"
    ],
    "apology_and_delay_notice": [
        "apologize for delayed delivery and provide new schedule",
        "send apology for late submission or report",
        "express regret for delay with explanation",
        "apologize for tardiness in delivering work"
    ],
    "inquire_information": [
        "ask for information regarding grants or fees",
        "inquire about conference details or registration",
        "request details about available funding or policy",
        "seek information about application process"
    ],
    "contract_negotiation": [
        "negotiate contract renewal and request volume discount",
        "discuss pricing terms for software license renewal",
        "propose discounted rate for enterprise renewal",
        "renew vendor agreement with modified terms"
    ],
    "equipment_request": [
        "request replacement hardware or office equipment",
        "ask IT helpdesk for computer charger replacement",
        "submit IT request for broken accessories",
        "order office supplies or hardware"
    ],
    "recommendation_request": [
        "request a letter of recommendation for program or job",
        "ask professor for reference letter for internship",
        "request academic recommendation before deadline",
        "inquire if professor can serve as reference"
    ],
    "informal_invitation_or_catchup": [
        "ask friend if free on friday to go outside and chat",
        "invite friend to meet up and catch up about life",
        "check if friend is available this weekend to hang out",
        "catch up with a friend over coffee or lunch",
        "ask friend to meet outside and talk about life"
    ]
}


class IntentClassifier:
    """
    Semantic Intent Interpreter using vector embeddings across intent clusters,
    with automatic fallbacks for open-ended extraction and ambiguity detection.
    """

    def __init__(self):
        self.embedding_engine = get_embedding_engine()
        self.uncertainty_triggers = ["weird", "stuff", "asap", "things", "whatever", "someone", "something"]

    def classify(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classifies intent using semantic embedding similarity against exemplars.
        If ambiguity is high, designates as uncertain_open_ended.
        If semantic fit is moderate or outside standard clusters, derives open-ended intent.
        """
        user_hints = user_hints or {}
        cleaned = text.strip()
        lower = cleaned.lower()

        # 1. Ambiguity / High Uncertainty check
        words = lower.split()
        if len(words) < 6:
            # Very short and ambiguous
            ambiguous_count = sum(1 for w in words if w in self.uncertainty_triggers)
            if ambiguous_count >= 2:
                return {
                    "intent": "uncertain_open_ended",
                    "intent_description": "Uncertain / ambiguous request requiring user clarification",
                    "confidence": 0.25,
                    "is_open_ended": True,
                    "matched_cluster": None
                }

        # 2. Semantic matching against exemplar clusters
        best_cluster = None
        best_score = 0.0

        for cluster_name, examples in INTENT_EXEMPLARS.items():
            cluster_scores = [self.embedding_engine.compute_similarity(cleaned, ex) for ex in examples]
            avg_score = sum(cluster_scores) / len(cluster_scores)
            max_score = max(cluster_scores)
            combined_score = 0.4 * avg_score + 0.6 * max_score

            if combined_score > best_score:
                best_score = combined_score
                best_cluster = cluster_name

        # Calibrate thresholds based on dense SBERT vs sparse N-gram vectorizer
        is_dense = getattr(self.embedding_engine, "_is_sbert_ready", False)
        high_thresh = 0.45 if is_dense else 0.10
        mid_thresh = 0.30 if is_dense else 0.06

        if best_score >= high_thresh and best_cluster:
            intent_label = best_cluster
            description = best_cluster.replace("_", " ").capitalize()
            confidence = round(min(0.98, best_score * (1.1 if is_dense else 3.5)), 3)
            is_open_ended = False
        elif best_score >= mid_thresh:
            # Open-ended extraction: derive intent description directly from text
            intent_label = "open_ended_communication"
            description = self._derive_open_ended_intent(cleaned)
            confidence = round(min(0.90, best_score * (1.1 if is_dense else 2.5)), 3)
            is_open_ended = True
        else:
            intent_label = "uncertain_open_ended"
            description = "Uncertain / general requirement"
            confidence = round(best_score, 3)
            is_open_ended = True

        return {
            "intent": intent_label,
            "intent_description": description,
            "confidence": confidence,
            "is_open_ended": is_open_ended,
            "matched_cluster": best_cluster
        }

    def _derive_open_ended_intent(self, text: str) -> str:
        """Derive natural language description of intent when outside standard clusters."""
        first_sentence = text.split(".")[0].strip()
        for prefix in ["write an email to ", "email ", "write a polite email to ", "ask ", "tell "]:
            if first_sentence.lower().startswith(prefix):
                return first_sentence[len(prefix):].capitalize()
        return first_sentence[:60]
