"""
ai/representation.py - Dynamic Structured Communication Representation
Implements the extensible Pydantic representation of user requirements,
schema generation, and serialization utilities.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from backend.schemas.email_schemas import StructuredRequirement


class CommunicationRepresentation(StructuredRequirement):
    """
    Subclass of StructuredRequirement with helper methods for validation,
    serialization, and research experiment diffing.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert representation to clean dictionary."""
        return self.model_dump()

    def get_essential_facts(self) -> List[str]:
        """Extract a consolidated list of facts, required actions, and constraints."""
        facts = list(self.required_facts)
        if self.required_action and self.required_action not in facts:
            facts.append(f"Action: {self.required_action}")
        if self.reason and self.reason not in facts:
            facts.append(f"Reason: {self.reason}")
        if self.deadline and self.deadline not in facts:
            facts.append(f"Deadline: {self.deadline}")
        if self.requested_duration and self.requested_duration not in facts:
            facts.append(f"Duration: {self.requested_duration}")
        return facts

    def is_ambiguous(self) -> bool:
        """Check if requirement has high uncertainty or missing vital fields."""
        return len(self.uncertainties) > 0 or self.intent in ("uncertain_open_ended", "unspecified", "other")


def create_default_representation(intent: str = "general_communication", tone: str = "polite and professional") -> CommunicationRepresentation:
    """Factory for an empty baseline representation."""
    return CommunicationRepresentation(
        intent=intent,
        tone=tone,
        recipient=None,
        topic=None,
        required_action=None,
        reason=None,
        deadline=None,
        requested_duration=None,
        required_facts=[],
        constraints=[],
        uncertainties=[],
        inferred_details={},
        clarification_questions=[],
        user_confirmed=False,
        custom_attributes={}
    )
