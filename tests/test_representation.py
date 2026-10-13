"""
tests/test_representation.py - Tests for Communication Representation
Verifies Pydantic schema validation, default creation, and dictionary conversion.
"""

import pytest
from pydantic import ValidationError
from backend.schemas.email_schemas import StructuredRequirement
from ai.representation import CommunicationRepresentation, create_default_representation


def test_valid_representation():
    rep = CommunicationRepresentation(
        intent="request_extension",
        tone="polite and professional",
        recipient="Professor",
        topic="Project submission",
        required_action="Request 2 days extension",
        reason="Technical issues",
        required_facts=["System crashed", "Submission pending"],
        constraints=["Do not invent names"]
    )
    assert rep.intent == "request_extension"
    assert len(rep.required_facts) == 2
    assert rep.is_ambiguous() is False
    assert len(rep.get_essential_facts()) >= 3


def test_ambiguity_detection():
    rep = CommunicationRepresentation(
        intent="uncertain_open_ended",
        uncertainties=["Recipient is missing", "No date specified"]
    )
    assert rep.is_ambiguous() is True


def test_schema_serialization():
    rep = create_default_representation()
    d = rep.to_dict()
    assert isinstance(d, dict)
    assert d["intent"] == "general_communication"
    assert d["user_confirmed"] is False


def test_missing_required_fields():
    with pytest.raises(ValidationError):
        # Intent is mandatory
        CommunicationRepresentation()
