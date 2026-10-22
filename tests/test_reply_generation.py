"""
tests/test_reply_generation.py - Dedicated Test Suite for Reply Button & Reply Generation Mode
Verifies:
1. Professional email asking for a project update.
2. Email containing a direct question.
3. Message requesting confirmation.
4. Informal message from a friend.
5. Complaint that requires a polite response.
6. Incoming email containing multiple questions.
7. Empty input validation.
8. Generation failure and resilient fallback handling.
9. Existing new-email generation preservation (no regression).
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.email_schemas import StructuredRequirement, GenerateRequest
from ai.prompt_builder import PromptBuilder
from ai.generation import EmailGenerationService, MockLLMProvider
from ai.semantic_interpretation import SemanticInterpreter

client = TestClient(app)


@pytest.fixture
def generation_service():
    return EmailGenerationService(provider_type="mock")


@pytest.fixture
def interpreter():
    return SemanticInterpreter()


def test_1_professional_email_project_update(generation_service, interpreter):
    """Case 1: Professional email asking for project update."""
    incoming_msg = (
        "Hi team, could you please provide a status update on the Q3 deliverables by tomorrow? "
        "We need to brief the executive committee."
    )
    analysis = interpreter.interpret(incoming_msg, user_preferences={"generation_mode": "reply"})
    rep = analysis["representation"]

    result = generation_service.generate(
        requirement=incoming_msg,
        representation=rep,
        generation_mode="reply",
        sender_name="Alex Morgan"
    )

    assert result["subject"].startswith("Re:")
    assert "deliverable" in result["body"].lower() or "project" in result["body"].lower()
    # Ensure it's not a mere paraphrase
    assert "could you please provide" not in result["body"].lower()
    assert "Alex Morgan" in result["body"] or "[Your Name]" not in result["body"]


def test_2_direct_question(generation_service, interpreter):
    """Case 2: Email containing a direct question."""
    incoming_msg = (
        "Can you confirm if you will be attending the team sync meeting on Thursday at 2 PM?"
    )
    analysis = interpreter.interpret(incoming_msg, user_preferences={"generation_mode": "reply"})
    rep = analysis["representation"]

    result = generation_service.generate(
        requirement=incoming_msg,
        representation=rep,
        generation_mode="reply"
    )

    assert result["subject"].startswith("Re:")
    assert "attending" in result["body"].lower() or "confirm" in result["body"].lower()
    # Directly answers rather than repeating question
    assert "can you confirm if" not in result["body"].lower()


def test_3_confirmation_request(generation_service, interpreter):
    """Case 3: Message requesting confirmation."""
    incoming_msg = (
        "Please confirm receipt of the attached contract documents for review."
    )
    analysis = interpreter.interpret(incoming_msg, user_preferences={"generation_mode": "reply"})
    rep = analysis["representation"]

    result = generation_service.generate(
        requirement=incoming_msg,
        representation=rep,
        generation_mode="reply"
    )

    assert result["subject"].startswith("Re:")
    assert "confirm" in result["body"].lower() and ("document" in result["body"].lower() or "receipt" in result["body"].lower())


def test_4_informal_message_from_friend(generation_service, interpreter):
    """Case 4: Informal message from a friend."""
    incoming_msg = (
        "Hey! Are we still on for lunch this Friday at noon? Let me know!"
    )
    analysis = interpreter.interpret(incoming_msg, user_preferences={"generation_mode": "reply"})
    rep = analysis["representation"]

    result = generation_service.generate(
        requirement=incoming_msg,
        representation=rep,
        generation_mode="reply"
    )

    assert "Re:" in result["subject"]
    assert any(w in result["body"].lower() for w in ["lunch", "friday", "noon", "catch up"])
    # Appropriate friendly greeting
    assert any(greeting in result["body"] for greeting in ["Hey", "Hi", "Hello", "Thank", "Thanks"])


def test_5_complaint_polite_response(generation_service, interpreter):
    """Case 5: Complaint that requires a polite response."""
    incoming_msg = (
        "I am extremely dissatisfied with the service delay on my order #98765. This is unacceptable."
    )
    analysis = interpreter.interpret(incoming_msg, user_preferences={"generation_mode": "reply"})
    rep = analysis["representation"]

    result = generation_service.generate(
        requirement=incoming_msg,
        representation=rep,
        generation_mode="reply"
    )

    assert result["subject"].startswith("Re:")
    # Empathetic, polite response
    assert any(w in result["body"].lower() for w in ["apolog", "sorry", "patience", "delay"])


def test_6_multiple_questions(generation_service, interpreter):
    """Case 6: Incoming email containing multiple questions."""
    incoming_msg = (
        "Could you clarify when the report is due, who should review it first, and if we need supplementary slides?"
    )
    analysis = interpreter.interpret(incoming_msg, user_preferences={"generation_mode": "reply"})
    rep = analysis["representation"]

    result = generation_service.generate(
        requirement=incoming_msg,
        representation=rep,
        generation_mode="reply"
    )

    assert result["subject"].startswith("Re:")
    # Structured response answering points
    assert ("due" in result["body"].lower() or "review" in result["body"].lower() or "slides" in result["body"].lower())


def test_7_empty_input_validation():
    """Case 7: Empty or whitespace input rejected."""
    # Test FastAPI API validation on /api/reply
    empty_payload = {
        "requirement": "   ",
        "representation": {
            "intent": "acknowledgement_and_reply",
            "tone": "polite and professional"
        },
        "generation_mode": "reply"
    }
    resp = client.post("/api/reply", json=empty_payload)
    assert resp.status_code == 422  # Unprocessable Entity (min_length=3 validation)


def test_8_generation_failure_handling(generation_service, interpreter):
    """Case 8: Generation failure produces safe fallback reply without crashing."""
    incoming_msg = "Please confirm our sync meeting tomorrow."
    analysis = interpreter.interpret(incoming_msg, user_preferences={"generation_mode": "reply"})
    rep = analysis["representation"]

    # Trigger fallback with unavailable provider
    result = generation_service.generate(
        requirement=incoming_msg,
        representation=rep,
        generation_mode="reply",
        provider_override="unknown_invalid_provider"
    )

    assert result["subject"].startswith("Re:")
    assert len(result["body"]) > 50
    assert "Thank you" in result["body"] or "received your" in result["body"]


def test_9_existing_new_email_generation_unaffected(generation_service, interpreter):
    """Case 9: Verify existing new-email generation mode ('new') is completely unaffected."""
    requirement = "Write a polite email to Professor Miller requesting two additional days for the project."
    analysis = interpreter.interpret(requirement, user_preferences={"generation_mode": "new"})
    rep = analysis["representation"]

    result = generation_service.generate(
        requirement=requirement,
        representation=rep,
        generation_mode="new"
    )

    # Standard new email should NOT start with "Re:"
    assert not result["subject"].startswith("Re:")
    assert "Professor" in result["body"]
    assert "extension" in result["body"].lower() or "days" in result["body"].lower()


def test_api_reply_endpoint_end_to_end():
    """Verifies the /api/reply REST endpoint end-to-end."""
    incoming_email = "Can you send me the latest presentation deck by 5 PM today?"
    
    # 1. Analyze with generation_mode='reply'
    a_resp = client.post("/api/analyze", json={
        "requirement": incoming_email,
        "generation_mode": "reply"
    })
    assert a_resp.status_code == 200
    a_data = a_resp.json()
    assert a_data["representation"]["intent"] == "acknowledgement_and_reply"
    assert a_data["is_underspecified"] is False

    # 2. Generate Reply via /api/reply
    g_resp = client.post("/api/reply", json={
        "requirement": incoming_email,
        "representation": a_data["representation"],
        "generation_mode": "reply",
        "sender_name": "Jordan Smith"
    })
    assert g_resp.status_code == 200
    g_data = g_resp.json()
    assert g_data["subject"].startswith("Re:")
    assert len(g_data["body"]) > 30
    assert "Jordan Smith" in g_data["body"]
