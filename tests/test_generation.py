"""
tests/test_generation.py - Tests for Prompt Builder and Generation Service
Verifies prompt construction, mock generation, and fallback behavior.
"""

import pytest
from ai.prompt_builder import PromptBuilder
from ai.generation import EmailGenerationService, MockLLMProvider
from backend.schemas.email_schemas import StructuredRequirement


@pytest.fixture
def sample_representation():
    return StructuredRequirement(
        intent="request_extension",
        tone="polite and professional",
        recipient="Professor",
        topic="Project submission",
        required_action="request two additional days",
        reason="technical issue",
        requested_duration="two additional days",
        required_facts=["System crashed"],
        constraints=["Do not invent excuses"]
    )


def test_prompt_builder_contains_critical_sections(sample_representation):
    req = "Write to professor for 2 days extension"
    prompt = PromptBuilder.build_proposed_prompt(req, sample_representation)

    assert "[SYSTEM CONSTRAINTS AND SAFETY BOUNDARIES]" in prompt
    assert "[VERIFIED STRUCTURED COMMUNICATION REPRESENTATION]" in prompt
    assert "Primary Intent: request_extension" in prompt
    assert "STRICT NEGATIVE CONSTRAINTS" in prompt


def test_mock_generation_service(sample_representation):
    service = EmailGenerationService()
    req = "Write to professor for 2 days extension"
    result = service.generate(req, sample_representation, system_version="proposed")

    assert "subject" in result
    assert "body" in result
    assert len(result["subject"]) > 5
    assert "Professor" in result["body"]
    assert "two additional days" in result["body"]


def test_baseline_generation_modes(sample_representation):
    service = EmailGenerationService()
    req = "Write to professor for 2 days extension"

    res_a = service.generate(req, sample_representation, system_version="baseline_a")
    res_b = service.generate(req, sample_representation, system_version="baseline_b")

    assert res_a["system_version"] == "baseline_a"
    assert res_b["system_version"] == "baseline_b"


def test_steve_catchup_extraction_and_generation():
    from ai.semantic_interpretation import SemanticInterpreter

    req = "I want to Write for My friend His name Steve.Willy you be free on friday and Go outside.To Conversation about my life"
    interpreter = SemanticInterpreter()
    analysis = interpreter.interpret(req)
    rep = analysis["representation"]

    assert rep.recipient == "Steve"
    assert rep.recipient != "Write"
    assert "friday" in (rep.deadline or "").lower()

    service = EmailGenerationService()
    result = service.generate(req, rep, sender_name="Sanjay")

    assert "Steve" in result["body"]
    assert "Write" not in result["body"]
    assert "Friday" in result["body"] or "friday" in result["body"]
    assert "Sanjay" in result["body"]
    assert "technical issue" not in result["body"]


def test_gemini_provider_response_parsing():
    from ai.generation import GeminiLLMProvider

    provider = GeminiLLMProvider(api_key="mock-key")

    # Case 1: Markdown fenced JSON
    json_block = "```json\n{\"subject\": \"Catching up on Friday\", \"body\": \"Hi Steve,\\nAre you free?\\nBest,\\nSanjay\"}\n```"
    s1, b1 = provider._parse_response(json_block)
    assert s1 == "Catching up on Friday"
    assert "Steve" in b1

    # Case 2: Plain text with Subject header
    plain_text = "Subject: Meet up this Friday\n\nHi Steve, let's go outside and chat!"
    s2, b2 = provider._parse_response(plain_text)
    assert s2 == "Meet up this Friday"
    assert "Steve" in b2


def test_gemini_provider_fallback_when_no_key(sample_representation):
    from backend.config import settings
    orig_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = None
    try:
        service = EmailGenerationService(provider_type="gemini")
        req = "Write to professor for 2 days extension"
        result = service.generate(req, sample_representation)
        assert "body" in result
        assert "Professor" in result["body"]
        assert "gemini API key missing" in result["model_identifier"].lower() or "mock-fallback" in result["model_identifier"].lower()
    finally:
        settings.GEMINI_API_KEY = orig_key


def test_wifi_notice_acknowledgement_generation():
    from ai.semantic_interpretation import SemanticInterpreter

    prompt = (
        "Dear Students, It has been observed that some students are sharing their Wi-Fi login credentials "
        "with their friends, resulting in unauthorized usage. Please note that each student is permitted to "
        "use up to 10 GB of data per day. Reply for this Mail Piltely thank yo sir,i will ot use 10GB"
    )

    interpreter = SemanticInterpreter()
    res = interpreter.interpret(prompt)
    rep = res["representation"]

    assert rep.recipient == "Sir"
    assert rep.recipient != "Use Up"

    service = EmailGenerationService()
    result = service.generate(prompt, rep, sender_name="Sakthi")

    assert "Sir" in result["body"]
    assert "Use Up" not in result["body"]
    assert "go outside" not in result["body"].lower()
    assert "conversation about life" not in result["body"].lower()
    assert "Wi-Fi" in result["subject"] or "10 GB" in result["body"] or "policy" in result["body"].lower()


def test_prompt_builder_with_revision_instructions(sample_representation):
    req = "Write to professor for 2 days extension"
    prev_draft = "Subject: Project Extension\n\nDear Professor,\nI need 2 days.\nBest,\nStudent"
    revision_instruction = "Make it much more formal and polite, and do not apologize."

    prompt = PromptBuilder.build_proposed_prompt(
        requirement=req,
        representation=sample_representation,
        previous_draft=prev_draft,
        revision_instruction=revision_instruction
    )

    assert "[PREVIOUS DRAFT (USER WAS NOT SATISFIED WITH THIS OUTPUT)]" in prompt
    assert "Dear Professor" in prompt
    assert "[USER REVISION FEEDBACK / RE-DO INSTRUCTIONS]" in prompt
    assert "Make it much more formal and polite" in prompt
    assert "CRITICAL RE-DO DIRECTIVE" in prompt


def test_redo_generation_service_with_revision(sample_representation):
    service = EmailGenerationService()
    req = "Write to professor for 2 days extension"

    # Initial draft
    res1 = service.generate(req, sample_representation, system_version="proposed")
    initial_body = res1["body"]

    # Redo draft with "shorter" instruction
    res2 = service.generate(
        req,
        sample_representation,
        system_version="proposed",
        previous_draft=f"Subject: {res1['subject']}\n\n{initial_body}",
        revision_instruction="Make it significantly shorter and concise"
    )

    assert res2["subject"] is not None
    assert res2["body"] is not None
    assert len(res2["body"]) > 0


def test_api_generate_redo_flow(sample_representation):
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    gen_payload = {
        "requirement": "Write to professor for 2 days extension",
        "representation": sample_representation.model_dump(),
        "previous_draft": "Subject: Old Subject\n\nOld body text that user was not happy with",
        "revision_instruction": "Make it more urgent and direct"
    }

    response = client.post("/api/generate", json=gen_payload)
    assert response.status_code == 200
    data = response.json()
    assert "subject" in data
    assert "body" in data
    assert len(data["body"]) > 10

