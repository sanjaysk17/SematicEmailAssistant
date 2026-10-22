"""
tests/test_validation.py - Tests for Output Validation Engine
Verifies intent preservation, factual completeness, tone suitability,
and unsupported information detection.
"""

import pytest
from ai.validation import OutputValidator
from backend.schemas.email_schemas import StructuredRequirement, ValidationStatusEnum, FactStatusEnum


@pytest.fixture
def validator():
    return OutputValidator()


@pytest.fixture
def sample_rep():
    return StructuredRequirement(
        intent="request_extension",
        tone="polite and professional",
        recipient="Professor",
        topic="Project submission",
        required_action="request two additional days",
        reason="technical issue",
        requested_duration="two additional days",
        required_facts=["could not complete project"]
    )


def test_validation_pass_case(validator, sample_rep):
    req = "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days."
    subject = "Project Submission - Requesting Two Additional Days"
    body = (
        "Dear Professor,\n\n"
        "I am writing to politely request two additional days regarding my project submission. "
        "Due to a technical issue, I could not complete the project on time. "
        "Thank you for your consideration.\n\n"
        "Sincerely,\nStudent Alex"
    )
    report = validator.validate(req, sample_rep, subject, body)

    assert report.intent_preserved is True
    assert report.tone_suitable is True
    assert report.completeness_score >= 0.8
    assert report.overall_status in (ValidationStatusEnum.PASS, ValidationStatusEnum.NEEDS_REVIEW)


def test_missing_fact_detection(validator, sample_rep):
    req = "Write a polite email to my professor requesting two additional days."
    subject = "Hello"
    # Body completely omits technical issue and two additional days
    body = "Dear Professor,\n\nI need more time. Thanks."
    report = validator.validate(req, sample_rep, subject, body)

    assert report.completeness_score < 0.6
    assert any("Duration" in f.fact for f in report.fact_evaluations if f.status == FactStatusEnum.MISSING)


def test_unsupported_attachment_warning(validator, sample_rep):
    req = "Write a polite email to my professor requesting two additional days."
    subject = "Project Extension"
    body = "Dear Professor,\n\nPlease find attached my complete report and medical certificate. Thanks."
    report = validator.validate(req, sample_rep, subject, body)

    assert len(report.unsupported_details) > 0
    assert any("attachment" in u.lower() for u in report.unsupported_details)


def test_validation_score_threshold_80_percent(validator, sample_rep):
    """Verifies that high-quality drafts score >= 0.80 (80%) while deficient drafts fall below 0.80."""
    req = "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days."
    good_subject = "Project Submission - Requesting Two Additional Days Extension"
    good_body = (
        "Dear Professor,\n\n"
        "I hope this message finds you well. I am writing to politely request two additional days "
        "to submit my project. Due to an unexpected technical issue, I was unable to complete the work "
        "by the initial deadline. I have now resolved the technical issue and will submit the final file promptly.\n\n"
        "Thank you very much for your understanding.\n\n"
        "Sincerely,\nAlex"
    )
    good_report = validator.validate(req, sample_rep, good_subject, good_body)
    assert good_report.validation_score >= 0.80
    assert good_report.passed_threshold is True

    # Deficient draft omitting reason and duration
    bad_subject = "Hi"
    bad_body = "Hi Professor, give me more time. Bye."
    bad_report = validator.validate(req, sample_rep, bad_subject, bad_body)
    assert bad_report.validation_score < 0.80
    assert bad_report.passed_threshold is False


def test_orchestrator_self_validation_before_delivery(sample_rep):
    """Verifies that orchestrator.generate() performs self-validation before delivering draft to user."""
    from backend.services.orchestrator import PipelineOrchestrator
    from backend.schemas.email_schemas import GenerateRequest

    orchestrator = PipelineOrchestrator()
    req = GenerateRequest(
        requirement="I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days.",
        representation=sample_rep,
        llm_provider="mock",
        sender_name="Alex"
    )

    resp = orchestrator.generate(req)
    assert resp.subject is not None
    assert resp.body is not None
    # Confirm project self-validated before delivering to user
    assert resp.validation_score is not None
    assert isinstance(resp.validation_score, float)
    assert resp.passed_validation is True
    assert resp.validation_score >= 0.80
    assert resp.validation_report is not None
