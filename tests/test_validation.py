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
