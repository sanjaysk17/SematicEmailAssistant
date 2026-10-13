"""
tests/test_semantic_interpretation.py - Tests for Semantic Understanding Pipeline
Verifies intent interpretation, recipient extraction, durations, and tone analysis.
"""

import pytest
from ai.semantic_interpretation import SemanticInterpreter


@pytest.fixture
def interpreter():
    return SemanticInterpreter()


def test_academic_extension_interpretation(interpreter):
    text = "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days."
    res = interpreter.interpret(text)
    rep = res["representation"]

    assert rep.intent == "request_extension"
    assert rep.recipient == "Professor"
    assert "polite" in rep.tone.lower()
    assert rep.requested_duration == "two additional days"
    assert "technical issue" in rep.reason.lower()
    assert len(rep.required_facts) > 0


def test_sick_leave_interpretation(interpreter):
    text = "I have a fever and doctor advised rest. Email my manager Sarah informing her I will be taking sick leave today Monday and tomorrow Tuesday."
    res = interpreter.interpret(text)
    rep = res["representation"]

    assert rep.intent == "request_sick_leave"
    assert "Sarah" in (rep.recipient or "")
    assert "fever" in (rep.reason or "").lower()


def test_ambiguous_request_triggers_uncertainty(interpreter):
    text = "Things happened and stuff is weird. Email someone to fix it ASAP."
    res = interpreter.interpret(text)
    rep = res["representation"]

    assert rep.intent == "uncertain_open_ended"
    assert len(rep.uncertainties) > 0
    assert len(rep.clarification_questions) > 0
