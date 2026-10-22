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


def test_friendly_invitation_tone_and_recipient(interpreter):
    text = "Write a casual and friendly email to an old college friend asking whether they are free this weekend to meet for coffee and catch up. Make it warm, natural, and informal without sounding overly sentimental."
    res = interpreter.interpret(text)
    rep = res["representation"]

    assert rep.intent == "informal_invitation_or_catchup"
    assert "casual" in rep.tone.lower() or "friendly" in rep.tone.lower()
    assert rep.recipient != "To"
    assert rep.recipient != "An"
    assert rep.recipient in (None, "Friend", "Old College Friend")


def test_tone_classification_respects_explicit_directives(interpreter):
    # Case 1: Prompt has explicit formal directive even though recipient is a friend
    text_formal = "Write a formal and serious email to an old college friend regarding payment of an invoice."
    res_formal = interpreter.interpret(text_formal)
    assert "formal" in res_formal["representation"].tone.lower()

    # Case 2: Prompt has no tone words, inferred from informal intent
    text_inferred = "Ask an old college friend if they want to meet for coffee this weekend."
    res_inferred = interpreter.interpret(text_inferred)
    assert "casual" in res_inferred["representation"].tone.lower() or "friendly" in res_inferred["representation"].tone.lower()

    # Case 3: User preference tone_hint overrides prompt
    res_hint = interpreter.interpret(text_inferred, user_preferences={"tone_hint": "urgent and professional"})
    assert res_hint["representation"].tone == "urgent and professional"

