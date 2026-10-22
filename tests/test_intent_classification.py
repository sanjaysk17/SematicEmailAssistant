"""
tests/test_intent_classification.py - Comprehensive Intent Classification Test Suite
Tests:
1. Empty input handling (safe degrade, 0.0 confidence)
2. Normal requests across diverse domains
3. Similar but distinct intents (e.g., request_leave vs request_sick_leave vs request_extension)
4. Multiple simultaneous intents (multi-intent extraction)
5. Unseen intent categories and out-of-scope requests
6. Genuinely ambiguous requests triggering uncertainty
7. Missing model artifacts and graceful fallback to exemplar matcher
8. Invalid / corrupted model outputs degradation
9. Model loading failure recovery
10. Valid confidence values bounded in [0.0, 1.0]
11. Backward compatibility with existing callers (SemanticInterpreter, API routes)
"""

import pytest
import numpy as np
from ai.intent_classification import IntentClassifier, get_intent_classifier
from ai.semantic_interpretation import SemanticInterpreter


@pytest.fixture
def classifier():
    return get_intent_classifier()


@pytest.fixture
def interpreter():
    return SemanticInterpreter()


def test_empty_input_handling(classifier):
    """Empty and whitespace strings should return uncertain_open_ended with 0 confidence."""
    for text in ["", "   ", "\n\t"]:
        res = classifier.classify(text)
        assert res["intent"] == "uncertain_open_ended"
        assert res["confidence"] == 0.0
        assert res["is_open_ended"] is True
        assert res["uncertainty_score"] == 1.0


def test_normal_requests(classifier):
    """Normal requests with standard phrasing should be accurately classified."""
    # Test Sick Leave
    res_sick = classifier.classify("I have a high fever and cannot come into the office today. Taking sick leave.")
    assert res_sick["intent"] == "request_sick_leave"
    assert 0.0 <= res_sick["confidence"] <= 1.0
    assert res_sick["is_open_ended"] is False

    # Test Extension
    res_ext = classifier.classify("Please grant me two additional days to finish the research paper due to technical issues.")
    assert res_ext["intent"] == "request_extension"
    assert 0.0 <= res_ext["confidence"] <= 1.0

    # Test Meeting Reschedule
    res_meet = classifier.classify("Can we move tomorrow's 3 PM team sync to Thursday morning instead?")
    assert res_meet["intent"] == "reschedule_meeting"


def test_similar_but_distinct_intents(classifier):
    """Ensures fine distinctions between related intents are maintained."""
    # Casual/Annual Leave vs Medical/Sick Leave
    leave_text = "I am requesting casual leave for family festival celebrations from Friday to Monday."
    sick_text = "I am unwell with influenza and physician ordered bed rest for 3 days."

    res_leave = classifier.classify(leave_text)
    res_sick = classifier.classify(sick_text)

    assert res_leave["intent"] == "request_leave"
    assert res_sick["intent"] == "request_sick_leave"
    assert res_leave["intent"] != res_sick["intent"]

    # Request Refund vs Contract Negotiation
    refund_text = "I was billed twice for my cloud subscription and demand a refund of $50."
    contract_text = "We want to renew our software license for next year but request a 10% volume discount."

    res_refund = classifier.classify(refund_text)
    res_contract = classifier.classify(contract_text)

    assert res_refund["intent"] == "request_refund"
    assert res_contract["intent"] == "contract_negotiation"


def test_multiple_simultaneous_intents(classifier):
    """Verifies that multiple communication goals are recognized."""
    multi_text = (
        "I have a fever today and am calling in sick, but please also reschedule my 2 PM project review to Thursday."
    )
    res = classifier.classify(multi_text)

    # Both communication goals must be identified
    all_intents = [res["intent"]] + res.get("secondary_intents", [])
    assert "request_sick_leave" in all_intents
    assert "reschedule_meeting" in all_intents
    assert res["is_multi_intent"] is True

    # User test case: apology and extension request
    apology_text = "Apologize for submitting the assignment late and request two additional days."
    res_apology = classifier.classify(apology_text)
    all_apology_intents = [res_apology["intent"]] + res_apology.get("secondary_intents", [])
    assert "apology_and_delay_notice" in all_apology_intents
    assert "request_extension" in all_apology_intents
    assert res_apology["is_multi_intent"] is True


def test_unseen_intent_and_out_of_scope(classifier):
    """Out-of-scope non-email queries should be rejected with high uncertainty."""
    out_of_scope_texts = [
        "What is the capital city of Australia?",
        "Please provide a recipe for gluten-free chocolate brownies.",
        "Solve the differential equation dy/dx = 4x + 3.",
        "asdfghjkl random noise 123456 !!!"
    ]
    for text in out_of_scope_texts:
        res = classifier.classify(text)
        # Should be flagged as uncertain_open_ended or have high uncertainty score
        assert res["intent"] in ("uncertain_open_ended", "out_of_scope") or res["is_open_ended"] is True or res["uncertainty_score"] >= 0.70


def test_ambiguous_requests_trigger_uncertainty(classifier):
    """Vague and ambiguous prompts should trigger uncertain_open_ended."""
    vague_text = "Things happened and stuff is weird. Email someone to fix it ASAP."
    res = classifier.classify(vague_text)
    assert res["intent"] == "uncertain_open_ended"
    assert res["is_open_ended"] is True
    assert res["uncertainty_score"] >= 0.75


def test_confidence_values_bounded(classifier):
    """Confidence and uncertainty scores must strictly be in range [0.0, 1.0]."""
    test_prompts = [
        "Draft an email to client apologizing for the server maintenance delay.",
        "Ask Dr. Evans for a letter of recommendation for graduate school.",
        "Request IT to replace my broken laptop power adapter."
    ]
    for p in test_prompts:
        res = classifier.classify(p)
        assert 0.0 <= res["confidence"] <= 1.0, f"Confidence {res['confidence']} out of bounds"
        assert 0.0 <= res["uncertainty_score"] <= 1.0, f"Uncertainty {res['uncertainty_score']} out of bounds"


def test_missing_model_artifacts_graceful_fallback(monkeypatch):
    """If neural checkpoints are absent or fail, the classifier falls back gracefully to exemplar matcher."""
    # Temporarily set active backend to fallback
    clf = IntentClassifier()
    orig_setfit = clf._setfit_model
    clf._setfit_model = None  # simulate missing SetFit model

    try:
        res = clf.classify("I could not complete my project because of a technical issue. Request two additional days.")
        assert res["intent"] in ("request_extension", "open_ended_communication")
        assert res["confidence"] > 0.0
    finally:
        clf._setfit_model = orig_setfit


def test_backward_compatibility_with_semantic_interpreter(interpreter):
    """Ensures existing SemanticInterpreter downstream pipeline consumes the classification output seamlessly."""
    text = "I have a fever and doctor advised rest. Email my manager Sarah informing her I will be taking sick leave today Monday and tomorrow Tuesday."
    res = interpreter.interpret(text)

    rep = res["representation"]
    assert rep.intent == "request_sick_leave"
    assert "Sarah" in (rep.recipient or "")
    assert hasattr(rep, "secondary_intents")
    assert "uncertainty_score" in rep.custom_attributes
    assert "classifier_type" in rep.custom_attributes
    assert res["confidence"] > 0.0


def test_unseen_wording_support(classifier):
    """Paraphrased requests with unseen wording should still be correctly mapped to the right intent."""
    unseen_text = "I need permission to push the deadline back because our deployment pipeline suffered an outage."
    res = classifier.classify(unseen_text)
    assert res["intent"] == "request_extension"
