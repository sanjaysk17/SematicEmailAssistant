"""
tests/test_preprocessing.py - Tests for Text Preprocessing and Normalization
Verifies preservation of dates, numbers, currencies, negations, and empty string handling.
"""

import pytest
from ai.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_empty_and_whitespace_input(preprocessor):
    assert preprocessor.normalize("") == ""
    assert preprocessor.normalize("   \n\t  ") == ""
    res = preprocessor.process("")
    assert res["word_count"] == 0
    assert not res["has_negation"]


def test_negation_preservation(preprocessor):
    text = "I could not complete my assignment and will never ask for extra time again."
    res = preprocessor.process(text)
    assert res["has_negation"] is True
    assert "not" in res["negations_found"]
    assert "never" in res["negations_found"]


def test_quantities_and_currency_preservation(preprocessor):
    text = "Duplicate charge of $49.99 for 50 enterprise seats due in 2 days."
    res = preprocessor.process(text)
    assert "$49.99" in res["quantities_found"]
    assert "50" in res["quantities_found"]
    assert "2" in res["quantities_found"]


def test_unicode_and_quotes_normalization(preprocessor):
    text = "“Special quote” and ‘single quote’ with hyphen—dash."
    cleaned = preprocessor.normalize(text)
    assert '"Special quote"' in cleaned
    assert "'single quote'" in cleaned
    assert "hyphen-dash" in cleaned
