"""
tests/test_answer_utils.py
===========================
Unit tests for capr.evaluation.answer_utils.
Run with:  python -m pytest tests/
"""

import sys
sys.path.insert(0, "src")

from capr.evaluation.answer_utils import extract_answer, answers_match


class TestExtractAnswer:

    def test_explicit_final_answer_tag(self):
        raw = "Step 1: ...\nStep 2: ...\nFINAL ANSWER: 42"
        assert extract_answer(raw, "arithmetic") == "42"

    def test_mcq_letter(self):
        raw = "The answer is B because machine learning is a subset of AI."
        assert extract_answer(raw, "knowledge") == "B"

    def test_yes_no_reasoning(self):
        assert extract_answer("Yes, all bloops are lazzles.", "reasoning") == "Yes"
        assert extract_answer("No, that is incorrect.", "reasoning") == "No"

    def test_numeric_arithmetic(self):
        raw = "After calculating: change = $3.00"
        assert extract_answer(raw, "arithmetic") == "3.00"

    def test_fallback(self):
        raw = "Unknown format response here"
        result = extract_answer(raw, "arithmetic")
        assert len(result) <= 60


class TestAnswersMatch:

    def test_exact_string_match(self):
        assert answers_match("B", "B") is True

    def test_case_insensitive(self):
        assert answers_match("yes", "Yes") is True

    def test_float_tolerance(self):
        assert answers_match("3.0", "3") is True
        assert answers_match("72.0", "72.0") is True
        assert answers_match("6.25", "6.25") is True

    def test_mismatch(self):
        assert answers_match("A", "B") is False
        assert answers_match("10.0", "12.0") is False
