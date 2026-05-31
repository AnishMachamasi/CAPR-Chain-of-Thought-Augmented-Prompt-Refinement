"""
tests/test_aggregation.py
==========================
Unit tests for capr.evaluation.aggregation.
Run with:  python -m pytest tests/
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from capr.evaluation.aggregation import majority_vote, confidence_weighted_vote, score_chain


class TestMajorityVote:

    def test_clear_majority(self):
        assert majority_vote(["A", "A", "B"]) == "A"

    def test_single_answer(self):
        assert majority_vote(["42"]) == "42"

    def test_empty(self):
        assert majority_vote([]) == ""


class TestConfidenceWeightedVote:

    def test_agrees_with_majority(self):
        # top-weighted and majority agree → return majority
        answers = ["42", "42", "10"]
        scores  = [0.9, 0.8, 0.5]
        assert confidence_weighted_vote(answers, scores) == "42"

    def test_disagrees_top_weighted_wins(self):
        # majority = "10", but highest score is "42"
        answers = ["10", "10", "42"]
        scores  = [0.3, 0.3, 0.95]
        assert confidence_weighted_vote(answers, scores) == "42"

    def test_empty(self):
        assert confidence_weighted_vote([], []) == ""


class TestScoreChain:

    def test_score_range(self):
        text = (
            "Step 1: I calculated 2+2=4.\n"
            "Step 2: Therefore the answer is 4.\n"
            "The premise is broadly accepted. I am HIGHLY CONFIDENT.\n"
            "FINAL ANSWER: 4"
        )
        score = score_chain(text)
        assert 0.0 <= score <= 1.0

    def test_short_chain_penalised(self):
        short = score_chain("4")
        normal = score_chain("Step 1: compute.\n" * 30)
        assert normal >= short
