# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Scorer implementations using fake data."""

import pytest
from inference_endpoint.accuracy.evaluators.scorers import PassAt1Scorer, RougeScorer


class TestPassAt1Scorer:
    """Test cases for PassAt1Scorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = PassAt1Scorer()

    def test_case_sensitive(self):
        """Test that scoring is case-sensitive."""
        assert self.scorer.score_sample("Correct", "correct") == 0
        assert self.scorer.score_sample("CORRECT", "correct") == 0

    def test_empty_strings(self):
        """Test behavior with empty strings."""
        assert self.scorer.score_sample("", "") == 1
        assert self.scorer.score_sample("answer", "") == 0
        assert self.scorer.score_sample("", "answer") == 0

    def test_whitespace_sensitive(self):
        """Test that whitespace is significant."""
        assert self.scorer.score_sample("answer", " answer") == 0
        assert self.scorer.score_sample("answer", "answer ") == 0
        assert self.scorer.score_sample(" answer ", "answer") == 0

    def test_multiline_text(self):
        """Test with multiline text."""
        multiline = "line1\nline2\nline3"
        assert self.scorer.score_sample(multiline, multiline) == 1
        assert self.scorer.score_sample("line1\nline2", multiline) == 0

    def test_special_characters(self):
        """Test with special characters."""
        special = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        assert self.scorer.score_sample(special, special) == 1
        assert self.scorer.score_sample("simple", special) == 0

    def test_unicode_characters(self):
        """Test with unicode characters."""
        unicode_text = "Hello 世界 🌍"
        assert self.scorer.score_sample(unicode_text, unicode_text) == 1
        assert self.scorer.score_sample("Hello", unicode_text) == 0


class TestRougeScorer:
    """Test cases for RougeScorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer_with_stemmer = RougeScorer(use_stemmer=True)
        self.scorer_without_stemmer = RougeScorer(use_stemmer=False)

    def test_returns_dict_with_all_metrics(self):
        """Test that scorer returns dictionary with all ROUGE metrics."""
        scores = self.scorer_with_stemmer.score_sample(
            "the cat sat on the mat", "a cat was sitting on a mat"
        )
        assert isinstance(scores, dict)
        assert "rouge1" in scores and isinstance(scores["rouge1"], float)
        assert "rouge2" in scores and isinstance(scores["rouge2"], float)
        assert "rougeL" in scores and isinstance(scores["rougeL"], float)
        assert 0.0 <= scores["rouge1"] <= 1.0
        assert 0.0 <= scores["rouge2"] <= 1.0
        assert 0.0 <= scores["rougeL"] <= 1.0

    def test_perfect_match_scores_one(self):
        """Test that identical texts score 1.0 for all metrics."""
        identical_text = "the cat sat on the mat"
        scores = self.scorer_with_stemmer.score_sample(identical_text, identical_text)
        assert scores["rouge1"] == pytest.approx(1.0)
        assert scores["rouge2"] == pytest.approx(1.0)
        assert scores["rougeL"] == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self):
        """Test that completely different texts score close to 0."""
        scores = self.scorer_with_stemmer.score_sample(
            "alpha beta gamma", "delta epsilon zeta"
        )
        assert scores["rouge1"] == pytest.approx(0.0)
        assert scores["rouge2"] == pytest.approx(0.0)
        assert scores["rougeL"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Test partial overlap produces intermediate scores."""
        scores = self.scorer_with_stemmer.score_sample(
            "the cat sat on the mat", "the dog sat on the floor"
        )
        # Should have some overlap (the, sat, on, the)
        assert 0.0 < scores["rouge1"] < 1.0
        # Rouge-2 might be lower due to bigram requirements
        assert 0.0 <= scores["rouge2"] < 1.0
        # Rouge-L should capture common subsequence
        assert 0.0 < scores["rougeL"] < 1.0

    def test_stemmer_improves_scores(self):
        """Test that stemmer improves scores for morphological variants."""
        sample = "running quickly"
        reference = "runs quick"

        scores_with_stem = self.scorer_with_stemmer.score_sample(sample, reference)
        scores_without_stem = self.scorer_without_stemmer.score_sample(
            sample, reference
        )

        # With stemmer should have better or equal scores
        assert scores_with_stem["rouge1"] >= scores_without_stem["rouge1"]

    def test_empty_strings(self):
        """Test behavior with empty strings."""
        scores = self.scorer_with_stemmer.score_sample("", "")
        # Empty strings should return valid float scores
        assert isinstance(scores["rouge1"], float | int)
        assert isinstance(scores["rouge2"], float | int)
        assert isinstance(scores["rougeL"], float | int)

        # Should return 0 for all metrics
        assert scores["rouge1"] == pytest.approx(0.0)
        assert scores["rouge2"] == pytest.approx(0.0)
        assert scores["rougeL"] == pytest.approx(0.0)

    def test_one_empty_string(self):
        """Test when one string is empty."""
        scores = self.scorer_with_stemmer.score_sample("some text here", "")
        # Should return 0 for all metrics
        assert scores["rouge1"] == pytest.approx(0.0)
        assert scores["rouge2"] == pytest.approx(0.0)
        assert scores["rougeL"] == pytest.approx(0.0)

    def test_single_word(self):
        """Test with single word inputs."""
        scores = self.scorer_with_stemmer.score_sample("cat", "cat")
        assert scores["rouge1"] == pytest.approx(1.0)
        # Rouge-2 requires at least 2 words, should be 0 or undefined
        assert 0.0 <= scores["rouge2"] <= 1.0
        assert scores["rougeL"] == pytest.approx(1.0)

    def test_long_text(self):
        """Test with longer text samples."""
        long_sample = " ".join([f"word{i}" for i in range(100)])
        long_reference = " ".join([f"word{i}" for i in range(50, 150)])

        scores = self.scorer_with_stemmer.score_sample(long_sample, long_reference)

        # Should have some overlap (word50-word99)
        assert 0.0 < scores["rouge1"] < 1.0
        assert scores["rouge2"] >= 0.0
        assert scores["rougeL"] > 0.0

    def test_case_insensitive(self):
        """Test that ROUGE is case-insensitive."""
        scores_lower = self.scorer_with_stemmer.score_sample(
            "the cat sat", "the cat sat"
        )
        scores_mixed = self.scorer_with_stemmer.score_sample(
            "The Cat Sat", "the cat sat"
        )
        # Should be identical or very close
        assert scores_lower["rouge1"] == pytest.approx(scores_mixed["rouge1"])
        assert scores_lower["rouge2"] == pytest.approx(scores_mixed["rouge2"])
        assert scores_lower["rougeL"] == pytest.approx(scores_mixed["rougeL"])

    def test_punctuation_handling(self):
        """Test how punctuation is handled."""
        with_punct = "Hello, world! How are you?"
        without_punct = "Hello world How are you"
        reference = "Hello world How are you"

        scores_with = self.scorer_with_stemmer.score_sample(with_punct, reference)
        scores_without = self.scorer_without_stemmer.score_sample(
            without_punct, reference
        )

        # Scores should be high for both since content is same
        assert scores_with["rouge1"] > 0.8
        assert scores_without["rouge1"] > 0.8

    @pytest.mark.parametrize(
        "sample,reference,expected_min_rouge1",
        [
            ("cat", "cat", 0.99),
            ("cat dog", "cat dog", 0.99),
            ("cat", "dog", 0.0),
            ("the cat", "the dog", 0.4),  # "the" overlaps
            ("a b c", "d e f", 0.0),
        ],
    )
    def test_parametrized_cases(self, sample, reference, expected_min_rouge1):
        """Test various sample cases with expected minimum ROUGE-1 scores."""
        scores = self.scorer_with_stemmer.score_sample(sample, reference)
        assert scores["rouge1"] >= expected_min_rouge1 - 0.1  # Allow small tolerance
