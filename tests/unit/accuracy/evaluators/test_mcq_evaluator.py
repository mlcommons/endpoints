# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MCQ Letter Choice evaluator components using fake data."""

import pytest
from inference_endpoint.accuracy.evaluators.MCQLetterChoice.entrypoint import (
    MCQLetterChoiceScorer,
    normalize_choice,
)


class TestNormalizeChoice:
    """Test cases for normalize_choice function."""

    def test_single_letter_normalization(self):
        """Test normalization of single letters."""
        assert normalize_choice("A") == "choice1"
        assert normalize_choice("B") == "choice2"
        assert normalize_choice("C") == "choice3"
        assert normalize_choice("D") == "choice4"

    def test_lowercase_letters(self):
        """Test that lowercase letters are normalized correctly."""
        assert normalize_choice("a") == "choice1"
        assert normalize_choice("b") == "choice2"
        assert normalize_choice("c") == "choice3"
        assert normalize_choice("d") == "choice4"

    def test_with_whitespace(self):
        """Test normalization with whitespace."""
        assert normalize_choice(" A ") == "choice1"
        assert normalize_choice("  B  ") == "choice2"
        assert normalize_choice("\tC\t") == "choice3"

    def test_with_parentheses(self):
        """Test normalization with parentheses."""
        assert normalize_choice("(A)") == "choice1"
        assert normalize_choice("(B)") == "choice2"

    def test_with_brackets(self):
        """Test normalization with brackets."""
        assert normalize_choice("[A]") == "choice1"
        assert normalize_choice("[B]") == "choice2"

    def test_with_period(self):
        """Test normalization with period."""
        assert normalize_choice("A.") == "choice1"
        assert normalize_choice("B.") == "choice2"

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        assert normalize_choice("") == ""

    def test_invalid_letter(self):
        """Test that invalid letters return empty string."""
        assert normalize_choice("E") == ""
        assert normalize_choice("Z") == ""
        assert normalize_choice("1") == ""
        assert normalize_choice("X") == ""

    def test_custom_options(self):
        """Test with custom options list."""
        assert normalize_choice("X", options=["X", "Y", "Z"]) == "choice1"
        assert normalize_choice("Y", options=["X", "Y", "Z"]) == "choice2"
        assert normalize_choice("Z", options=["X", "Y", "Z"]) == "choice3"

    def test_custom_options_different_count(self):
        """Test with different number of options."""
        assert normalize_choice("P", options=["P", "Q"]) == "choice1"
        assert normalize_choice("Q", options=["P", "Q"]) == "choice2"

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("A", "choice1"),
            ("B", "choice2"),
            ("C", "choice3"),
            ("D", "choice4"),
            ("a", "choice1"),
            (" A ", "choice1"),
            ("(B)", "choice2"),
            ("[C]", "choice3"),
            ("D.", "choice4"),
            ("", ""),
            ("E", ""),
            ("invalid", ""),
        ],
    )
    def test_parametrized_normalization(self, input_text, expected):
        """Test various normalization cases with parametrization."""
        assert normalize_choice(input_text) == expected


class TestMCQLetterChoiceScorer:
    """Test cases for MCQLetterChoiceScorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = MCQLetterChoiceScorer()

    def test_correct_answer_a(self):
        """Test correct answer A."""
        assert self.scorer.score_sample("A", "choice1") == 1

    def test_correct_answer_b(self):
        """Test correct answer B."""
        assert self.scorer.score_sample("B", "choice2") == 1

    def test_correct_answer_c(self):
        """Test correct answer C."""
        assert self.scorer.score_sample("C", "choice3") == 1

    def test_correct_answer_d(self):
        """Test correct answer D."""
        assert self.scorer.score_sample("D", "choice4") == 1

    def test_incorrect_answer(self):
        """Test incorrect answers."""
        assert self.scorer.score_sample("A", "choice2") == 0
        assert self.scorer.score_sample("B", "choice3") == 0
        assert self.scorer.score_sample("C", "choice4") == 0
        assert self.scorer.score_sample("D", "choice1") == 0

    def test_lowercase_input(self):
        """Test that lowercase input works correctly."""
        assert self.scorer.score_sample("a", "choice1") == 1
        assert self.scorer.score_sample("b", "choice2") == 1

    def test_with_formatting(self):
        """Test input with various formatting."""
        assert self.scorer.score_sample(" A ", "choice1") == 1
        assert self.scorer.score_sample("(B)", "choice2") == 1
        assert self.scorer.score_sample("[C]", "choice3") == 1
        assert self.scorer.score_sample("D.", "choice4") == 1

    def test_invalid_input(self):
        """Test with invalid input."""
        assert self.scorer.score_sample("E", "choice1") == 0
        assert self.scorer.score_sample("", "choice1") == 0
        assert self.scorer.score_sample("invalid", "choice1") == 0

    def test_empty_strings(self):
        """Test with empty strings."""
        # Both empty: normalize_choice("") returns "", PassAt1 matches "" == "" → 1
        assert self.scorer.score_sample("", "") == 1
        # Empty ground truth doesn't match normalized empty string
        assert self.scorer.score_sample("A", "") == 0
        # Empty output: normalize_choice("") returns "", doesn't match "choice1"
        assert self.scorer.score_sample("", "choice1") == 0

    def test_inheritance_from_pass_at_1(self):
        """Test that MCQLetterChoiceScorer properly inherits from PassAt1Scorer."""
        # After normalization, it should use PassAt1Scorer's exact match
        # A normalized to choice1, which matches ground truth
        assert self.scorer.score_sample("A", "choice1") == 1

        # B normalized to choice2, which doesn't match choice1
        assert self.scorer.score_sample("B", "choice1") == 0

    @pytest.mark.parametrize(
        "sample_output,ground_truth,expected_score",
        [
            ("A", "choice1", 1),
            ("B", "choice2", 1),
            ("C", "choice3", 1),
            ("D", "choice4", 1),
            ("a", "choice1", 1),
            (" A ", "choice1", 1),
            ("(B)", "choice2", 1),
            ("A", "choice2", 0),
            ("B", "choice1", 0),
            ("E", "choice1", 0),
            ("", "choice1", 0),
        ],
    )
    def test_parametrized_scoring(self, sample_output, ground_truth, expected_score):
        """Test various scoring scenarios with parametrization."""
        assert self.scorer.score_sample(sample_output, ground_truth) == expected_score

    def test_batch_scoring(self):
        """Test scoring a batch of samples."""
        test_cases = [
            ("A", "choice1", 1),
            ("B", "choice2", 1),
            ("C", "choice3", 1),
            ("D", "choice4", 1),
            ("A", "choice2", 0),
            ("B", "choice3", 0),
        ]

        for sample, truth, expected in test_cases:
            score = self.scorer.score_sample(sample, truth)
            assert score == expected, f"Failed for {sample} vs {truth}"

    def test_realistic_mcq_scenario(self):
        """Test a realistic MCQ evaluation scenario."""
        # Simulate 10 questions with various answers
        samples = [
            ("A", "choice1"),  # Correct
            ("B", "choice1"),  # Wrong
            ("C", "choice3"),  # Correct
            ("D", "choice4"),  # Correct
            ("A", "choice2"),  # Wrong
            ("B", "choice2"),  # Correct
            ("D", "choice3"),  # Wrong
            ("C", "choice3"),  # Correct
            ("A", "choice1"),  # Correct
            ("B", "choice4"),  # Wrong
        ]

        scores = [self.scorer.score_sample(sample, truth) for sample, truth in samples]

        # Should have 6 correct out of 10
        assert sum(scores) == 6
        assert sum(scores) / len(scores) == 0.6  # 60% accuracy

    def test_edge_case_all_same_answer(self):
        """Test edge case where model always predicts same answer."""
        # Model always predicts "A"
        ground_truths = ["choice1", "choice2", "choice3", "choice4", "choice1"]

        scores = [self.scorer.score_sample("A", truth) for truth in ground_truths]

        # Should only get 2 correct (when ground truth is choice1)
        assert sum(scores) == 2
        assert scores == [1, 0, 0, 0, 1]

    def test_mixed_case_and_formatting(self):
        """Test with mixed case and formatting in batch."""
        test_cases = [
            ("a", "choice1"),
            ("B", "choice2"),
            (" C ", "choice3"),
            ("(D)", "choice4"),
            ("[A]", "choice1"),
            ("b.", "choice2"),
        ]

        scores = [
            self.scorer.score_sample(sample, truth) for sample, truth in test_cases
        ]

        # All should be correct
        assert all(score == 1 for score in scores)
        assert sum(scores) == len(test_cases)
