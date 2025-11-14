# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for base evaluator and registry pattern."""

import pytest
from inference_endpoint.eval.evaluator import Evaluator

# Import evaluators to trigger registration
from inference_endpoint.eval.evaluators import (
    AIMEEvaluator,
    GPQAEvaluator,
    LiveCodeBenchEvaluator,
)


class TestEvaluatorRegistry:
    """Test evaluator registry pattern."""

    def test_evaluators_registered(self):
        """Test that evaluators are auto-registered."""
        available = Evaluator.get_available_evaluators()

        assert "gpqa" in available
        assert "aime" in available
        assert "livecodebench" in available

    def test_get_evaluator_gpqa(self):
        """Test getting GPQA evaluator from registry."""
        evaluator_cls = Evaluator.get_evaluator("gpqa")
        assert evaluator_cls == GPQAEvaluator

        # Can instantiate
        evaluator = evaluator_cls()
        assert isinstance(evaluator, GPQAEvaluator)
        assert isinstance(evaluator, Evaluator)

    def test_get_evaluator_aime(self):
        """Test getting AIME evaluator from registry."""
        evaluator_cls = Evaluator.get_evaluator("aime")
        assert evaluator_cls == AIMEEvaluator

    def test_get_evaluator_livecodebench(self):
        """Test getting LiveCodeBench evaluator from registry."""
        evaluator_cls = Evaluator.get_evaluator("livecodebench")
        assert evaluator_cls == LiveCodeBenchEvaluator

    def test_get_evaluator_not_found(self):
        """Test error when evaluator not in registry."""
        with pytest.raises(KeyError, match="not found"):
            Evaluator.get_evaluator("nonexistent")

    def test_get_available_evaluators(self):
        """Test getting list of available evaluators."""
        available = Evaluator.get_available_evaluators()

        assert isinstance(available, list)
        assert len(available) >= 3
        assert sorted(available) == available  # Should be sorted


class TestGPQAEvaluator:
    """Test GPQA evaluator functionality."""

    def test_extract_answer_simple(self):
        """Test extracting simple answer."""
        evaluator = GPQAEvaluator()

        response = "Answer: B"
        extracted = evaluator.extract_answer(response)
        assert extracted == "B"

    def test_extract_answer_at_end(self):
        """Test extracting answer at end of response."""
        evaluator = GPQAEvaluator()

        response = "After careful analysis, I believe the correct option is D"
        _ = evaluator.extract_answer(response)
        # Note: This may extract D from the pattern, testing that it doesn't crash

    def test_score_correct(self):
        """Test scoring correct answer."""
        evaluator = GPQAEvaluator()

        result = evaluator.score("B", "B")
        assert result["correct"] is True
        assert result["extracted"] == "B"
        assert result["ground_truth"] == "B"

    def test_score_incorrect(self):
        """Test scoring incorrect answer."""
        evaluator = GPQAEvaluator()

        result = evaluator.score("A", "B")
        assert result["correct"] is False
        assert result["extracted"] == "A"
        assert result["ground_truth"] == "B"

    def test_score_none_extracted(self):
        """Test scoring when extraction fails."""
        evaluator = GPQAEvaluator()

        result = evaluator.score(None, "B")
        assert result["correct"] is False
        assert result["extracted"] is None

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = GPQAEvaluator()

        responses = [
            "Answer: A",
            "Answer: B",
            "Answer: C",
        ]
        ground_truths = ["A", "B", "D"]

        result = evaluator.evaluate_batch(responses, ground_truths, k=1)

        assert result["total"] == 3
        assert result["metrics"]["EM"] == pytest.approx(2 / 3)
        assert result["metrics"]["per_sample_EM"] == pytest.approx(2 / 3)
        assert result["metadata"]["correct_attempts"] == 2
        assert result["metadata"]["correct_samples"] == 2
        assert len(result["per_sample"]) == 3


class TestAIMEEvaluator:
    """Test AIME evaluator functionality."""

    def test_extract_boxed(self):
        """Test extracting boxed answer."""
        evaluator = AIMEEvaluator()

        response = "The solution is \\boxed{42}"
        extracted = evaluator.extract_answer(response)
        assert extracted == "42"

    def test_extract_fallback(self):
        """Test extracting with fallback to last integer."""
        evaluator = AIMEEvaluator()

        response = "Some reasoning and the final result is 123 units"
        extracted = evaluator.extract_answer(response)
        assert extracted == "123"

    def test_score_correct(self):
        """Test scoring correct numeric answer."""
        evaluator = AIMEEvaluator()

        result = evaluator.score("42", "42")
        assert result["correct"] is True

    def test_evaluate_batch(self):
        """Test batch evaluation for AIME."""
        evaluator = AIMEEvaluator()

        responses = [
            "\\boxed{42}",
            "The answer is 123",
            "\\boxed{999}",
        ]
        ground_truths = ["42", "123", "500"]

        result = evaluator.evaluate_batch(responses, ground_truths, k=1)

        assert result["total"] == 3
        assert result["metrics"]["EM"] == pytest.approx(2 / 3)
        assert result["metrics"]["per_sample_EM"] == pytest.approx(2 / 3)
        assert result["metadata"]["correct_attempts"] == 2
