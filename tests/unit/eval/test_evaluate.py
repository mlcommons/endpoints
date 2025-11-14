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

"""Tests for core evaluate_results function."""

import pandas as pd
import pytest

from inference_endpoint.eval.evaluate import evaluate_results, EvaluationReport


class TestEvaluateResults:
    """Test core evaluate_results function."""
    
    def test_basic_evaluation_gpqa(self):
        """Test basic GPQA evaluation."""
        results = pd.DataFrame({
            "response_output": ["The answer is A", "I choose B", "The answer is C"],
            "ground_truth": ["A", "B", "D"],
        })
        
        report = evaluate_results(results, evaluator_name="gpqa", k=1)
        
        assert isinstance(report, EvaluationReport)
        assert report.total == 3
        assert report.evaluator_name == "gpqa"
        # Check metrics dict (new structure)
        assert "EM" in report.metrics
        assert "per_sample_EM" in report.metrics
        assert report.metrics["EM"] == pytest.approx(2/3)  # Final metric (pass@1)
        assert report.metrics["per_sample_EM"] == pytest.approx(2/3)  # Per-attempt
        # Check metadata
        assert report.metadata["k"] == 1
        assert report.metadata["dataset_size"] == 3
        assert report.metadata["repeats"] == 1
        assert report.metadata["correct_attempts"] == 2
        assert report.metadata["correct_samples"] == 2
    
    def test_evaluation_with_single_dict_input(self):
        """Test evaluation with single dict input (single sample)."""
        results = {
            "response_output": "Answer is A",
            "ground_truth": "A",
        }
        
        report = evaluate_results(results, evaluator_name="gpqa", k=1)
        
        assert report.total == 1
        assert report.metrics["EM"] == 1.0
        assert report.metrics["per_sample_EM"] == 1.0
        assert report.metadata["correct_attempts"] == 1
    
    def test_evaluation_with_list_input(self):
        """Test evaluation with list of dicts."""
        results = [
            {"response_output": "Answer is A", "ground_truth": "A"},
            {"response_output": "Answer is B", "ground_truth": "B"},
        ]
        
        report = evaluate_results(results, evaluator_name="gpqa", k=1)
        
        assert report.total == 2
        assert report.metrics["EM"] == 1.0
        assert report.metrics["per_sample_EM"] == 1.0
        assert report.metadata["correct_attempts"] == 2
    
    def test_evaluation_with_pass_k(self):
        """Test evaluation with pass@k."""
        # This test simulates what would happen internally in the evaluator
        # when it receives responses and ground_truths separately
        from inference_endpoint.eval.evaluators import GPQAEvaluator
        
        evaluator = GPQAEvaluator()
        
        # 2 samples, 2 repeats each
        responses = [
            "Answer is A",  # sample 0, rep 0
            "Answer is B",  # sample 1, rep 0
            "Answer is A",  # sample 0, rep 1
            "Answer is C",  # sample 1, rep 1
        ]
        ground_truths = ["A", "B"]  # 2 unique ground truths
        
        result = evaluator.evaluate_batch(responses, ground_truths, k=1)
        
        assert result["total"] == 4
        # New structure
        assert result["metrics"]["per_sample_EM"] == 0.75  # 3/4 correct
        assert result["metrics"]["EM"] == 1.0  # Both samples pass (at least 1 correct)
        assert result["metadata"]["k"] == 1
        assert result["metadata"]["dataset_size"] == 2
        assert result["metadata"]["repeats"] == 2
        assert result["metadata"]["correct_attempts"] == 3
        assert result["metadata"]["correct_samples"] == 2
    
    def test_evaluation_aime(self):
        """Test AIME evaluation."""
        results = pd.DataFrame({
            "response_output": ["\\boxed{42}", "The answer is 100", "\\boxed{999}"],
            "ground_truth": ["42", "100", "500"],
        })
        
        report = evaluate_results(results, evaluator_name="aime", k=1)
        
        assert report.total == 3
        assert report.metrics["EM"] == pytest.approx(2/3)
        assert report.metrics["per_sample_EM"] == pytest.approx(2/3)
        assert report.metadata["correct_attempts"] == 2
    
    def test_evaluator_not_found(self):
        """Test error when evaluator not in registry."""
        results = pd.DataFrame({
            "response_output": ["answer"],
            "ground_truth": ["truth"],
        })
        
        with pytest.raises(KeyError, match="not found"):
            evaluate_results(results, evaluator_name="nonexistent")
    
    def test_missing_response_column(self):
        """Test error when response column missing."""
        results = pd.DataFrame({
            "no_response": ["answer"],
            "ground_truth": ["truth"],
        })
        
        with pytest.raises(ValueError, match="response"):
            evaluate_results(results, evaluator_name="gpqa")
    
    def test_missing_ground_truth_column(self):
        """Test error when ground_truth column missing."""
        results = pd.DataFrame({
            "response_output": ["answer"],
            "no_ground_truth": ["truth"],
        })
        
        with pytest.raises(ValueError, match="ground_truth"):
            evaluate_results(results, evaluator_name="gpqa")
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        results = pd.DataFrame({
            "response_output": ["Answer is A", "Answer is B"],
            "ground_truth": ["A", "B"],
        })
        
        report = evaluate_results(results, evaluator_name="gpqa", k=1)
        report_dict = report.to_dict()
        
        assert "evaluator" in report_dict
        assert "metrics" in report_dict
        assert "total" in report_dict
        assert "per_sample" in report_dict
        assert "metadata" in report_dict
        assert report_dict["evaluator"] == "gpqa"
        assert "EM" in report_dict["metrics"]
        assert "per_sample_EM" in report_dict["metrics"]
    
    def test_report_summary(self):
        """Test report summary generation."""
        results = pd.DataFrame({
            "response_output": ["Answer is A"],
            "ground_truth": ["A"],
        })
        
        report = evaluate_results(results, evaluator_name="gpqa", k=1)
        summary = report.summary()
        
        assert isinstance(summary, str)
        assert "Evaluation Results" in summary
        assert "gpqa" in summary
        assert "EM" in summary
        assert "pass@1" in summary

