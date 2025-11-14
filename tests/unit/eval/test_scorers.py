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

"""Tests for scoring functions."""

import pytest

from inference_endpoint.eval.scorers import exact_match_score


class TestExactMatchScore:
    """Test exact_match_score function."""
    
    def test_basic_scoring(self):
        """Test basic exact match scoring."""
        responses = ["A", "B", "C"]
        ground_truths = ["A", "B", "D"]
        
        result = exact_match_score(responses, ground_truths)
        
        assert result["total"] == 3
        assert result["correct"] == 2
        assert result["accuracy"] == pytest.approx(2/3)
        assert result["scores"] == [1, 1, 0]
    
    def test_all_correct(self):
        """Test when all answers are correct."""
        responses = ["A", "B", "C"]
        ground_truths = ["A", "B", "C"]
        
        result = exact_match_score(responses, ground_truths)
        
        assert result["accuracy"] == 1.0
        assert result["correct"] == 3
    
    def test_all_incorrect(self):
        """Test when all answers are incorrect."""
        responses = ["A", "B", "C"]
        ground_truths = ["D", "D", "D"]
        
        result = exact_match_score(responses, ground_truths)
        
        assert result["accuracy"] == 0.0
        assert result["correct"] == 0
    
    def test_with_none_responses(self):
        """Test when some responses are None (extraction failed)."""
        responses = ["A", None, "C"]
        ground_truths = ["A", "B", "C"]
        
        result = exact_match_score(responses, ground_truths)
        
        assert result["total"] == 3
        assert result["correct"] == 2
        assert result["scores"] == [1, 0, 1]
    
    def test_length_mismatch(self):
        """Test error when responses not divisible by ground truths."""
        responses = ["A", "B", "C"]
        ground_truths = ["A", "B"]  # 3 is not divisible by 2
        
        with pytest.raises(ValueError, match="must be divisible"):
            exact_match_score(responses, ground_truths)
    
    def test_empty_lists(self):
        """Test with empty lists."""
        result = exact_match_score([], [], k=1)
        
        assert result["total"] == 0
        assert result["correct"] == 0
        assert result["accuracy"] == 0.0
        # With empty lists, dataset_size=0, pass_at_k=0
        assert result["pass_at_k"] == 0.0
        assert result["dataset_size"] == 0


class TestPassAtK:
    """Test pass@k calculation in exact_match_score."""
    
    def test_pass_at_k_simple(self):
        """Test pass@k with 2 samples, 2 repeats, pass@1."""
        # Dataset: 2 samples with repeats=2
        # Ground truths: [sample0, sample1] (unique)
        # Responses: [sample0_rep0, sample1_rep0, sample0_rep1, sample1_rep1]
        responses = ["A", "B", "A", "C"]
        ground_truths = ["A", "B"]  # 2 unique ground truths
        
        result = exact_match_score(responses, ground_truths, k=1)
        
        # Sample 0 (gt=A): responses=[A,A], 2/2 correct -> passes
        # Sample 1 (gt=B): responses=[B,C], 1/2 correct -> passes (at least 1)
        assert result["pass_at_k"] == 1.0
        assert result["dataset_size"] == 2
        assert result["repeats"] == 2
        assert result["k"] == 1
    
    def test_pass_at_k_all_fail(self):
        """Test pass@k when all samples fail."""
        # 2 samples, 2 repeats, pass@2 (need both correct)
        responses = ["A", "B", "C", "D"]
        ground_truths = ["A", "B"]  # 2 unique ground truths
        
        result = exact_match_score(responses, ground_truths, k=2)
        
        # Sample 0 (gt=A): responses=[A,C], 1/2 correct -> fails (needs 2)
        # Sample 1 (gt=B): responses=[B,D], 1/2 correct -> fails (needs 2)
        assert result["pass_at_k"] == 0.0
        assert result["dataset_size"] == 2
        assert result["repeats"] == 2
    
    def test_pass_at_k_partial(self):
        """Test pass@k with partial success."""
        # 3 samples, 3 repeats, pass@2
        responses = [
            "A", "B", "C",  # rep 0
            "A", "B", "C",  # rep 1
            "A", "X", "C",  # rep 2
        ]
        ground_truths = ["A", "B", "C"]  # 3 unique ground truths
        
        result = exact_match_score(responses, ground_truths, k=2)
        
        # Sample 0 (A): 3/3 correct -> passes
        # Sample 1 (B): 2/3 correct -> passes  
        # Sample 2 (C): 3/3 correct -> passes
        assert result["pass_at_k"] == 1.0
        assert result["dataset_size"] == 3
        assert result["repeats"] == 3
    
    def test_pass_at_k_strict(self):
        """Test pass@k with strict requirements."""
        # 2 samples, 3 repeats, pass@3 (must be perfect)
        responses = [
            "A", "B",  # rep 0
            "A", "B",  # rep 1
            "A", "X",  # rep 2
        ]
        ground_truths = ["A", "B"]  # 2 unique ground truths
        
        result = exact_match_score(responses, ground_truths, k=3)
        
        # Sample 0 (gt=A): responses=[A,A,A], 3/3 -> passes
        # Sample 1 (gt=B): responses=[B,B,X], 2/3 -> fails (needs 3)
        assert result["pass_at_k"] == 0.5
        assert result["dataset_size"] == 2
    
    def test_pass_k_out_of_range(self):
        """Test error when pass_k is out of valid range."""
        responses = ["A", "B", "A", "B"]
        ground_truths = ["A", "B"]  # 2 unique, so repeats=2
        
        # pass_k=3 but only 2 repeats
        with pytest.raises(ValueError, match="k must be in"):
            exact_match_score(responses, ground_truths, k=3)
    
    def test_pass_k_zero(self):
        """Test error when pass_k is zero."""
        responses = ["A", "B"]
        ground_truths = ["A", "B"]
        
        with pytest.raises(ValueError, match="k must be in"):
            exact_match_score(responses, ground_truths, k=0)


class TestExactMatchScoreEdgeCases:
    """Test edge cases for exact_match_score."""
    
    def test_single_sample_default_pass_k(self):
        """Test single sample with default pass@k=1."""
        result = exact_match_score(["A"], ["A"])
        
        assert result["accuracy"] == 1.0
        # pass@k is always calculated (default=1)
        assert result["pass_at_k"] == 1.0
        assert result["dataset_size"] == 1
        assert result["repeats"] == 1
        assert result["k"] == 1
    
    def test_single_sample_with_explicit_k(self):
        """Test single sample with explicitly set k=1."""
        result = exact_match_score(["A"], ["A"], k=1)
        
        assert result["accuracy"] == 1.0
        assert result["pass_at_k"] == 1.0
        assert result["dataset_size"] == 1
        assert result["repeats"] == 1

