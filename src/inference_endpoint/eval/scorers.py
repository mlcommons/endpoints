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

"""Scoring functions for evaluating model responses."""

from typing import Any


def exact_match_score(
    responses: list[str | None],
    ground_truths: list[str | None],
    k: int = 1
) -> dict[str, Any]:
    """Score responses using exact match comparison.
    
    Binary scoring: 1 if response exactly matches ground truth, 0 otherwise.
    
    Always calculates pass@k metric by grouping responses by original sample.
    Responses must be organized as [sample0, sample1, ..., sample0, sample1, ...]
    when repeats > 1.
    
    Args:
        responses: List of extracted answers (or None if extraction failed)
        ground_truths: List of unique ground truth answers (or None)
        k: k value for pass@k calculation (default: 1, must be in [1, repeats])
    
    Returns:
        Dictionary with scoring results:
        - "scores": list of binary scores (1 or 0)
        - "accuracy": overall accuracy (per-attempt)
        - "pass_at_k": pass@k metric
        - "dataset_size": number of unique samples
        - "repeats": number of repeats per sample
        - "k": k value used
        - "total": total number of responses
        - "correct": number of correct responses
    
    Raises:
        ValueError: If lengths don't match or k is invalid
    
    Examples:
        >>> exact_match_score(["A", "B", "C"], ["A", "B", "D"])
        {'scores': [1, 1, 0], 'accuracy': 0.666..., 'pass_at_k': 0.666..., 'k': 1, ...}
        
        >>> # With repeats=2, k=1: [sample0_rep0, sample1_rep0, sample0_rep1, sample1_rep1]
        >>> exact_match_score(["A", "B", "A", "C"], ["A", "B"], k=1)
        {'scores': [1, 1, 1, 0], 'accuracy': 0.75, 'pass_at_k': 1.0, 'k': 1, ...}
    """
    total_responses = len(responses)
    dataset_size = len(ground_truths)
    
    # Handle empty lists
    if total_responses == 0 and dataset_size == 0:
        return {
            "scores": [],
            "accuracy": 0.0,
            "pass_at_k": 0.0,
            "dataset_size": 0,
            "repeats": 0,
            "k": k,
            "total": 0,
            "correct": 0,
        }
    
    # Check if responses is a multiple of ground_truths (for repeats)
    if dataset_size == 0 or total_responses % dataset_size != 0:
        raise ValueError(
            f"Number of responses ({total_responses}) must be divisible by "
            f"number of ground truths ({dataset_size})"
        )
    
    repeats = total_responses // dataset_size
    
    # Calculate per-response scores
    # Match each response with corresponding ground truth using modulo
    scores = []
    for i, resp in enumerate(responses):
        gt_idx = i % dataset_size
        gt = ground_truths[gt_idx]
        score = 1 if resp is not None and gt is not None and resp == gt else 0
        scores.append(score)
    
    correct_count = sum(scores)
    accuracy = correct_count / total_responses if total_responses > 0 else 0.0
    
    # Validate k
    if k < 1 or k > repeats:
        raise ValueError(
            f"k must be in [1, repeats], got k={k}, repeats={repeats}"
        )
    
    # Group scores by original sample index
    sample_scores = {}
    for i, score in enumerate(scores):
        sample_idx = i % dataset_size
        if sample_idx not in sample_scores:
            sample_scores[sample_idx] = []
        sample_scores[sample_idx].append(score)
    
    # Calculate pass@k: at least k correct out of N repeats
    pass_at_k_count = 0
    for sample_idx in sorted(sample_scores.keys()):
        sample_score_list = sample_scores[sample_idx]
        correct_in_sample = sum(sample_score_list)
        if correct_in_sample >= k:
            pass_at_k_count += 1
    
    pass_at_k_metric = pass_at_k_count / dataset_size if dataset_size > 0 else 0.0
    
    return {
        "scores": scores,
        "accuracy": accuracy,
        "pass_at_k": pass_at_k_metric,
        "dataset_size": dataset_size,
        "repeats": repeats,
        "k": k,
        "total": total_responses,
        "correct": correct_count,
    }


def rouge_score(
    responses: list[str | None],
    ground_truths: list[str]
) -> dict[str, Any]:
    """Score responses using ROUGE similarity metrics.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    overlap between generated and reference text. Commonly used for
    summarization tasks.
    
    NOTE: pass@k is NOT supported for ROUGE as it's a similarity metric,
    not a binary pass/fail metric.
    
    Args:
        responses: List of generated responses
        ground_truths: List of reference texts
    
    Returns:
        Dictionary with ROUGE scores:
        - "rouge1": ROUGE-1 (unigram) scores
        - "rouge2": ROUGE-2 (bigram) scores
        - "rougeL": ROUGE-L (longest common subsequence) scores
        - "avg_rouge1_f1": Average ROUGE-1 F1 score
        - "avg_rouge2_f1": Average ROUGE-2 F1 score
        - "avg_rougeL_f1": Average ROUGE-L F1 score
    
    Raises:
        ValueError: If lengths don't match
        ImportError: If rouge-score package not installed
    
    Note:
        Requires: pip install rouge-score
    """
    if len(responses) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(responses)} responses vs {len(ground_truths)} ground truths"
        )
    
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score package not installed. Install with: pip install rouge-score"
        )
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for response, gt in zip(responses, ground_truths):
        if response is None:
            response = ""
        
        scores = scorer.score(gt, response)
        rouge1_scores.append({
            "precision": scores['rouge1'].precision,
            "recall": scores['rouge1'].recall,
            "fmeasure": scores['rouge1'].fmeasure,
        })
        rouge2_scores.append({
            "precision": scores['rouge2'].precision,
            "recall": scores['rouge2'].recall,
            "fmeasure": scores['rouge2'].fmeasure,
        })
        rougeL_scores.append({
            "precision": scores['rougeL'].precision,
            "recall": scores['rougeL'].recall,
            "fmeasure": scores['rougeL'].fmeasure,
        })
    
    # Calculate averages
    avg_rouge1_f1 = sum(s["fmeasure"] for s in rouge1_scores) / len(rouge1_scores)
    avg_rouge2_f1 = sum(s["fmeasure"] for s in rouge2_scores) / len(rouge2_scores)
    avg_rougeL_f1 = sum(s["fmeasure"] for s in rougeL_scores) / len(rougeL_scores)
    
    return {
        "rouge1": rouge1_scores,
        "rouge2": rouge2_scores,
        "rougeL": rougeL_scores,
        "avg_rouge1_f1": avg_rouge1_f1,
        "avg_rouge2_f1": avg_rouge2_f1,
        "avg_rougeL_f1": avg_rougeL_f1,
        "total": len(responses),
    }

