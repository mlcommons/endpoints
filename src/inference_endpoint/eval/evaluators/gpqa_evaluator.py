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

"""GPQA (Graduate-Level Google-Proof Q&A) evaluator.

Based on OpenAI's GPT-OSS GPQA implementation.

References:
- Paper: https://arxiv.org/abs/2311.12022
- OpenAI implementation: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
- Answer extraction: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/abcd_grader.py
"""

from typing import Any

from ..evaluator import Evaluator
from ..extractors import extract_abcd
from ..normalizers import normalize_multiple_choice
from ..scorers import exact_match_score


class GPQAEvaluator(Evaluator, name="gpqa"):
    """Evaluator for GPQA multiple choice questions.
    
    GPQA consists of graduate-level questions in science with four options (A/B/C/D).
    The evaluator uses OpenAI's extract_abcd pattern to extract the chosen option
    and compares it to the ground truth using exact match.
    
    Implementation follows:
    https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
    
    Scoring:
    - Binary exact match (1 if correct, 0 if incorrect)
    - Supports pass@k for repeated evaluations
    
    Example:
        >>> evaluator = GPQAEvaluator()
        >>> response = "After careful consideration, I believe the answer is B."
        >>> result = evaluator.score(evaluator.extract_answer(response), "B")
        >>> result["correct"]
        True
    """
    
    def extract_answer(self, response: str) -> str | None:
        """Extract ABCD answer from response.
        
        Uses OpenAI's extract_abcd pattern matching with priority-based
        selection and fallback to first character if no pattern matches.
        
        Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/abcd_grader.py
        
        Args:
            response: Model response text
        
        Returns:
            Extracted answer as uppercase letter (A/B/C/D), or None if extraction failed
        """
        # extract_abcd already returns uppercase A/B/C/D or None
        # Following OpenAI's implementation, we apply minimal normalization
        extracted = extract_abcd(response)
        if extracted:
            # Normalize to ensure it's a clean single uppercase letter
            return normalize_multiple_choice(extracted)
        return None
    
    def score(
        self,
        extracted_answer: str | None,
        ground_truth: str,
        k: int = 1
    ) -> dict[str, Any]:
        """Score extracted answer against ground truth.
        
        Following OpenAI's scoring logic:
        score = 1.0 if extracted_answer == correct_answer else 0.0
        
        Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
        
        Args:
            extracted_answer: Extracted answer (A/B/C/D or None)
            ground_truth: Ground truth answer (A/B/C/D)
            pass_k: Not used for single sample scoring (used in evaluate_batch)
        
        Returns:
            Dictionary with:
            - "correct": bool indicating if answer is correct
            - "extracted": extracted answer
            - "ground_truth": ground truth answer
        """
        # Normalize ground truth for comparison
        normalized_gt = normalize_multiple_choice(ground_truth)
        
        # Binary scoring: 1.0 if match, 0.0 otherwise (following OpenAI)
        correct = (
            extracted_answer is not None
            and extracted_answer == normalized_gt
        )
        
        return {
            "correct": correct,
            "extracted": extracted_answer,
            "ground_truth": normalized_gt,
        }
    
    def evaluate_batch(
        self,
        responses: list[str],
        ground_truths: list[str],
        k: int = 1
    ) -> dict[str, Any]:
        """Evaluate a batch of GPQA responses.
        
        Returns redesigned structure:
        - metrics: {"EM": pass@k_score, "per_sample_EM": per_attempt_accuracy}
        - metadata: {dataset_size, repeats, k, total, correct_attempts, correct_samples}
        - per_sample: detailed per-response results
        - total: total number of responses
        
        Args:
            responses: List of model responses
            ground_truths: List of unique ground truth answers
            k: k value for pass@k (default: 1)
        
        Returns:
            Dictionary with:
            - "metrics": dict of metric names to values
            - "metadata": dict of evaluation settings
            - "per_sample": list of per-response details
            - "total": total number of responses
        """
        # Extract all answers
        extracted_answers = [self.extract_answer(resp) for resp in responses]
        
        # Normalize all ground truths
        normalized_gts = [normalize_multiple_choice(gt) for gt in ground_truths]
        
        # Use exact_match_score (always calculates pass@k)
        score_result = exact_match_score(extracted_answers, normalized_gts, k=k)
        
        # Build per-sample details
        per_sample = []
        for i, (resp, extracted) in enumerate(zip(responses, extracted_answers)):
            gt_idx = i % len(normalized_gts)
            gt = normalized_gts[gt_idx]
            per_sample.append({
                "response": resp,
                "extracted": extracted,
                "ground_truth": gt,
                "correct": score_result["scores"][i] == 1,
            })
        
        # Build metrics dictionary
        metrics = {
            "EM": score_result["pass_at_k"],  # Final metric (pass@k)
            "per_sample_EM": score_result["accuracy"],  # Per-attempt accuracy
        }
        
        # Build metadata dictionary
        metadata = {
            "dataset_size": score_result["dataset_size"],
            "repeats": score_result["repeats"],
            "k": score_result["k"],
            "total": score_result["total"],
            "correct_attempts": score_result["correct"],  # Correct attempts out of total
            "correct_samples": int(score_result["pass_at_k"] * score_result["dataset_size"]),  # Samples passing pass@k
        }
        
        return {
            "metrics": metrics,
            "metadata": metadata,
            "per_sample": per_sample,
            "total": score_result["total"],
        }

