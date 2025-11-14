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

"""AIME (American Invitational Mathematics Examination) evaluator.

Based on OpenAI's GPT-OSS AIME implementation.

References:
- Dataset: https://huggingface.co/datasets/opencompass/AIME2025
- OpenAI implementation: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py

AIME problems require integer answers in the range 000-999.
"""

from typing import Any

from ..evaluator import Evaluator
from ..extractors import extract_boxed_text
from ..normalizers import normalize_number
from ..scorers import exact_match_score


class AIMEEvaluator(Evaluator, name="aime"):
    """Evaluator for AIME math problems.

    AIME consists of challenging math problems requiring integer answers
    in the range 000-999.

    Implementation follows OpenAI's AIME25Eval:
    https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py

    Extraction:
    - Uses extract_boxed (based on OpenAI's extract_boxed_text)
    - Converts to integer for comparison

    Scoring:
    - Binary exact match: 1.0 if match, 0.0 otherwise
    - Supports pass@k for repeated evaluations

    Example:
        >>> evaluator = AIMEEvaluator()
        >>> response = "The solution is \\\\boxed{042}"
        >>> result = evaluator.score(evaluator.extract_answer(response), "42")
        >>> result["correct"]
        True
    """

    def extract_answer(self, response: str) -> str | None:
        """Extract numeric answer from response.

        Following OpenAI's pattern:
        1. Extract from \\boxed{...} or \\framebox{...}
        2. Fallback to last integer if no boxed notation
        3. Normalize to extract leading digits

        Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py

        Args:
            response: Model response text

        Returns:
            Extracted and normalized numeric string, or None if extraction failed
        """
        # extract_boxed_text follows OpenAI's extract_boxed_text pattern
        extracted = extract_boxed_text(response)

        if extracted:
            # Normalize to get leading digits (following OpenAI's normalize_number)
            normalized = normalize_number(extracted)
            return normalized

        return None

    def score(
        self, extracted_answer: str | None, ground_truth: str, k: int = 1
    ) -> dict[str, Any]:
        """Score extracted answer against ground truth.

        Following OpenAI's scoring logic:
        - Normalize ground truth answer
        - Convert both to integers for comparison
        - score = 1.0 if extracted_answer == correct_answer else 0.0

        Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py

        Args:
            extracted_answer: Extracted numeric answer (or None)
            ground_truth: Ground truth numeric answer
            pass_k: Not used for single sample scoring (used in evaluate_batch)

        Returns:
            Dictionary with:
            - "correct": bool indicating if answer is correct
            - "extracted": extracted answer (as string)
            - "ground_truth": normalized ground truth
        """
        # Normalize ground truth (following OpenAI: normalize_number if string)
        if isinstance(ground_truth, str):
            normalized_gt = normalize_number(ground_truth)
        else:
            normalized_gt = str(ground_truth)

        # Try to convert both to integers for comparison (following OpenAI)
        correct = False
        try:
            if extracted_answer is not None and normalized_gt is not None:
                extracted_int = int(extracted_answer)
                gt_int = int(normalized_gt)
                correct = extracted_int == gt_int
        except (ValueError, TypeError):
            # If conversion fails, fall back to string comparison
            correct = (
                extracted_answer is not None
                and normalized_gt is not None
                and extracted_answer == normalized_gt
            )

        return {
            "correct": correct,
            "extracted": extracted_answer,
            "ground_truth": normalized_gt,
        }

    def evaluate_batch(
        self, responses: list[str], ground_truths: list[str], k: int = 1
    ) -> dict[str, Any]:
        """Evaluate a batch of AIME responses.

        Following OpenAI's pattern:
        - Extract boxed answers
        - Normalize both extracted and ground truth
        - Convert to integers for comparison

        Returns redesigned structure:
        - metrics: {"EM": pass@k_score, "per_sample_EM": per_attempt_accuracy}
        - metadata: {dataset_size, repeats, k, total, correct_attempts, correct_samples}

        Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py

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

        # Normalize all ground truths (following OpenAI's approach)
        normalized_gts = []
        for gt in ground_truths:
            if isinstance(gt, str):
                norm_gt = normalize_number(gt)
            else:
                norm_gt = str(gt)
            normalized_gts.append(norm_gt)

        # Convert to integers for comparison (following OpenAI)
        # Use string representation for exact_match_score
        integer_extracted = []
        integer_gts = []

        for extracted, gt in zip(extracted_answers, normalized_gts, strict=False):
            # Try to convert to int then back to string for normalization
            try:
                if extracted is not None:
                    integer_extracted.append(str(int(extracted)))
                else:
                    integer_extracted.append(None)
            except (ValueError, TypeError):
                integer_extracted.append(extracted)

            try:
                if gt is not None:
                    integer_gts.append(str(int(gt)))
                else:
                    integer_gts.append(None)
            except (ValueError, TypeError):
                integer_gts.append(gt)

        # Use exact_match_score (always calculates pass@k)
        score_result = exact_match_score(integer_extracted, integer_gts, k=k)

        # Build per-sample details
        per_sample = []
        for i, (resp, extracted) in enumerate(
            zip(responses, integer_extracted, strict=False)
        ):
            gt_idx = i % len(integer_gts)
            gt = integer_gts[gt_idx]
            per_sample.append(
                {
                    "response": resp,
                    "extracted": extracted,
                    "ground_truth": gt,
                    "correct": score_result["scores"][i] == 1,
                }
            )

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
            "correct_attempts": score_result["correct"],
            "correct_samples": int(
                score_result["pass_at_k"] * score_result["dataset_size"]
            ),
        }

        return {
            "metrics": metrics,
            "metadata": metadata,
            "per_sample": per_sample,
            "total": score_result["total"],
        }
