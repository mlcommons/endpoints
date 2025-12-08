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

"""Generic Multiple Choice Question Evaluator using Letter Choice format (ABCD).

This evaluator is used to evaluate the accuracy of a model's responses to multiple choice questions.
It is a generic evaluator that can be used to evaluate the accuracy of a model's responses to any
multiple choice question.

It is based on the OpenAI GPT-OSS evaluator for multiple choice questions.

Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/abcd_grader.py
"""

from ..base import ABCDExtractor, Evaluator
from ..scorers import PassAt1Scorer


def normalize_choice(
    extracted_text: str,
    options: list[str] | tuple[str, ...] = ("A", "B", "C", "D"),
) -> str:
    """Normalize the extracted text to 'choiceN' format, which is the column name in the dataset."""
    if not extracted_text:
        return ""

    # Normalize to single uppercase letter
    # Strip whitespace and common punctuation
    normalized = extracted_text.strip().strip("()[]").strip(".")

    # Convert to 'choiceN' format
    for i, option in enumerate(options):
        if normalized.upper() == option.upper():
            return f"choice{i+1}"

    return ""


class MCQLetterChoiceScorer(PassAt1Scorer):
    """Scorer for multiple choice questions using letter choice format (ABCD).

    This scorer extends PassAt1Scorer by normalizing extracted answers (A/B/C/D)
    to 'choiceN' format before performing exact string matching against the ground truth.

    Uses pass@1 scoring: returns 1 if correct, 0 otherwise.
    """

    def score_sample(self, sample_output: str, ground_truth: str) -> int:
        """
        Score extracted answer against ground truth.

        Normalizes the sample output from letter format (A/B/C/D) to 'choiceN' format,
        then uses PassAt1Scorer's exact match logic.

        Following OpenAI's scoring logic:
        score = 1.0 if extracted_answer == correct_answer else 0.0

        Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py

        Args:
            sample_output: Extracted answer (A/B/C/D or None)
            ground_truth: Ground truth answer in 'choiceN' format

        Returns:
            int: 1 if extracted answer is correct, 0 otherwise
        """
        # Normalize the extracted answer to 'choiceN' format
        extracted_answer = normalize_choice(sample_output)
        # Use parent's exact match scoring
        return super().score_sample(extracted_answer, ground_truth)


class MCQLetterChoiceEvaluator(Evaluator):
    """Evaluator for multiple choice questions using letter choice format (ABCD).

    This evaluator extracts ABCD answers from model outputs and scores them against
    ground truth using pass@1 scoring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(ABCDExtractor, MCQLetterChoiceScorer(), *args, **kwargs)
