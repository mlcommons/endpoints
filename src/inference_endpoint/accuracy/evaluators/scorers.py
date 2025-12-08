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

"""Concrete implementations of Scorer classes for various evaluation metrics."""

from rouge_score import rouge_scorer

from .base import Scorer


class PassAt1Scorer(Scorer):
    """Implements pass@1 scoring as defined by Artificial Analysis.

    pass@1 means the model gets exactly one attempt to produce the correct answer.
    The score is 1 if the output matches the ground truth exactly, 0 otherwise.

    This is the standard scoring method for multiple-choice questions and other
    tasks where there is a single correct answer.

    Reference: https://artificialanalysis.ai/methodology/intelligence-benchmarking
    """

    def score_sample(self, sample_output: str, ground_truth: str) -> int:
        """
        Score using exact string match (pass@1).

        Args:
            sample_output: The extracted output from the model.
            ground_truth: The ground truth answer.

        Returns:
            1 if exact match, 0 otherwise.
        """
        return 1 if sample_output == ground_truth else 0


class RougeScorer(Scorer):
    """Production-ready ROUGE scorer that returns multiple metrics (ROUGE-1, ROUGE-2, ROUGE-L).

    This scorer uses Google Research's `rouge-score` library for accurate ROUGE computation.
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is commonly used for
    evaluating text summarization and generation tasks.

    The scorer returns F1 scores for ROUGE-1 (unigram overlap), ROUGE-2 (bigram overlap),
    and ROUGE-L (longest common subsequence).

    Reference: https://github.com/google-research/google-research/tree/master/rouge
    """

    def __init__(self, use_stemmer: bool = True):
        """
        Initialize the ROUGE scorer.

        Args:
            use_stemmer: Whether to use Porter stemmer for token normalization.
                        Default is True for better matching across word variations.
        """
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
        )

    def score_sample(self, sample_output: str, ground_truth: str) -> dict[str, float]:
        """
        Calculate ROUGE metrics between sample output and ground truth using rouge-score library.

        Args:
            sample_output: The generated text from the model.
            ground_truth: The reference text.

        Returns:
            Dictionary with keys 'rouge1', 'rouge2', 'rougeL' mapping to F1 scores.
            Each score is a float between 0.0 and 1.0.
        """
        # Calculate scores using rouge-score library
        scores = self._scorer.score(ground_truth, sample_output)

        # Extract F1 scores (fmeasure) for each metric
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
