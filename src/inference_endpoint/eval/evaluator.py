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

"""Base evaluator class with registry pattern for accuracy evaluation."""

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """Base class for dataset-specific evaluators.

    Evaluators implement dataset-specific logic for:
    1. Extracting answers from model responses
    2. Scoring answers against ground truth
    3. Batch evaluation of multiple responses

    Evaluator implementations auto-register via __init_subclass__ by specifying
    the name parameter. This enables runtime selection of evaluators:

        evaluator_cls = Evaluator.get_evaluator("gpqa")
        evaluator = evaluator_cls()
        results = evaluator.evaluate_batch(responses, ground_truths)

    Built-in evaluators: see evaluators/__init__.py

    Attributes:
        _IMPL_MAP: Class-level registry mapping evaluator names (str) to Evaluator classes.

    Example:
        class GPQAEvaluator(Evaluator, name="gpqa"):
            def extract_answer(self, response):
                return extract_multiple_choice(response)

            def score(self, extracted, ground_truth):
                return exact_match_score(extracted, ground_truth)
    """

    # Registry for evaluator implementations (populated via __init_subclass__)
    # Uses string keys instead of enums to avoid cyclic imports
    _IMPL_MAP: dict[str, type["Evaluator"]] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        """Auto-register evaluator implementations.

        Args:
            name: Evaluator name to register (e.g., "gpqa", "aime", "livecodebench")

        Raises:
            ValueError: If name already registered
        """
        super().__init_subclass__(**kwargs)

        if name is not None:
            if name in Evaluator._IMPL_MAP:
                raise ValueError(
                    f"Cannot register {cls.__name__} with name '{name}' - "
                    f"Already registered to {Evaluator._IMPL_MAP[name].__name__}"
                )
            Evaluator._IMPL_MAP[name] = cls

    @classmethod
    def get_evaluator(cls, name: str) -> type["Evaluator"]:
        """Get evaluator implementation by name.

        Args:
            name: Evaluator name (e.g., "gpqa", "aime", "livecodebench")

        Returns:
            Evaluator subclass

        Raises:
            KeyError: If evaluator name not found in registry
        """
        if name not in cls._IMPL_MAP:
            available = ", ".join(sorted(cls._IMPL_MAP.keys()))
            raise KeyError(
                f"Evaluator '{name}' not found. Available evaluators: {available}"
            )
        return cls._IMPL_MAP[name]

    @classmethod
    def get_available_evaluators(cls) -> list[str]:
        """Get list of registered evaluator names.

        Returns:
            Sorted list of available evaluator names
        """
        return sorted(cls._IMPL_MAP.keys())

    @abstractmethod
    def extract_answer(self, response: str) -> str | None:
        """Extract answer from model response.

        Args:
            response: Raw model response text

        Returns:
            Extracted answer string, or None if extraction failed
        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self, extracted_answer: str | None, ground_truth: str, k: int = 1
    ) -> dict[str, Any]:
        """Score extracted answer against ground truth.

        Args:
            extracted_answer: Extracted answer from response (or None if extraction failed)
            ground_truth: Ground truth answer
            k: k value for pass@k (default: 1)

        Returns:
            Dictionary with score information:
            - "correct": bool (for binary scorers) or float (for similarity scorers)
            - "extracted": extracted answer
            - "ground_truth": ground truth answer
            - Additional scorer-specific fields
        """
        raise NotImplementedError

    def evaluate_batch(
        self, responses: list[str], ground_truths: list[str], k: int = 1
    ) -> dict[str, Any]:
        """Evaluate a batch of responses.

        NOTE: Subclasses should override this for efficient batch processing.

        Args:
            responses: List of model responses
            ground_truths: List of unique ground truth answers
            k: k value for pass@k (default: 1)

        Returns:
            Dictionary with evaluation results:
            - "metrics": dict of metric names to values
            - "metadata": dict of evaluation settings
            - "per_sample": list of per-response details
            - "total": total number of responses

        Raises:
            ValueError: If responses not divisible by ground_truths
        """
        # This is a fallback implementation - subclasses should override
        raise NotImplementedError(
            "Subclasses must implement evaluate_batch for efficient batch processing"
        )
