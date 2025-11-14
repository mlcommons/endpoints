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

"""Shared evaluation logic called from benchmark, eval, and eval-results commands."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Import evaluators to register them
from . import evaluators  # noqa: F401
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class EvaluationReport:
    """Container for evaluation results.

    Redesigned structure:
    - Metrics: Core evaluation scores (e.g., "EM", "ROUGE-1", etc.)
    - Metadata: Evaluation settings and context (dataset_size, repeats, k, etc.)
    - Per-sample results: Detailed breakdown

    For binary exact match evaluators:
    - Metrics: {"EM": pass@k_score, "per_sample_EM": per_attempt_accuracy}
    - Metadata: {dataset_size, repeats, k, total, correct_attempts, correct_samples}

    For similarity evaluators (e.g., ROUGE):
    - Metrics: {"rouge1_f1": 0.45, "rouge2_f1": 0.30, ...}
    - Metadata: {dataset_size, total}

    Attributes:
        metrics: Dictionary mapping metric names to values
        total: Total number of responses evaluated
        per_sample_results: Detailed results for each sample
        evaluator_name: Name of evaluator used
        metadata: Evaluation settings and context
    """

    def __init__(
        self,
        metrics: dict[str, float],
        total: int,
        evaluator_name: str,
        per_sample_results: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize evaluation report.

        Args:
            metrics: Dictionary of metric names to values
            total: Total responses
            evaluator_name: Evaluator used
            per_sample_results: Per-sample details
            metadata: Evaluation settings (dataset_size, repeats, k, etc.)
        """
        self.metrics = metrics
        self.total = total
        self.evaluator_name = evaluator_name
        self.per_sample_results = per_sample_results
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dictionary representation of the report
        """
        result = {
            "evaluator": self.evaluator_name,
            "metrics": self.metrics,
            "total": self.total,
            "per_sample": self.per_sample_results,
        }

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "Evaluation Results",
            "=" * 60,
            f"Evaluator: {self.evaluator_name}",
        ]

        # Add metadata about dataset structure
        if "dataset_size" in self.metadata and "repeats" in self.metadata:
            ds_size = self.metadata["dataset_size"]
            repeats = self.metadata["repeats"]
            lines.append(
                f"Dataset: {ds_size} samples × {repeats} repeats = {self.total} total"
            )

        # Add all metrics
        for metric_name, metric_value in self.metrics.items():
            if metric_name == "per_sample_EM":
                # Skip per-sample EM in summary (not useful for large datasets)
                continue
            elif metric_name == "EM":
                # Final exact match metric (pass@k)
                k = self.metadata.get("k", "?")
                correct_samples = self.metadata.get("correct_samples", None)
                dataset_size = self.metadata.get("dataset_size", None)
                if correct_samples is not None and dataset_size is not None:
                    lines.append(
                        f"EM (pass@{k}): {metric_value:.2%} ({correct_samples}/{dataset_size})"
                    )
                else:
                    lines.append(f"EM (pass@{k}): {metric_value:.2%}")
            else:
                # Generic metric display
                if isinstance(metric_value, float):
                    lines.append(f"{metric_name}: {metric_value:.4f}")
                else:
                    lines.append(f"{metric_name}: {metric_value}")

        lines.append("=" * 60)
        return "\n".join(lines)


def evaluate_results(
    results: pd.DataFrame | dict | list[dict],
    evaluator_name: str,
    k: int = 1,
    output_path: Path | None = None,
) -> EvaluationReport:
    """Core evaluation function shared across commands.

    This function is called by:
    - `benchmark` command after collecting responses (TestMode.ACC/BOTH)
    - `eval` command after running benchmark
    - `eval-results` command with loaded results

    Args:
        results: Benchmark results containing responses and ground truths.
                Can be DataFrame or dict/list of dicts with keys:
                - "response_output" or "response": model response
                - "ground_truth": expected answer
        evaluator_name: Which evaluator to use (from registry)
        k: k value for pass@k calculation (default: 1)
        output_path: Optional path to save detailed results

    Returns:
        EvaluationReport with metrics and metadata

    Raises:
        KeyError: If evaluator not found in registry
        ValueError: If required fields missing from results

    Example:
        >>> results_df = pd.DataFrame({
        ...     "response_output": ["A", "B", "C"],
        ...     "ground_truth": ["A", "B", "D"]
        ... })
        >>> report = evaluate_results(results_df, "gpqa", k=1)
        >>> print(report.summary())
    """
    # Get evaluator from registry
    try:
        evaluator_cls = Evaluator.get_evaluator(evaluator_name)
        evaluator = evaluator_cls()
        logger.info(f"Using evaluator: {evaluator_name}")
    except KeyError:
        available = ", ".join(Evaluator.get_available_evaluators())
        logger.error(f"Evaluator '{evaluator_name}' not found. Available: {available}")
        raise

    # Convert results to DataFrame if needed
    if isinstance(results, list):
        results = pd.DataFrame(results)
    elif isinstance(results, dict):
        results = pd.DataFrame([results])

    # Extract responses and ground truths
    if "response_output" in results.columns:
        responses = results["response_output"].tolist()
    elif "response" in results.columns:
        responses = results["response"].tolist()
    else:
        raise ValueError("Results must contain 'response_output' or 'response' column")

    if "ground_truth" not in results.columns:
        raise ValueError("Results must contain 'ground_truth' column")

    ground_truths = results["ground_truth"].tolist()

    logger.info(f"Evaluating {len(responses)} responses")
    logger.info(f"Calculating EM with pass@{k}")

    # Evaluate using the selected evaluator
    eval_result = evaluator.evaluate_batch(responses, ground_truths, k=k)

    # Extract metrics and metadata from eval_result
    # Evaluators now return a clean structure with metrics and metadata separated
    metrics = eval_result["metrics"]
    metadata = eval_result["metadata"]

    report = EvaluationReport(
        metrics=metrics,
        total=eval_result["total"],
        evaluator_name=evaluator_name,
        per_sample_results=eval_result["per_sample"],
        metadata=metadata,
    )

    # Save detailed results if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved detailed results to: {output_path}")

    # Log summary
    logger.info(f"\n{report.summary()}")

    return report
