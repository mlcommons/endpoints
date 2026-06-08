# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""BFCL v4 Multi-Turn Scorer.

Scores multi-turn conversations using bfcl-eval's multi_turn_checker, which
executes both model and ground truth function calls against simulated classes
and compares resulting states.

Produces per-subset accuracy and an overall multi_turn category score
(unweighted mean of subset scores), matching evalscope's aggregation.
"""

import logging
from collections import defaultdict
from typing import Any

from ..dataset_manager.predefined.bfcl_v4.multi_turn import MULTI_TURN_SUBSETS

logger = logging.getLogger(__name__)

try:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )
except ImportError:
    multi_turn_checker = None

_MODEL_NAME_FOR_SCORING = "mlcommons_endpoints_eval"


class BFCLv4MultiTurnScorer:
    """Scores BFCL v4 multi-turn conversation results.

    Takes the accumulated results from BFCLMultiTurnRunner and evaluates each
    entry using bfcl-eval's multi_turn_checker. Produces per-subset and
    category-level accuracy scores.
    """

    def __init__(self):
        if multi_turn_checker is None:
            raise ImportError(
                "bfcl-eval is required for BFCL v4 multi-turn scoring. "
                "Install with: pip install inference-endpoint[bfcl]"
            )

    def score(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Score all multi-turn conversation results.

        Args:
            results: List of result dicts from BFCLMultiTurnRunner.run_all().
                Each dict contains: entry_id, subset, model_results_per_turn,
                ground_truth, initial_config, involved_classes, force_terminated.

        Returns:
            Dict with:
                - overall_accuracy: category-level score (unweighted mean of subsets)
                - subset_scores: {subset_name: accuracy_pct}
                - per_entry_scores: [{entry_id, valid, error_type, ...}]
                - total_samples: total entries evaluated
        """
        per_entry_scores: list[dict[str, Any]] = []
        subset_correct: dict[str, int] = defaultdict(int)
        subset_total: dict[str, int] = defaultdict(int)

        for result in results:
            entry_id = result["entry_id"]
            subset = result["subset"]
            subset_total[subset] += 1

            # Force-terminated entries are automatically failed
            if result["force_terminated"]:
                per_entry_scores.append(
                    {
                        "entry_id": entry_id,
                        "subset": subset,
                        "valid": False,
                        "error_type": "force_terminated",
                        "error_message": (
                            f"Force terminated after {result['num_turns_completed']} "
                            f"of {result['num_turns_expected']} turns"
                        ),
                    }
                )
                continue

            # Check turn count mismatch
            model_results = result["model_results_per_turn"]
            ground_truth = result["ground_truth"]

            if len(model_results) != len(ground_truth):
                per_entry_scores.append(
                    {
                        "entry_id": entry_id,
                        "subset": subset,
                        "valid": False,
                        "error_type": "turn_count_mismatch",
                        "error_message": (
                            f"Model produced {len(model_results)} turns, "
                            f"expected {len(ground_truth)}"
                        ),
                    }
                )
                continue

            # Build test_entry dict for multi_turn_checker
            test_entry = {
                "id": entry_id,
                "initial_config": result["initial_config"],
                "involved_classes": result["involved_classes"],
            }

            # Call multi_turn_checker
            try:
                checker_result = multi_turn_checker(
                    multi_turn_model_result_list_decoded=model_results,
                    multi_turn_ground_truth_list=ground_truth,
                    test_entry=test_entry,
                    test_category=subset,
                    model_name=_MODEL_NAME_FOR_SCORING,
                )
            except Exception as exc:
                logger.warning("multi_turn_checker failed for %s: %s", entry_id, exc)
                per_entry_scores.append(
                    {
                        "entry_id": entry_id,
                        "subset": subset,
                        "valid": False,
                        "error_type": "checker_exception",
                        "error_message": str(exc),
                    }
                )
                continue

            valid = checker_result.get("valid", False)
            if valid:
                subset_correct[subset] += 1

            entry_score = {
                "entry_id": entry_id,
                "subset": subset,
                "valid": valid,
            }
            if not valid:
                entry_score["error_type"] = checker_result.get("error_type", "unknown")
                entry_score["error_message"] = checker_result.get("error_message", "")

            per_entry_scores.append(entry_score)

        # Compute per-subset accuracy
        subset_scores: dict[str, str] = {}
        for subset in MULTI_TURN_SUBSETS:
            total = subset_total.get(subset, 0)
            if total > 0:
                acc = subset_correct.get(subset, 0) / total * 100
                subset_scores[subset] = f"{acc:.2f}"

        # Category aggregate: unweighted mean of subset scores (matching evalscope)
        valid_subset_accs = [float(v) for v in subset_scores.values()]
        overall = (
            sum(valid_subset_accs) / len(valid_subset_accs)
            if valid_subset_accs
            else 0.0
        )

        return {
            "overall_accuracy": f"{overall:.2f}",
            "category_scores": {
                "multi_turn": f"{overall:.2f}",
            },
            "subset_scores": subset_scores,
            "per_entry_scores": per_entry_scores,
            "total_samples": len(results),
        }
