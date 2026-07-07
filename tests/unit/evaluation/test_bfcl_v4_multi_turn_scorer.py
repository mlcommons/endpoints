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

"""Unit tests for BFCLv4MultiTurnScorer.score().

bfcl-eval is an optional, conflicting dependency, so ``multi_turn_checker`` is
stubbed here (the same approach the runner/execution-bridge tests use). This
pins the force-terminated / turn-count-mismatch / checker-exception fail paths,
the valid/invalid bookkeeping, the unweighted-mean category aggregation, the
empty-results zero branch, and the string-typed ``subset_scores`` contract that
compliance and plotting must coerce.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from inference_endpoint.evaluation import bfcl_v4_multi_turn_scorer as mts

pytestmark = pytest.mark.unit


def _result(
    entry_id,
    subset,
    *,
    model_results=None,
    ground_truth=None,
    force_terminated=False,
    num_turns_completed=0,
    num_turns_expected=0,
):
    return {
        "entry_id": entry_id,
        "subset": subset,
        "model_results_per_turn": model_results if model_results is not None else [[]],
        "ground_truth": ground_truth if ground_truth is not None else [[]],
        "initial_config": {},
        "involved_classes": [],
        "force_terminated": force_terminated,
        "num_turns_completed": num_turns_completed,
        "num_turns_expected": num_turns_expected,
    }


@contextmanager
def _scorer_with(checker):
    """Yield a scorer whose bfcl-eval checker is ``checker`` for __init__ + score()."""
    with patch.object(mts, "multi_turn_checker", checker):
        yield mts.BFCLv4MultiTurnScorer()


def test_force_terminated_entry_is_failed_without_calling_checker():
    calls = []

    def checker(**kwargs):
        calls.append(kwargs)
        return {"valid": True}

    with _scorer_with(checker) as scorer:
        out = scorer.score(
            [
                _result(
                    "multi_turn_base_0",
                    "multi_turn_base",
                    force_terminated=True,
                    num_turns_completed=1,
                    num_turns_expected=3,
                )
            ]
        )

    assert calls == []  # never scored via the checker
    entry = out["per_entry_scores"][0]
    assert entry["valid"] is False
    assert entry["error_type"] == "force_terminated"
    assert out["subset_scores"]["multi_turn_base"] == "0.00"
    assert out["overall_accuracy"] == "0.00"


def test_turn_count_mismatch_is_failed_without_calling_checker():
    calls = []

    def checker(**kwargs):
        calls.append(kwargs)
        return {"valid": True}

    with _scorer_with(checker) as scorer:
        out = scorer.score(
            [
                _result(
                    "multi_turn_base_0",
                    "multi_turn_base",
                    model_results=[[], []],  # 2 turns
                    ground_truth=[[]],  # 1 turn expected
                )
            ]
        )

    assert calls == []
    entry = out["per_entry_scores"][0]
    assert entry["valid"] is False
    assert entry["error_type"] == "turn_count_mismatch"


def test_checker_exception_is_caught_and_marked_invalid():
    def checker(**kwargs):
        raise RuntimeError("boom")

    with _scorer_with(checker) as scorer:
        out = scorer.score([_result("multi_turn_base_0", "multi_turn_base")])

    entry = out["per_entry_scores"][0]
    assert entry["valid"] is False
    assert entry["error_type"] == "checker_exception"
    assert "boom" in entry["error_message"]


def test_valid_and_invalid_bookkeeping_within_a_subset():
    def checker(**kwargs):
        # First entry valid, second invalid — keyed off the entry id.
        return {"valid": kwargs["test_entry"]["id"].endswith("_0")}

    with _scorer_with(checker) as scorer:
        out = scorer.score(
            [
                _result("multi_turn_base_0", "multi_turn_base"),
                _result("multi_turn_base_1", "multi_turn_base"),
            ]
        )

    assert out["subset_scores"]["multi_turn_base"] == "50.00"
    assert out["overall_accuracy"] == "50.00"


def test_category_aggregate_is_unweighted_mean_of_subsets():
    # multi_turn_base: 2/2 valid -> 100.00; multi_turn_miss_func: 0/1 -> 0.00.
    # Unweighted mean is 50.00; a sample-weighted mean would be 66.67.
    def checker(**kwargs):
        return {"valid": kwargs["test_category"] == "multi_turn_base"}

    with _scorer_with(checker) as scorer:
        out = scorer.score(
            [
                _result("multi_turn_base_0", "multi_turn_base"),
                _result("multi_turn_base_1", "multi_turn_base"),
                _result("multi_turn_miss_func_0", "multi_turn_miss_func"),
            ]
        )

    assert out["subset_scores"]["multi_turn_base"] == "100.00"
    assert out["subset_scores"]["multi_turn_miss_func"] == "0.00"
    assert out["overall_accuracy"] == "50.00"
    assert out["category_scores"]["multi_turn"] == "50.00"


def test_empty_results_yields_zero_overall_and_no_subsets():
    with _scorer_with(lambda **k: {"valid": True}) as scorer:
        out = scorer.score([])

    assert out["overall_accuracy"] == "0.00"
    assert out["subset_scores"] == {}
    assert out["total_samples"] == 0


def test_subset_scores_are_formatted_strings():
    with _scorer_with(lambda **k: {"valid": True}) as scorer:
        out = scorer.score([_result("multi_turn_base_0", "multi_turn_base")])

    assert isinstance(out["subset_scores"]["multi_turn_base"], str)
    assert isinstance(out["overall_accuracy"], str)
    assert out["subset_scores"]["multi_turn_base"] == "100.00"
