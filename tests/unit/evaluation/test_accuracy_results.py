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

"""Tests for the shared accuracy-breakdown helpers."""

from __future__ import annotations

import pytest
from inference_endpoint.evaluation.accuracy_results import (
    ACCURACY_METRIC_KEYS,
    average_accuracy,
    build_breakdown,
    find_accuracy_breakdown,
    find_accuracy_entry,
    to_float,
)


@pytest.mark.unit
class TestToFloat:
    def test_none(self):
        assert to_float(None) is None

    def test_numeric(self):
        assert to_float(3) == 3.0
        assert to_float(1.5) == 1.5

    def test_string_number(self):
        assert to_float("86.23") == 86.23

    def test_non_numeric(self):
        assert to_float("abc") is None
        assert to_float([1]) is None


@pytest.mark.unit
class TestBuildBreakdown:
    def test_shape(self):
        # No overall_accuracy — the headline lives on the entry's scalar score.
        assert build_breakdown({"a": 70.0, "b": 88.89}, 100) == {
            "subset_scores": {"a": 70.0, "b": 88.89},
            "total_samples": 100,
            "complete": True,
        }

    def test_rounds_to_two_dp(self):
        bd = build_breakdown({"a": 70.111}, 5)
        assert bd["subset_scores"] == {"a": 70.11}

    def test_incomplete(self):
        bd = build_breakdown({}, 0, complete=False)
        assert bd["subset_scores"] == {}
        assert bd["complete"] is False
        assert "overall_accuracy" not in bd


@pytest.mark.unit
class TestFindAccuracyBreakdown:
    def test_no_scores(self):
        assert find_accuracy_breakdown({}) is None
        assert find_accuracy_breakdown({"accuracy_scores": None}) is None
        assert find_accuracy_breakdown({"accuracy_scores": []}) is None

    def test_breakdown_block(self):
        block = {"subset_scores": {"a": 90.0}, "total_samples": 5}
        results = {
            "accuracy_scores": [
                {"dataset_name": "plain", "score": 0.5},
                {"dataset_name": "gptoss", "score": 0.9, "breakdown": block},
            ]
        }
        assert find_accuracy_breakdown(results) is block

    def test_recognized_without_overall(self):
        """A DeepSeek-shaped breakdown (subset_scores, no overall_accuracy) is
        still found; the entry carries the headline score."""
        entry = {
            "dataset_name": "ds",
            "score": 81.0,
            "breakdown": {"subset_scores": {"aime": 80.0}, "total_samples": 2},
        }
        results = {"accuracy_scores": [entry]}
        assert find_accuracy_entry(results) is entry
        assert find_accuracy_breakdown(results) is entry["breakdown"]

    def test_no_breakdown_key_is_ignored(self):
        results = {"accuracy_scores": [{"dataset_name": "x", "score": 0.5}]}
        assert find_accuracy_breakdown(results) is None
        assert find_accuracy_entry(results) is None


@pytest.mark.unit
class TestAverageAccuracy:
    def test_multi_component_mean(self):
        scores = [
            {"dataset_name": "aime25::gptoss", "score": 83.33},
            {"dataset_name": "gpqa::gptoss", "score": 74.75},
            {"dataset_name": "livecodebench::gptoss", "score": 84.74},
        ]
        assert average_accuracy(scores) == pytest.approx((83.33 + 74.75 + 84.74) / 3)

    def test_single_component_equals_itself(self):
        assert average_accuracy([{"dataset_name": "dsr1", "score": 81.04}]) == 81.04

    def test_excludes_performance_and_non_numeric(self):
        scores = [
            {"dataset_name": "performance", "score": 999.0},
            {"dataset_name": "rouge", "score": {"rougeL": 1.0}},  # non-numeric
            {"dataset_name": "flag", "score": True},  # bool is not a score
            {"dataset_name": "aime", "score": 80.0},
        ]
        assert average_accuracy(scores) == 80.0

    def test_none_when_nothing_numeric(self):
        assert average_accuracy([]) is None
        assert average_accuracy([{"dataset_name": "performance", "score": 5.0}]) is None


@pytest.mark.unit
def test_metric_keys():
    assert ACCURACY_METRIC_KEYS["bfcl_overall_accuracy"] == "overall_accuracy"
    assert (
        ACCURACY_METRIC_KEYS["bfcl_normalized_accuracy"]
        == "normalized_single_turn_score"
    )
