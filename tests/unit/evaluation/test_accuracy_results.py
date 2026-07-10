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
    build_breakdown,
    find_accuracy_breakdown,
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
        assert build_breakdown(82.34, {"a": 70.0, "b": 88.89}, 100) == {
            "overall_accuracy": 82.34,
            "subset_scores": {"a": 70.0, "b": 88.89},
            "total_samples": 100,
            "complete": True,
        }

    def test_rounds_to_two_dp(self):
        bd = build_breakdown(82.3456, {"a": 70.111}, 5)
        assert bd["overall_accuracy"] == 82.35
        assert bd["subset_scores"] == {"a": 70.11}

    def test_none_overall_and_extra(self):
        bd = build_breakdown(
            None, {}, 0, complete=False, per_subset_status={"a": "unscored"}
        )
        assert bd["overall_accuracy"] is None
        assert bd["complete"] is False
        assert bd["per_subset_status"] == {"a": "unscored"}


@pytest.mark.unit
class TestFindAccuracyBreakdown:
    def test_no_scores(self):
        assert find_accuracy_breakdown({}) is None
        assert find_accuracy_breakdown({"accuracy_scores": None}) is None
        assert find_accuracy_breakdown({"accuracy_scores": []}) is None

    def test_breakdown_block(self):
        block = {"overall_accuracy": 90.0, "subset_scores": {}, "total_samples": 5}
        results = {
            "accuracy_scores": [
                {"dataset_name": "plain", "score": 0.5},
                {"dataset_name": "gptoss", "score": 0.9, "breakdown": block},
            ]
        }
        assert find_accuracy_breakdown(results) is block

    def test_no_overall_key_is_ignored(self):
        results = {"accuracy_scores": [{"dataset_name": "x", "score": 0.5}]}
        assert find_accuracy_breakdown(results) is None


@pytest.mark.unit
def test_metric_keys():
    assert ACCURACY_METRIC_KEYS["bfcl_overall_accuracy"] == "overall_accuracy"
    assert (
        ACCURACY_METRIC_KEYS["bfcl_normalized_accuracy"]
        == "normalized_single_turn_score"
    )
