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
    build_breakdown,
    find_accuracy_breakdown,
    find_accuracy_entry,
    samples_weighted_average_accuracy,
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
    def test_multi_component_weighted_by_unit_samples(self):
        # Components are weighted by unique-problem count (unit_samples), not
        # averaged flat: LiveCodeBench (1055 problems) dominates AIME (30).
        scores = [
            {"dataset_name": "aime25::gptoss", "score": 83.33, "unit_samples": 30},
            {"dataset_name": "gpqa::gptoss", "score": 74.75, "unit_samples": 198},
            {
                "dataset_name": "livecodebench::gptoss",
                "score": 84.74,
                "unit_samples": 1055,
            },
        ]
        weighted = (83.33 * 30 + 74.75 * 198 + 84.74 * 1055) / (30 + 198 + 1055)
        assert samples_weighted_average_accuracy(scores) == pytest.approx(weighted)
        # The weighted mean is materially different from the flat mean.
        assert samples_weighted_average_accuracy(scores) != pytest.approx(
            (83.33 + 74.75 + 84.74) / 3
        )

    def test_mlperf_reference_exact_match(self):
        # Regression lock to the MLCommons gpt-oss reference calculation.
        # aime25: 197/240 correct over 8 repeats (30 unique problems)
        # gpqa_diamond: 752/990 over 5 repeats (198 unique)
        # livecodebench_v6: 2684/3165 over 3 repeats (1055 unique)
        # Reference final score: 1069.69 / 1283.00 = 83.374% ('exact_match': 83.374).
        scores = [
            {"dataset_name": "aime25", "score": 197 / 240, "unit_samples": 30},
            {"dataset_name": "gpqa_diamond", "score": 752 / 990, "unit_samples": 198},
            {
                "dataset_name": "livecodebench_v6",
                "score": 2684 / 3165,
                "unit_samples": 1055,
            },
        ]
        avg = samples_weighted_average_accuracy(scores)
        assert avg is not None
        assert round(avg * 100, 3) == 83.374
        # Must NOT collapse to the old unweighted mean (~80.95%); this fails if the
        # sample weighting is ever reverted to a flat mean.
        unweighted = (197 / 240 + 752 / 990 + 2684 / 3165) / 3
        assert avg != pytest.approx(unweighted)

    def test_weight_defaults_to_one_when_unit_samples_missing(self):
        # Legacy artifacts predating unit_samples fall back to an unweighted mean.
        scores = [
            {"dataset_name": "a", "score": 0.6},
            {"dataset_name": "b", "score": 0.8},
        ]
        assert samples_weighted_average_accuracy(scores) == pytest.approx(0.7)

    def test_explicit_none_unit_samples_defaults_to_one(self):
        # An explicit None weight behaves like an absent key (legacy fallback).
        scores = [
            {"dataset_name": "a", "score": 0.6, "unit_samples": None},
            {"dataset_name": "b", "score": 0.8, "unit_samples": None},
        ]
        assert samples_weighted_average_accuracy(scores) == pytest.approx(0.7)

    def test_nonpositive_or_bad_weight_entries_are_skipped(self):
        # A present-but-corrupt weight (<=0, or a non-numeric/bool type) must NOT
        # be counted as one sample — the entry is dropped from the mean entirely,
        # so only the well-formed component (weight 40) contributes. The negative
        # and non-numeric cases are the real regression locks; `0` is a boundary
        # (skipping and a zero weight yield the same mean) included for completeness.
        for bad in (0, -5, True, "30", [30]):
            scores = [
                {"dataset_name": "good", "score": 0.9, "unit_samples": 40},
                {"dataset_name": "bad", "score": 0.1, "unit_samples": bad},
            ]
            assert samples_weighted_average_accuracy(scores) == pytest.approx(0.9)

    def test_all_weights_corrupt_returns_none(self):
        # Every component has a present-but-invalid weight -> nothing counted.
        scores = [
            {"dataset_name": "a", "score": 0.5, "unit_samples": 0},
            {"dataset_name": "b", "score": 0.7, "unit_samples": -1},
        ]
        assert samples_weighted_average_accuracy(scores) is None

    def test_partial_missing_weight_degrades_to_one(self):
        # Mixed shape (one real weight, one absent) is not tool-producible but must
        # be well-defined: the absent component contributes weight 1.0, not 0.
        scores = [
            {"dataset_name": "big", "score": 1.0, "unit_samples": 1055},
            {"dataset_name": "legacy", "score": 0.0},  # no unit_samples -> 1.0
        ]
        assert samples_weighted_average_accuracy(scores) == pytest.approx(1055 / 1056)

    def test_total_samples_is_not_used_as_weight(self):
        # Weighting must use unit_samples (unique problems), never total_samples
        # (issued attempts) — the latter reproduces the reference's raw accuracy,
        # not the submitted score. A large total_samples on a low-score dataset
        # must not drag the result down when its unit_samples is small.
        scores = [
            {
                "dataset_name": "aime",
                "score": 0.9,
                "unit_samples": 30,
                "total_samples": 240,
            },
            {
                "dataset_name": "lcb",
                "score": 0.6,
                "unit_samples": 30,
                "total_samples": 3000,
            },
        ]
        # Equal unit_samples -> plain mean 0.75, regardless of total_samples.
        assert samples_weighted_average_accuracy(scores) == pytest.approx(0.75)

    def test_single_component_equals_itself(self):
        assert (
            samples_weighted_average_accuracy(
                [{"dataset_name": "dsr1", "score": 81.04}]
            )
            == 81.04
        )

    def test_excludes_performance_and_non_numeric(self):
        scores = [
            {"dataset_name": "perf", "score": 999.0, "dataset_type": "performance"},
            {"dataset_name": "rouge", "score": {"rougeL": 1.0}},  # non-numeric
            {"dataset_name": "flag", "score": True},  # bool is not a score
            {"dataset_name": "aime", "score": 80.0},
        ]
        assert samples_weighted_average_accuracy(scores) == 80.0

    def test_homogeneous_fraction_scores_average(self):
        scores = [
            {"dataset_name": "aime", "score": 0.8},
            {"dataset_name": "gpqa", "score": 0.9},
        ]
        assert samples_weighted_average_accuracy(scores) == pytest.approx(0.85)

    def test_all_percentage_scores_average_incl_near_zero(self):
        # A percentage-scale set with a near-zero component must still average —
        # the removed magnitude guard used to wrongly omit this.
        scores = [
            {"dataset_name": "lcb", "score": 0.0},
            {"dataset_name": "aime", "score": 80.0},
        ]
        assert samples_weighted_average_accuracy(scores) == pytest.approx(40.0)

    def test_excludes_by_type_not_name(self):
        # A dataset legitimately named "performance" but of accuracy type is still
        # counted — exclusion is by dataset_type, not dataset_name.
        scores = [
            {"dataset_name": "performance", "score": 90.0, "dataset_type": "accuracy"}
        ]
        assert samples_weighted_average_accuracy(scores) == 90.0

    def test_none_when_nothing_numeric(self):
        assert samples_weighted_average_accuracy([]) is None
        assert (
            samples_weighted_average_accuracy(
                [{"dataset_name": "perf", "score": 5.0, "dataset_type": "performance"}]
            )
            is None
        )
