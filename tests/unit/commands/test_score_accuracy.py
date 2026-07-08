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

"""Tests for accuracy grouping/consolidation dispatch in finalize_benchmark."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from inference_endpoint.commands.benchmark.execute import (
    AccuracyConfiguration,
    _accuracy_group,
    _score_accuracy,
)


class _FakeDataset:
    def __init__(self, n: int, score: float):
        self._n = n
        self.score = score
        self.data = list(range(n))

    def num_samples(self) -> int:
        return self._n


class _FakeScorer:
    def __init__(
        self, name, dataset, report_dir, extractor=None, ground_truth_column=None, **x
    ):
        self._d = dataset
        self.complete = True

    def score(self):
        return self._d.score, 1

    def score_breakdown(self):
        return None


def _cfg(name: str, n: int, score: float, group: str | None, tmp):
    return AccuracyConfiguration(
        _FakeScorer,  # type: ignore[arg-type]  # duck-typed stand-in
        None,
        name,
        _FakeDataset(n, score),  # type: ignore[arg-type]
        tmp,
        None,
        1,
        {},
        group=group,
    )


def _ctx(cfgs):
    return SimpleNamespace(eval_configs=cfgs)


_RESULT = SimpleNamespace(perf_results=[])


@pytest.mark.unit
class TestAccuracyGroup:
    def test_explicit_group_wins(self):
        assert _accuracy_group(SimpleNamespace(group="gptoss")) == "gptoss"

    def test_no_group(self):
        assert _accuracy_group(SimpleNamespace(group=None)) is None
        assert _accuracy_group(None) is None

    def test_variant_suffix_is_not_a_group(self):
        # The ::variant suffix must NOT implicitly consolidate — group comes only
        # from an explicit accuracy_config.group.
        cfg = SimpleNamespace(group=None, name="open_orca::llama2_70b")
        assert _accuracy_group(cfg) is None


@pytest.mark.unit
class TestScoreAccuracyGrouping:
    def test_group_of_three_consolidates(self, tmp_path):
        cfgs = [
            _cfg("aime25::gptoss", 30, 0.8, "gptoss", tmp_path),
            _cfg("gpqa::gptoss", 198, 0.9, "gptoss", tmp_path),
            _cfg("livecodebench::gptoss", 1055, 0.6, "gptoss", tmp_path),
        ]
        scores = _score_accuracy(_ctx(cfgs), _RESULT)
        # Consolidated entry PLUS per-subset audit entries.
        assert set(scores) == {
            "gptoss",
            "aime25::gptoss",
            "gpqa::gptoss",
            "livecodebench::gptoss",
        }
        bd = scores["gptoss"]["breakdown"]
        assert bd["subset_scores"] == {
            "aime25": 80.0,
            "gpqa": 90.0,
            "livecodebench": 60.0,
        }
        assert bd["total_samples"] == 1283
        # Per-subset entries carry the raw [0,1] score and no breakdown.
        assert scores["aime25::gptoss"]["score"] == 0.8
        assert scores["livecodebench::gptoss"]["score"] == 0.6
        assert "breakdown" not in scores["aime25::gptoss"]

    def test_single_member_group_stays_singleton(self, tmp_path):
        # A group with only one member is below the >=2 threshold: scored on its
        # own, keeping its full ::gptoss name and no consolidated breakdown.
        cfgs = [_cfg("aime25::gptoss", 30, 0.8, "gptoss", tmp_path)]
        scores = _score_accuracy(_ctx(cfgs), _RESULT)
        assert set(scores) == {"aime25::gptoss"}
        assert "breakdown" not in scores["aime25::gptoss"]

    def test_ungrouped_dataset_not_consolidated(self, tmp_path):
        cfgs = [
            _cfg("aime25::gptoss", 30, 0.8, "gptoss", tmp_path),
            _cfg("gpqa::gptoss", 198, 0.9, "gptoss", tmp_path),
            _cfg("cnn_dailymail::llama3_8b", 100, 0.5, None, tmp_path),
        ]
        scores = _score_accuracy(_ctx(cfgs), _RESULT)
        assert set(scores) == {
            "gptoss",
            "aime25::gptoss",
            "gpqa::gptoss",
            "cnn_dailymail::llama3_8b",
        }
        assert "breakdown" not in scores["cnn_dailymail::llama3_8b"]
