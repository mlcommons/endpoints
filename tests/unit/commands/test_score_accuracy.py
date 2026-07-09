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

"""Tests for per-dataset accuracy scoring in finalize_benchmark."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from inference_endpoint.commands.benchmark.execute import (
    AccuracyConfiguration,
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
    """Duck-typed scorer stand-in with no breakdown."""

    def __init__(
        self, name, dataset, report_dir, extractor=None, ground_truth_column=None, **x
    ):
        self._d = dataset
        self.complete = True

    def score(self):
        return self._d.score, 1

    def score_breakdown(self):
        return None


class _FakeBreakdownScorer(_FakeScorer):
    """Scorer that returns a breakdown (like the composite gpt-oss scorer)."""

    def score_breakdown(self):
        return {"overall_accuracy": 80.0, "subset_scores": {"x": 80.0}}


def _cfg(name: str, n: int, score: float, tmp, scorer=_FakeScorer):
    return AccuracyConfiguration(
        scorer,  # type: ignore[arg-type]  # duck-typed stand-in
        None,
        name,
        _FakeDataset(n, score),  # type: ignore[arg-type]
        tmp,
        None,
        1,
        {},
    )


def _ctx(cfgs):
    return SimpleNamespace(eval_configs=cfgs)


_RESULT = SimpleNamespace(perf_results=[])


@pytest.mark.unit
class TestScoreAccuracy:
    def test_each_dataset_gets_its_own_entry(self, tmp_path):
        cfgs = [
            _cfg("aime25::gptoss", 30, 0.8, tmp_path),
            _cfg("gpqa::gptoss", 198, 0.9, tmp_path),
            _cfg("cnn_dailymail::llama3_8b", 100, 0.5, tmp_path),
        ]
        scores = _score_accuracy(_ctx(cfgs), _RESULT)
        # One entry per dataset, no consolidated/group entry.
        assert set(scores) == {
            "aime25::gptoss",
            "gpqa::gptoss",
            "cnn_dailymail::llama3_8b",
        }
        assert scores["aime25::gptoss"]["score"] == 0.8
        assert scores["gpqa::gptoss"]["num_samples"] == 198
        assert "breakdown" not in scores["aime25::gptoss"]

    def test_breakdown_attached_only_when_scorer_provides_it(self, tmp_path):
        cfgs = [
            _cfg("plain", 10, 0.7, tmp_path),
            _cfg(
                "gptoss_120b_accuracy", 10, 0.83, tmp_path, scorer=_FakeBreakdownScorer
            ),
        ]
        scores = _score_accuracy(_ctx(cfgs), _RESULT)
        assert "breakdown" not in scores["plain"]
        assert scores["gptoss_120b_accuracy"]["breakdown"]["overall_accuracy"] == 80.0
