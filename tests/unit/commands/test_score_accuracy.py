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


def _cfg(name: str, n: int, score: float, tmp, scorer=_FakeScorer, repeats: int = 1):
    return AccuracyConfiguration(
        scorer,  # type: ignore[arg-type]  # duck-typed stand-in
        None,
        name,
        _FakeDataset(n, score),  # type: ignore[arg-type]
        tmp,
        None,
        repeats,
        {},
    )


def _by_name(scores: list[dict]) -> dict[str, dict]:
    return {e["dataset_name"]: e for e in scores}


def _ctx(cfgs):
    return SimpleNamespace(eval_configs=cfgs)


_RESULT = SimpleNamespace(perf_results=[], phase_results=[])


@pytest.mark.unit
class TestScoreAccuracy:
    def test_each_dataset_gets_its_own_entry(self, tmp_path):
        cfgs = [
            _cfg("aime25::gptoss", 30, 0.8, tmp_path, repeats=8),
            _cfg("gpqa::gptoss", 198, 0.9, tmp_path, repeats=5),
            _cfg("cnn_dailymail::llama3_8b", 100, 0.5, tmp_path),
        ]
        scores = _score_accuracy(_ctx(cfgs), _RESULT)
        assert isinstance(scores, list)
        by = _by_name(scores)
        assert set(by) == {
            "aime25::gptoss",
            "gpqa::gptoss",
            "cnn_dailymail::llama3_8b",
        }
        assert by["aime25::gptoss"]["score"] == 0.8
        # unit_samples = single instance; total = unit × repeats.
        assert by["aime25::gptoss"]["unit_samples"] == 30
        assert by["aime25::gptoss"]["num_repeats"] == 8
        assert by["aime25::gptoss"]["total_samples"] == 240
        assert by["gpqa::gptoss"]["total_samples"] == 990
        assert "breakdown" not in by["aime25::gptoss"]

    def test_breakdown_attached_only_when_scorer_provides_it(self, tmp_path):
        cfgs = [
            _cfg("plain", 10, 0.7, tmp_path),
            _cfg("with_bd", 10, 0.83, tmp_path, scorer=_FakeBreakdownScorer),
        ]
        by = _by_name(_score_accuracy(_ctx(cfgs), _RESULT))
        assert "breakdown" not in by["plain"]
        assert by["with_bd"]["breakdown"]["overall_accuracy"] == 80.0

    def test_performance_entry_uses_issued_count_for_total(self, tmp_path):
        # The "performance" dataset totals the perf phases' issued counts, not
        # unit × repeats. unit_samples still reports its own dataset length (3).
        cfg = _cfg("performance", 3, 0.6, tmp_path)
        result = SimpleNamespace(
            perf_results=[
                SimpleNamespace(issued_count=40),
                SimpleNamespace(issued_count=88),
            ],
            phase_results=[
                SimpleNamespace(
                    name="performance",
                    start_time_ns=2_000_000_000,
                    end_time_ns=5_000_000_000,
                ),
            ],
        )
        by = _by_name(_score_accuracy(_ctx([cfg]), result))
        assert by["performance"]["unit_samples"] == 3
        assert by["performance"]["num_repeats"] == 1
        assert by["performance"]["total_samples"] == 128
        assert by["performance"]["duration_s"] == 3.0

    def test_empty_when_no_datasets(self, tmp_path):
        assert _score_accuracy(_ctx([]), _RESULT) == []

    def test_accuracy_entry_has_phase_duration(self, tmp_path):
        """Each entry carries its issue phase's wall-clock (seconds), matched by
        phase name == dataset_name."""
        cfg = _cfg("aime25::gptoss", 30, 0.8, tmp_path)
        result = SimpleNamespace(
            perf_results=[],
            phase_results=[
                SimpleNamespace(
                    name="aime25::gptoss",
                    start_time_ns=1_000_000_000,
                    end_time_ns=6_500_000_000,
                ),
            ],
        )
        entry = _by_name(_score_accuracy(_ctx([cfg]), result))["aime25::gptoss"]
        assert entry["duration_s"] == 5.5

    def test_numpy_score_coerced_to_serializable(self, tmp_path):
        """A scorer returning a numpy scalar (e.g. np.mean) must yield a native
        float so the entry serializes via both json and msgspec — regression:
        Report.to_json crashed with "Encoding objects of type numpy.float64 is
        unsupported"."""
        import json

        import msgspec.json
        import numpy as np

        class _NumpyScorer(_FakeScorer):
            def score(self):
                return np.float64(0.5), 1

        cfg = _cfg("np::ds", 10, 0.0, tmp_path, scorer=_NumpyScorer)
        entry = _by_name(_score_accuracy(_ctx([cfg]), _RESULT))["np::ds"]
        # np.floating is a float subclass, so isinstance(..., float) is not
        # enough — assert it is specifically NOT a numpy scalar.
        assert not isinstance(entry["score"], np.floating)
        assert entry["score"] == 0.5
        # Both serializers used downstream (results.json / results_summary.json)
        # must accept the coerced entry.
        json.dumps(entry)
        msgspec.json.encode(entry)
