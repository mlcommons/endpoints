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

import sys
from types import SimpleNamespace

import pytest
from inference_endpoint.commands.benchmark.execute import (
    AccuracyConfiguration,
    _load_osl_tokenizer,
    _phase_osl_stats,
    _score_accuracy,
)

# Module object for the tests that monkeypatch execute's own module-level symbols
# (_load_osl_tokenizer / AutoTokenizer) so _score_accuracy resolves the patched
# one. Taken from sys.modules to avoid importing execute under both the
# ``from ... import`` and ``import ...`` styles.
execute_mod = sys.modules[_score_accuracy.__module__]


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


class _FakeOSLScorer(_FakeScorer):
    """Scorer exposing the get_outputs()/sample_index_map the OSL path reads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Two completed samples: outputs of 2 and 4 "tokens" (words).
        self.sample_index_map = {"u1": 0, "u2": 1}

    def get_outputs(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {"sample_uuid": "u1", "output": "a b"},
                {"sample_uuid": "u2", "output": "a b c d"},
                {"sample_uuid": "other", "output": "not in this phase"},
            ]
        )


def _cfg(
    name: str,
    n: int,
    score: float,
    tmp,
    scorer=_FakeScorer,
    repeats: int = 1,
    scale: float = 1.0,
):
    return AccuracyConfiguration(
        scorer,  # type: ignore[arg-type]  # duck-typed stand-in
        None,
        name,
        _FakeDataset(n, score),  # type: ignore[arg-type]
        tmp,
        None,
        repeats,
        {},
        scale,
    )


def _by_name(scores: list[dict]) -> dict[str, dict]:
    return {e["dataset_name"]: e for e in scores}


def _ctx(cfgs, tokenizer_name=None):
    # tokenizer_name None => OSL is skipped (the fake scorers have no get_outputs).
    return SimpleNamespace(eval_configs=cfgs, tokenizer_name=tokenizer_name)


class _WordTokenizer:
    """Stand-in tokenizer: one token per whitespace-delimited word."""

    model_max_length = 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


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

    def test_scale_reports_score_as_percentage(self, tmp_path):
        """accuracy_config.scale multiplies the scalar score (e.g. 100 turns a
        [0,1] pass@1 into a percentage); default scale=1.0 leaves it untouched."""
        cfgs = [
            _cfg("aime25::gptoss", 30, 0.8, tmp_path, scale=100),
            _cfg("plain", 10, 0.8, tmp_path),  # default scale=1.0
        ]
        by = _by_name(_score_accuracy(_ctx(cfgs), _RESULT))
        assert by["aime25::gptoss"]["score"] == pytest.approx(80.0)
        assert by["plain"]["score"] == pytest.approx(0.8)

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

    def test_no_osl_without_tokenizer(self, tmp_path):
        # tokenizer_name None (the default) => no output_sequence_lengths attached,
        # and the OSL path is never entered (fake scorers have no get_outputs).
        cfg = _cfg("aime25::gptoss", 30, 0.8, tmp_path)
        entry = _by_name(_score_accuracy(_ctx([cfg]), _RESULT))["aime25::gptoss"]
        assert "output_sequence_lengths" not in entry

    def test_osl_attached_with_tokenizer(self, tmp_path, monkeypatch):
        """With a tokenizer, each accuracy entry gets an output_sequence_lengths
        block (same shape as the perf report) from the phase's completions."""
        monkeypatch.setattr(
            execute_mod, "_load_osl_tokenizer", lambda name: _WordTokenizer()
        )
        cfg = _cfg("aime25::gptoss", 2, 0.8, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(_score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "aime25::gptoss"
        ]
        # Outputs "a b" (2) and "a b c d" (4); "other" is not in sample_index_map.
        osl = entry["output_sequence_lengths"]
        assert osl["avg"] == 3.0
        assert osl["min"] == 2
        assert osl["max"] == 4
        # Same block shape/keys as the perf report's output_sequence_lengths.
        assert {"median", "std_dev", "total", "percentiles", "histogram"} <= set(osl)
        # Tokenization is timed and attached alongside the OSL stats.
        assert isinstance(entry["osl_tokenize_s"], float)
        assert entry["osl_tokenize_s"] >= 0.0

    def test_osl_skipped_for_performance_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            execute_mod, "_load_osl_tokenizer", lambda name: _WordTokenizer()
        )
        cfg = _cfg("performance", 2, 0.6, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(_score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "performance"
        ]
        assert "output_sequence_lengths" not in entry

    def test_osl_dropped_when_get_outputs_raises(self, tmp_path, monkeypatch):
        """A read/tokenize failure only drops OSL — scoring still succeeds."""
        monkeypatch.setattr(
            execute_mod, "_load_osl_tokenizer", lambda name: _WordTokenizer()
        )

        class _RaisingScorer(_FakeOSLScorer):
            def get_outputs(self):
                raise RuntimeError("events.jsonl unreadable")

        cfg = _cfg("ds", 1, 0.8, tmp_path, scorer=_RaisingScorer)
        entry = _by_name(_score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "ds"
        ]
        assert "output_sequence_lengths" not in entry
        assert entry["score"] == pytest.approx(0.8)  # scoring unaffected


@pytest.mark.unit
class TestPhaseOslStats:
    def test_returns_perf_shaped_block(self):
        uuid_to_text = {"a": "x y z", "b": "x", "c": "not in phase"}
        block = _phase_osl_stats(["a", "b", "missing"], uuid_to_text, _WordTokenizer())
        assert block["avg"] == 2.0
        assert block["min"] == 1
        assert block["max"] == 3
        assert block["total"] == 4  # 3 + 1 tokens

    def test_batching_over_256(self):
        # >256 texts exercises the multi-batch tokenization loop; all are counted.
        uuid_to_text = {str(i): "w" for i in range(300)}  # 1 token each
        block = _phase_osl_stats(
            [str(i) for i in range(300)], uuid_to_text, _WordTokenizer()
        )
        assert block["total"] == 300
        assert block["avg"] == 1.0

    def test_none_when_no_matching_outputs(self):
        assert _phase_osl_stats([], {}, _WordTokenizer()) is None
        assert _phase_osl_stats(["x"], {"y": "a b"}, _WordTokenizer()) is None

    def test_skips_empty_outputs(self):
        # A failed/empty completion (output == "") is excluded, matching the
        # perf-side OslTrigger — it is not counted as a 0-token sample.
        block = _phase_osl_stats(["a", "b"], {"a": "", "b": "x y"}, _WordTokenizer())
        assert block["total"] == 2  # only "x y" counted
        assert block["min"] == 2
        assert block["avg"] == 2.0

    def test_none_when_all_outputs_empty(self):
        assert _phase_osl_stats(["a"], {"a": ""}, _WordTokenizer()) is None


@pytest.mark.unit
class TestLoadOslTokenizer:
    def test_none_name_returns_none(self):
        assert _load_osl_tokenizer(None) is None

    def test_load_failure_returns_none(self, monkeypatch):
        class _BoomAutoTokenizer:
            @staticmethod
            def from_pretrained(name):
                raise OSError("no such tokenizer")

        monkeypatch.setattr(execute_mod, "AutoTokenizer", _BoomAutoTokenizer)
        assert _load_osl_tokenizer("bad/model") is None

    def test_success_raises_model_max_length(self, monkeypatch):
        class _FakeTok:
            model_max_length = 5

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name):
                return _FakeTok()

        monkeypatch.setattr(execute_mod, "AutoTokenizer", _FakeAutoTokenizer)
        tok = _load_osl_tokenizer("some/model")
        # Cap raised so counting long outputs can't trip the length warning.
        assert tok.model_max_length == int(1e12)

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
        # Both serializers used downstream (results.json / result_summary.json)
        # must accept the coerced entry.
        json.dumps(entry)
        msgspec.json.encode(entry)
