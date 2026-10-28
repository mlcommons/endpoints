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

# ruff: noqa: I001

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import msgspec.json
import pytest
from inference_endpoint.commands.benchmark.accuracy import (
    AccuracyConfiguration,
    PhaseServerOsl,
    _accuracy_uuid_bound,
    _phase_osl_stats,
    _phase_response_counts,
    _phase_server_osl_stats,
    score_accuracy,
    write_accuracy_results,
)
from inference_endpoint.config import schema as config_schema
from inference_endpoint.config.schema import (
    DatasetType,
    EndpointConfig,
    ModelParams,
    ScorerMethod,
)
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput, TokenCountSource, basis_key
from inference_endpoint.evaluation import scoring as evaluation_scoring
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.metrics.token_count_basis import ServerUsage

from tests.test_helpers import LOGGED_SERVER_USAGE_CASES

# The module that defines score_accuracy (commands.benchmark.accuracy), resolved
# via __module__ so the tests monkeypatch its module-level load_reference_backend
# and score_accuracy picks up the patched one — without importing the module under
# both the ``from ... import`` and ``import ...`` styles.
scoring_mod = sys.modules[score_accuracy.__module__]
CLIENT, SERVER = TokenCountSource.CLIENT, TokenCountSource.SERVER


class _FakeDataset:
    def __init__(self, n: int, score: float):
        self._n = n
        self.score = score
        self.data = list(range(n))

    def num_samples(self) -> int:
        return self._n


class _FakeScorer:
    """Duck-typed scorer stand-in with no breakdown."""

    SKIP_ENDPOINT_PHASE = False

    def __init__(
        self, name, dataset, report_dir, extractor=None, ground_truth_column=None, **x
    ):
        self._d = dataset
        self.complete = True

    def score(self):
        return self._d.score, 1

    def score_breakdown(self):
        return None


class _FakeSWEBenchScorer(_FakeScorer):
    SCORER_ID = ScorerMethod.SWE_BENCH.value
    SKIP_ENDPOINT_PHASE = True
    received_kwargs: dict = {}

    @classmethod
    def external_sample_count(cls, extras):
        return None

    def __init__(self, *args, **kwargs):
        type(self).received_kwargs = kwargs
        super().__init__(*args, **kwargs)


class _FakeBreakdownScorer(_FakeScorer):
    """Scorer that returns a breakdown (like the composite gpt-oss scorer)."""

    def score_breakdown(self):
        return {"overall_accuracy": 80.0, "subset_scores": {"x": 80.0}}


class _FakeOSLScorer(_FakeScorer):
    """Scorer exposing the iter_complete_outputs()/sample_index_map the OSL
    path reads."""

    # Two completed samples: outputs of 2 and 4 "tokens" (words).
    SAMPLE_INDEX_MAP = {"u1": 0, "u2": 1}
    # (uuid, response text, server completion_tokens when the usage is usable)
    OUTPUTS: list[tuple[str, str, int | None]] = [
        ("u1", "a b", None),
        ("u2", "a b c d", None),
        ("other", "not in this phase", None),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_index_map = dict(self.SAMPLE_INDEX_MAP)

    def iter_complete_outputs(self, uuids):
        yield from self.OUTPUTS


class _EmptyOSLScorer(_FakeOSLScorer):
    """Every completion empty (all requests failed): OSL yields None."""

    OUTPUTS = [("u1", "", None), ("u2", "", None)]


class _MixedOSLScorer(_FakeOSLScorer):
    """One scored, one empty (COMPLETE, blank), one missing (no COMPLETE)."""

    SAMPLE_INDEX_MAP = {"u1": 0, "u2": 1, "u3": 2}
    # u3 has no COMPLETE event -> missing
    OUTPUTS = [("u1", "x y", None), ("u2", "", None)]


class _AllMissingScorer(_FakeOSLScorer):
    """No COMPLETE event for this phase, as when the bound filters everything."""

    SAMPLE_INDEX_MAP = {"u1": 0, "u2": 1, "u3": 2}
    OUTPUTS = []


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
        dataset_type=(
            DatasetType.PERFORMANCE if name == "performance" else DatasetType.ACCURACY
        ),
    )


def _by_name(scores: list[dict]) -> dict[str, dict]:
    return {e["dataset_name"]: e for e in scores}


def _ctx(
    cfgs,
    tokenizer_name=None,
    report_dir=None,
    test_mode: config_schema.TestMode = config_schema.TestMode.ACC,
    token_count_source: TokenCountSource = TokenCountSource.CLIENT,
):
    # tokenizer_name None => client OSL is skipped (and _FakeScorer has no
    # iter_complete_outputs, so its entries carry no OSL block at all).
    # report_dir None => the uuid bound falls back to an unbounded read (no map).
    return SimpleNamespace(
        eval_configs=cfgs,
        tokenizer_name=tokenizer_name,
        report_dir=report_dir,
        test_mode=test_mode,
        token_count_source=token_count_source,
    )


class _ServerOSLScorer(_FakeOSLScorer):
    """u1 scored with usage, u2 scored without usage, u3 empty with usage."""

    SAMPLE_INDEX_MAP = {"u1": 0, "u2": 1, "u3": 2}
    OUTPUTS = [("u1", "a b", 3), ("u2", "a b c d", None), ("u3", "", 1), ("x", "y", 9)]


class _EventsScorer(_FakeScorer):
    """Reads its phase's outputs from events.jsonl through the real reader."""

    _iter_complete_records = Scorer._iter_complete_records
    iter_complete_outputs = Scorer.iter_complete_outputs

    def __init__(self, name, dataset, report_dir, **kwargs):
        super().__init__(name, dataset, report_dir, **kwargs)
        self.report_dir = report_dir
        idx_map = msgspec.json.decode((report_dir / "sample_idx_map.json").read_bytes())
        self.sample_index_map = idx_map[name]


_EVENT_ENCODER = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


def _complete(uuid: str, text: str, prompt=msgspec.UNSET, completion=msgspec.UNSET):
    return EventRecord(
        event_type=SampleEventType.COMPLETE,
        sample_uuid=uuid,
        data=TextModelOutput(output=text),
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


def _write_run(report_dir: Path, phases: dict[str, list[EventRecord]]) -> list[bytes]:
    """Write sample_idx_map.json and events.jsonl, an ISSUED line before each
    COMPLETE; return the COMPLETE lines."""
    idx_map = {
        name: {r.sample_uuid: i for i, r in enumerate(records)}
        for name, records in phases.items()
    }
    (report_dir / "sample_idx_map.json").write_bytes(msgspec.json.encode(idx_map))
    complete_lines = []
    with (report_dir / "events.jsonl").open("wb") as f:
        for records in phases.values():
            for record in records:
                issued = EventRecord(
                    event_type=SampleEventType.ISSUED, sample_uuid=record.sample_uuid
                )
                f.write(_EVENT_ENCODER.encode(issued) + b"\n")
                complete_lines.append(_EVENT_ENCODER.encode(record) + b"\n")
                f.write(complete_lines[-1])
    return complete_lines


class _WordBackend:
    """Stand-in tokenizers backend: one token per whitespace-delimited word."""

    def encode_batch(self, texts, add_special_tokens=False):
        return [SimpleNamespace(ids=t.split()) for t in texts]


class _RaisingBackend:
    """Stand-in tokenizers backend that fails on every batch."""

    def encode_batch(self, texts, add_special_tokens=False):
        raise RuntimeError("tokenizer blew up")


class _CountingDecoder:
    """Wraps the events decoder and records every line it decodes."""

    def __init__(self, inner):
        self._inner = inner
        self.lines: list[bytes] = []

    def decode(self, line):
        self.lines.append(line)
        return self._inner.decode(line)


_RESULT = SimpleNamespace(perf_results=[], phase_results=[])


@pytest.mark.unit
class TestScoreAccuracy:
    @pytest.mark.parametrize(
        "test_mode", [config_schema.TestMode.ACC, config_schema.TestMode.BOTH]
    )
    def test_swebench_receives_typed_runtime_model_and_endpoint(
        self, tmp_path, test_mode
    ):
        model_params = ModelParams(
            name="test-model",
            temperature=0.2,
            seed=7,
            max_new_tokens=2048,
            chat_template_kwargs={"enable_thinking": False},
        )
        endpoint_config = EndpointConfig(
            endpoints=["http://endpoint:8000"],
            api_key="runtime-secret",
        )
        cfg = AccuracyConfiguration(
            scorer=_FakeSWEBenchScorer,  # type: ignore[arg-type]
            extractor=None,
            dataset_name="swe_bench",
            dataset=_FakeDataset(1, 1.0),  # type: ignore[arg-type]
            report_dir=tmp_path,
            ground_truth_column=None,
            num_repeats=1,
            extras={"swebench_service_auth_token": "service-secret"},
            model_params=model_params,
            endpoint_config=endpoint_config,
        )

        score_accuracy(_ctx([cfg], test_mode=test_mode), _RESULT)

        assert _FakeSWEBenchScorer.received_kwargs == {
            "extractor": None,
            "ground_truth_column": None,
            "swebench_service_auth_token": "service-secret",
            "model_params": model_params,
            "endpoint_config": endpoint_config,
        }

    def test_each_dataset_gets_its_own_entry(self, tmp_path):
        cfgs = [
            _cfg("aime25::gptoss", 30, 0.8, tmp_path, repeats=8),
            _cfg("gpqa::gptoss", 198, 0.9, tmp_path, repeats=5),
            _cfg("cnn_dailymail::llama3_8b", 100, 0.5, tmp_path),
        ]
        scores = score_accuracy(_ctx(cfgs), _RESULT)
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
        by = _by_name(score_accuracy(_ctx(cfgs), _RESULT))
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
        by = _by_name(score_accuracy(_ctx([cfg]), result))
        assert by["performance"]["unit_samples"] == 3
        assert by["performance"]["num_repeats"] == 1
        assert by["performance"]["total_samples"] == 128
        assert by["performance"]["duration_s"] == 3.0

    def test_empty_when_no_datasets(self, tmp_path):
        assert score_accuracy(_ctx([]), _RESULT) == []

    def test_perf_mode_skips_only_external_scoring(self, tmp_path):
        ordinary_cfg = _cfg("aime25::gptoss", 30, 0.8, tmp_path)
        external_cfg = _cfg("swe_bench", 1, 1.0, tmp_path, scorer=_FakeSWEBenchScorer)
        _FakeSWEBenchScorer.received_kwargs = {}
        scores = score_accuracy(
            _ctx(
                [ordinary_cfg, external_cfg],
                test_mode=config_schema.TestMode.PERF,
            ),
            _RESULT,
        )
        assert {entry["dataset_name"]: entry["score"] for entry in scores} == {
            "aime25::gptoss": 0.8
        }
        assert _FakeSWEBenchScorer.received_kwargs == {}

    def test_no_osl_without_tokenizer(self, tmp_path):
        # tokenizer_name None (the default) => no output_sequence_lengths attached,
        # and the OSL path stops at the read (_FakeScorer has no
        # iter_complete_outputs).
        cfg = _cfg("aime25::gptoss", 30, 0.8, tmp_path)
        entry = _by_name(score_accuracy(_ctx([cfg]), _RESULT))["aime25::gptoss"]
        assert "output_sequence_lengths" not in entry

    def test_osl_attached_with_tokenizer(self, tmp_path, monkeypatch):
        """With a tokenizer, each accuracy entry gets an output_sequence_lengths
        block (same shape as the perf report) from the phase's completions."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        cfg = _cfg("aime25::gptoss", 2, 0.8, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
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
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        cfg = _cfg("performance", 2, 0.6, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "performance"
        ]
        assert "output_sequence_lengths" not in entry

    def test_osl_dropped_when_events_read_raises(self, tmp_path, monkeypatch):
        """A read failure only drops OSL — scoring still succeeds."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )

        class _RaisingScorer(_FakeOSLScorer):
            def iter_complete_outputs(self, uuids):
                raise RuntimeError("events.jsonl unreadable")

        cfg = _cfg("ds", 1, 0.8, tmp_path, scorer=_RaisingScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "ds"
        ]
        assert "output_sequence_lengths" not in entry
        assert "response_counts" not in entry  # dropped with OSL on the same read
        assert entry["score"] == pytest.approx(0.8)  # scoring unaffected

    def test_response_counts_without_tokenizer(self, tmp_path):
        """response_counts must be published even with no tokenizer configured —
        failure visibility cannot depend on OSL being enabled."""
        cfg = _cfg("aime25::gptoss", 2, 0.8, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg]), _RESULT))["aime25::gptoss"]
        assert entry["response_counts"] == {
            "issued": 2,
            "scored": 2,
            "empty": 0,
            "missing": 0,
        }
        assert "output_sequence_lengths" not in entry  # no tokenizer

    def test_response_counts_published_when_all_empty(self, tmp_path, monkeypatch):
        """Masking regression: every response empty => OSL returns None, but
        response_counts must still publish scored=0 rather than omitting all."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        cfg = _cfg("aime25::gptoss", 2, 0.8, tmp_path, scorer=_EmptyOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "aime25::gptoss"
        ]
        assert "output_sequence_lengths" not in entry  # all empty -> OSL None
        assert entry["response_counts"] == {
            "issued": 2,
            "scored": 0,
            "empty": 2,
            "missing": 0,
        }

    def test_response_counts_classifies_missing(self, tmp_path, monkeypatch):
        """scored/empty/missing partition the issued samples; OSL tokenizes only
        the one scored (non-empty) response."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        cfg = _cfg("ds", 1, 0.8, tmp_path, scorer=_MixedOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "ds"
        ]
        assert entry["response_counts"] == {
            "issued": 3,
            "scored": 1,
            "empty": 1,
            "missing": 1,
        }
        assert entry["output_sequence_lengths"]["total"] == 2  # only "x y"

    def test_response_counts_skipped_for_performance_entry(self, tmp_path):
        cfg = _cfg("performance", 2, 0.6, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg]), _RESULT))["performance"]
        assert "response_counts" not in entry

    def test_all_missing_still_publishes_response_counts(self, tmp_path, monkeypatch):
        """An accuracy phase with no COMPLETE rows (empty, columned frame) must
        still publish response_counts (all missing) rather than dropping them."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        cfg = _cfg("ds", 1, 0.8, tmp_path, scorer=_AllMissingScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "ds"
        ]
        assert entry["response_counts"] == {
            "issued": 3,
            "scored": 0,
            "empty": 0,
            "missing": 3,
        }
        assert "output_sequence_lengths" not in entry  # nothing to tokenize

    def test_no_fast_backend_disables_osl_keeps_counts(self, tmp_path, monkeypatch):
        """A tokenizer with no fast backend (load_reference_backend -> None) skips
        OSL but still publishes response_counts."""
        monkeypatch.setattr(scoring_mod, "load_reference_backend", lambda name: None)
        cfg = _cfg("aime25::gptoss", 2, 0.8, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "aime25::gptoss"
        ]
        assert "output_sequence_lengths" not in entry
        assert entry["response_counts"]["scored"] == 2

    def test_tokenizer_load_failure_disables_osl_not_scoring(
        self, tmp_path, monkeypatch
    ):
        """A tokenizer load exception disables OSL but never fails scoring."""

        def _boom(name):
            raise OSError("no tokenizer")

        monkeypatch.setattr(scoring_mod, "load_reference_backend", _boom)
        cfg = _cfg("aime25::gptoss", 2, 0.8, tmp_path, scorer=_FakeOSLScorer)
        entry = _by_name(score_accuracy(_ctx([cfg], tokenizer_name="fake"), _RESULT))[
            "aime25::gptoss"
        ]
        assert "output_sequence_lengths" not in entry
        assert entry["response_counts"]["scored"] == 2
        assert entry["score"] == pytest.approx(0.8)

    def test_uuid_bound_passed_to_events_read(self, tmp_path, monkeypatch):
        """End-to-end: score_accuracy computes the accuracy uuid bound from
        sample_idx_map.json and passes it into iter_complete_outputs."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        (tmp_path / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({"aime25::gptoss": {"u1": 0, "u2": 1}})
        )
        captured: dict = {}

        class _CaptureScorer(_FakeOSLScorer):
            def iter_complete_outputs(self, uuids):
                captured["wanted"] = uuids
                return super().iter_complete_outputs(uuids)

        cfg = _cfg("aime25::gptoss", 2, 0.8, tmp_path, scorer=_CaptureScorer)
        score_accuracy(_ctx([cfg], tokenizer_name="fake", report_dir=tmp_path), _RESULT)
        assert captured["wanted"] == {"u1", "u2"}

    def test_empty_uuid_bound_stays_bounded(self, tmp_path):
        (tmp_path / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({"swe_bench": {}})
        )
        captured: dict = {}

        class _CaptureScorer(_FakeOSLScorer):
            def iter_complete_outputs(self, uuids):
                captured["wanted"] = uuids
                return super().iter_complete_outputs(uuids)

        cfg = _cfg("swe_bench", 1, 0.8, tmp_path, scorer=_CaptureScorer)
        score_accuracy(_ctx([cfg], report_dir=tmp_path), _RESULT)

        assert captured["wanted"] == set()

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_accuracy_keys_follow_rule(self, tmp_path, monkeypatch, primary):
        """Each basis's block and sample count take the base name on the
        primary basis and the suffixed name on the other; missing usage is
        basis-independent and marks the server block partial only when the
        server basis is the other one."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        cfg = _cfg("ds", 3, 0.8, tmp_path, scorer=_ServerOSLScorer)

        entry = _by_name(
            score_accuracy(
                _ctx([cfg], tokenizer_name="fake", token_count_source=primary),
                _RESULT,
            )
        )["ds"]

        def key(base: str, basis: TokenCountSource) -> str:
            return basis_key(base, basis, primary=primary)

        assert entry["response_counts"] == {
            "issued": 3,
            "scored": 2,
            "empty": 1,
            "missing": 0,
        }
        # Per basis: (total tokens, samples counted) over the scored samples.
        for basis, (total, n_samples) in {CLIENT: (6, 2), SERVER: (3, 1)}.items():
            assert entry[key("output_sequence_lengths", basis)]["total"] == total
            assert entry[key("n_osl_samples", basis)] == n_samples
        assert entry["n_missing_usage"] == 1
        server_block = entry[key("output_sequence_lengths", SERVER)]
        assert ("partial" in server_block) is (primary is CLIENT)
        assert "partial" not in entry[key("output_sequence_lengths", CLIENT)]
        assert isinstance(entry["osl_tokenize_s"], float)
        assert not [k for k in entry if k.endswith(f"_{primary.value}")]

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_server_osl_needs_no_tokenizer(self, tmp_path, primary):
        cfg = _cfg("ds", 3, 0.8, tmp_path, scorer=_ServerOSLScorer)

        entry = _by_name(
            score_accuracy(_ctx([cfg], token_count_source=primary), _RESULT)
        )["ds"]

        def key(base: str, basis: TokenCountSource) -> str:
            return basis_key(base, basis, primary=primary)

        assert entry[key("output_sequence_lengths", SERVER)]["avg"] == 3.0
        assert key("output_sequence_lengths", CLIENT) not in entry
        assert "osl_tokenize_s" not in entry
        assert (
            entry[key("n_osl_samples", SERVER)],
            entry[key("n_osl_samples", CLIENT)],
            entry["n_missing_usage"],
        ) == (1, 0, 1)

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_accuracy_tokenizer_failure_keeps_server_block(
        self, tmp_path, monkeypatch, caplog, primary
    ):
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _RaisingBackend()
        )
        cfg = _cfg("ds", 3, 0.8, tmp_path, scorer=_ServerOSLScorer)

        with caplog.at_level("WARNING"):
            entry = _by_name(
                score_accuracy(
                    _ctx([cfg], tokenizer_name="fake", token_count_source=primary),
                    _RESULT,
                )
            )["ds"]

        def key(base: str, basis: TokenCountSource) -> str:
            return basis_key(base, basis, primary=primary)

        assert entry[key("output_sequence_lengths", SERVER)]["total"] == 3
        assert key("output_sequence_lengths", CLIENT) not in entry
        assert "osl_tokenize_s" not in entry
        assert entry["response_counts"]["scored"] == 2
        assert (
            entry[key("n_osl_samples", SERVER)],
            entry[key("n_osl_samples", CLIENT)],
            entry["n_missing_usage"],
        ) == (1, 0, 1)
        assert "tokenizer blew up" in caplog.text

    def test_accuracy_single_pass_over_events(self, tmp_path, monkeypatch):
        """Two accuracy phases share one read of events.jsonl, which decodes
        each COMPLETE line once and no other line."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        complete_lines = _write_run(
            tmp_path,
            {
                "ds1": [_complete("u1", "a b", 1, 2), _complete("u2", "c")],
                "ds2": [_complete("u3", "d e f", 1, 3)],
            },
        )
        decoder = _CountingDecoder(evaluation_scoring._EVENT_DECODER)
        monkeypatch.setattr(evaluation_scoring, "_EVENT_DECODER", decoder)
        cfgs = [
            _cfg("ds1", 2, 0.5, tmp_path, scorer=_EventsScorer),
            _cfg("ds2", 1, 0.5, tmp_path, scorer=_EventsScorer),
        ]

        by = _by_name(
            score_accuracy(
                _ctx(cfgs, tokenizer_name="fake", report_dir=tmp_path), _RESULT
            )
        )

        assert sorted(decoder.lines) == sorted(complete_lines)
        assert by["ds1"]["output_sequence_lengths"]["total"] == 3
        assert by["ds2"]["output_sequence_lengths"]["total"] == 3
        assert (by["ds1"]["n_missing_usage"], by["ds2"]["n_missing_usage"]) == (1, 0)

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    @pytest.mark.parametrize(
        ("prompt", "completion", "output", "expected"),
        [pytest.param(*case[1:], id=case[0]) for case in LOGGED_SERVER_USAGE_CASES],
    )
    def test_accuracy_missing_usage_matches_rule(
        self, tmp_path, primary, prompt, completion, output, expected
    ):
        """Accuracy OSL counts each scored sample's usage by the shared rule."""
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="u1",
            data=output,
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
        _write_run(tmp_path, {"ds": [record]})
        cfg = _cfg("ds", 1, 0.8, tmp_path, scorer=_EventsScorer)

        entry = _by_name(
            score_accuracy(
                _ctx([cfg], report_dir=tmp_path, token_count_source=primary), _RESULT
            )
        )["ds"]

        server_key = basis_key("output_sequence_lengths", SERVER, primary=primary)
        assert entry["n_missing_usage"] == int(expected is ServerUsage.MISSING)
        assert (server_key in entry) is (expected is ServerUsage.USABLE)

    def test_accuracy_unsuffixed_entry_identical_with_and_without_usage_keys(
        self, tmp_path, monkeypatch
    ):
        """On the client basis, server usage only adds keys: the shared keys
        hold the same values whether or not the log carries usage."""
        monkeypatch.setattr(
            scoring_mod, "load_reference_backend", lambda name: _WordBackend()
        )
        with_usage = [
            _complete("u1", "a b", 1, 2),
            _complete("u2", "c d e"),
            _complete("u3", "", 1, 0),
            _complete("u4", "f", 1, 0),
        ]
        without_usage = [_complete(r.sample_uuid, str(r.data)) for r in with_usage]
        entries = []
        for name, records in (("with", with_usage), ("without", without_usage)):
            run = tmp_path / name
            run.mkdir()
            _write_run(run, {"ds": records})
            cfg = _cfg("ds", 4, 0.5, run, scorer=_EventsScorer)
            ctx = _ctx([cfg], tokenizer_name="fake", report_dir=run)
            entries.append(_by_name(score_accuracy(ctx, _RESULT))["ds"])

        shared = ("response_counts", "output_sequence_lengths")
        assert {k: entries[0][k] for k in shared} == {k: entries[1][k] for k in shared}
        assert entries[0]["output_sequence_lengths"]["total"] == 6


@pytest.mark.unit
def test_phase_server_osl_counts_scored_samples_with_usable_usage():
    """Each scored sample counts its usable completion_tokens or is missing
    usage; an empty sample is outside the scored population."""
    phase = _phase_server_osl_stats(
        ["a", "b", "c", "d"],
        {"a": "x y", "b": "z", "c": "", "d": "w"},
        {"b": 4, "c": 2},
    )
    assert isinstance(phase, PhaseServerOsl)
    assert phase.block is not None and phase.block["total"] == 4
    assert (phase.n_counted, phase.n_missing_usage) == (1, 2)


@pytest.mark.unit
class TestPhaseOslStats:
    def test_returns_perf_shaped_block(self):
        uuid_to_text = {"a": "x y z", "b": "x", "c": "not in phase"}
        block = _phase_osl_stats(["a", "b", "missing"], uuid_to_text, _WordBackend())
        assert block["avg"] == 2.0
        assert block["min"] == 1
        assert block["max"] == 3
        assert block["total"] == 4  # 3 + 1 tokens

    def test_counts_large_population(self):
        # >256 texts exercises the batched encode loop; all are counted.
        uuid_to_text = {str(i): "w" for i in range(300)}  # 1 token each
        block = _phase_osl_stats(
            [str(i) for i in range(300)], uuid_to_text, _WordBackend()
        )
        assert block["total"] == 300
        assert block["avg"] == 1.0

    def test_none_when_no_matching_outputs(self):
        assert _phase_osl_stats([], {}, _WordBackend()) is None
        assert _phase_osl_stats(["x"], {"y": "a b"}, _WordBackend()) is None

    def test_skips_empty_outputs(self):
        # A failed/empty completion (output == "") is excluded, matching the
        # perf-side OslTrigger — it is not counted as a 0-token sample.
        block = _phase_osl_stats(["a", "b"], {"a": "", "b": "x y"}, _WordBackend())
        assert block["total"] == 2  # only "x y" counted
        assert block["min"] == 2
        assert block["avg"] == 2.0

    def test_none_when_all_outputs_empty(self):
        assert _phase_osl_stats(["a"], {"a": ""}, _WordBackend()) is None


@pytest.mark.unit
class TestPhaseResponseCounts:
    def test_classifies_scored_empty_missing(self):
        counts = _phase_response_counts(["u1", "u2", "u3"], {"u1": "x y", "u2": ""})
        assert counts == {"issued": 3, "scored": 1, "empty": 1, "missing": 1}

    def test_invariant_issued_equals_sum(self):
        counts = _phase_response_counts(
            ["a", "b", "c", "d"], {"a": "x", "b": "", "c": "y"}
        )
        assert (
            counts["issued"] == counts["scored"] + counts["empty"] + counts["missing"]
        )

    def test_empty_iterable(self):
        assert _phase_response_counts([], {}) == {
            "issued": 0,
            "scored": 0,
            "empty": 0,
            "missing": 0,
        }

    def test_scored_matches_osl_population(self):
        # scored uses the same truthiness test as _phase_osl_stats, so it equals
        # the number of texts OSL would tokenize (the OSL population).
        uuid_to_text = {"a": "x y", "b": "", "c": "z"}
        uuids = ["a", "b", "c", "missing"]
        counts = _phase_response_counts(uuids, uuid_to_text)
        osl = _phase_osl_stats(uuids, uuid_to_text, _WordBackend())
        assert counts["scored"] == 2  # "a" and "c"
        assert osl is not None  # a non-empty population exists


@pytest.mark.unit
class TestAccuracyUuidBound:
    def test_unions_accuracy_datasets_only(self, tmp_path):
        (tmp_path / "sample_idx_map.json").write_bytes(
            msgspec.json.encode(
                {
                    "aime25::gptoss": {"u1": 0, "u2": 1},
                    "gpqa::gptoss": {"u3": 0},
                    "performance": {"p1": 0},  # perf entry excluded from the bound
                }
            )
        )
        cfgs = [
            _cfg("aime25::gptoss", 2, 0.8, tmp_path),
            _cfg("gpqa::gptoss", 1, 0.9, tmp_path),
            _cfg("performance", 1, 0.6, tmp_path),
        ]
        assert _accuracy_uuid_bound(tmp_path, cfgs) == {"u1", "u2", "u3"}

    def test_none_report_dir_returns_empty_without_warning(self, caplog):
        with caplog.at_level("WARNING"):
            assert _accuracy_uuid_bound(None, []) is None
        assert "uuid bound unavailable" not in caplog.text

    def test_missing_map_warns_and_falls_back(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            assert (
                _accuracy_uuid_bound(tmp_path, [_cfg("ds", 1, 0.8, tmp_path)]) is None
            )
        assert "uuid bound unavailable" in caplog.text

    def test_corrupt_map_warns_and_falls_back(self, tmp_path, caplog):
        (tmp_path / "sample_idx_map.json").write_text("{not valid json")
        with caplog.at_level("WARNING"):
            assert (
                _accuracy_uuid_bound(tmp_path, [_cfg("ds", 1, 0.8, tmp_path)]) is None
            )
        assert "uuid bound unavailable" in caplog.text

    def test_non_dict_map_warns_and_falls_back(self, tmp_path, caplog):
        # Valid JSON of the wrong shape (a list) must not raise out of finalize.
        (tmp_path / "sample_idx_map.json").write_bytes(msgspec.json.encode(["nope"]))
        with caplog.at_level("WARNING"):
            assert (
                _accuracy_uuid_bound(tmp_path, [_cfg("ds", 1, 0.8, tmp_path)]) is None
            )
        assert "not an object" in caplog.text

    def test_non_dict_dataset_value_is_skipped(self, tmp_path):
        # A dataset whose value is not an object (null) is skipped, not crashed on.
        (tmp_path / "sample_idx_map.json").write_bytes(
            msgspec.json.encode({"ds": None, "other": {"u1": 0}})
        )
        cfgs = [_cfg("ds", 1, 0.8, tmp_path), _cfg("other", 1, 0.9, tmp_path)]
        assert _accuracy_uuid_bound(tmp_path, cfgs) == {"u1"}


@pytest.mark.unit
class TestPhaseDuration:
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
        entry = _by_name(score_accuracy(_ctx([cfg]), result))["aime25::gptoss"]
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
        entry = _by_name(score_accuracy(_ctx([cfg]), _RESULT))["np::ds"]
        # np.floating is a float subclass, so isinstance(..., float) is not
        # enough — assert it is specifically NOT a numpy scalar.
        assert not isinstance(entry["score"], np.floating)
        assert entry["score"] == 0.5
        # Both serializers used downstream (results.json / result_summary.json)
        # must accept the coerced entry.
        json.dumps(entry)
        msgspec.json.encode(entry)


@pytest.mark.unit
def test_write_accuracy_results_keeps_stable_average_accuracy_key(tmp_path):
    # The Python aggregator was renamed to samples_weighted_average_accuracy, but
    # the JSON artifact key is intentionally frozen as "average_accuracy" so
    # existing accuracy_results.json consumers keep working. Lock the key name and
    # that its value is the unit_samples-weighted mean (627/1030, not the flat 0.75).
    scores = [
        {"dataset_name": "aime", "score": 0.9, "unit_samples": 30},
        {"dataset_name": "lcb", "score": 0.6, "unit_samples": 1000},
    ]
    write_accuracy_results(tmp_path, scores, token_count_source=TokenCountSource.SERVER)
    data = msgspec.json.decode(
        (tmp_path / "accuracy" / "accuracy_results.json").read_bytes()
    )
    assert "average_accuracy" in data
    assert "samples_weighted_average_accuracy" not in data
    expected = (0.9 * 30 + 0.6 * 1000) / (30 + 1000)
    assert data["average_accuracy"] == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("source", list(TokenCountSource))
def test_write_accuracy_results_labels_the_token_basis(tmp_path, source):
    scores = [{"dataset_name": "ds", "score": 0.5, "unit_samples": 2}]
    write_accuracy_results(tmp_path, scores, token_count_source=source)
    data = msgspec.json.decode(
        (tmp_path / "accuracy" / "accuracy_results.json").read_bytes()
    )
    assert data["token_count_source"] == source.value
    assert data["accuracy_scores"] == scores
