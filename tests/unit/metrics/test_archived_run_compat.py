# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run archived before server token counts existed keeps its numbers.

``tests/assets/archived_runs/echo_char_streaming`` is a real report directory
(client tokenizer counts only, no ``usage`` anywhere). Every post-run consumer
must still decode it and recompute exactly what that run reported. A failure
here means an archived ``events.jsonl`` or ``final_snapshot.json`` changed
meaning, which is never acceptable; fix the code, not the pinned values.
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import msgspec
import pytest
from inference_endpoint.commands.benchmark.full_run_osl import full_run_osl_for_report
from inference_endpoint.core.record import EventRecord, EventType
from inference_endpoint.core.types import TokenCountSource
from inference_endpoint.metrics import steady_state_diagnostics as ssd
from inference_endpoint.metrics.report import Report

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_DIR = _REPO_ROOT / "tests/assets/archived_runs/echo_char_streaming"
_EVENTS = _RUN_DIR / "events.jsonl"
_SUMMARY = _RUN_DIR / "performance/result_summary.json"
_CHAR_TOKENIZER_DIR = _REPO_ROOT / "tests/assets/tokenizers/char"
_SUMMARY_FIELDS = (
    "n_samples_issued",
    "n_samples_completed",
    "n_samples_failed",
    "duration_ns",
    "state",
    "complete",
    "ttft",
    "tpot",
    "latency",
    "output_sequence_lengths",
    "input_sequence_lengths",
    "legacy_loadgen_window_duration_ns",
    "qps",
    "tps",
    "e2e_avg_interactivity",
    "finish_reason_counts",
    "steady_state",
)


def _json(path):
    return json.loads(Path(path).read_bytes())


def _char_counts(texts):
    # The char tokenizer emits one token per ASCII character.
    return [len(t) for t in texts]


def _load_es_script():
    spec = importlib.util.spec_from_file_location(
        "early_stopping_estimate_from_events_archived",
        _REPO_ROOT / "scripts/early_stopping_estimate_from_events.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_line_decodes_as_event_record():
    decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
    records = [decoder.decode(line) for line in _EVENTS.read_bytes().splitlines()]
    assert Counter(r.event_type.topic for r in records) == {
        "session.started": 1,
        "session.phase_start": 1,
        "session.start_performance_tracking": 1,
        "sample.issued": 15,
        "sample.recv_first": 12,
        "sample.complete": 15,
        "session.stop_performance_tracking": 1,
        "session.ended": 1,
    }
    for record in records:
        assert getattr(record, "prompt_tokens", msgspec.UNSET) is msgspec.UNSET
        assert getattr(record, "completion_tokens", msgspec.UNSET) is msgspec.UNSET


def test_report_from_archived_snapshot_reproduces_archived_summary():
    summary = _json(_SUMMARY)
    report = Report.from_snapshot(
        _json(_RUN_DIR / "metrics/final_snapshot.json"),
        run_config=summary["run_config"],
        use_legacy_loadgen_qps_metrics=False,  # concurrency run
        # Archived runs counted every token with the client tokenizer.
        token_count_source=TokenCountSource.CLIENT,
    )
    rebuilt = json.loads(report.to_json())
    for field in _SUMMARY_FIELDS:
        assert rebuilt[field] == summary[field], field
    assert (report.n_samples_issued, report.n_samples_completed) == (15, 15)
    assert (report.n_samples_failed, report.duration_ns) == (0, 7837666)
    assert (report.state, report.complete) == ("complete", True)
    osl = report.output_sequence_lengths
    assert (osl["total"], osl["min"], osl["max"], osl["avg"]) == (423, 22, 49, 35.25)
    assert report.input_sequence_lengths["total"] == 423
    assert report.tpot["total"] == pytest.approx(50977.06920610349, rel=1e-12)
    assert report.tpot["avg"] == pytest.approx(4248.089100508624, rel=1e-12)
    assert (report.ttft["total"], report.latency["total"]) == (10320914, 13232749)
    assert report.qps == pytest.approx(1913.8350626321662, rel=1e-12)
    assert report.tps == pytest.approx(53970.14876622709, rel=1e-12)
    assert report.finish_reason_counts["stop"] == 15
    assert report.steady_state is not None
    assert report.steady_state.found is False
    assert report.steady_state.superpass_size == 5
    assert report.steady_state.n_super_passes == 3


def test_full_run_osl_recomputes_archived_block():
    block = full_run_osl_for_report(
        _RUN_DIR, str(_CHAR_TOKENIZER_DIR), token_count_source=TokenCountSource.CLIENT
    )
    archived = _json(_SUMMARY)["output_sequence_lengths_full_run"]
    assert block is not None
    for key in (
        "output_sequence_lengths",
        "n_turns_counted",
        "n_empty",
        "n_errors",
        "n_undecodable",
        "n_missing",
        "partial",
    ):
        assert json.loads(json.dumps(block[key])) == archived[key], key
    assert (block["n_turns_counted"], block["n_empty"]) == (12, 3)
    assert block["partial"] is False
    osl = block["output_sequence_lengths"]
    assert (osl["total"], osl["min"], osl["max"], osl["avg"]) == (423, 22, 49, 35.25)


def test_steady_state_replay_rebuilds_archived_series():
    size = ssd.superpass_size_from_events(str(_EVENTS))
    assert size == 5
    series = ssd.build_super_pass_series(
        str(_EVENTS), size, _char_counts, basis=TokenCountSource.CLIENT
    ).super_passes
    assert [
        (sp.n_issued, len(sp.ttft_ns), len(sp.tpot_ns), len(sp.latency_ns))
        for sp in series
    ] == [(5, 4, 4, 5)] * 3
    assert [sp.out_tokens for sp in series] == [116, 116, 116]
    snap = {
        m["name"]: m for m in _json(_RUN_DIR / "metrics/final_snapshot.json")["metrics"]
    }
    assert sum(sum(sp.tpot_ns) for sp in series) == pytest.approx(
        snap["tpot_ns"]["total"], rel=1e-12
    )
    assert sum(sum(sp.ttft_ns) for sp in series) == snap["ttft_ns"]["total"]
    assert snap["ttft_ns"]["total"] == 10320914
    assert sum(sum(sp.latency_ns) for sp in series) == 13232749


def test_steady_state_cli_replays_archived_verdict(tmp_path):
    out = tmp_path / "steady_state.json"
    argv = [str(_RUN_DIR), "--tokenizer", str(_CHAR_TOKENIZER_DIR), "--json", str(out)]
    assert ssd.main(argv) == 0
    replayed = _json(out)["steady_state"]
    assert replayed == _json(_SUMMARY)["steady_state"]
    assert (replayed["found"], replayed["n_super_passes"]) == (False, 3)


def test_early_stopping_script_reproduces_archived_estimates(tmp_path):
    es = _load_es_script()
    series = es.compute_series(
        str(_EVENTS),
        count_tokens=_char_counts,
        basis=TokenCountSource.CLIENT,
    )
    assert {k: len(v) for k, v in series.items()} == {
        "ttft_ns": 12,
        "tpot_ns": 12,
        "sample_latency_ns": 15,
    }
    assert sum(series["ttft_ns"]) == 10320914
    assert sum(series["sample_latency_ns"]) == 13232749
    assert sum(series["tpot_ns"]) == pytest.approx(50977.06920610349, rel=1e-12)

    out = tmp_path / "augmented.json"
    es.main(
        [
            str(_EVENTS),
            "--summary",
            str(_SUMMARY),
            "--tokenizer",
            str(_CHAR_TOKENIZER_DIR),
            "--json",
            str(out),
        ]
    )
    augmented, summary = _json(out), _json(_SUMMARY)
    for field in ("ttft", "tpot", "latency"):
        esp = augmented[field]["early_stopping_percentiles"]
        assert esp == summary[field]["early_stopping_percentiles"], field
        assert esp["90.0"] is None
    assert augmented["ttft"]["early_stopping_percentiles"]["50.0"] == 3258417
    assert augmented["latency"]["early_stopping_percentiles"]["50.0"] == 3045958
    assert augmented["tpot"]["early_stopping_percentiles"]["50.0"] == pytest.approx(
        8921.529411764706, rel=1e-12
    )
