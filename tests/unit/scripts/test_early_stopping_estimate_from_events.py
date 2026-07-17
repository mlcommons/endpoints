# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/early_stopping_estimate_from_events.py.

The script is a thin composition of product pieces (typed EventRecord decode,
TextModelOutput chunk semantics — covered in tests/unit/test_core_types.py — and the
ES math — covered in tests/unit/metrics/test_early_stopping.py). These tests pin only
what the script itself owns: aggregator-parity event gating, robustness to imperfect
historical logs, the product-mapping drift guard, and the CLI contract.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from inference_endpoint.core.types import TextModelOutput

pytestmark = pytest.mark.unit


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "early_stopping_estimate_from_events",
        Path("scripts/early_stopping_estimate_from_events.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_script()


def _ev(event_type, ts, uuid="", data=None):
    return json.dumps(
        {
            "event_type": event_type,
            "timestamp_ns": ts,
            "sample_uuid": uuid,
            "data": data,
        }
    )


def _tmo(output=None, tool_calls=None):
    # wire form of TextModelOutput (tagged, array_like) as the JSONL writer emits it
    return ["TextModelOutput", output, None, tool_calls]


def _ws_counts(texts):  # deterministic stand-in tokenizer: whitespace tokens
    return [len(t.split()) for t in texts]


def _write(tmp_path, lines):
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_extract_tpot_text_gates_on_decoded_type():
    # The script owns only the gate; chunk semantics are the product's
    # text_after_first_chunk (tested with TextModelOutput itself).
    assert mod.extract_tpot_text(TextModelOutput(output=("a", "b c"))) == "b c"
    assert (
        mod.extract_tpot_text(TextModelOutput(output=("a",), tool_calls=(("t",),)))
        is None
    )
    assert mod.extract_tpot_text(None) is None
    assert mod.extract_tpot_text("not a model output") is None


def test_metric_mapping_comes_from_product():
    # The series -> summary-field mapping must be report.py's own object (it
    # builds result_summary.json from it) — a copy would drift.
    from inference_endpoint.metrics.report import SERIES_TO_SUMMARY_FIELD

    assert mod.SERIES_TO_SUMMARY_FIELD is SERIES_TO_SUMMARY_FIELD
    assert set(mod._UNITS) == set(SERIES_TO_SUMMARY_FIELD.values())


def test_tracking_window_gating_and_metric_deltas(tmp_path):
    # Aggregator parity for the happy path: only samples ISSUED inside the
    # tracking window count; completions after stop still count for open rows;
    # ttft/latency/tpot use the aggregator's exact formulas.
    events = _write(
        tmp_path,
        [
            _ev("sample.issued", 5, "w"),  # BEFORE tracking -> excluded entirely
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.issued", 300, "c"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.recv_first", 50, "w"),
            _ev("sample.complete", 400, "a", _tmo(["x", "one two", "three"])),
            _ev("sample.complete", 500, "w", _tmo(["x", "y"])),
            _ev("session.stop_performance_tracking", 550),
            _ev("sample.complete", 700, "c", _tmo("nonstream full")),  # no recv_first
        ],
    )
    s = mod.compute_series(events, count_tokens=_ws_counts)
    assert s["ttft_ns"] == [50]  # a only; w untracked, c never streamed
    assert sorted(s["sample_latency_ns"]) == [300, 400]  # a and c
    assert s["tpot_ns"] == [(400 - 150) / 2]  # "one twothree" -> 2 ws tokens
    # without a tokenizer the tool degrades to ttft/latency only
    assert mod.compute_series(events, count_tokens=None)["tpot_ns"] == []


def test_imperfect_log_robustness(tmp_path, capsys):
    # Historical logs are the input domain: corrupt lines and retries must be
    # counted/warned — never crash, never silently change the series.
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            "this is not json {",  # malformed 1
            "123",  # valid JSON, not an event record -> malformed 2
            json.dumps(
                {"event_type": 42, "timestamp_ns": 1}
            ),  # decode_hook escape -> 3
            _ev("sample.issued", 100, "a"),
            json.dumps(
                {"event_type": "sample.issued", "timestamp_ns": 1}
            ),  # no uuid -> 4
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.issued", 200, "a"),  # retry: keep row, refresh issue ts
            _ev("sample.recv_first", 250, "a"),  # retry re-emits its first chunk
            _ev("sample.issued", 300, "b"),
            _ev("sample.recv_first", 360, "b"),
            _ev("sample.complete", 400, "b", _tmo(["x", "y"], tool_calls=[["tc"]])),
            _ev("sample.complete", 600, "a", _tmo(["x", "one two"])),
            _ev("session.stop_performance_tracking", 700),
        ],
    )
    s = mod.compute_series(events, count_tokens=_ws_counts)
    # aggregator parity: BOTH recv_first events fire ttft (150-100, 250-200)
    assert sorted(s["ttft_ns"]) == [50, 50, 60]
    # a: complete - REFRESHED issue ts (duplicate-ISSUED parity)
    assert sorted(s["sample_latency_ns"]) == [100, 400]
    # a's TPOT window starts at the RETRY's first chunk; b skipped (tool call)
    assert s["tpot_ns"] == [(600 - 250) / 2]
    err = capsys.readouterr().err
    assert "4" in err and "malformed" in err
    assert "tool-call" in err


def test_cross_check_against_inband_map(capsys):
    # --summary cross-check must flag real divergence from a run's in-band map,
    # and skip unexpected shapes (pre-release artifacts) instead of crashing.
    ours = {"99.0": 123.0}

    def summary(esp):
        return {"ttft": {"early_stopping_percentiles": esp}}

    mod._cross_check(summary({"99.0": 123.0}), "ttft_ns", dict(ours))
    assert "EXACT MATCH" in capsys.readouterr().out
    mod._cross_check(summary({"99.0": -1.0}), "ttft_ns", dict(ours))
    assert "MISMATCH" in capsys.readouterr().out
    mod._cross_check(summary([{"legacy": "shape"}]), "ttft_ns", dict(ours))
    assert "MATCH" not in capsys.readouterr().out


def test_main_augments_summary_in_run_shape(tmp_path, capsys):
    # --json must emit the run's summary AUGMENTED in place: same document, with
    # each recorded metric gaining early_stopping_percentiles directly after its
    # own grid, keyed and ordered by that grid (int-style keys preserved).
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.complete", 400, "a", _tmo(["x", "one two"])),
            _ev("session.stop_performance_tracking", 500),
        ],
    )
    summary = {
        "n_samples_completed": 1,
        "ttft": {
            "min": 50,
            "percentiles": {"99": 50.0, "90": 50.0, "50": 50.0},
            "histogram": {"buckets": [], "counts": []},
        },
        "tpot": {},  # not recorded by the run -> must stay untouched
        "latency": {
            "min": 300,
            "percentiles": {"99.0": 300.0, "50.0": 300.0},
            "histogram": {"buckets": [], "counts": []},
        },
    }
    spath = tmp_path / "result_summary.json"
    spath.write_text(json.dumps(summary))
    out = tmp_path / "augmented.json"
    result = mod.main([str(events), "--summary", str(spath), "--json", str(out)])
    augmented = json.loads(out.read_text())
    assert augmented == result
    # ttft: grid-keyed (int style preserved), grid order, placed after percentiles
    ttft = augmented["ttft"]
    assert list(ttft) == [
        "min",
        "percentiles",
        "early_stopping_percentiles",
        "histogram",
    ]
    assert list(ttft["early_stopping_percentiles"]) == ["99", "90", "50"]
    assert all(v is None for v in ttft["early_stopping_percentiles"].values())  # n=1
    assert augmented["tpot"] == {}  # untouched
    assert list(augmented["latency"]["early_stopping_percentiles"]) == ["99.0", "50.0"]
    # everything else byte-identical to the input summary
    stripped = json.loads(out.read_text())
    for field in ("ttft", "latency"):
        stripped[field].pop("early_stopping_percentiles")
    assert stripped == summary
    capsys.readouterr()  # drain


def test_fallback_percentiles_dedupe_on_value(tmp_path, capsys):
    # "99" and "99.0" are the same target — one computation, one canonical key.
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.complete", 400, "a", _tmo(["x", "one two"])),
            _ev("session.stop_performance_tracking", 500),
        ],
    )
    mod.main([str(events), "--percentiles", "99,99.0,90"])
    out = capsys.readouterr().out
    # one row per metric section for the canonical key; a bare "p99:" row would
    # mean "99" survived as a second target alongside "99.0"
    assert out.count("p99.0:") == out.count("p90.0:") > 0
    assert "p99: " not in out


def test_main_cli_contract(tmp_path, capsys):
    # invalid percentile/confidence must exit up front (they would hang or
    # corrupt the math); --json without --summary is a contract error.
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _ev("sample.complete", 400, "a", _tmo(["x", "one two"])),
            _ev("session.stop_performance_tracking", 500),
        ],
    )
    with pytest.raises(SystemExit):
        mod.main([str(events), "--json", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        mod.main([str(events), "--percentiles", "150"])  # grid convention: (0, 100)
    with pytest.raises(SystemExit):
        # pre-#423 fraction style must hard-error, not silently mean p0.99%
        mod.main([str(events), "--percentiles", "0.99"])
    with pytest.raises(SystemExit):
        mod.main([str(events), "--confidence", "1.0"])
    capsys.readouterr()  # drain
