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

import msgspec
import pytest
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput, TokenCountSource
from inference_endpoint.metrics.token_count_basis import ServerUsage

from tests.test_helpers import LOGGED_SERVER_USAGE_CASES

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


def _line(record: dict) -> str:
    """One event-log line, compact like the JSONL writer's."""
    return json.dumps(record, separators=(",", ":"))


def _ev(event_type, ts, uuid="", data=None):
    return _line(
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


_CLIENT = TokenCountSource.CLIENT
_SERVER = TokenCountSource.SERVER


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
    s = mod.compute_series(events, count_tokens=_ws_counts, basis=_CLIENT)
    assert s["ttft_ns"] == [50]  # a only; w untracked, c never streamed
    assert sorted(s["sample_latency_ns"]) == [300, 400]  # a and c
    assert s["tpot_ns"] == [(400 - 150) / 2]  # "one twothree" -> 2 ws tokens
    # without a tokenizer the tool degrades to ttft/latency only
    tokenless = mod.compute_series(events, count_tokens=None, basis=_CLIENT)
    assert tokenless["tpot_ns"] == []


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
    s = mod.compute_series(events, count_tokens=_ws_counts, basis=_CLIENT)
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
        # fraction style must hard-error, not silently mean p0.99%
        mod.main([str(events), "--percentiles", "0.99"])
    with pytest.raises(SystemExit):
        mod.main([str(events), "--confidence", "1.0"])
    capsys.readouterr()  # drain


def _usage(line, completion_tokens, prompt_tokens=1):
    rec = json.loads(line)
    rec["prompt_tokens"] = prompt_tokens
    rec["completion_tokens"] = completion_tokens
    return _line(rec)


def _summary(run_dir, primary, **fields):
    """The run's ``result_summary.json``, recording ``primary`` as its basis."""
    path = run_dir / "performance" / "result_summary.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"token_count_source": primary.value, **fields}))
    return path


def test_server_basis_tpot_from_completion_tokens(tmp_path, capsys):
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _usage(_ev("sample.complete", 400, "a", _tmo(["x", "one two"])), 3),
            _ev("sample.issued", 300, "b"),
            _ev("sample.recv_first", 360, "b"),
            _usage(
                _ev("sample.complete", 600, "b", _tmo(["x", "y"], tool_calls=[["tc"]])),
                5,
            ),
            _ev("sample.issued", 320, "c"),
            _ev("sample.recv_first", 380, "c"),
            _usage(_ev("sample.complete", 500, "c", _tmo(["x"])), 1),  # <= 1: skipped
            _ev("sample.issued", 340, "d"),
            _ev("sample.recv_first", 390, "d"),
            _ev("sample.complete", 700, "d", _tmo(["x", "y z"])),  # no usage: skipped
            _ev("session.stop_performance_tracking", 800),
        ],
    )
    s = mod.compute_series(events, count_tokens=None, basis=_SERVER)
    assert sorted(s["tpot_ns"]) == [
        60.0,
        125.0,
    ]  # b: 240 / 4 (tool call kept), a: 250 / 2
    assert "tool-call" not in capsys.readouterr().err


def test_main_reads_server_basis_from_sibling_summary(tmp_path, capsys):
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _usage(_ev("sample.complete", 400, "a", _tmo(["x", "one two"])), 3),
            _ev("session.stop_performance_tracking", 500),
        ],
    )
    _summary(tmp_path, _SERVER)
    mod.main([str(events)])
    captured = capsys.readouterr()
    assert "tpot (from tpot_ns, n=1)" in captured.out
    assert "TPOT skipped" not in captured.err


def test_server_basis_counts_missing_usage(tmp_path, capsys):
    events = _write(
        tmp_path,
        [
            _ev("sample.issued", 5, "w"),  # untracked: never counted
            _ev("sample.complete", 50, "w", _tmo(["x"])),
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _usage(_ev("sample.complete", 400, "a", _tmo(["x", "one two"])), 3),
            _ev("sample.issued", 300, "b"),
            _ev("sample.recv_first", 360, "b"),
            _ev("sample.complete", 600, "b", _tmo(["x", "y"])),  # streamed, no usage
            _ev("sample.issued", 320, "c"),
            _ev("sample.complete", 500, "c", _tmo("whole")),  # non-streaming, no usage
            _ev("sample.issued", 340, "d"),
            _ev("sample.complete", 700, "d", None),  # error: not a missing-usage sample
            _ev("session.stop_performance_tracking", 800),
        ],
    )
    s = mod.compute_series(events, count_tokens=None, basis=_SERVER)
    assert s["tpot_ns"] == [125.0]  # missing usage is never filled in from a tokenizer
    err = capsys.readouterr().err
    assert "WARN: 2 tracked samples returned no server token usage" in err


_ENCODER = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


@pytest.mark.parametrize(
    ("prompt", "completion", "output", "expected"),
    [pytest.param(*case[1:], id=case[0]) for case in LOGGED_SERVER_USAGE_CASES],
)
def test_replay_missing_usage_matches_live_rule(
    tmp_path, capsys, prompt, completion, output, expected
):
    """The server-basis replay reads usage by ``classify_server_usage``, as live."""
    complete = EventRecord(
        event_type=SampleEventType.COMPLETE,
        timestamp_ns=400,
        sample_uuid="a",
        data=output,
        prompt_tokens=prompt,
        completion_tokens=completion,
    )
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _ENCODER.encode(complete).decode(),
            _ev("session.stop_performance_tracking", 500),
        ],
    )
    s = mod.compute_series(events, count_tokens=None, basis=_SERVER)
    usable = expected is ServerUsage.USABLE
    assert s["tpot_ns"] == ([250 / (completion - 1)] if usable else [])
    missing = expected is ServerUsage.MISSING
    err = capsys.readouterr().err
    assert ("1 tracked samples returned no server token usage" in err) is missing


def _usage_run_events(run_dir, *, usage):
    """Streamed samples whose COMPLETE lines carry server usage iff ``usage``."""
    lines = [_ev("session.start_performance_tracking", 10)]
    for i in range(4):
        t, u = 100 * (i + 1), f"s{i}"
        complete = _ev("sample.complete", t + 70 + i, u, _tmo(["x", "one two"]))
        lines += [
            _ev("sample.issued", t, u),
            _ev("sample.recv_first", t + 20, u),
            _usage(complete, 9) if usage else complete,
        ]
    run_dir.mkdir()
    return _write(run_dir, [*lines, _ev("session.stop_performance_tracking", 900)])


def test_client_primary_replay_identical_with_and_without_usage_keys(tmp_path):
    with_usage, without_usage = (
        mod.compute_series(
            _usage_run_events(tmp_path / f"usage_{usage}", usage=usage),
            count_tokens=_ws_counts,
            basis=_CLIENT,
        )
        for usage in (True, False)
    )
    assert with_usage == without_usage
    assert len(with_usage["tpot_ns"]) == 4


@pytest.mark.parametrize("primary", list(TokenCountSource))
def test_es_script_refuses_json_off_primary(tmp_path, capsys, monkeypatch, primary):
    # The summary holds TPOT ES maps on the run's basis. A replay on the other
    # basis must not be cross-checked against them, nor written into them.
    events = _write(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _usage(_ev("sample.complete", 400, "a", _tmo(["x", "one two"])), 4),
            _ev("session.stop_performance_tracking", 500),
        ],
    )
    tpot = {"percentiles": {"50.0": 83.0}, "early_stopping_percentiles": {"50.0": 1.0}}
    summary = _summary(tmp_path, primary, tpot=tpot)
    monkeypatch.setattr(mod, "_make_counter", lambda path: _ws_counts)
    off = primary.other
    argv = [str(events), "--summary", str(summary), "--tokenizer", "unused"]
    argv += ["--token-count-source", off.value]
    with pytest.raises(SystemExit, match=f"on the {primary.value} token basis"):
        mod.main([*argv, "--json", str(tmp_path / "o.json")])

    augmented = mod.main(argv)
    out = capsys.readouterr().out
    assert "tpot (from tpot_ns, n=1)" in out  # the off-primary replay still prints
    assert (
        f"XCHECK in-band: skipped (TPOT replayed on the {off.value} basis; "
        f"the run reported {primary.value})"
    ) in out
    assert "MISMATCH" not in out
    assert augmented["tpot"] == tpot  # the run's own maps are untouched


def _copied_run(tmp_path, summary_basis, tpot):
    """An event log copied away from its run; the run's summary stays behind."""
    (tmp_path / "copy").mkdir()
    events = _write(
        tmp_path / "copy",
        [
            _ev("session.start_performance_tracking", 10),
            _ev("sample.issued", 100, "a"),
            _ev("sample.recv_first", 150, "a"),
            _usage(_ev("sample.complete", 400, "a", _tmo(["x", "one two"])), 6),
            _ev("session.stop_performance_tracking", 500),
        ],
    )
    summary = tmp_path / "run" / "result_summary.json"
    summary.parent.mkdir()
    summary.write_text(json.dumps({"token_count_source": summary_basis, "tpot": tpot}))
    return events, summary


def test_es_script_summary_outside_the_run_dir_sets_the_basis(
    tmp_path, capsys, monkeypatch
):
    # Nothing beside the copied log records a basis; --summary records server.
    tpot = {"percentiles": {"50.0": 50.0}}
    events, summary = _copied_run(tmp_path, _SERVER.value, tpot)
    monkeypatch.setattr(mod, "_make_counter", lambda path: _ws_counts)
    argv = [str(events), "--summary", str(summary), "--tokenizer", "unused"]
    off_primary = [*argv, "--token-count-source", _CLIENT.value]
    with pytest.raises(SystemExit, match="on the server token basis"):
        mod.main([*off_primary, "--json", str(tmp_path / "o.json")])

    out = tmp_path / "augmented.json"
    mod.main([*argv, "--json", str(out)])
    captured = capsys.readouterr()
    assert "token basis: server" in captured.out
    assert "no token-count basis recorded" not in captured.err
    assert "tpot (from tpot_ns, n=1)" in captured.out
    augmented = json.loads(out.read_text())
    assert augmented["token_count_source"] == _SERVER.value
    assert augmented["tpot"]["early_stopping_percentiles"] == {"50.0": None}  # n=1


def test_es_script_unreadable_summary_basis_with_override_never_labels_it(
    tmp_path, capsys, monkeypatch
):
    events, summary = _copied_run(tmp_path, "bogus", {})
    monkeypatch.setattr(mod, "_make_counter", lambda path: _ws_counts)
    argv = [str(events), "--summary", str(summary), "--tokenizer", "unused"]
    with pytest.raises(SystemExit) as exc:
        mod.main(argv)
    assert exc.value.code == 2
    assert "pass --token-count-source" in capsys.readouterr().err
    chosen = [*argv, "--token-count-source", _SERVER.value]
    # The summary's own basis is unknown, so the chosen one cannot label it.
    with pytest.raises(SystemExit, match="could not be read"):
        mod.main([*chosen, "--json", str(tmp_path / "o.json")])
    mod.main(chosen)
    assert "token basis: server" in capsys.readouterr().out


def test_es_script_unreadable_summary_exits_with_hint(tmp_path, capsys):
    events = _write(tmp_path, [_ev("session.start_performance_tracking", 10)])
    (tmp_path / "performance").mkdir()
    (tmp_path / "performance" / "result_summary.json").write_text("{not json")
    with pytest.raises(SystemExit) as exc:
        mod.main([str(events)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot read the token-count basis" in err
    assert "pass --token-count-source" in err
    # The hint holds: choosing the basis replays the run.
    mod.main([str(events), "--token-count-source", _SERVER.value])
    capsys.readouterr()  # drain


def test_es_script_missing_summary_fails_before_the_replay(tmp_path, monkeypatch):
    events = _write(tmp_path, [_ev("session.start_performance_tracking", 10)])
    monkeypatch.setattr(
        mod, "compute_series", lambda *a, **k: pytest.fail("log replayed")
    )
    missing = tmp_path / "nope.json"
    with pytest.raises(SystemExit, match=f"FATAL: --summary {missing} not found"):
        mod.main([str(events), "--summary", str(missing)])


def test_es_script_override_needs_no_archived_basis_note(tmp_path, capsys):
    events = _write(tmp_path, [_ev("session.start_performance_tracking", 10)])
    mod.main([str(events)])
    assert "no token-count basis recorded" in capsys.readouterr().err
    mod.main([str(events), "--token-count-source", _SERVER.value])
    err = capsys.readouterr().err
    assert "no token-count basis recorded" not in err
    assert "NOTE: token basis: server" in err
