# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/synth_events.py.

The generator plants a known steady region / ramp / drift / staircase into an
``events.jsonl`` whose wire schema matches ``scripts/steady_state_diagnostics.py``. These
tests round-trip generated runs back through the diagnostic (with a whitespace token
counter, so 1 chunk == 1 word == ~1 token) and assert the reconstructed TTFT/TPOT recover
the planted base values and that a flat run is detected as steady.
"""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, Path(rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


synth = _load("synth_events", "scripts/synth_events.py")
diag = _load("steady_state_diagnostics", "scripts/steady_state_diagnostics.py")


def _words(texts):
    """Fake tokenizer: token count == whitespace word count (matches the emitted chunks)."""
    return [len(t.split()) for t in texts]


def _args(**overrides):
    """Small, fast, flat-by-default run suitable for steady detection."""
    base = {
        "mode": "concurrency",
        "concurrency": 8,
        "rate": 50.0,
        "duration_s": 20.0,
        "seed": 42,
        "base_ttft_ms": 50.0,
        "base_tpot_ms": 10.0,
        "prefill_per_tok_ns": 0.0,
        "ttft_noise": 0.04,
        "tpot_noise": 0.04,
        "osl_median": 30.0,
        "osl_sigma": 0.3,
        "osl_cap": 8192,
        "isl_median": 256.0,
        "isl_sigma": 0.4,
        "isl_cap": 131072,
        "ramp_s": 5.0,
        "ramp_start": 2.0,
        "drift_end": 1.0,
        "step_at_s": None,
        "step_mag": 1.15,
        "turns": 1,
    }
    base.update(overrides)
    ns = type("NS", (), base)()
    return synth.planted_from_args(ns)


def _generate(tmp_path, **overrides):
    planted = _args(**overrides)
    gen = synth.build_events(planted)
    gt = synth.groundtruth(planted, gen)
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = str(tmp_path / "events.jsonl")
    synth.write_run(out, gen, gt)
    return out, planted, gt


# --------------------------------------------------------------------------- #
# (a) generated events parse; TTFT/TPOT recover the planted base in the steady region
# --------------------------------------------------------------------------- #


def test_generated_events_parse_and_recover_base_metrics(tmp_path):
    out, planted, gt = _generate(tmp_path)
    series = diag.build_super_pass_series(out, superpass_size=40, count_tokens=_words)
    assert len(series) >= 4

    # Pool the back half (past the ramp) and compare medians to the planted base.
    lo = len(series) // 2
    ttft = diag.pooled(series, lo, len(series), "ttft_ns")
    tpot = diag.pooled(series, lo, len(series), "tpot_ns")
    assert ttft and tpot

    assert statistics.median(ttft) == pytest.approx(planted.base_ttft_ns, rel=0.15)
    assert statistics.median(tpot) == pytest.approx(planted.base_tpot_ns, rel=0.15)


def test_events_have_expected_schema(tmp_path):
    out, _, _ = _generate(tmp_path)
    types = set()
    saw_complete_list = False
    with open(out) as fh:
        for line in fh:
            rec = json.loads(line)
            types.add(rec["event_type"])
            assert isinstance(rec["timestamp_ns"], int)
            if rec["event_type"] == "sample.complete":
                assert rec["data"][0] == "TextModelOutput"
                assert isinstance(rec["data"][1], list)
                saw_complete_list = True
    assert saw_complete_list
    assert {
        "session.start_performance_tracking",
        "session.stop_performance_tracking",
        "sample.issued",
        "sample.recv_first",
        "sample.complete",
    } <= types


def test_issued_events_are_chronological(tmp_path):
    out, _, _ = _generate(tmp_path)
    issued_ts = [
        json.loads(line)["timestamp_ns"]
        for line in open(out)
        if json.loads(line)["event_type"] == "sample.issued"
    ]
    assert issued_ts == sorted(issued_ts)


# --------------------------------------------------------------------------- #
# (b) groundtruth sidecar schema
# --------------------------------------------------------------------------- #


def test_groundtruth_sidecar_keys(tmp_path):
    out, planted, gt = _generate(tmp_path)
    sidecar = Path(out + ".groundtruth.json")
    assert sidecar.is_file()
    loaded = json.loads(sidecar.read_text())
    expected = {
        "mode",
        "concurrency",
        "rate",
        "duration_s",
        "seed",
        "base_ttft_ns",
        "base_tpot_ns",
        "steady_per_user_tps",
        "ramp_s",
        "ramp_start",
        "drift_end",
        "step_at_s",
        "step_mag",
        "osl_median",
        "osl_sigma",
        "osl_cap",
        "isl_median",
        "isl_sigma",
        "ttft_noise",
        "tpot_noise",
        "turns",
        "n_samples",
        "run_wall_clock_s",
        "steady_window_s",
        "steady_is_flat",
    }
    assert expected <= set(loaded)
    assert loaded["mode"] == "concurrency"
    assert loaded["steady_per_user_tps"] == pytest.approx(1e9 / planted.base_tpot_ns)
    assert loaded["steady_window_s"] == [planted.ramp_s, planted.duration_s]
    assert loaded["steady_is_flat"] is True


def test_groundtruth_staircase_window_stops_at_step(tmp_path):
    _, planted, gt = _generate(tmp_path, step_at_s=12.0, step_mag=1.3)
    assert gt["steady_window_s"] == [planted.ramp_s, 12.0]
    assert gt["steady_is_flat"] is False


def test_groundtruth_drift_marks_not_flat(tmp_path):
    _, _, gt = _generate(tmp_path, drift_end=1.3)
    assert gt["steady_is_flat"] is False


# --------------------------------------------------------------------------- #
# (c) determinism
# --------------------------------------------------------------------------- #


def test_same_seed_identical_bytes(tmp_path):
    out_a, _, _ = _generate(tmp_path / "a")
    out_b, _, _ = _generate(tmp_path / "b")
    assert Path(out_a).read_bytes() == Path(out_b).read_bytes()
    assert (
        Path(out_a + ".groundtruth.json").read_bytes()
        == Path(out_b + ".groundtruth.json").read_bytes()
    )


def test_different_seed_differs(tmp_path):
    out_a, _, _ = _generate(tmp_path / "a", seed=1)
    out_b, _, _ = _generate(tmp_path / "b", seed=2)
    assert Path(out_a).read_bytes() != Path(out_b).read_bytes()


# --------------------------------------------------------------------------- #
# (d) planted-steady no-drift run -> build_steady_state finds a window
# --------------------------------------------------------------------------- #


def test_flat_run_detected_as_steady(tmp_path):
    out, _, _ = _generate(tmp_path)
    result = diag.run(out, superpass_size=40, count_tokens=_words)
    ss = result["steady_state"]
    assert ss["found"] is True
    assert ss["window"] is not None
    assert ss["window"]["n_super_passes"] >= 4
    assert ss["anomaly"]["detected"] is False


def test_poisson_mode_generates_and_parses(tmp_path):
    out, planted, gt = _generate(tmp_path, mode="poisson", rate=40.0, duration_s=20.0)
    assert gt["mode"] == "poisson"
    assert gt["concurrency"] is None
    assert gt["rate"] == 40.0
    series = diag.build_super_pass_series(out, superpass_size=40, count_tokens=_words)
    assert sum(sp.n_issued for sp in series) == gt["n_samples"]


def test_multiturn_emits_conversation_and_turn(tmp_path):
    out, _, gt = _generate(tmp_path, turns=3, duration_s=15.0)
    convs = set()
    turns = set()
    with open(out) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["event_type"] == "sample.issued":
                assert "conversation_id" in rec
                convs.add(rec["conversation_id"])
                turns.add(rec["turn"])
    assert len(convs) >= 1
    assert turns <= {1, 2, 3}
    assert max(turns) >= 2  # multi-turn trajectories actually issued
