# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/steady_state_diagnostics.py.

The script is self-contained (no ``inference_endpoint`` import), so these tests pin
everything it owns: the plain-JSON event parse (wire shapes referenced from
core/record.py + core/types.py), super-pass bucketing, TTFT/TPOT reconstruction via an
injected token counter, the trend-detection algorithms, and the CoV table.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "steady_state_diagnostics",
        Path("scripts/steady_state_diagnostics.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field annotations (PEP 563 strings) resolve.
    sys.modules[spec.name] = mod
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


def _words(texts):
    """Fake tokenizer: token count == whitespace word count."""
    return [len(t.split()) for t in texts]


# --------------------------------------------------------------------------- #
# text_after_first_chunk — port of TextModelOutput.text_after_first_chunk over
# the parsed JSON array [tag, output, reasoning?, tool_calls?].
# --------------------------------------------------------------------------- #


def test_text_after_first_chunk_streaming_output_drops_first():
    data = ["TextModelOutput", ["hello ", "world ", "again"]]
    assert mod.text_after_first_chunk(data) == "world again"


def test_text_after_first_chunk_reasoning_first_keeps_all_output():
    # reasoning is a tuple (streaming) -> first chunk lived in reasoning, so all
    # output chunks are post-first-chunk and are kept.
    data = ["TextModelOutput", ["out1 ", "out2"], ["think1 ", "think2 "]]
    assert mod.text_after_first_chunk(data) == "think2 out1 out2"


def test_text_after_first_chunk_non_streaming_str_has_no_first_chunk():
    data = ["TextModelOutput", "the whole answer"]
    assert mod.text_after_first_chunk(data) == ""


# --------------------------------------------------------------------------- #
# Super-pass series construction
# --------------------------------------------------------------------------- #


def _write_events(tmp_path, lines):
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_build_super_pass_series_buckets_by_issue_order(tmp_path):
    lines = [
        _ev("session.start_performance_tracking", 0),
        # super-pass 0 (superpass_size=2): samples A, B
        _ev("sample.issued", 1000, "A"),
        _ev("sample.issued", 1100, "B"),
        _ev("sample.recv_first", 1300, "B"),
        _ev("sample.recv_first", 1500, "A"),
        _ev("sample.complete", 2100, "B", ["TextModelOutput", ["x ", "y y y"]]),
        _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b b"]]),
        # super-pass 1: sample C
        _ev("sample.issued", 1200, "C"),
        _ev("sample.recv_first", 1400, "C"),
        _ev("sample.complete", 2000, "C", ["TextModelOutput", ["p ", "q"]]),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=2, count_tokens=_words)

    assert len(series) == 2
    sp0, sp1 = series
    assert sp0.n_issued == 2
    assert sorted(sp0.ttft_ns) == [200.0, 500.0]
    # TPOT: A -> "b b" 2 tokens over (3000-1500)=1500 -> 750; B -> "y y y" 3 over
    # (2100-1300)=800 -> 266.66...
    assert sorted(round(v, 2) for v in sp0.tpot_ns) == [266.67, 750.0]
    assert sp0.out_tokens == 5
    assert sp1.n_issued == 1
    assert sp1.ttft_ns == [200.0]


def test_build_super_pass_series_records_e2e_latency(tmp_path):
    lines = [
        _ev("session.start_performance_tracking", 0),
        _ev("sample.issued", 1000, "A"),
        _ev("sample.recv_first", 1500, "A"),
        _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b b"]]),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=4, count_tokens=_words)
    assert series[0].latency_ns == [2000.0]  # complete(3000) - issued(1000)


def test_warm_turn_ttft_excludes_cold_first_turn(tmp_path):
    def evt(et, ts, uuid, turn, data=None):
        return json.dumps(
            {
                "event_type": et,
                "timestamp_ns": ts,
                "sample_uuid": uuid,
                "turn": turn,
                "data": data,
            }
        )

    lines = [
        _ev("session.start_performance_tracking", 0),
        evt("sample.issued", 1000, "A", 1),
        evt("sample.recv_first", 1200, "A", 1),  # cold turn 1 -> ttft 200
        evt("sample.complete", 3000, "A", 1, ["TextModelOutput", ["a ", "b"]]),
        evt("sample.issued", 1100, "B", 2),
        evt("sample.recv_first", 1150, "B", 2),  # warm turn 2 -> ttft 50
        evt("sample.complete", 2000, "B", 2, ["TextModelOutput", ["c ", "d"]]),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=4, count_tokens=_words)
    assert sorted(series[0].ttft_ns) == [50.0, 200.0]  # all turns
    assert series[0].ttft_warm_ns == [50.0]  # turn 1 discarded


def test_build_super_pass_series_ignores_events_outside_tracking(tmp_path):
    lines = [
        _ev("sample.issued", 500, "PRE"),  # before tracking -> ignored
        _ev("session.start_performance_tracking", 900),
        _ev("sample.issued", 1000, "A"),
        _ev("sample.recv_first", 1500, "A"),
        _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b"]]),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=4, count_tokens=_words)
    assert len(series) == 1
    assert series[0].n_issued == 1


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def test_percentile_lower_nearest_rank():
    s = [1.0, 2.0, 3.0, 4.0]
    assert mod.percentile_lower(s, 0.5) == 2.0
    assert mod.percentile_lower(s, 0.0) == 1.0
    # nearest-rank-lower: int(0.99*3) == 2 -> 3.0 (matches reference percentile_lower)
    assert mod.percentile_lower(s, 0.99) == 3.0
    assert mod.percentile_lower(s, 1.0) == 4.0


def test_cov_flat_series_is_zero():
    assert mod.cov([5.0, 5.0, 5.0]) == 0.0


def test_cov_positive_for_varied_series():
    assert mod.cov([1.0, 2.0, 3.0]) > 0.0


# --------------------------------------------------------------------------- #
# Trend algorithms — verdict in {"up", "steady", "down"}
# --------------------------------------------------------------------------- #

STRONG_UP = [float(i) for i in range(12)]
STRONG_DOWN = [float(11 - i) for i in range(12)]
FLAT = [5.0] * 12


def test_mann_kendall_detects_upward_trend():
    assert mod.mann_kendall(STRONG_UP).verdict == "up"


def test_mann_kendall_detects_downward_trend():
    assert mod.mann_kendall(STRONG_DOWN).verdict == "down"


def test_mann_kendall_flat_is_steady():
    assert mod.mann_kendall(FLAT).verdict == "steady"


def test_hamed_rao_inflates_variance_vs_plain_mk():
    # Autocorrelation correction can only widen (or equal) the MK variance.
    plain = mod.mann_kendall(STRONG_UP)
    corrected = mod.mann_kendall_hamed_rao(STRONG_UP)
    assert corrected.variance >= plain.variance
    assert corrected.verdict == "up"


def test_hamed_rao_does_not_fabricate_trend_on_oscillation():
    # A sawtooth "climbs" in raw issue order, but that trend is an ordering artifact.
    # Naive MK is fooled; the Hamed-Rao autocorrelation correction must inflate the
    # variance (never collapse it) and return steady.
    saw = [0.0, 20.0, 2.0, 22.0, 4.0, 24.0, 6.0, 26.0, 8.0, 28.0, 10.0, 30.0]
    plain = mod.mann_kendall(saw)
    hr = mod.mann_kendall_hamed_rao(saw)
    assert plain.verdict == "up"  # uncorrected MK false-positives
    assert hr.verdict == "steady"  # corrected test refuses
    assert hr.variance >= plain.variance  # inflated, not collapsed to a sliver


def test_theil_sen_recovers_exact_slope():
    assert mod.theil_sen([0.0, 2.0, 4.0, 6.0, 8.0]).slope == 2.0


def test_theil_sen_flat_is_steady():
    assert mod.theil_sen(FLAT).verdict == "steady"


def test_newey_west_detects_trend_and_flat():
    assert mod.newey_west(STRONG_UP).verdict == "up"
    assert mod.newey_west(FLAT).verdict == "steady"


def test_slope_vs_scatter_matches_reference_formula():
    # Clean line: huge SNR, large rel-drift -> up.
    assert mod.slope_vs_scatter(STRONG_UP).verdict == "up"
    assert mod.slope_vs_scatter(STRONG_DOWN).verdict == "down"
    # Flat -> steady.
    assert mod.slope_vs_scatter(FLAT).verdict == "steady"


def test_trend_algorithms_report_insufficient_below_min_n():
    for fn in (mod.mann_kendall, mod.theil_sen, mod.newey_west, mod.slope_vs_scatter):
        assert fn([1.0, 2.0, 3.0]).verdict == "insufficient"


# --------------------------------------------------------------------------- #
# Rolling scan + CoV table
# --------------------------------------------------------------------------- #


def test_rolling_windows_enumerates_all_positions():
    # n=6, window=4 -> starts 0,1,2 -> [0,4),[1,5),[2,6)
    windows = mod.rolling_windows(n=6, window=4)
    assert windows == [(0, 4), (1, 5), (2, 6)]


def test_cov_table_pass_fail_against_bounds():
    # A per-super-pass p50 series that is dead flat -> CoV 0 -> passes every bound.
    flat_series = [3.0, 3.0, 3.0, 3.0]
    row = mod.cov_pass_row(flat_series, bounds=(0.03, 0.05, 0.08))
    assert row == {0.03: True, 0.05: True, 0.08: True}

    # A noisy series whose CoV exceeds 0.03 but not 0.08.
    noisy = [3.0, 3.2, 2.9, 3.15]
    row2 = mod.cov_pass_row(noisy, bounds=(0.03, 0.5))
    assert row2[0.03] is False
    assert row2[0.5] is True


# --------------------------------------------------------------------------- #
# Top-level run() + render_text() contract
# --------------------------------------------------------------------------- #


def _synthetic_events(tmp_path, n, ttft_ns_fn):
    """One sample per super-pass (superpass_size=1), n super-passes."""
    lines = [_ev("session.start_performance_tracking", 0)]
    t = 1000
    for i in range(n):
        uuid = f"s{i}"
        issue = t
        recv = issue + int(ttft_ns_fn(i))
        complete = recv + 1000
        lines.append(_ev("sample.issued", issue, uuid))
        lines.append(_ev("sample.recv_first", recv, uuid))
        lines.append(
            _ev("sample.complete", complete, uuid, ["TextModelOutput", ["a ", "b b"]])
        )
        t += 10_000
    lines.append(_ev("session.stop_performance_tracking", t))
    return _write_events(tmp_path, lines)


def test_run_result_structure(tmp_path):
    # ttft climbs steadily -> an upward drift the scan should surface.
    path = _synthetic_events(tmp_path, n=8, ttft_ns_fn=lambda i: 100 + 20 * i)
    result = mod.run(
        path, superpass_size=1, count_tokens=_words, window_sizes=[4], warmup=1
    )
    assert result["n_super_passes"] == 8
    assert result["n_post_warmup"] == 7
    assert "ttft_p50" in result["trajectories"]
    # window 4 over 7 post-warmup super-passes -> 4 rolling positions
    rolling = result["drift"]["4"]["ttft_p50"]["rolling"]
    assert [r["window"] for r in rolling] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    # CoV table carries a pass/fail cell per bound and a gate flag
    cell = result["cov"]["4"]["ttft_p50"]
    assert cell["gated"] is True
    assert set(cell["passes"]) == {"0.03", "0.05", "0.08"}
    # p99 is present but marked diagnostic (not gated)
    assert result["cov"]["4"]["ttft_p99"]["gated"] is False


def test_text_after_first_chunk_empty_reasoning_str_output_is_first_chunk():
    # reasoning=[] is falsy -> str output IS the sole first chunk -> excluded.
    data = ["TextModelOutput", "the whole answer", []]
    assert mod.text_after_first_chunk(data) == ""


def test_retried_sample_counts_ttft_once(tmp_path):
    lines = [
        _ev("session.start_performance_tracking", 0),
        _ev("sample.issued", 1000, "A"),
        _ev("sample.recv_first", 1500, "A"),  # ttft 500 (the real first token)
        _ev("sample.issued", 2000, "A"),  # retry: refresh issue ts only
        _ev("sample.recv_first", 2600, "A"),  # must NOT add a second ttft
        _ev("sample.complete", 4000, "A", ["TextModelOutput", ["a ", "b b"]]),
        _ev("session.stop_performance_tracking", 5000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=4, count_tokens=_words)
    assert series[0].ttft_ns == [500.0]


def test_valid_json_line_missing_timestamp_is_skipped(tmp_path):
    # A syntactically valid event object missing timestamp_ns must be skipped, not crash.
    partial = json.dumps({"event_type": "sample.issued", "sample_uuid": "BAD"})
    lines = [
        _ev("session.start_performance_tracking", 0),
        partial,
        _ev("sample.issued", 1000, "A"),
        _ev("sample.recv_first", 1500, "A"),
        _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b"]]),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=4, count_tokens=_words)
    assert sum(sp.n_issued for sp in series) == 1  # only the well-formed sample


def test_cov_pass_row_insufficient_window_is_inconclusive():
    # Fewer than 2 points -> CoV undefined -> cells inconclusive (None), not True.
    assert mod.cov_pass_row([3.0], bounds=(0.03, 0.05)) == {0.03: None, 0.05: None}


def test_run_rejects_negative_warmup(tmp_path):
    path = _synthetic_events(tmp_path, n=6, ttft_ns_fn=lambda i: 100.0)
    with pytest.raises(ValueError):
        mod.run(path, superpass_size=1, count_tokens=_words, warmup=-1)


def test_super_pass_tracks_timestamps(tmp_path):
    lines = [
        _ev("session.start_performance_tracking", 0),
        _ev("sample.issued", 1000, "A"),
        _ev("sample.recv_first", 1500, "A"),
        _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b b"]]),
        _ev("sample.issued", 1100, "B"),
        _ev("sample.recv_first", 1400, "B"),
        _ev("sample.complete", 2500, "B", ["TextModelOutput", ["c ", "d"]]),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=4, count_tokens=_words)
    assert series[0].first_issue_ns == 1000  # earliest issue
    assert series[0].last_event_ns == 3000  # latest event (A's complete)


def test_window_elapsed_and_pooling(tmp_path):
    lines = [
        _ev("session.start_performance_tracking", 0),
        _ev("sample.issued", 1000, "A"),
        _ev("sample.recv_first", 1500, "A"),
        _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b b"]]),
        _ev("sample.issued", 11000, "B"),
        _ev("sample.recv_first", 11500, "B"),
        _ev("sample.complete", 13000, "B", ["TextModelOutput", ["c ", "d d d"]]),
        _ev("session.stop_performance_tracking", 20000),
    ]
    path = _write_events(tmp_path, lines)
    series = mod.build_super_pass_series(path, superpass_size=1, count_tokens=_words)
    assert len(series) == 2
    # pooled ttft over both super-passes
    assert sorted(mod.pooled(series, 0, 2, "ttft_ns")) == [500.0, 500.0]
    assert mod.pooled_out_tokens(series, 0, 2) == 5  # 2 + 3
    # elapsed = last_event(sp1) - first_issue(sp0) = 13000 - 1000
    assert mod.window_elapsed_ns(series, 0, 2) == 12000


def test_histogram_bins_sum_to_count():
    vals = [1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    hist = mod.histogram(vals, nbins=4)
    assert sum(b["count"] for b in hist) == len(vals)
    assert hist[0]["lo"] == 1.0
    assert hist[-1]["hi"] == 13.0


def test_histogram_degenerate_single_value():
    hist = mod.histogram([5.0, 5.0, 5.0], nbins=4)
    assert sum(b["count"] for b in hist) == 3


def test_summarize_reports_stats_and_histogram():
    vals = [float(i) for i in range(1, 101)]
    s = mod.summarize(vals)
    assert s["count"] == 100
    assert s["mean"] == 50.5
    assert s["p50"] == 50.0  # nearest-rank-lower
    assert s["p99"] == 99.0
    assert sum(b["count"] for b in s["histogram"]) == 100


def test_batch_means_ci_zero_variance_is_point():
    lo, hi = mod.batch_means_ci([10.0, 10.0, 10.0, 10.0])
    assert lo == 10.0 and hi == 10.0


def test_batch_means_ci_brackets_mean():
    lo, hi = mod.batch_means_ci([8.0, 10.0, 12.0, 10.0])
    assert lo < 10.0 < hi


def test_pettitt_detects_level_shift():
    series = [0.0] * 8 + [10.0] * 8
    res = mod.pettitt(series)
    assert res["significant"] is True
    assert res["change_point"] == 8  # first segment has 8 elements


def test_pettitt_flat_series_no_change():
    res = mod.pettitt([5.0] * 16)
    assert res["significant"] is False


def test_tps_formulas():
    # per-user = 1e9 / mean_tpot_ns; 2e6 ns/token -> 500 tok/s/user
    assert mod.per_user_tps(2_000_000.0) == 500.0
    # system = out_tokens / (elapsed_ns / 1e9); 5000 tokens over 10s -> 500 tok/s
    assert mod.system_tps(5000, 10_000_000_000) == 500.0


def _mk_series(levels, samples=40):
    """Build a SuperPassRollup list; ``levels`` = list of (tpot, ttft) per super-pass."""
    series = []
    for i, (tp, tt) in enumerate(levels):
        sp = mod.SuperPassRollup(index=i)
        sp.tpot_ns = [float(tp)] * samples
        sp.ttft_ns = [float(tt)] * samples
        sp.out_tokens = samples * 10
        sp.n_issued = samples
        sp.first_issue_ns = i * 1000
        sp.last_issue_ns = i * 1000 + 100  # issue span within the super-pass
        sp.last_event_ns = i * 1000 + 900  # completions land later (drain)
        series.append(sp)
    return series


def test_window_issue_span_excludes_the_drain():
    series = _mk_series([(100, 50)] * 3)
    # issue span: last_issue(SP2)=2100 - first_issue(SP0)=0 = 2100
    assert mod.window_issue_span_ns(series, 0, 3) == 2100
    # completion span is larger (includes the drain): last_event(SP2)=2900 - 0
    assert mod.window_elapsed_ns(series, 0, 3) == 2900
    assert mod.window_issue_span_ns(series, 0, 3) < mod.window_elapsed_ns(series, 0, 3)


GATE = "mk_hamed_rao"
BOUNDS = (0.03, 0.05, 0.08)


def test_window_admissible_flat_yes_spanning_jump_no():
    series = _mk_series([(100, 50)] * 4 + [(200, 60)] * 4)
    assert mod.window_admissible(series, 0, 4, GATE, BOUNDS) is True
    # a window straddling the 100->200 jump has high CoV -> inadmissible
    assert mod.window_admissible(series, 2, 6, GATE, BOUNDS) is False


def test_segment_plateaus_splits_staircase():
    series = _mk_series([(100, 50)] * 6 + [(200, 60)] * 6)
    plateaus = mod.segment_plateaus(series, GATE, BOUNDS)
    assert plateaus == [(0, 6), (6, 12)]


def test_segment_plateaus_single_flat_run():
    series = _mk_series([(100, 50)] * 8)
    assert mod.segment_plateaus(series, GATE, BOUNDS) == [(0, 8)]


def test_detect_level_shift_flags_staircase():
    series = _mk_series([(100, 50)] * 6 + [(200, 60)] * 6)
    plateaus = mod.segment_plateaus(series, GATE, BOUNDS)
    shift = mod.detect_level_shift(series, plateaus)
    assert shift["detected"] is True
    assert shift["change_point_sp"] == 6
    assert shift["delta_pct"] > 0  # degradation (TPOT rose)


def test_detect_level_shift_none_on_single_plateau():
    series = _mk_series([(100, 50)] * 8)
    plateaus = mod.segment_plateaus(series, GATE, BOUNDS)
    assert mod.detect_level_shift(series, plateaus)["detected"] is False


def test_build_steady_state_reports_first_plateau_and_anomaly():
    series = _mk_series([(100, 50)] * 6 + [(200, 60)] * 6)
    ss = mod.build_steady_state(series, GATE, BOUNDS)
    assert ss["found"] is True
    assert ss["window"]["sp_lo"] == 0 and ss["window"]["sp_hi"] == 6  # first plateau
    assert ss["tps"]["per_user"] > 0 and ss["tps"]["system"] > 0
    assert ss["ttft"]["count"] == 6 * 40
    assert ss["anomaly"]["detected"] is True  # the 100->200 step is surfaced


def test_adaptive_warmup_crops_tpot_ramp():
    # TPOT ramps up to a flat 50 -> symmetric band drops the leading below-steady ramp.
    series = _mk_series([(20, 50), (30, 50), (40, 50)] + [(50, 50)] * 7)
    # steady (back-half median) = 50; band 0.05 -> keep from first SP within 5% (48 -> SP3)
    assert mod.adaptive_warmup(series, driver="tpot_p50", band=0.05) == 3


def test_adaptive_warmup_flat_run_returns_min():
    series = _mk_series([(50, 50)] * 8)
    assert mod.adaptive_warmup(series, driver="tpot_p50", band=0.05) == 1


def test_adaptive_warmup_capped_at_max_frac():
    # A never-settling monotonic ramp is capped so it can't crop the whole run.
    series = _mk_series([(10 * (i + 1), 50) for i in range(10)])
    assert mod.adaptive_warmup(series, driver="tpot_p50", band=0.05, max_frac=0.5) == 5


def test_run_auto_warmup_resolves_to_int(tmp_path):
    path = _synthetic_events(tmp_path, n=8, ttft_ns_fn=lambda i: 100.0)
    result = mod.run(path, superpass_size=1, count_tokens=_words, warmup="auto")
    assert isinstance(result["warmup"], int)
    assert result["warmup_mode"] == "auto"


def test_build_steady_state_flags_global_drift_after_first_plateau():
    # First plateau is flat, but TPOT ramps up for the rest of the run (the C22528
    # pattern): a local plateau exists, yet the metric drifts up globally.
    series = _mk_series([(100, 50)] * 6 + [(100 + 25 * i, 50) for i in range(1, 8)])
    ss = mod.build_steady_state(series, GATE, BOUNDS)
    assert ss["found"] is True
    assert ss["window"]["sp_lo"] == 0 and ss["window"]["sp_hi"] == 6
    assert ss["anomaly"]["detected"] is False  # gradual ramp, not a discrete staircase
    assert "tpot_p50" in ss["drifting_up"]  # global Drifting-Up gate catches it
    assert "ttft_p50" not in ss["drifting_up"]  # TTFT is flat


def test_build_steady_state_no_global_drift_on_flat_run():
    series = _mk_series([(100, 50)] * 8)
    ss = mod.build_steady_state(series, GATE, BOUNDS)
    assert ss["drifting_up"] == []


def test_build_steady_state_none_when_run_never_settles():
    # per-super-pass TPOT ramps every step -> no length-4 window is within CoV
    series = _mk_series([(100 + 20 * i, 50) for i in range(8)])
    ss = mod.build_steady_state(series, GATE, BOUNDS)
    assert ss["found"] is False


def test_run_result_has_steady_state_block(tmp_path):
    path = _synthetic_events(tmp_path, n=8, ttft_ns_fn=lambda i: 100.0)
    result = mod.run(
        path, superpass_size=1, count_tokens=_words, window_sizes=[4], warmup=1
    )
    assert "steady_state" in result
    assert "anomaly" in result["steady_state"]


def test_render_text_has_section_headers(tmp_path):
    path = _synthetic_events(tmp_path, n=8, ttft_ns_fn=lambda i: 100 + 20 * i)
    result = mod.run(
        path, superpass_size=1, count_tokens=_words, window_sizes=[4], warmup=1
    )
    text = mod.render_text(result, cov_bounds=[0.03, 0.05, 0.08])
    assert "window size 4" in text
    assert "CoV steadiness" in text
    assert "drift (whole-run" in text
    assert "ttft_p50" in text


# --------------------------------------------------------------------------- #
# CLI simplification: model registry, profiles, sidecar auto-detect, NATL
# --------------------------------------------------------------------------- #


def test_resolve_tokenizer_matches_and_none():
    assert mod.resolve_tokenizer("Kimi-K3-Instruct") == ("moonshotai/Kimi-K3", True)
    assert mod.resolve_tokenizer("/models/gpt-oss-120b") == (
        "openai/gpt-oss-120b",
        False,
    )
    assert mod.resolve_tokenizer("some-deepseek-r1-distill") == (
        "deepseek-ai/DeepSeek-R1",
        False,
    )
    assert mod.resolve_tokenizer("llama-3-70b") is None


def test_profile_for_load_pattern():
    assert mod.profile_for_load_pattern("agentic_inference").name == "agentic"
    assert mod.profile_for_load_pattern("agentic_inference").metric == "natl"
    assert mod.profile_for_load_pattern("poisson").name == "poisson"
    assert mod.profile_for_load_pattern("max_throughput").name == "offline"
    assert mod.profile_for_load_pattern("offline").name == "offline"
    assert mod.profile_for_load_pattern("concurrency").name == "concurrency"
    assert mod.profile_for_load_pattern("something-unknown").name == "concurrency"


def test_find_run_files_from_dir_and_events(tmp_path):
    client = tmp_path / "client"
    client.mkdir()
    (client / "events.jsonl").write_text("{}\n")
    (client / "config.yaml").write_text("model_params:\n  name: /models/Kimi-K3\n")
    (client / "run_meta.json").write_text('{"dataset_size": 613}')
    ev, cfg, meta = mod.find_run_files(str(tmp_path))
    assert ev.endswith("client/events.jsonl")
    assert cfg is not None and cfg.endswith("config.yaml")
    assert meta is not None and meta.endswith("run_meta.json")
    # passing the events path directly resolves the same sidecars
    ev2, cfg2, meta2 = mod.find_run_files(ev)
    assert (ev2, cfg2, meta2) == (ev, cfg, meta)


def test_find_run_files_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.find_run_files(str(tmp_path))


def test_read_run_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "model_params:\n"
        "  name: /models/Kimi-K3\n"
        "  tokenizer_name: /models/Kimi-K3\n"
        "settings:\n"
        "  load_pattern:\n"
        "    type: agentic_inference\n"
        "datasets:\n"
        "  - name: agentic_combined\n"
        "    agentic_inference:\n"
        "      num_trajectories_to_issue: 613\n"
    )
    meta = tmp_path / "run_meta.json"
    meta.write_text('{"dataset_size": 6396}')
    c = mod.read_run_config(str(cfg), str(meta))
    assert c["model"] == "/models/Kimi-K3"
    assert c["load_pattern"] == "agentic_inference"
    assert c["num_trajectories"] == 613
    assert c["dataset_size"] == 6396


def test_read_run_config_top_level_load_pattern(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "load_pattern:\n  type: concurrency\nmodel_params:\n  name: gpt-oss\n"
    )
    c = mod.read_run_config(str(cfg), None)
    assert c["load_pattern"] == "concurrency"
    assert c["model"] == "gpt-oss"
    assert c["dataset_size"] is None


def _conv_ev(et, ts, uuid, conv, turn=None, data=None):
    return json.dumps(
        {
            "event_type": et,
            "timestamp_ns": ts,
            "sample_uuid": uuid,
            "conversation_id": conv,
            "turn": turn,
            "data": data,
        }
    )


def test_build_trajectory_natl(tmp_path):
    lines = [
        _ev("session.start_performance_tracking", 0),
        # c1: two turns, 1s each -> sum latency 2s; tokens 2 + 3 = 5 -> NATL 2.5
        _conv_ev("sample.issued", 0, "a", "c1", 1),
        _conv_ev(
            "sample.complete",
            1_000_000_000,
            "a",
            "c1",
            1,
            ["TextModelOutput", ["a ", "b"]],
        ),
        _conv_ev("sample.issued", 2_000_000_000, "b", "c1", 2),
        _conv_ev(
            "sample.complete",
            3_000_000_000,
            "b",
            "c1",
            2,
            ["TextModelOutput", ["c ", "d ", "e"]],
        ),
        # c2: one turn, 2s; tokens 4 -> NATL 2.0
        _conv_ev("sample.issued", 0, "d", "c2", 1),
        _conv_ev(
            "sample.complete",
            2_000_000_000,
            "d",
            "c2",
            1,
            ["TextModelOutput", ["w ", "x ", "y ", "z"]],
        ),
        _ev("session.stop_performance_tracking", 9_000_000_000),
    ]
    path = _write_events(tmp_path, lines)
    pairs = mod.build_trajectory_natl(path, _words)
    # sorted by completion: c2 completes at 2e9, c1 at 3e9
    assert [round(n, 3) for _, n in pairs] == [2.0, 2.5]
    assert [t for t, _ in pairs] == [2_000_000_000, 3_000_000_000]


def test_build_natl_result_flat_and_varied():
    flat = [(i, 5.0) for i in range(100)]
    r = mod.build_natl_result(flat, superpass_trajectories=10, cov_bounds=(0.10, 0.15))
    assert r["n_trajectories"] == 100
    assert r["n_super_passes"] == 10
    assert len(r["sp_median"]) == 10
    assert r["across_cov"] == 0.0
    assert r["found"] is True
    assert r["distribution"]["p50"] == 5.0
    # a wildly varying series across super-passes -> high across-CoV -> not found
    varied = [(i, float(1 + (i // 10) * 5)) for i in range(100)]  # 1,6,11,...,46
    r2 = mod.build_natl_result(
        varied, superpass_trajectories=10, cov_bounds=(0.10, 0.15)
    )
    assert r2["across_cov"] > 0.15
    assert r2["found"] is False
