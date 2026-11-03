# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``inference_endpoint.metrics.steady_state_diagnostics``.

Pins everything the detector owns: the event-log parse (wire shapes referenced
from core/record.py + core/types.py), super-pass bucketing, TTFT/TPOT
reconstruction via an injected token counter or the server usage, the
trend-detection algorithms, and the CoV table.
"""

import json

import msgspec
import pytest
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput, TokenCountSource, basis_key
from inference_endpoint.metrics import steady_state_diagnostics as mod
from inference_endpoint.metrics.token_count_basis import ServerUsage

from tests.test_helpers import LOGGED_SERVER_USAGE_CASES

pytestmark = pytest.mark.unit


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


def _words(texts):
    """Fake tokenizer: token count == whitespace word count."""
    return [len(t.split()) for t in texts]


def _tokenized_series(path, superpass_size):
    """Super-passes replayed on the client basis with the whitespace tokenizer."""
    return mod.build_super_pass_series(
        path, superpass_size, _words, basis=TokenCountSource.CLIENT
    ).super_passes


def _tokenized_run(path, **kwargs):
    """``run`` on a client-basis run, one sample per super-pass."""
    return mod.run(
        path,
        superpass_size=1,
        count_tokens=_words,
        basis=TokenCountSource.CLIENT,
        primary=TokenCountSource.CLIENT,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# text_after_first_chunk — TextModelOutput.text_after_first_chunk without the
# tool calls; full_output_text — the whole output, tool calls as logged.
# --------------------------------------------------------------------------- #


def test_text_after_first_chunk_streaming_output_drops_first():
    output = TextModelOutput(output=("hello ", "world ", "again"))
    assert mod.text_after_first_chunk(output) == "world again"


def test_text_after_first_chunk_reasoning_first_keeps_all_output():
    # The first delta contains only reasoning, so content has an empty first entry.
    output = TextModelOutput(
        output=("", "out1 ", "out2"), reasoning=("think1 ", "think2 ")
    )
    assert mod.text_after_first_chunk(output) == "think2 out1 out2"


def test_text_after_first_chunk_joined_output_follows_streamed_reasoning():
    output = TextModelOutput(output="out1 out2", reasoning=("think1 ", "think2 "))
    assert mod.text_after_first_chunk(output) == "think2 out1 out2"


def test_text_after_first_chunk_non_streaming_str_has_no_first_chunk():
    output = TextModelOutput(output="the whole answer")
    assert mod.text_after_first_chunk(output) == ""


def test_text_after_first_chunk_leaves_out_tool_calls():
    tool_calls = (({"index": 0, "function": {"name": "f"}},),)
    output = TextModelOutput(output=("a ", "b"), tool_calls=tool_calls)
    assert mod.text_after_first_chunk(output) == "b"


def test_full_output_text_is_reasoning_output_then_logged_tool_calls():
    tool_calls = [[{"index": 0, "function": {"name": "f", "arguments": "{}"}}]]
    output = msgspec.json.decode(
        msgspec.json.encode(
            ["TextModelOutput", ["a ", "b"], ["r1 ", "r2 "], tool_calls]
        ),
        type=TextModelOutput,
    )
    assert mod.full_output_text(output) == "r1 r2 a b" + json.dumps(tool_calls)
    assert mod.full_output_text(TextModelOutput(output="whole")) == "whole"


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
    series = _tokenized_series(path, 2)

    assert len(series) == 2
    sp0, sp1 = series
    assert sp0.n_issued == 2
    assert sorted(sp0.ttft_ns) == [200.0, 500.0]
    # TPOT: A -> "b b" 2 tokens over (3000-1500)=1500 -> 750; B -> "y y y" 3 over
    # (2100-1300)=800 -> 266.66...
    assert sorted(round(v, 2) for v in sp0.tpot_ns) == [266.67, 750.0]
    assert sp0.out_tokens == 5
    assert sp1.n_issued == 1
    assert list(sp1.ttft_ns) == [200.0]


def test_build_super_pass_series_records_e2e_latency(tmp_path):
    lines = [
        _ev("session.start_performance_tracking", 0),
        _ev("sample.issued", 1000, "A"),
        _ev("sample.recv_first", 1500, "A"),
        _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b b"]]),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    series = _tokenized_series(path, 4)
    assert list(series[0].latency_ns) == [2000.0]  # complete(3000) - issued(1000)


def test_warm_turn_ttft_excludes_cold_first_turn(tmp_path):
    def evt(et, ts, uuid, turn, data=None):
        return _line(
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
    series = _tokenized_series(path, 4)
    assert sorted(series[0].ttft_ns) == [50.0, 200.0]  # all turns
    assert list(series[0].ttft_warm_ns) == [50.0]  # turn 1 discarded


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
    series = _tokenized_series(path, 4)
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
    result = _tokenized_run(path, warmup=1)
    assert result.steady_state.n_super_passes == 8
    assert result.steady_state.warmup == 1
    assert result.drift["ttft_p50"]["mk_hamed_rao"] == "up"
    # CoV cells carry a pass/fail per bound and a gate flag. Admissibility gates on
    # TPOT only, so tpot_p50 is gated while ttft_p50 is now diagnostic.
    cov = result.steady_state.cov
    assert cov["tpot_p50"].gated is True
    cell = cov["ttft_p50"]
    assert cell.gated is False
    assert set(cell.passes) == {"0.03", "0.05", "0.08"}
    # p99 is present but marked diagnostic (not gated)
    assert cov["ttft_p99"].gated is False


def test_text_after_first_chunk_empty_reasoning_str_output_is_first_chunk():
    # reasoning=() is falsy -> str output IS the sole first chunk -> excluded.
    output = TextModelOutput(output="the whole answer", reasoning=())
    assert mod.text_after_first_chunk(output) == ""


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
    series = _tokenized_series(path, 4)
    assert list(series[0].ttft_ns) == [500.0]


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
    series = _tokenized_series(path, 4)
    assert sum(sp.n_issued for sp in series) == 1  # only the well-formed sample


def test_truncated_complete_line_is_skipped(tmp_path):
    # The last line of a killed run can be a cut-off COMPLETE: both replays skip it.
    cut = _conv_ev("sample.complete", 3100, "B", "c2", 1, ["TextModelOutput", ["x"]])
    lines = [
        _ev("session.start_performance_tracking", 0),
        _conv_ev("sample.issued", 1000, "A", "c1", 1),
        _conv_ev("sample.recv_first", 1500, "A", "c1", 1),
        _conv_ev(
            "sample.complete", 3000, "A", "c1", 1, ["TextModelOutput", ["a ", "b"]]
        ),
        _conv_ev("sample.issued", 1100, "B", "c2", 1),
        cut[: len(cut) // 2],
    ]
    path = _write_events(tmp_path, lines)
    (sp,) = _tokenized_series(path, 4)
    assert (sp.n_issued, list(sp.latency_ns)) == (2, [2000.0])
    natl = mod.build_trajectory_natl(path, _words, basis=TokenCountSource.CLIENT)
    assert [t for t, _ in natl.pairs] == [3000]


def test_replays_refuse_a_non_compact_complete_line(tmp_path, capsys):
    # COMPLETE lines are found by their compact prefix; a spaced one fails loudly.
    lines = [
        _ev("session.start_performance_tracking", 0),
        _conv_ev("sample.issued", 1000, "A", "c1", 1),
        _conv_ev("sample.recv_first", 1500, "A", "c1", 1),
        json.dumps(
            json.loads(
                _conv_ev("sample.complete", 3000, "A", "c1", 1, ["TextModelOutput"])
            )
        ),
        _ev("session.stop_performance_tracking", 4000),
    ]
    path = _write_events(tmp_path, lines)
    with pytest.raises(ValueError, match="compact"):
        _tokenized_series(path, 1)
    with pytest.raises(ValueError, match="compact"):
        mod.build_trajectory_natl(path, _words, basis=TokenCountSource.CLIENT)
    _summary(tmp_path, TokenCountSource.SERVER)
    argv = [str(tmp_path), "--superpass-size", "1", "--warmup", "0"]
    for profile in ([], ["--profile", "agentic"]):
        with pytest.raises(SystemExit) as exc:
            mod.main([*argv, *profile])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert f"{path}: a sample.complete line is not in the JSONL writer" in err


def test_cov_pass_row_insufficient_window_is_inconclusive():
    # Fewer than 2 points -> CoV undefined -> cells inconclusive (None), not True.
    assert mod.cov_pass_row([3.0], bounds=(0.03, 0.05)) == {0.03: None, 0.05: None}


def test_run_rejects_negative_warmup(tmp_path):
    path = _synthetic_events(tmp_path, n=6, ttft_ns_fn=lambda i: 100.0)
    with pytest.raises(ValueError):
        _tokenized_run(path, warmup=-1)


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
    series = _tokenized_series(path, 4)
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
    series = _tokenized_series(path, 1)
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
        sp.latency_ns = [float(tp) + float(tt)] * samples
        sp.osl = [10.0] * samples
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


def _analyse(series, **kw):
    """compute_steady_state_metrics over an already-post-warmup series."""
    kw.setdefault("warmup", 0)
    return mod.compute_steady_state_metrics(
        series, superpass_size=40, gate_algo=GATE, cov_bounds=BOUNDS, **kw
    )


def test_window_admissible_flat_yes_spanning_jump_no():
    series = _mk_series([(100, 50)] * 4 + [(200, 60)] * 4)
    assert mod.window_admissible(series, 0, 4, GATE, BOUNDS) is True
    # a window straddling the 100->200 jump has high CoV -> inadmissible
    assert mod.window_admissible(series, 2, 6, GATE, BOUNDS) is False


def test_window_admissible_tolerates_small_drift_but_breaks_large():
    # ~3% end-to-end monotonic drift, low scatter: the rank gate calls it a trend, but it
    # is below the effect-size floor (0.05) -> admissible (not fragmented). CoV is fine.
    small = _mk_series(
        [
            (100.0, 50.0),
            (100.6, 50.0),
            (101.2, 50.0),
            (101.8, 50.0),
            (102.4, 50.0),
            (103.0, 50.0),
        ]
    )
    assert mod.window_admissible(small, 0, 6, GATE, BOUNDS) is True
    # ~15% end-to-end drift -> above the floor -> inadmissible (a real drift still breaks).
    big = _mk_series(
        [
            (100.0, 50.0),
            (103.0, 50.0),
            (106.0, 50.0),
            (109.0, 50.0),
            (112.0, 50.0),
            (115.0, 50.0),
        ]
    )
    assert mod.window_admissible(big, 0, 6, GATE, BOUNDS) is False


def test_ttft_not_gated_but_still_drift_warned():
    # TPOT flat, TTFT climbs steeply. Admissibility gates on TPOT only, so TTFT does not
    # fragment the run -> one plateau. But the whole-run drift warning still surfaces TTFT.
    series = _mk_series(
        [(100.0, 50.0 + 15.0 * i) for i in range(8)]
    )  # (tpot flat, ttft ramps)
    assert (
        mod.window_admissible(series, 0, 8, GATE, BOUNDS) is True
    )  # TTFT ignored by the gate
    ss = _analyse(series, enforce_min_duration=False)
    assert ss.found is True
    assert "ttft_p50" in ss.drifting_up  # TTFT drift surfaced as a soft warning
    assert "tpot_p50" not in ss.drifting_up  # TPOT is flat


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
    assert shift.detected is True
    assert shift.change_point_sp == 6
    assert shift.delta_pct > 0  # degradation (TPOT rose)


def test_detect_level_shift_none_on_single_plateau():
    series = _mk_series([(100, 50)] * 8)
    plateaus = mod.segment_plateaus(series, GATE, BOUNDS)
    assert mod.detect_level_shift(series, plateaus).detected is False


def test_steady_state_reports_first_plateau_and_anomaly():
    series = _mk_series([(100, 50)] * 6 + [(200, 60)] * 6)
    ss = _analyse(series, enforce_min_duration=False)
    assert ss.found is True
    assert ss.window.sp_lo == 0 and ss.window.sp_hi == 6  # first plateau
    assert ss.tps.per_user > 0 and ss.tps.system > 0
    assert ss.ttft["count"] == 6 * 40
    assert ss.anomaly.detected is True  # the 100->200 step is surfaced


def test_steady_state_reports_osl_latency_and_window_bounds():
    series = _mk_series([(100, 50)] * 6)
    ss = _analyse(series, enforce_min_duration=False)
    w = ss.window
    # Wall-clock bounds come from the window's own issue timestamps.
    assert (w.start_ns, w.end_ns) == (
        series[0].first_issue_ns,
        series[5].last_issue_ns,
    )
    # OSL and latency are summarized per sample over the same window as TTFT/TPOT.
    assert ss.osl["count"] == ss.ttft["count"] == 6 * 40
    assert ss.osl["p50"] == 10.0
    assert ss.latency["p50"] == 150.0


def test_steady_state_cov_scores_the_reported_window():
    series = _mk_series([(100, 50)] * 6)
    ss = _analyse(series, enforce_min_duration=False)
    # Dead-flat window -> CoV 0 on every tracked metric, passing every bound.
    assert set(ss.cov) == {m.key for m in mod.TRACKED_METRICS}
    assert ss.cov["tpot_p50"] == mod.CovCell(
        gated=True,
        n=6,
        cov=0.0,
        passes={"0.03": True, "0.05": True, "0.08": True},
    )
    assert ss.cov["ttft_p99"].gated is False


def test_steady_state_carries_the_run_shape():
    series = _mk_series([(100, 50)] * 8)
    ss = mod.compute_steady_state_metrics(
        series,
        superpass_size=512,
        warmup=2,
        cov_bounds=BOUNDS,
        enforce_min_duration=False,
    )
    assert (ss.superpass_size, ss.n_super_passes, ss.warmup) == (512, 8, 2)
    assert ss.window.n_super_passes == 6  # indices are post-warmup


def test_a_flat_but_noisy_run_reports_its_cov_over_the_whole_span():
    """Scatter and drift fail for different reasons, so say which one it was.

    A run that is trend-steady throughout yet yielded no plateau was rejected on
    CoV. The verdict reports that CoV over the span the trend test cleared, so
    "no admissible steady plateau" stops being the whole answer.
    """
    # No trend, but the super-pass means swing +-10% -> CoV above every bound.
    series = _mk_series([(50 if i % 2 else 60, 50) for i in range(8)])
    ss = _analyse(series)
    assert ss.found is False
    assert ss.cov, "a trend-steady run should carry its CoV"
    basis = ss.cov_basis
    assert basis is not None
    assert basis == mod.CovBasis(sp_lo=0, sp_hi=8, n_super_passes=8)
    assert "CoV" in (ss.reason or "")
    assert ss.osl is None and ss.latency is None


def test_a_drifting_run_claims_no_cov_and_names_the_drifter():
    """Nothing was steady, so there is no span a CoV would describe."""
    series = _mk_series([(10 * (i + 1), 50) for i in range(8)])
    ss = _analyse(series)
    assert ss.found is False
    assert ss.cov == {}
    assert ss.cov_basis is None
    assert "trends across the run" in (ss.reason or "")


def test_nothing_trend_steady_reports_no_cov_at_all():
    """Too short to trend-test at all: CoV is not the story, so none is claimed."""
    ss = _analyse(_mk_series([(50, 50)] * (mod.MIN_TREND_N - 1)))
    assert ss.found is False
    assert ss.cov == {}
    assert ss.cov_basis is None


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
    result = _tokenized_run(path, warmup="auto")
    assert isinstance(result.steady_state.warmup, int)


def test_steady_state_flags_global_drift_after_first_plateau():
    # First plateau is flat, but TPOT ramps up for the rest of the run (the C22528
    # pattern): a local plateau exists, yet the metric drifts up globally.
    series = _mk_series([(100, 50)] * 6 + [(100 + 25 * i, 50) for i in range(1, 8)])
    ss = _analyse(series, enforce_min_duration=False)
    assert ss.found is True
    assert ss.window.sp_lo == 0 and ss.window.sp_hi == 6
    assert ss.anomaly.detected is False  # gradual ramp, not a discrete staircase
    assert "tpot_p50" in ss.drifting_up  # global Drifting-Up gate catches it
    assert "ttft_p50" not in ss.drifting_up  # TTFT is flat


def test_steady_state_no_global_drift_on_flat_run():
    series = _mk_series([(100, 50)] * 8)
    ss = _analyse(series, enforce_min_duration=False)
    assert ss.drifting_up == []


def test_steady_state_none_when_run_never_settles():
    # per-super-pass TPOT ramps every step -> no length-4 window is within CoV
    series = _mk_series([(100 + 20 * i, 50) for i in range(8)])
    ss = _analyse(series, enforce_min_duration=False)
    assert ss.found is False


def _spanned_series(
    n, tpot=100.0, per_sp_span_s=1.0, gap_s=0.0, latency_s=0.1, samples=40
):
    """SuperPassRollups with explicit wall-time: per-SP offered span + inter-SP gap +
    a constant sample latency. Lets the min-duration gate see realistic seconds."""
    series = []
    t = 0.0
    for i in range(n):
        sp = mod.SuperPassRollup(index=i)
        sp.tpot_ns = [float(tpot)] * samples
        sp.ttft_ns = [50.0] * samples
        sp.latency_ns = [latency_s * 1e9] * samples
        sp.out_tokens = samples * 10
        sp.n_issued = samples
        sp.first_issue_ns = int(t * 1e9)
        sp.last_issue_ns = int((t + per_sp_span_s) * 1e9)
        sp.last_event_ns = int((t + per_sp_span_s + latency_s) * 1e9)
        series.append(sp)
        t += per_sp_span_s + gap_s
    return series


def test_min_steady_duration_floor_dominates_clean_short_run():
    sw = mod.min_steady_duration(
        _spanned_series(6, per_sp_span_s=1.0, latency_s=0.1), 0, 6
    )
    assert sw.dominant == "floor"
    assert sw.min_duration_s == mod.MIN_DUR_FLOOR_S
    assert sw.kstar == mod.MIN_DUR_KSTAR_FLOOR  # flat metric -> k* pinned at floor
    assert sw.is_short is True  # ~6s window << 600s floor


def test_min_steady_duration_relaxation_dominates_on_long_tail():
    # p99 sample latency 200s -> relax = 5 * 200 = 1000s > floor
    sw = mod.min_steady_duration(
        _spanned_series(6, per_sp_span_s=1.0, latency_s=200.0), 0, 6
    )
    assert sw.dominant == "relaxation"
    assert abs(sw.min_duration_s - 1000.0) < 1e-3


def test_min_steady_duration_precision_dominates_on_noisy_metric():
    series = _spanned_series(8, per_sp_span_s=50.0, latency_s=0.1)
    for i, sp in enumerate(series):
        sp.tpot_ns = [
            100.0 if i % 2 == 0 else 160.0
        ] * 40  # high CoV_b across super-passes
    sw = mod.min_steady_duration(series, 0, 8)
    assert sw.kstar > mod.MIN_DUR_KSTAR_FLOOR  # k* self-raises with CoV
    assert sw.dominant == "precision"


def test_min_steady_duration_precision_exempt_at_kstar_floor():
    # Flat metric -> k* floors at MIN_TREND_N. Even with a large per-super-pass span (so
    # k*·τ would be 4·200 = 800s > floor), the precision term is exempt at the floor, so
    # min_duration falls back to the 600s floor and the ~800s window is NOT short.
    sw = mod.min_steady_duration(
        _spanned_series(4, per_sp_span_s=200.0, latency_s=1.0), 0, 4
    )
    assert sw.kstar == mod.MIN_DUR_KSTAR_FLOOR
    assert sw.t_precision_s == 0.0
    assert sw.dominant == "floor"
    assert sw.min_duration_s == mod.MIN_DUR_FLOOR_S
    assert sw.is_short is False


def test_min_steady_duration_not_short_when_window_long_enough():
    # 6 super-passes each 120s offered -> window span ~720s >= 600s floor
    sw = mod.min_steady_duration(
        _spanned_series(6, per_sp_span_s=120.0, latency_s=1.0), 0, 6
    )
    assert sw.is_short is False


def test_steady_state_hard_fails_short_window():
    series = _mk_series([(100, 50)] * 8)  # flat plateau, microsecond spans
    ss = _analyse(series)  # enforce default True
    assert ss.found is False
    assert ss.reason is not None and "too short" in ss.reason
    assert ss.short_window.is_short is True
    assert ss.window is not None  # window summary kept for context


def test_steady_state_soft_reports_short_window():
    series = _mk_series([(100, 50)] * 8)
    ss = _analyse(series, enforce_min_duration=False)
    assert ss.found is True  # not rejected, only flagged
    assert ss.short_window.is_short is True


def test_steady_state_accepts_long_enough_window():
    series = _spanned_series(8, per_sp_span_s=120.0, latency_s=1.0)  # flat, ~960s span
    ss = _analyse(series)  # enforce default True
    assert ss.found is True
    assert ss.short_window.is_short is False


def _concat_plateaus(segs):
    """Chain SuperPassRollup segments into one series: renumber indices and offset each
    segment's timestamps so they run monotonically, with a 1 s gap between segments."""
    out = []
    t_off = 0
    idx = 0
    for seg in segs:
        base = min(sp.first_issue_ns for sp in seg)
        last = 0
        for sp in seg:
            sp.index = idx
            idx += 1
            sp.first_issue_ns = sp.first_issue_ns - base + t_off
            sp.last_issue_ns = sp.last_issue_ns - base + t_off
            sp.last_event_ns = sp.last_event_ns - base + t_off
            last = max(last, sp.last_event_ns)
            out.append(sp)
        t_off = last + 1_000_000_000
    return out


def test_hard_path_skips_short_plateau_for_next_admissible():
    # Plateau A: 4 super-passes, ~1 s each (4 s window -> short). Plateau B: 8 super-passes,
    # 120 s each (~960 s window -> long enough). Distinct TPOT levels segment them.
    A = _spanned_series(4, tpot=5.0, per_sp_span_s=1.0, latency_s=0.1)
    B = _spanned_series(8, tpot=6.5, per_sp_span_s=120.0, latency_s=1.0)
    ss = _analyse(_concat_plateaus([A, B]))
    assert ss.found is True
    assert ss.window.plateau_index == 1  # first (short) plateau skipped
    assert ss.window.skipped_short == 1
    assert ss.window.sp_lo == 4 and ss.window.sp_hi == 12
    assert ss.short_window.is_short is False


def test_hard_path_rejects_when_all_plateaus_short():
    A = _spanned_series(4, tpot=5.0, per_sp_span_s=1.0, latency_s=0.1)
    B = _spanned_series(5, tpot=6.5, per_sp_span_s=1.0, latency_s=0.1)  # also short
    ss = _analyse(_concat_plateaus([A, B]))
    assert ss.found is False
    assert "too short" in ss.reason
    assert ss.window is not None  # longest candidate kept for context


def test_soft_path_reports_first_plateau_even_when_short():
    A = _spanned_series(4, tpot=5.0, per_sp_span_s=1.0, latency_s=0.1)
    B = _spanned_series(8, tpot=6.5, per_sp_span_s=120.0, latency_s=1.0)
    ss = _analyse(_concat_plateaus([A, B]), enforce_min_duration=False)
    assert ss.found is True
    assert ss.window.plateau_index == 0  # first plateau, no skipping
    assert ss.window.skipped_short == 0
    assert ss.short_window.is_short is True  # flagged, advisory only


def test_run_result_has_steady_state_block(tmp_path):
    path = _synthetic_events(tmp_path, n=8, ttft_ns_fn=lambda i: 100.0)
    result = _tokenized_run(path, warmup=1)
    assert result.steady_state.anomaly is not None


def test_render_text_has_section_headers(tmp_path):
    path = _synthetic_events(tmp_path, n=8, ttft_ns_fn=lambda i: 100 + 20 * i)
    result = _tokenized_run(path, warmup=1)
    text = mod.render_text(result, cov_bounds=[0.03, 0.05, 0.08])
    assert "CoV" in text
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
    # Fails closed: an unclassified workload inherits no verdict.
    assert mod.profile_for_load_pattern("something-unknown") is None


def test_find_run_files_from_dir_and_events(tmp_path):
    client = tmp_path / "client"
    client.mkdir()
    (client / "events.jsonl").write_text("{}\n")
    (client / "config.yaml").write_text("model_params:\n  name: /models/Kimi-K3\n")
    ev, cfg = mod.find_run_files(str(tmp_path))
    assert ev.endswith("client/events.jsonl")
    assert cfg is not None and cfg.endswith("config.yaml")
    # passing the events path directly resolves the same sidecar
    assert mod.find_run_files(ev) == (ev, cfg)


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
    c = mod.read_run_config(str(cfg))
    assert c["model"] == "/models/Kimi-K3"
    assert c["load_pattern"] == "agentic_inference"
    assert c["num_trajectories"] == 613


def test_read_run_config_top_level_load_pattern(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "load_pattern:\n  type: concurrency\nmodel_params:\n  name: gpt-oss\n"
    )
    c = mod.read_run_config(str(cfg))
    assert c["load_pattern"] == "concurrency"
    assert c["model"] == "gpt-oss"


def test_superpass_size_comes_from_the_performance_phase_start(tmp_path):
    """A hand re-run needs no arguments: the size is on the wire."""

    def phase(kind, turns, ts):
        return json.dumps(
            {
                "event_type": "session.phase_start",
                "timestamp_ns": ts,
                "data": ["PhaseData", kind, True, turns, 0],
            }
        )

    path = _write_events(
        tmp_path, [phase("warmup", 99, 1), phase("performance", 613, 2)]
    )
    assert mod.superpass_size_from_events(path) == 613
    assert mod.superpass_size_from_events(_write_events(tmp_path, ["{}"])) is None


def _conv_ev(et, ts, uuid, conv, turn=None, data=None):
    return _line(
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
    trajectory = mod.build_trajectory_natl(path, _words, basis=TokenCountSource.CLIENT)
    assert trajectory.n_missing_usage is None  # the client basis never reads usage
    pairs = trajectory.pairs
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


# --------------------------------------------------------------------------- #
# Token-count basis: server usage from COMPLETE lines, the run-basis reader
# --------------------------------------------------------------------------- #

_ENCODER = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


def _logged(record: EventRecord) -> str:
    """``record`` as the JSONL writer logs it."""
    return _ENCODER.encode(record).decode()


def _with_usage(line, completion_tokens, prompt_tokens=1):
    rec = json.loads(line)
    rec["prompt_tokens"] = prompt_tokens
    rec["completion_tokens"] = completion_tokens
    return _line(rec)


def _summary(run_dir, primary):
    """The run's ``result_summary.json``, recording ``primary`` as its basis."""
    (run_dir / "performance").mkdir()
    (run_dir / "performance" / "result_summary.json").write_text(
        json.dumps({"token_count_source": primary.value})
    )


def test_server_replay_uses_completion_tokens(tmp_path):
    path = _write_events(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 0),
            _ev("sample.issued", 1000, "A"),
            _ev("sample.recv_first", 1500, "A"),
            _with_usage(
                _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b"]]), 4
            ),
            _ev("sample.issued", 1100, "B"),
            _ev("sample.recv_first", 1300, "B"),
            # completion_tokens <= 1: no TPOT
            _with_usage(
                _ev("sample.complete", 2100, "B", ["TextModelOutput", ["x "]]), 1
            ),
            _ev("sample.issued", 1200, "C"),
            _ev("sample.recv_first", 1400, "C"),
            # no usage: no TPOT
            _ev("sample.complete", 2000, "C", ["TextModelOutput", ["p ", "q"]]),
            _ev("session.stop_performance_tracking", 4000),
        ],
    )
    replay = mod.build_super_pass_series(path, 4, None, basis=TokenCountSource.SERVER)
    (sp,) = replay.super_passes
    assert (list(sp.tpot_ns), list(sp.osl), sp.out_tokens, len(sp.latency_ns)) == (
        [500.0],
        [3.0],
        3,
        3,
    )
    assert replay.n_missing_usage == 1
    with pytest.raises(ValueError, match="count_tokens"):
        mod.build_super_pass_series(path, 4, None, basis=TokenCountSource.CLIENT)


def test_build_trajectory_natl_server_basis(tmp_path):
    out = ["TextModelOutput", ["a"]]
    path = _write_events(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 0),
            _conv_ev("sample.issued", 0, "a", "c1", 1),
            _with_usage(
                _conv_ev("sample.complete", 1_000_000_000, "a", "c1", 1, out), 3
            ),
            _conv_ev("sample.issued", 2_000_000_000, "b", "c1", 2),
            _with_usage(
                _conv_ev("sample.complete", 3_000_000_000, "b", "c1", 2, out), 5
            ),
            _conv_ev("sample.issued", 0, "d", "c2", 1),
            # no usage: no tokens
            _conv_ev("sample.complete", 2_000_000_000, "d", "c2", 1, out),
            _ev("session.stop_performance_tracking", 9_000_000_000),
        ],
    )
    assert mod.build_trajectory_natl(
        path, None, basis=TokenCountSource.SERVER
    ) == mod.ReplayTrajectory(
        pairs=[(3_000_000_000, 4.0)], n_missing_usage=1, n_undecodable=0
    )


@pytest.mark.parametrize(
    ("prompt", "completion", "output", "expected"),
    [pytest.param(*case[1:], id=case[0]) for case in LOGGED_SERVER_USAGE_CASES],
)
def test_replay_missing_usage_matches_live_rule(
    tmp_path, prompt, completion, output, expected
):
    """Both replays read usage by ``classify_server_usage``, as the aggregator does."""
    complete = EventRecord(
        event_type=SampleEventType.COMPLETE,
        timestamp_ns=3000,
        sample_uuid="A",
        conversation_id="c1",
        data=output,
        prompt_tokens=prompt,
        completion_tokens=completion,
    )
    path = _write_events(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 0),
            _conv_ev("sample.issued", 1000, "A", "c1"),
            _conv_ev("sample.recv_first", 1500, "A", "c1"),
            _logged(complete),
            _ev("session.stop_performance_tracking", 4000),
        ],
    )
    series = mod.build_super_pass_series(path, 1, None, basis=TokenCountSource.SERVER)
    natl = mod.build_trajectory_natl(path, None, basis=TokenCountSource.SERVER)
    n_missing = int(expected is ServerUsage.MISSING)
    assert (series.n_missing_usage, natl.n_missing_usage) == (n_missing, n_missing)
    usable = expected is ServerUsage.USABLE
    (sp,) = series.super_passes
    assert list(sp.osl) == ([completion - 1] if usable else [])
    latency_s = (3000 - 1000) / 1e9
    assert natl.pairs == ([(3000, completion / latency_s)] if usable else [])


def test_main_server_run_needs_no_tokenizer(tmp_path, monkeypatch):
    lines = [_ev("session.start_performance_tracking", 0)]
    for i in range(8):
        t, u = 1000 * (i + 1), f"s{i}"
        lines += [
            _ev("sample.issued", t, u),
            _ev("sample.recv_first", t + 100, u),
            _with_usage(
                _ev("sample.complete", t + 1100, u, ["TextModelOutput", ["a ", "b"]]),
                11,
            ),
        ]
    _write_events(tmp_path, [*lines, _ev("session.stop_performance_tracking", 20_000)])
    _summary(tmp_path, TokenCountSource.SERVER)
    monkeypatch.setattr(
        mod, "_make_token_counter", lambda *a, **k: pytest.fail("tokenizer loaded")
    )
    argv = [
        str(tmp_path),
        "--superpass-size",
        "2",
        "--warmup",
        "0",
        "--no-min-duration",
    ]
    assert mod.main(argv) == 0
    with pytest.raises(SystemExit):  # client override: no model/tokenizer resolvable
        mod.main([*argv, "--token-count-source", TokenCountSource.CLIENT.value])


def _missing_usage_events(tmp_path):
    """Server run: A has usage; B (streamed) and C (non-streaming) do not; D errored."""
    return _write_events(
        tmp_path,
        [
            _ev("sample.issued", 10, "W"),  # untracked: never counted
            _ev("sample.complete", 20, "W", ["TextModelOutput", ["w"]]),
            _ev("session.start_performance_tracking", 0),
            _ev("sample.issued", 1000, "A"),
            _ev("sample.recv_first", 1500, "A"),
            _with_usage(
                _ev("sample.complete", 3000, "A", ["TextModelOutput", ["a ", "b"]]), 4
            ),
            _ev("sample.issued", 1100, "B"),
            _ev("sample.recv_first", 1300, "B"),
            _ev("sample.complete", 2100, "B", ["TextModelOutput", ["x ", "y"]]),
            _ev("sample.issued", 1200, "C"),
            _ev("sample.complete", 2000, "C", ["TextModelOutput", "whole answer"]),
            _ev("sample.issued", 1300, "D"),
            _ev("sample.complete", 1900, "D", None),
            _ev("session.stop_performance_tracking", 4000),
        ],
    )


def test_run_labels_the_basis_and_counts_missing_usage(tmp_path):
    path = _missing_usage_events(tmp_path)
    server = mod.run(
        path,
        superpass_size=1,
        count_tokens=None,
        warmup=0,
        basis=TokenCountSource.SERVER,
        primary=TokenCountSource.SERVER,
    )
    assert (server.basis, server.n_missing_usage) == (
        TokenCountSource.SERVER,
        2,
    )
    replay = mod.build_super_pass_series(path, 1, None, basis=TokenCountSource.SERVER)
    tpot = [v for sp in replay.super_passes for v in sp.tpot_ns]
    assert tpot == [500.0]  # missing usage is never filled in from a tokenizer
    text = mod.render_text(server, cov_bounds=list(mod.DEFAULT_COV_BOUNDS))
    assert text.splitlines()[0] == "token basis: server"
    client = _tokenized_run(path, warmup=0)
    # The client replay never reads usage, so it reports no missing-usage count.
    assert (client.basis, client.n_missing_usage) == (
        TokenCountSource.CLIENT,
        None,
    )
    tokenized = mod.build_super_pass_series(
        path, 1, _words, basis=TokenCountSource.CLIENT
    )
    assert tokenized.n_missing_usage is None


def test_main_json_is_labelled_with_the_basis(tmp_path, capsys):
    _missing_usage_events(tmp_path)
    _summary(tmp_path, TokenCountSource.SERVER)
    out = tmp_path / "ss.json"
    argv = [str(tmp_path), "--superpass-size", "2", "--warmup", "0"]
    assert mod.main([*argv, "--no-min-duration", "--json", str(out)]) == 0
    blob = json.loads(out.read_text())
    # Top level, beside the steady_state block (which stays summary-comparable).
    assert (blob["token_count_source"], blob["n_missing_usage"]) == ("server", 2)
    assert "token_count_source" not in blob["steady_state"]
    assert "n_undecodable" not in blob
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "token basis: server"
    assert "2 tracked samples returned no server token usage" in captured.err
    assert "NATL and system TPS are understated" in captured.err
    assert "could not be decoded" not in captured.err


@pytest.mark.parametrize("primary", list(TokenCountSource))
def test_main_json_names_the_verdict_by_basis(tmp_path, monkeypatch, capsys, primary):
    """The verdict keeps its name on the run's basis; the other basis suffixes it."""
    _missing_usage_events(tmp_path)
    _summary(tmp_path, primary)
    monkeypatch.setattr(mod, "_make_token_counter", lambda *a, **k: _words)
    argv = [str(tmp_path), "--tokenizer", "words", "--superpass-size", "2"]
    argv += ["--warmup", "0", "--no-min-duration"]
    for override in (None, *TokenCountSource):
        replayed = primary if override is None else override
        flag = [] if override is None else ["--token-count-source", override.value]
        out = tmp_path / f"{override}.json"
        assert mod.main([*argv, *flag, "--json", str(out)]) == 0
        blob = json.loads(out.read_text())
        assert blob["token_count_source"] == replayed.value
        assert basis_key("steady_state", replayed, primary=primary) in blob
        assert basis_key("steady_state", replayed.other, primary=primary) not in blob
    capsys.readouterr()  # drain


def _usage_run(run_dir, *, usage: bool) -> None:
    """A client-basis run whose COMPLETE lines carry server usage iff ``usage``."""
    lines = [_ev("session.start_performance_tracking", 0)]
    for i in range(8):
        t, u = 1000 * (i + 1), f"s{i}"
        complete = _ev(
            "sample.complete", t + 1100 + 10 * i, u, ["TextModelOutput", ["a ", "b c"]]
        )
        lines += [
            _ev("sample.issued", t, u),
            _ev("sample.recv_first", t + 100, u),
            _with_usage(complete, 7) if usage else complete,
        ]
    run_dir.mkdir()
    _write_events(run_dir, [*lines, _ev("session.stop_performance_tracking", 20_000)])
    _summary(run_dir, TokenCountSource.CLIENT)


def test_client_primary_replay_identical_with_and_without_usage_keys(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(mod, "_make_token_counter", lambda *a, **k: _words)
    blobs = []
    for usage in (True, False):
        run_dir = tmp_path / f"usage_{usage}"
        _usage_run(run_dir, usage=usage)
        out = run_dir / "ss.json"
        argv = [str(run_dir), "--tokenizer", "words", "--superpass-size", "2"]
        argv += ["--warmup", "0", "--no-min-duration", "--json", str(out)]
        assert mod.main(argv) == 0
        blobs.append(out.read_bytes())
    assert blobs[0] == blobs[1]
    assert json.loads(blobs[0])["steady_state"]["n_super_passes"] == 4
    capsys.readouterr()  # drain


def test_steady_state_cli_unreadable_summary_exits_with_hint(
    tmp_path, monkeypatch, capsys
):
    _missing_usage_events(tmp_path)
    (tmp_path / "performance").mkdir()
    (tmp_path / "performance" / "result_summary.json").write_text("{not json")
    monkeypatch.setattr(mod, "_make_token_counter", lambda *a, **k: _words)
    argv = [str(tmp_path), "--tokenizer", "words", "--superpass-size", "2"]
    argv += ["--warmup", "0", "--no-min-duration"]
    with pytest.raises(SystemExit) as exc:
        mod.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot read the token-count basis" in err
    assert "pass --token-count-source" in err
    # The hint holds: choosing the basis replays the run.
    assert mod.main([*argv, "--token-count-source", TokenCountSource.SERVER.value]) == 0


def test_main_natl_json_is_labelled_with_the_basis(tmp_path, capsys):
    out_text = ["TextModelOutput", ["a"]]
    _write_events(
        tmp_path,
        [
            _ev("session.start_performance_tracking", 0),
            _conv_ev("sample.issued", 0, "a", "c1", 1),
            _with_usage(
                _conv_ev("sample.complete", 1_000_000_000, "a", "c1", 1, out_text), 3
            ),
            _conv_ev("sample.issued", 0, "d", "c2", 1),
            _conv_ev("sample.complete", 2_000_000_000, "d", "c2", 1, out_text),
            _ev("session.stop_performance_tracking", 9_000_000_000),
        ],
    )
    _summary(tmp_path, TokenCountSource.SERVER)
    out = tmp_path / "natl.json"
    assert mod.main([str(tmp_path), "--profile", "agentic", "--json", str(out)]) == 0
    blob = json.loads(out.read_text())
    assert (blob["token_count_source"], blob["n_missing_usage"]) == ("server", 1)
    assert blob["n_trajectories"] == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "token basis: server"
    assert "1 tracked samples returned no server token usage" in captured.err


def _undecodable(line):
    """``line`` with a field of the wrong type: still a COMPLETE line, but one
    the typed decode rejects."""
    rec = json.loads(line)
    rec["finish_reason"] = 5
    return _line(rec)


def _undecodable_events(tmp_path):
    """Four tracked samples with usage; D's COMPLETE does not decode."""
    lines = [_ev("session.start_performance_tracking", 0)]
    for i, uuid in enumerate("ABCD"):
        t = 1000 * (i + 1)
        complete = _with_usage(
            _conv_ev(
                "sample.complete",
                t + 600,
                uuid,
                f"c{i}",
                1,
                ["TextModelOutput", ["a ", "b"]],
            ),
            4,
        )
        lines += [
            _conv_ev("sample.issued", t, uuid, f"c{i}", 1),
            _conv_ev("sample.recv_first", t + 100, uuid, f"c{i}", 1),
            _undecodable(complete) if uuid == "D" else complete,
        ]
    return _write_events(
        tmp_path, [*lines, _ev("session.stop_performance_tracking", 9000)]
    )


@pytest.mark.parametrize("basis", list(TokenCountSource))
def test_replays_count_undecodable_complete_lines(tmp_path, basis):
    path = _undecodable_events(tmp_path)
    count_tokens = None if basis is TokenCountSource.SERVER else _words
    series = mod.build_super_pass_series(path, 4, count_tokens, basis=basis)
    natl = mod.build_trajectory_natl(path, count_tokens, basis=basis)
    assert (series.n_undecodable, natl.n_undecodable) == (1, 1)
    (sp,) = series.super_passes
    assert (len(sp.latency_ns), len(natl.pairs)) == (3, 3)


@pytest.mark.parametrize("profile", ["concurrency", "agentic"])
def test_main_warns_on_undecodable_lines_and_records_them(tmp_path, capsys, profile):
    _undecodable_events(tmp_path)
    _summary(tmp_path, TokenCountSource.SERVER)
    out = tmp_path / "ss.json"
    argv = [str(tmp_path), "--profile", profile, "--superpass-size", "2"]
    argv += ["--warmup", "0", "--no-min-duration", "--json", str(out)]
    assert mod.main(argv) == 0
    assert json.loads(out.read_text())["n_undecodable"] == 1
    err = capsys.readouterr().err
    assert "WARNING: 1 sample.complete lines could not be decoded" in err


def test_main_override_needs_no_archived_basis_note(tmp_path, capsys):
    _missing_usage_events(tmp_path)
    argv = [str(tmp_path), "--superpass-size", "2", "--warmup", "0"]
    argv += ["--no-min-duration", "--token-count-source", "server"]
    assert mod.main(argv) == 0
    err = capsys.readouterr().err
    assert "no token-count basis recorded" not in err
    assert "[token basis: server]" in err
