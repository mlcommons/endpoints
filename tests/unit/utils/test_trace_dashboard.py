# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TDD coverage for the trace dashboard's count + lifecycle logic.

These tests target the symptoms the user has repeatedly hit:
    * "in-flight count stuck / wrong"
    * "N (stage count) stuck"
    * "complete=0 even though events are flowing"

The dashboard logic lives in ``inference_endpoint.utils.trace_dashboard``
so we can drive it with synthetic frame buffers and assert invariants
without spinning up a real benchmark.
"""

# ruff: noqa: I001 — see scripts/trace_dashboard.py for why this file is
# pinned: the pre-commit ruff (v0.3.3) and the local ruff (v0.15.8)
# disagree on `inference_endpoint` first-party detection.
from __future__ import annotations

import struct
import time
import uuid

import pytest
from inference_endpoint.utils.trace import (
    FRAME_SIZE,
    MAIN_PROC_LOOP_ID,
    PACKER,
    Event,
)
from inference_endpoint.utils.trace_dashboard import Dashboard

# All events arrive on a single ascending clock for these tests.
_t = [1000]


def _sid_from_uuid(req_id: str) -> int:
    return int(req_id[:16], 16)


def _frame(event: Event, sid: int, ts: int | None = None) -> bytes:
    if ts is None:
        _t[0] += 1
        ts = _t[0]
    return PACKER.pack(int(event), sid, ts)


def _loop_lag_sid(worker_id: int, lag_ns: int) -> int:
    return ((worker_id & 0xFF) << 56) | (lag_ns & ((1 << 56) - 1))


def _drop_sid(proc_id: int, dropped_bytes: int) -> int:
    return ((proc_id & 0xFF) << 56) | (dropped_bytes & ((1 << 56) - 1))


def _full_lifecycle(sid: int) -> bytes:
    """Offline lifecycle: no RESPONSE_DONE (RESPONSE_BYTES is the full body)."""
    return b"".join(
        _frame(ev, sid)
        for ev in (
            Event.ISSUED,
            Event.WORKER_RECEIVED,
            Event.CONN_ACQUIRED,
            Event.WRITTEN,
            Event.RESPONSE_HEADERS,
            Event.RESPONSE_BYTES,
            Event.MAIN_RECEIVED,
            Event.COMPLETE,
        )
    )


def _inflight(sid: int) -> bytes:
    """Issued + written (payload sent), not yet complete — counts as
    on-the-wire in-flight."""
    return _frame(Event.ISSUED, sid) + _frame(Event.WRITTEN, sid)


def _full_streaming_lifecycle(sid: int) -> bytes:
    """Streaming lifecycle: RESPONSE_BYTES = 1st chunk, RESPONSE_DONE = last."""
    return b"".join(
        _frame(ev, sid)
        for ev in (
            Event.ISSUED,
            Event.WORKER_RECEIVED,
            Event.CONN_ACQUIRED,
            Event.WRITTEN,
            Event.RESPONSE_HEADERS,
            Event.RESPONSE_BYTES,
            Event.RECV_FIRST,
            Event.RESPONSE_DONE,
            Event.MAIN_RECEIVED,
            Event.COMPLETE,
        )
    )


def _new_sid() -> int:
    return _sid_from_uuid(uuid.uuid4().hex)


def _dash() -> Dashboard:
    """Test factory: zero fold defer so finalize_completed() folds
    immediately without needing to advance the wall clock."""
    return Dashboard(fold_defer_ns=0)


# ---------------------------------------------------------------------------
# In-flight counter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInFlightCounter:
    def test_starts_at_zero(self) -> None:
        d = _dash()
        assert d.in_flight == 0
        assert d.n_issued == 0
        assert d.n_complete_seen == 0

    def test_written_increments_in_flight(self) -> None:
        d = _dash()
        sid = _new_sid()
        d.ingest_frames(_inflight(sid))  # issued + written
        assert d.in_flight == 1
        assert d.n_issued == 1
        assert d.n_complete_seen == 0

    def test_issued_only_not_in_flight(self) -> None:
        # Issued but not yet written (IPC backlog) is NOT on-the-wire
        # in-flight — only written-but-not-complete counts.
        d = _dash()
        d.ingest_frames(_frame(Event.ISSUED, _new_sid()))
        assert d.in_flight == 0
        assert d.n_issued == 1

    def test_complete_brings_in_flight_to_zero(self) -> None:
        d = _dash()
        sid = _new_sid()
        d.ingest_frames(_inflight(sid))
        d.ingest_frames(_frame(Event.COMPLETE, sid))
        assert d.in_flight == 0
        assert d.n_complete_seen == 1

    def test_many_written_then_many_complete(self) -> None:
        d = _dash()
        sids = [_new_sid() for _ in range(500)]
        for sid in sids:
            d.ingest_frames(_inflight(sid))
        assert d.in_flight == 500
        for sid in sids:
            d.ingest_frames(_frame(Event.COMPLETE, sid))
        assert d.in_flight == 0
        assert d.n_issued == 500
        assert d.n_complete_seen == 500

    def test_in_flight_never_negative_and_excludes_orphan_complete(self) -> None:
        # A COMPLETE with no preceding ISSUED (warmup bleed: ISSUED
        # cleared at PERF_START) is not counted, so in_flight clamps at
        # zero and never goes negative.
        d = _dash()
        d.ingest_frames(_frame(Event.COMPLETE, _new_sid()))
        assert d.in_flight == 0
        assert d.n_complete_seen == 0  # orphan COMPLETE ignored
        # A written-but-not-complete request is on-the-wire in-flight.
        d.ingest_frames(_inflight(_new_sid()))
        assert d.in_flight == 1

    def test_in_flight_is_constant_time_at_scale(self) -> None:
        # User's complaint: at 40k+ entries the in-flight count visibly
        # lagged because we iterated the dict. After moving to direct
        # counters this should be a plain int subtraction regardless of
        # dict size. Sanity-check the timing.
        d = _dash()
        sids = [_new_sid() for _ in range(50_000)]
        for sid in sids:
            d.ingest_frames(_frame(Event.ISSUED, sid))
        assert d.lifecycle_count() == 50_000
        t0 = time.monotonic_ns()
        for _ in range(1000):
            _ = d.in_flight
        elapsed_us = (time.monotonic_ns() - t0) / 1000
        # 1000 reads should take < 50 ms total (i.e. < 50 µs each) even
        # at 50k lifecycle entries — generous bound that catches any
        # O(N) regression.
        assert (
            elapsed_us < 50_000
        ), f"in_flight read took {elapsed_us:.0f} µs / 1000 calls — O(N) regression"

    def test_in_flight_counts_written_minus_complete_at_scale(self) -> None:
        # 21,578 fully-completed lifecycles + 958,187 written-but-not-
        # complete. On-the-wire in-flight is the latter.
        d = _dash()
        complete_sids = [_new_sid() for _ in range(21_578)]
        in_flight_sids = [_new_sid() for _ in range(958_187)]
        d.ingest_frames(b"".join(_full_lifecycle(s) for s in complete_sids))
        d.ingest_frames(b"".join(_inflight(s) for s in in_flight_sids))
        assert d.n_issued == 979_765
        assert d.n_complete_seen == 21_578
        assert d.in_flight == 958_187

    def test_rates_track_ingest(self) -> None:
        d = _dash()
        # Force a known elapsed window.
        d._start_ns = time.monotonic_ns() - 4_000_000_000  # 4 s ago
        sids = [_new_sid() for _ in range(2000)]
        for s in sids:
            d.ingest_frames(_frame(Event.ISSUED, s))
        # COMPLETE 500 of the issued sids (a COMPLETE only counts when
        # its ISSUED was seen first).
        for s in sids[:500]:
            d.ingest_frames(_frame(Event.COMPLETE, s))
        # 2000/4 = 500 issue/s, 500/4 = 125 complete/s
        assert 400 < d.issuance_rate < 600
        assert 100 < d.completion_rate < 150

    def test_in_flight_is_written_minus_complete(self) -> None:
        # On-the-wire in-flight = WRITTEN − COMPLETE.
        d = _dash()
        sids = [_new_sid() for _ in range(100)]
        d.ingest_frames(b"".join(_inflight(s) for s in sids))
        assert d.in_flight == 100  # all written, none complete
        for s in sids[:60]:
            d.ingest_frames(_frame(Event.COMPLETE, s))
        assert d.in_flight == 40
        assert d.n_complete_seen == 60

    def test_in_flight_clamped_when_written_exceeds_issued(self) -> None:
        # Under FIFO drops, WRITTEN frames (worker-proc) can survive for
        # requests whose ISSUED (main-proc) was dropped → n_written >
        # n_issued. in_flight must never exceed issued − complete.
        d = _dash()
        for s in (_new_sid() for _ in range(50)):
            d.ingest_frames(_frame(Event.ISSUED, s))
        # 80 WRITTEN frames, 30 of them for requests with no ISSUED seen.
        for s in (_new_sid() for _ in range(80)):
            d.ingest_frames(_frame(Event.WRITTEN, s))
        assert d.in_flight == 50  # min(80, 50) − 0, not 80

    def test_in_flight_zero_when_all_completed(self) -> None:
        # End-of-benchmark invariant: once every ISSUED has its
        # COMPLETE, in_flight MUST be 0 — regardless of what stage
        # folding or eviction did along the way.
        d = _dash()
        sids = [_new_sid() for _ in range(10_000)]
        d.ingest_frames(b"".join(_inflight(s) for s in sids))
        assert d.in_flight == 10_000
        # Render a few times mid-flight (simulating the render thread
        # interleaving with ingestion). The fold queue is empty so
        # nothing folds, in-flight stays put.
        for _ in range(5):
            d.finalize_completed()
        assert d.in_flight == 10_000
        # Now everything completes.
        d.ingest_frames(b"".join(_frame(Event.COMPLETE, s) for s in sids))
        assert d.in_flight == 0
        # Further finalize ticks don't change it.
        for _ in range(5):
            d.finalize_completed()
        assert d.in_flight == 0

    def test_orphan_complete_without_issued_not_counted(self) -> None:
        # Warmup bleed: a COMPLETE whose ISSUED was cleared at PERF_START
        # must not inflate the perf-window counters.
        d = _dash()
        d.ingest_frames(_frame(Event.COMPLETE, _new_sid()))
        assert d.n_complete_seen == 0


# ---------------------------------------------------------------------------
# Stage N count via the folding path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStageN:
    def test_no_fold_before_render_tick(self) -> None:
        # Lifecycles sit in the dict until finalize_completed runs.
        d = _dash()
        sid = _new_sid()
        d.ingest_frames(_full_lifecycle(sid))
        assert d.stage_n("backpressure") == 0
        assert d.n_complete_folded == 0

    def test_fold_happens_on_first_finalize_with_zero_defer(self) -> None:
        d = _dash()  # fold_defer_ns=0
        sid = _new_sid()
        d.ingest_frames(_full_lifecycle(sid))
        d.finalize_completed()
        for key in (
            "backpressure",
            "socket_write",
            "server_headers",
            "server_resp",
            "tail_offline",
            "e2e",
        ):
            assert d.stage_n(key) == 1, f"stage {key} did not fold"
        assert d.n_complete_folded == 1

    def test_streaming_lifecycle_folds_split_tail(self) -> None:
        # RESPONSE_DONE present → token-gen (1st→last chunk) and the
        # client tail (last chunk→complete) fold separately; the offline
        # combined tail stays empty.
        d = _dash()
        d.ingest_frames(_full_streaming_lifecycle(_new_sid()))
        d.finalize_completed()
        assert d.stage_n("stream_gen") == 1
        assert d.stage_n("tail_stream") == 1
        assert d.stage_n("server_resp") == 1  # headers -> 1st chunk
        assert d.stage_n("tail_offline") == 1  # also folds, just not shown

    def test_streaming_render_uses_split_labels(self) -> None:
        d = _dash()
        for _ in range(5):
            d.ingest_frames(_full_streaming_lifecycle(_new_sid()))
        text = d.render().plain
        assert "1st chunk -> last chunk" in text
        assert "last chunk -> ipc_2_main -> complete" in text
        assert "headers -> response" not in text  # offline-only label

    def test_offline_render_uses_response_labels(self) -> None:
        d = _dash()
        for _ in range(5):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        text = d.render().plain
        assert "headers recvd -> response" in text
        assert "response -> ipc_2_main -> complete" in text
        assert "1st chunk -> last chunk" not in text  # streaming-only label

    def test_fold_defer_holds_completes_until_deadline(self) -> None:
        # With a non-zero fold defer, a COMPLETE just observed should
        # NOT be folded until the deadline has passed. This is what
        # lets late worker frames catch up before we pop the lifecycle.
        defer_ns = 50_000_000  # 50 ms
        d = Dashboard(fold_defer_ns=defer_ns)
        sid = _new_sid()
        d.ingest_frames(_full_lifecycle(sid))
        d.finalize_completed()
        assert d.n_complete_folded == 0  # not yet
        time.sleep(0.075)  # past the 50 ms defer
        d.finalize_completed()
        assert d.n_complete_folded == 1

    def test_n_grows_monotonically_with_completions(self) -> None:
        d = _dash()
        for _ in range(100):
            sid = _new_sid()
            d.ingest_frames(_full_lifecycle(sid))
        # First render marks all 100 with complete_seen_at; second
        # render folds them all.
        d.finalize_completed()
        d.finalize_completed()
        assert d.stage_n("e2e") == 100
        assert d.n_complete_folded == 100

    def test_partial_lifecycle_does_not_inflate_stage_n(self) -> None:
        # Partial frames in: backpressure (issue -> tcp conn_acquired)
        # folds only when COMPLETE finally lands AND CONN_ACQUIRED was
        # seen. Invariant: N never exceeds n_complete_folded.
        d = _dash()
        sid = _new_sid()
        d.ingest_frames(_frame(Event.ISSUED, sid))
        d.ingest_frames(_frame(Event.WORKER_RECEIVED, sid))
        d.ingest_frames(_frame(Event.CONN_ACQUIRED, sid))
        for _ in range(5):
            d.finalize_completed()
        assert d.stage_n("backpressure") == 0
        assert d.n_complete_folded == 0
        d.ingest_frames(_frame(Event.COMPLETE, sid))
        d.finalize_completed()
        d.finalize_completed()
        assert d.stage_n("backpressure") == 1
        assert d.n_complete_folded == 1

    def test_stage_n_unchanged_by_extra_events_on_already_folded_sid(self) -> None:
        # After fold, the sid is popped. A late event for that sid
        # should create a new lifecycle (and inflate in-flight by 1
        # until it ages out), but must NOT double-count any stage.
        d = _dash()
        sid = _new_sid()
        d.ingest_frames(_full_lifecycle(sid))
        d.finalize_completed()
        d.finalize_completed()
        assert d.stage_n("backpressure") == 1
        # Late stray frame
        d.ingest_frames(_frame(Event.WORKER_RECEIVED, sid))
        d.finalize_completed()
        d.finalize_completed()
        assert d.stage_n("backpressure") == 1


# ---------------------------------------------------------------------------
# Drop counter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDrops:
    def test_no_drops_initially(self) -> None:
        d = _dash()
        assert d.dropped_frames == 0

    def test_trace_drops_sums_across_procs(self) -> None:
        d = _dash()
        # 34 bytes dropped on main, 17 bytes on worker 5
        d.ingest_frames(_frame(Event.TRACE_DROPS, _drop_sid(MAIN_PROC_LOOP_ID, 34)))
        d.ingest_frames(_frame(Event.TRACE_DROPS, _drop_sid(5, 17)))
        assert d.dropped_frames == (34 + 17) // FRAME_SIZE  # 3

    def test_trace_drops_per_proc_is_cumulative_latest(self) -> None:
        # The payload is a per-proc CUMULATIVE total re-sent each tick.
        # Same proc reporting 34 then 510 → latest wins (not summed),
        # so a lost frame self-heals on the next.
        d = _dash()
        d.ingest_frames(_frame(Event.TRACE_DROPS, _drop_sid(5, 34)))
        d.ingest_frames(_frame(Event.TRACE_DROPS, _drop_sid(5, 510)))
        assert d.dropped_frames == 510 // FRAME_SIZE  # 30, not (34+510)
        # A stale/reordered lower value does not regress the count.
        d.ingest_frames(_frame(Event.TRACE_DROPS, _drop_sid(5, 100)))
        assert d.dropped_frames == 510 // FRAME_SIZE

    def test_trace_drops_does_not_create_lifecycle(self) -> None:
        d = _dash()
        d.ingest_frames(_frame(Event.TRACE_DROPS, _drop_sid(MAIN_PROC_LOOP_ID, 34)))
        assert d.lifecycle_count() == 0
        assert d.in_flight == 0


# ---------------------------------------------------------------------------
# LOOP_LAG demux
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoopLag:
    def test_demux_per_worker(self) -> None:
        d = _dash()
        d.ingest_frames(_frame(Event.LOOP_LAG, _loop_lag_sid(0, 1_000_000)))
        d.ingest_frames(_frame(Event.LOOP_LAG, _loop_lag_sid(0, 2_000_000)))
        d.ingest_frames(_frame(Event.LOOP_LAG, _loop_lag_sid(1, 500_000)))
        d.ingest_frames(
            _frame(Event.LOOP_LAG, _loop_lag_sid(MAIN_PROC_LOOP_ID, 3_000_000))
        )
        assert d.loop_lag_n(0) == 2
        assert d.loop_lag_n(1) == 1
        assert d.loop_lag_n(MAIN_PROC_LOOP_ID) == 1

    def test_loop_lag_does_not_create_lifecycle(self) -> None:
        d = _dash()
        d.ingest_frames(_frame(Event.LOOP_LAG, _loop_lag_sid(0, 1_000_000)))
        assert d.lifecycle_count() == 0
        assert d.in_flight == 0
        assert d.n_issued == 0


# ---------------------------------------------------------------------------
# Frame parsing robustness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameParsing:
    def test_ignores_trailing_partial_frame(self) -> None:
        d = _dash()
        sid = _new_sid()
        whole = _frame(Event.ISSUED, sid)
        partial = struct.pack("<BQ", int(Event.COMPLETE), sid)  # only 9 bytes
        d.ingest_frames(whole + partial)
        # Reader gives us only complete-frame multiples; partial bytes
        # at the tail are simply not unpacked.
        assert d.n_issued == 1
        assert d.n_complete_seen == 0

    def test_handles_zero_bytes(self) -> None:
        d = _dash()
        d.ingest_frames(b"")
        assert d.lifecycle_count() == 0
        assert d.in_flight == 0


# ---------------------------------------------------------------------------
# Burst scenarios — mirror the user's offline-burst failure modes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBurst:
    def test_burst_then_completion(self) -> None:
        d = _dash()
        sids = [_new_sid() for _ in range(1000)]
        # Phase 1: all ISSUED arrive in a burst — issued but not written
        # is IPC backlog, not on-the-wire in-flight.
        d.ingest_frames(b"".join(_frame(Event.ISSUED, s) for s in sids))
        assert d.in_flight == 0
        # Render at this point — no folds yet.
        d.finalize_completed()
        assert d.n_complete_folded == 0
        # Phase 2: worker events arrive; WRITTEN puts them on the wire.
        for ev in (
            Event.WORKER_RECEIVED,
            Event.CONN_ACQUIRED,
            Event.WRITTEN,
            Event.RESPONSE_HEADERS,
            Event.RESPONSE_BYTES,
        ):
            d.ingest_frames(b"".join(_frame(ev, s) for s in sids))
        assert d.in_flight == 1000  # written, not yet complete
        # Phase 3: COMPLETE arrives
        d.ingest_frames(b"".join(_frame(Event.MAIN_RECEIVED, s) for s in sids))
        d.ingest_frames(b"".join(_frame(Event.COMPLETE, s) for s in sids))
        assert d.in_flight == 0
        # Now fold (two cycles of defer logic)
        d.finalize_completed()
        d.finalize_completed()
        assert d.n_complete_folded == 1000
        assert d.stage_n("e2e") == 1000
        assert d.stage_n("backpressure") == 1000

    def test_render_header_shows_correct_counts(self) -> None:
        # End-to-end: ingest a known set of events, render once, parse
        # the header from the rich Text, assert the numbers the user
        # actually sees on screen. 30 written-not-complete (in-flight),
        # 70 fully complete → 100 issued.
        d = _dash()
        for _ in range(30):
            d.ingest_frames(_inflight(_new_sid()))
        for _ in range(70):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        text = d.render().plain
        # First header line: complete + req/s
        complete_line = next(line for line in text.splitlines() if "complete" in line)
        assert " 70" in complete_line, complete_line
        # Second header line: issued / in-flight
        inflight_line = next(line for line in text.splitlines() if "in-flight" in line)
        assert " 30" in inflight_line, inflight_line  # in-flight
        assert " 100" in inflight_line, inflight_line  # issued
        assert "queued" not in text
        assert "processing" not in text

    def test_time_line_attributes_queue_wait_correctly(self) -> None:
        # Long IPC backpressure must NOT be billed as client overhead;
        # it shows up in the "backpressure" column.
        d = _dash()
        sid = _new_sid()
        ts = [
            (Event.ISSUED, 0),
            (Event.WORKER_RECEIVED, 20_000_000_000),  # 20 s backpressure
            (Event.CONN_ACQUIRED, 20_000_005_000),
            (Event.WRITTEN, 20_000_010_000),  # 10 us client_pre
            (Event.RESPONSE_HEADERS, 25_000_010_000),  # 5 s server
            (Event.RESPONSE_BYTES, 25_000_010_500),
            (Event.MAIN_RECEIVED, 25_000_011_000),
            (Event.COMPLETE, 25_000_012_000),
        ]
        d.ingest_frames(b"".join(_frame(ev, sid, t) for ev, t in ts))
        d.finalize_completed()
        text = d.render().plain
        time_line = next(line for line in text.splitlines() if "backpressure" in line)
        assert "client work" in time_line
        assert "server work" in time_line
        assert "0.0%" in time_line  # client work
        assert "80.0%" in time_line  # backpressure
        assert "20.0%" in time_line  # server work

    def test_rate_line_shows_issued_completed_backlog(self) -> None:
        d = _dash()
        d._start_ns = time.monotonic_ns() - 10_000_000_000
        for _ in range(70_000):
            d.ingest_frames(_frame(Event.ISSUED, _new_sid()))
        for _ in range(2_000):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        text = d.render().plain
        rate_line = next(line for line in text.splitlines() if "backlog " in line)
        assert "issued " in rate_line
        assert "completed " in rate_line
        assert "+" in rate_line  # positive backlog

    def test_rate_line_suppressed_during_warmup(self) -> None:
        d = _dash()
        for _ in range(50):
            d.ingest_frames(_frame(Event.ISSUED, _new_sid()))
        text = d.render().plain
        assert not any(
            "issued " in line and "backlog" in line for line in text.splitlines()
        )


@pytest.mark.unit
class TestLoadgenComparison:
    """Final-frame comparison: trace-measured vs loadgen-recorded metrics."""

    def _snapshot(
        self,
        *,
        completed: int = 1000,
        tracked: int = 1000,
        tracked_duration_ns: int = 10_000_000_000,
        e2e_p50_ns: float = 100_000_000,
        e2e_p99_ns: float = 250_000_000,
        ttft_p50_ns: float | None = None,
        ttft_p99_ns: float | None = None,
    ) -> dict:
        # Mirror the on-wire shape produced by
        # snapshot_to_dict: counters live in `metrics` with
        # type="counter", not under a top-level dict.
        metrics: list[dict] = [
            {"type": "counter", "name": "total_samples_completed", "value": completed},
            {"type": "counter", "name": "tracked_samples_completed", "value": tracked},
            {
                "type": "counter",
                "name": "tracked_duration_ns",
                "value": tracked_duration_ns,
            },
            {
                "type": "counter",
                "name": "total_duration_ns",
                "value": tracked_duration_ns,
            },
            {
                "type": "series",
                "name": "sample_latency_ns",
                "count": tracked,
                "total": e2e_p50_ns * tracked,
                "min": 0.0,
                "max": e2e_p99_ns,
                "sum_sq": 0.0,
                "percentiles": {"50.0": e2e_p50_ns, "99.0": e2e_p99_ns},
                "histogram": [],
            },
        ]
        if ttft_p50_ns is not None:
            metrics.append(
                {
                    "type": "series",
                    "name": "ttft_ns",
                    "count": tracked,
                    "total": ttft_p50_ns * tracked,
                    "min": 0.0,
                    "max": ttft_p99_ns or ttft_p50_ns,
                    "sum_sq": 0.0,
                    "percentiles": {
                        "50.0": ttft_p50_ns,
                        "99.0": ttft_p99_ns or ttft_p50_ns,
                    },
                    "histogram": [],
                }
            )
        return {
            "counter": 0,  # snapshot frame number (int), not the per-metric counters
            "timestamp_ns": 0,
            "state": "complete",
            "n_pending_tasks": 0,
            "metrics": metrics,
        }

    def test_no_comparison_section_until_attached(self) -> None:
        d = _dash()
        d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        text = d.render().plain
        assert "LOADGEN vs TRACE" not in text

    def test_comparison_section_appears_after_attach(self) -> None:
        d = _dash()
        for _ in range(1000):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        d.attach_loadgen_snapshot(
            self._snapshot(completed=1000, tracked=1000, tracked_duration_ns=10**10)
        )
        text = d.render().plain
        assert "LOADGEN vs TRACE" in text
        # Two-section layout: counts/rates + latency (with src column).
        assert "counts / rates" in text
        assert "loadgen" in text
        assert "trace" in text
        assert "Δ" in text
        assert "throughput (req/s)" in text
        assert "samples completed" in text
        # Latency block exposes min/p50/p99/max + Δmax.
        assert "latency" in text
        assert "Δmax" in text
        # Unit is auto-picked from the observed max so the label suffix
        # varies (ms/µs/s); just confirm the metric name is present.
        assert "e2e (" in text
        for col in ("min", "p50", "p99", "max"):
            assert col in text

    def test_comparison_values_match_when_in_agreement(self) -> None:
        # Snapshot says 1000 completed; drive trace ingest with same
        # counts so Δ on the samples row is ~0%.
        d = _dash()
        d._start_ns = time.monotonic_ns() - 10_000_000_000  # 10 s elapsed
        for _ in range(1000):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        d.attach_loadgen_snapshot(
            self._snapshot(completed=1000, tracked=1000, tracked_duration_ns=10**10)
        )
        text = d.render().plain
        # One row carries the label + both values + delta.
        samples_line = next(
            line for line in text.splitlines() if "samples completed" in line
        )
        # Loadgen and trace each contribute one "1,000" cell on the
        # same row; the third token is the Δ.
        assert samples_line.count("1,000") == 2, samples_line
        assert "+0.0%" in samples_line or "0.0%" in samples_line

    def test_comparison_throughput_falls_back_to_total_duration(self) -> None:
        # Snapshot with tracked_duration_ns = 0 but total_duration_ns
        # populated: throughput must NOT be reported as 0.
        d = _dash()
        for _ in range(100):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        snap = self._snapshot(completed=100, tracked=100, tracked_duration_ns=0)
        # _snapshot() puts the same value in both counters. Force
        # tracked_duration_ns back to 0 while keeping total_duration_ns
        # non-zero to mimic the real-world "tracked block not yet
        # closed" case.
        for m in snap["metrics"]:
            if m.get("name") == "tracked_duration_ns":
                m["value"] = 0
            elif m.get("name") == "total_duration_ns":
                m["value"] = 10_000_000_000  # 10 s
        d.attach_loadgen_snapshot(snap)
        text = d.render().plain
        tput_line = next(
            line for line in text.splitlines() if "throughput (req/s)" in line
        )
        # 100 samples / 10 s = 10 req/s. Must not be 0.0 and the
        # loadgen cell must not be an em-dash.
        assert "10.0" in tput_line, tput_line
        assert "—" not in tput_line.split("throughput (req/s)")[1].split()[0], tput_line

    def test_comparison_shows_tpot_and_tps_loadgen_only(self) -> None:
        # tpot + tps come from loadgen only; the trace cell stays
        # em-dashed because we don't measure per-token timings.
        d = _dash()
        for _ in range(100):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        snap = self._snapshot(
            completed=100, tracked=100, tracked_duration_ns=10_000_000_000
        )
        # Add tpot and osl series so tps + tpot rows render.
        snap["metrics"].extend(
            [
                {
                    "type": "series",
                    "name": "tpot_ns",
                    "count": 100,
                    "total": 100.0,
                    "min": 0.0,
                    "max": 100.0,
                    "sum_sq": 0.0,
                    "percentiles": {"50.0": 5_000_000, "99.0": 12_000_000},  # 5 / 12 ms
                    "histogram": [],
                },
                {
                    "type": "series",
                    "name": "osl",
                    "count": 100,
                    "total": 50_000.0,  # 500 tokens × 100 samples
                    "min": 0.0,
                    "max": 1000.0,
                    "sum_sq": 0.0,
                    "percentiles": {"50.0": 500.0, "99.0": 1000.0},
                    "histogram": [],
                },
            ]
        )
        d.attach_loadgen_snapshot(snap)
        text = d.render().plain
        lines = text.splitlines()
        # tpot block: loadgen row carries p50 = 5 ms; trace row is em-dash
        # all the way across (no per-token timings emitted as trace events).
        assert "tpot (ms)" in text
        tpot_block = [i for i, line in enumerate(lines) if "tpot (ms)" in line]
        assert tpot_block, "expected a tpot (ms) row"
        loadgen_line = lines[tpot_block[0]]
        trace_line = lines[tpot_block[0] + 1]
        assert "loadgen" in loadgen_line
        assert "5.00" in loadgen_line  # p50 = 5 ms
        assert "trace" in trace_line
        assert "—" in trace_line, trace_line

        tps_line = next(line for line in lines if "tok/s" in line)
        # 50,000 tokens / 10 s = 5,000 tok/s
        assert "5,000.0" in tps_line, tps_line

    def test_comparison_flags_drift(self) -> None:
        # Loadgen says 1000 completed but trace only saw 500. Δ on
        # the samples row should be -50%.
        d = _dash()
        for _ in range(500):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        d.attach_loadgen_snapshot(
            self._snapshot(completed=1000, tracked=1000, tracked_duration_ns=10**10)
        )
        text = d.render().plain
        samples_line = next(
            line for line in text.splitlines() if "samples completed" in line
        )
        assert "1,000" in samples_line  # loadgen value
        assert "500" in samples_line  # trace value
        assert "-50" in samples_line or "−50" in samples_line, samples_line

    def test_comparison_skips_rows_with_no_data(self) -> None:
        # Streaming-off offline run: snapshot has no ttft / tpot / osl
        # series. The ttft / tpot / tok-throughput rows must not render.
        d = _dash()
        for _ in range(100):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        d.attach_loadgen_snapshot(
            self._snapshot(completed=100, tracked=100, tracked_duration_ns=10**10)
        )
        text = d.render().plain
        assert "ttft (ms)" not in text
        assert "tpot (ms)" not in text
        assert "throughput (tok/s)" not in text

    def test_render_header_after_drops(self) -> None:
        d = _dash()
        d.ingest_frames(_frame(Event.ISSUED, _new_sid()))
        # 17 bytes = 1 frame dropped
        d.ingest_frames(
            _frame(Event.TRACE_DROPS, _drop_sid(MAIN_PROC_LOOP_ID, FRAME_SIZE))
        )
        text = d.render().plain
        assert "dropped" in text
        dropped_line = next(line for line in text.splitlines() if "dropped" in line)
        assert "1" in dropped_line

    def test_interleaved_burst(self) -> None:
        # Realistic high-QPS pattern: ISSUED for new sids and COMPLETE
        # for older sids interleaved. in-flight tracks the running
        # difference accurately.
        d = _dash()
        live: list[int] = []
        for _ in range(100):
            new = [_new_sid() for _ in range(10)]
            live.extend(new)
            d.ingest_frames(b"".join(_full_lifecycle(s) for s in new[:5]))
            d.ingest_frames(b"".join(_inflight(s) for s in new[5:]))
            d.finalize_completed()
            d.finalize_completed()
        # Five out of every ten are fully resolved (folded); the other
        # five are written-but-not-complete and stay in-flight.
        assert d.n_issued == 1000
        assert d.n_complete_seen == 500
        assert d.in_flight == 500
        assert d.n_complete_folded == 500
        assert d.stage_n("e2e") == 500

    def test_complete_arriving_after_eviction_still_folds(self) -> None:
        # USER'S BUG (real pattern): in offline burst the loadgen
        # issues millions of requests in a tight loop while responses
        # trickle in over many seconds. The render cycle fires while
        # ingestion is still going — the lifecycle dict grows past
        # MAX_INFLIGHT, eviction pops the still-in-flight ISSUED
        # entries, *then* their COMPLETE eventually arrives, finds no
        # ISSUED in stages, and the fold gate rejects them.
        # Result: complete=N, stage N=tiny.
        d = _dash()
        # 1) Flood the dict with ISSUED-only entries until eviction
        #    kicks in. This mirrors the loadgen burst racing ahead of
        #    the server.
        burst = [_new_sid() for _ in range(250_000)]
        d.ingest_frames(b"".join(_frame(Event.ISSUED, s) for s in burst))
        # 2) Render — the old code would evict 150k partials here.
        d.finalize_completed()
        # 3) Now those same requests complete. With the broken
        #    behaviour, the lifecycle has no ISSUED, gets rejected,
        #    and stage N never grows.
        for sid in burst[:500]:
            d.ingest_frames(
                b"".join(
                    _frame(ev, sid)
                    for ev in (
                        Event.WORKER_RECEIVED,
                        Event.CONN_ACQUIRED,
                        Event.WRITTEN,
                        Event.RESPONSE_HEADERS,
                        Event.RESPONSE_BYTES,
                        Event.MAIN_RECEIVED,
                        Event.COMPLETE,
                    )
                )
            )
        d.finalize_completed()
        d.finalize_completed()
        assert d.n_complete_seen == 500
        assert (
            d.n_complete_folded == 500
        ), f"folded {d.n_complete_folded}/500 — eviction lost ISSUED context"
        assert d.stage_n("e2e") == 500
        assert d.stage_n("backpressure") == 500

    def test_huge_in_flight_does_not_starve_folds(self) -> None:
        # USER'S BUG: at 980k in-flight + 21k complete, stage N is stuck
        # at 2.5k. Reason was MAX_INFLIGHT-triggered eviction that ran
        # *before* each request's COMPLETE arrived — so the ISSUED was
        # popped from the dict, then COMPLETE arrived for a sid with
        # no ISSUED in stages, missed the fold gate, and the request
        # never made it into the stage histograms.
        #
        # Invariant: with N issued + M complete (M ≤ N), after
        # ingesting + rendering, stage N must reach M (not get
        # throttled by dict-size eviction).
        d = _dash()
        completed_sids = [_new_sid() for _ in range(500)]
        outstanding_sids = [_new_sid() for _ in range(200_000)]
        # Interleave issuance with completions to simulate the user's
        # 980k-in-flight + 21k-complete scenario in miniature.
        # Phase 1: issue all sids
        d.ingest_frames(b"".join(_frame(Event.ISSUED, s) for s in completed_sids))
        d.ingest_frames(b"".join(_frame(Event.ISSUED, s) for s in outstanding_sids))
        # Phase 2: complete the first batch
        for sid in completed_sids:
            d.ingest_frames(
                b"".join(
                    _frame(ev, sid)
                    for ev in (
                        Event.WORKER_RECEIVED,
                        Event.CONN_ACQUIRED,
                        Event.WRITTEN,
                        Event.RESPONSE_HEADERS,
                        Event.RESPONSE_BYTES,
                        Event.MAIN_RECEIVED,
                        Event.COMPLETE,
                    )
                )
            )
        # Two render ticks should fold every completed lifecycle, no
        # matter how many partial lifecycles are sitting alongside.
        d.finalize_completed()
        d.finalize_completed()
        assert d.n_complete_seen == 500
        assert (
            d.n_complete_folded == 500
        ), f"folded only {d.n_complete_folded}/500 — eviction starved folds"
        assert d.stage_n("e2e") == 500
        assert d.stage_n("backpressure") == 500


@pytest.mark.unit
class TestTailIndicator:
    """`is_tail` flips when ISSUED stops arriving and in_flight > 0."""

    def test_false_before_any_issued(self) -> None:
        d = _dash()
        assert d.is_tail is False

    def test_false_while_issuance_active(self) -> None:
        d = _dash()
        d.ingest_frames(_frame(Event.ISSUED, _new_sid()))
        # Same monotonic clock → quiet window is 0; not yet "tail".
        assert d.is_tail is False

    def test_true_once_quiet_window_elapses(self) -> None:
        from inference_endpoint.utils.trace_dashboard import _TAIL_QUIET_NS

        d = _dash()
        sid = _new_sid()
        d.ingest_frames(_inflight(sid))
        d._last_issued_ns = time.monotonic_ns() - _TAIL_QUIET_NS - 1
        assert d.in_flight == 1
        assert d.is_tail is True

    def test_done_once_activity_ceases(self) -> None:
        # Activity-based: issuance quiet AND no ISSUED/COMPLETE for
        # _DONE_QUIET_NS → is_done (and is_tail clears). Independent of
        # in-flight, so it fires even when COMPLETE frames were dropped.
        from inference_endpoint.utils.trace_dashboard import (
            _DONE_QUIET_NS,
            _TAIL_QUIET_NS,
        )

        d = _dash()
        d.ingest_frames(_inflight(_new_sid()))
        now = time.monotonic_ns()
        # Issuance quiet but a completion arrived recently → still draining.
        d._last_issued_ns = now - _TAIL_QUIET_NS - 1
        assert d.is_tail is True
        assert d.is_done is False
        # No lifecycle event for the done window → run has stopped → done.
        d._last_lifecycle_ns = now - _DONE_QUIET_NS - 1
        assert d.is_done is True
        assert d.is_tail is False

    def test_header_shows_tail_chip(self) -> None:
        from inference_endpoint.utils.trace_dashboard import _TAIL_QUIET_NS

        d = _dash()
        d.ingest_frames(_inflight(_new_sid()))
        d._last_issued_ns = time.monotonic_ns() - _TAIL_QUIET_NS - 1
        text = d.render().plain
        assert "TAIL" in text
        assert "draining" in text


@pytest.mark.unit
class TestBackpressure:
    """``is_backpressured`` flips when the first stage (ISSUED →
    CONN_ACQUIRED) takes ≥ _BACKPRESSURE_PCT of E2E."""

    def _ingest(self, d: Dashboard, e2e_ns: int, first_stage_ns: int) -> None:
        """Synthesise lifecycles with a given issue→conn_acquired gap."""
        for _ in range(200):
            sid = _new_sid()
            issued_ts = 1
            conn_ts = issued_ts + first_stage_ns
            for ev, ts in (
                (Event.ISSUED, issued_ts),
                (Event.WORKER_RECEIVED, issued_ts + first_stage_ns // 2),
                (Event.CONN_ACQUIRED, conn_ts),
                (Event.WRITTEN, conn_ts + 1),
                (Event.RESPONSE_HEADERS, conn_ts + 2),
                (Event.RESPONSE_BYTES, conn_ts + 3),
                (Event.MAIN_RECEIVED, conn_ts + 4),
                (Event.COMPLETE, issued_ts + e2e_ns),
            ):
                d.ingest_frames(_frame(ev, sid, ts))
        d.finalize_completed()

    def test_false_when_first_stage_below_threshold(self) -> None:
        d = _dash()
        # First stage = 2% of e2e — below 20%.
        self._ingest(d, e2e_ns=1_000_000, first_stage_ns=20_000)
        assert d.is_backpressured is False

    def test_true_when_first_stage_heavy(self) -> None:
        d = _dash()
        # First stage = 50% of e2e.
        self._ingest(d, e2e_ns=1_000_000, first_stage_ns=500_000)
        assert d.is_backpressured is True

    def test_true_survives_dropped_intermediate_frames(self) -> None:
        # First stage folds from ISSUED + CONN_ACQUIRED endpoints only,
        # so it triggers even when WORKER_RECEIVED frames are lost.
        d = _dash()
        for _ in range(50):
            sid = _new_sid()
            d.ingest_frames(_frame(Event.ISSUED, sid, 1))
            d.ingest_frames(_frame(Event.CONN_ACQUIRED, sid, 600_000))
            d.ingest_frames(_frame(Event.COMPLETE, sid, 1_000_000))
        d.finalize_completed()
        assert d.is_backpressured is True

    def test_header_chip_shows_backpressure(self) -> None:
        d = _dash()
        self._ingest(d, e2e_ns=1_000_000, first_stage_ns=500_000)
        text = d.render().plain
        assert "BACKPRESSURE" in text
        assert "(tcp)" not in text and "(worker)" not in text


@pytest.mark.unit
class TestPerfStartReset:
    """PERF_START drops warmup state so LOADGEN vs TRACE aligns with
    loadgen's tracked window."""

    def test_metrics_and_counters_reset_on_perf_start(self) -> None:
        d = _dash()
        for _ in range(50):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        assert d.n_issued == 50
        assert d.n_complete_seen == 50
        assert d.stage_n("e2e") == 50

        # Phase boundary marker. sid=0; ts irrelevant.
        d.ingest_frames(_frame(Event.PERF_START, 0))
        assert d.n_issued == 0
        assert d.n_complete_seen == 0
        assert d.stage_n("e2e") == 0
        assert d.in_flight == 0

    def test_loop_lag_survives_reset(self) -> None:
        # Loop lag is per-worker process health, not per-request — it
        # must not get wiped by a phase boundary.
        d = _dash()
        d.ingest_frames(_frame(Event.LOOP_LAG, _loop_lag_sid(0, 1_000_000)))
        d.ingest_frames(_frame(Event.PERF_START, 0))
        assert 0 in d._loop_lag
        assert d._loop_lag[0].total == 1

    def test_warmup_request_completing_after_perf_start_not_counted(self) -> None:
        # A warmup request: ISSUED before PERF_START (cleared by reset),
        # COMPLETE after. Its COMPLETE must not bleed into the perf
        # window's counters or fold into the stage histograms.
        d = _dash()
        warmup_sid = _new_sid()
        d.ingest_frames(_frame(Event.ISSUED, warmup_sid))
        d.ingest_frames(_frame(Event.PERF_START, 0))  # clears the ISSUED
        # Perf-phase request, fully traced.
        perf_sid = _new_sid()
        d.ingest_frames(_full_lifecycle(perf_sid))
        # Warmup request's late COMPLETE lands now.
        d.ingest_frames(_frame(Event.COMPLETE, warmup_sid))
        d.finalize_completed()
        assert d.n_issued == 1  # only the perf request
        assert d.n_complete_seen == 1  # warmup COMPLETE excluded
        assert d.stage_n("e2e") == 1
        assert d.in_flight == 0


@pytest.mark.unit
class TestErrorsCounter:
    """``errors`` row in LOADGEN vs TRACE picks up tracked_samples_failed."""

    def test_errors_row_renders_failure_count(self) -> None:
        d = _dash()
        d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        d.attach_loadgen_snapshot(
            {
                "counter": 0,
                "timestamp_ns": 0,
                "state": "complete",
                "n_pending_tasks": 0,
                "metrics": [
                    {
                        "type": "counter",
                        "name": "total_samples_completed",
                        "value": 100,
                    },
                    {
                        "type": "counter",
                        "name": "tracked_samples_completed",
                        "value": 100,
                    },
                    {
                        "type": "counter",
                        "name": "tracked_samples_failed",
                        "value": 7,
                    },
                    {
                        "type": "counter",
                        "name": "tracked_duration_ns",
                        "value": 10**10,
                    },
                    {
                        "type": "counter",
                        "name": "total_duration_ns",
                        "value": 10**10,
                    },
                ],
            }
        )
        text = d.render().plain
        errors_line = next(line for line in text.splitlines() if "errors" in line)
        assert "7" in errors_line

    def test_errors_row_skipped_when_zero(self) -> None:
        d = _dash()
        d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        d.attach_loadgen_snapshot(
            {
                "counter": 0,
                "timestamp_ns": 0,
                "state": "complete",
                "n_pending_tasks": 0,
                "metrics": [
                    {
                        "type": "counter",
                        "name": "tracked_samples_completed",
                        "value": 100,
                    },
                    {
                        "type": "counter",
                        "name": "tracked_duration_ns",
                        "value": 10**10,
                    },
                ],
            }
        )
        text = d.render().plain
        for line in text.splitlines():
            assert "errors" not in line, line
