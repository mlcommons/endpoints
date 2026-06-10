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
from inference_endpoint.utils.trace_dashboard import Dashboard, _heat
from rich.text import Text

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

    def test_flush_pending_folds_refreshes_frozen_stats(self) -> None:
        # Regression: render() freezes stage stats when is_done first fires,
        # but that fires inside the fold-defer window — a COMPLETE seen just
        # before PERF_END is still queued and excluded from the freeze. The
        # end-of-run flush_pending_folds() folds it; without invalidating the
        # freeze the closing frame would keep the stale (N=0) frozen copy and
        # disagree with the LOADGEN panel.
        d = Dashboard(fold_defer_ns=50_000_000)  # 50 ms defer
        sid = _new_sid()
        d.ingest_frames(_full_lifecycle(sid))  # COMPLETE just observed → deferred
        d.ingest_frames(_frame(Event.PERF_END, 0))  # is_done
        d.render()  # freezes before the deferred COMPLETE is folded
        assert d._frozen_stats is not None
        assert d._frozen_stats["e2e"].n == 0  # deferred completion excluded
        d.flush_pending_folds()  # folds it AND invalidates the freeze
        assert d._frozen_stats is None
        d.render()  # re-freezes with the folded completion
        assert d._frozen_stats is not None
        assert d._frozen_stats["e2e"].n == 1  # now in the closing frame

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
        # End-to-end: ingest a known set of events, render once, parse the
        # header. With no loadgen snapshot the header sources `issued` from
        # the trace (30 in-flight + 70 complete = 100); request counts
        # otherwise live in the LOADGEN panel, not the header.
        d = _dash()
        for _ in range(30):
            d.ingest_frames(_inflight(_new_sid()))
        for _ in range(70):
            d.ingest_frames(_full_lifecycle(_new_sid()))
        text = d.render().plain
        lines = text.splitlines()
        issued_line = next(
            line for line in lines if "issued" in line and "issued/s" not in line
        )
        assert " 100" in issued_line, issued_line
        # Health chip occupies the middle column; run is active (not
        # quiesced/finished) here, so it is neither TAIL nor DONE.
        assert "status" in text
        assert "TAIL" not in text and "DONE" not in text
        # in-flight / completed are no longer header labels.
        assert "in-flight" not in text
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

    def test_no_loadgen_section_until_attached(self) -> None:
        d = _dash()
        d.ingest_frames(_full_lifecycle(_new_sid()))
        d.finalize_completed()
        assert "LOADGEN" not in d.render().plain

    def test_loadgen_section_is_loadgen_only(self) -> None:
        # Panel shows fresh loadgen stats only — no trace column, no Δ.
        d = _dash()
        d.attach_loadgen_snapshot(
            self._snapshot(completed=1000, tracked=1000, tracked_duration_ns=10**10)
        )
        text = d.render().plain
        assert "LOADGEN" in text
        assert "vs TRACE" not in text
        assert "Δ" not in text
        assert "completed" in text and "completed/s" in text
        assert "1,000" in text  # loadgen completed
        assert "latency (ms)" in text and "e2e" in text
        for col in ("min", "p50", "p99", "max"):
            assert col in text

    @pytest.mark.parametrize(
        ("tracked_dur", "total_dur", "expected"),
        [
            (10**10, 10**10, "100.0"),  # tracked window: 1000 / 10 s
            (0, 10**10, "100.0"),  # tracked=0 mid-run → fall back to total
        ],
    )
    def test_completed_per_s_uses_tracked_then_total_duration(
        self, tracked_dur: int, total_dur: int, expected: str
    ) -> None:
        d = _dash()
        snap = self._snapshot(
            completed=1000, tracked=1000, tracked_duration_ns=tracked_dur
        )
        for m in snap["metrics"]:
            if m.get("name") == "total_duration_ns":
                m["value"] = total_dur
        d.attach_loadgen_snapshot(snap)
        rate_line = next(
            line for line in d.render().plain.splitlines() if "completed/s" in line
        )
        assert expected in rate_line, rate_line

    def test_loadgen_tpot_and_tps(self) -> None:
        d = _dash()
        snap = self._snapshot(
            completed=100, tracked=100, tracked_duration_ns=10_000_000_000
        )
        snap["metrics"].extend(
            [
                {
                    "type": "series",
                    "name": "tpot_ns",
                    "count": 100,
                    "total": 100.0,
                    "min": 0.0,
                    "max": 12_000_000,
                    "sum_sq": 0.0,
                    "percentiles": {"50.0": 5_000_000, "99.0": 12_000_000},
                    "histogram": [],
                },
                {
                    "type": "series",
                    "name": "osl",
                    "count": 100,
                    "total": 50_000.0,
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
        assert "tpot" in text
        tpot_line = next(
            line for line in text.splitlines() if line.strip().startswith("tpot")
        )
        assert "5.00" in tpot_line  # p50 = 5 ms
        tps_line = next(line for line in text.splitlines() if "tok/s" in line)
        assert "5,000.0" in tps_line, tps_line  # 50,000 tok / 10 s

    def test_loadgen_shows_all_latency_rows_even_empty(self) -> None:
        # Offline (streaming-off) snapshot has no ttft / tpot series. They
        # still render as — placeholders so absence reads as "no data",
        # not "the panel is broken".
        d = _dash()
        d.attach_loadgen_snapshot(
            self._snapshot(completed=100, tracked=100, tracked_duration_ns=10**10)
        )
        text = d.render().plain
        assert "ttft" in text and "tpot" in text and "e2e" in text
        assert "—" in text  # placeholder for the empty ttft / tpot series
        assert "tok/s" in text

    def test_render_header_after_drops(self) -> None:
        d = _dash()
        d.ingest_frames(_frame(Event.ISSUED, _new_sid()))
        # Report 42 frames' worth of dropped bytes (a value that can't be
        # confused with the issued/uptime fields on the same row).
        d.ingest_frames(
            _frame(Event.TRACE_DROPS, _drop_sid(MAIN_PROC_LOOP_ID, 42 * FRAME_SIZE))
        )
        assert d.dropped_frames == 42
        dropped_line = next(
            line for line in d.render().plain.splitlines() if "dropped frames" in line
        )
        # Value sits right after the "dropped frames" label in its column.
        assert "42" in dropped_line.split("dropped frames", 1)[1]

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

    def test_perf_end_freezes_deterministically(self) -> None:
        # PERF_END (main proc, run over) is the authoritative freeze — it
        # fires regardless of frame drops or a stale loadgen snapshot.
        from inference_endpoint.utils.trace_dashboard import _TAIL_QUIET_NS

        d = _dash()
        d.ingest_frames(_inflight(_new_sid()))
        d._last_issued_ns = time.monotonic_ns() - _TAIL_QUIET_NS - 1
        assert d.is_tail is True
        assert d.is_done is False
        d.ingest_frames(_frame(Event.PERF_END, 0))
        assert d.is_done is True
        assert d.is_tail is False

    def test_loadgen_terminal_state_is_done_backup(self) -> None:
        # Backup to PERF_END: a non-terminal loadgen snapshot is NOT done,
        # a terminal one is. (Guards against false-DONE when PERF_END is
        # lost but the aggregator's terminal frame arrives.)
        from inference_endpoint.utils.trace_dashboard import _TAIL_QUIET_NS

        d = _dash()
        d.ingest_frames(_inflight(_new_sid()))
        d._last_issued_ns = time.monotonic_ns() - _TAIL_QUIET_NS - 1
        d.attach_loadgen_snapshot({"state": "draining", "metrics": []})
        assert d.is_done is False  # loadgen says still running
        assert d.is_tail is True
        d.attach_loadgen_snapshot({"state": "complete", "metrics": []})
        assert d.is_done is True
        assert d.is_tail is False

    def test_header_shows_tail_chip(self) -> None:
        from inference_endpoint.utils.trace_dashboard import _TAIL_QUIET_NS

        d = _dash()
        d.ingest_frames(_inflight(_new_sid()))
        d._last_issued_ns = time.monotonic_ns() - _TAIL_QUIET_NS - 1
        text = d.render().plain
        # TAIL surfaces as the status chip in the header middle column.
        assert "TAIL" in text


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
    """The LOADGEN counts row always carries an ``errors`` column sourced from
    tracked_samples_failed — rendering 0 when clean so the column is anchored."""

    @pytest.mark.parametrize("failed", [0, 7])
    def test_errors_column_reflects_failure_count(self, failed: int) -> None:
        d = _dash()
        metrics = [
            {"type": "counter", "name": "tracked_samples_completed", "value": 100},
            {"type": "counter", "name": "tracked_duration_ns", "value": 10**10},
        ]
        if failed:
            metrics.append(
                {"type": "counter", "name": "tracked_samples_failed", "value": failed}
            )
        d.attach_loadgen_snapshot(
            {
                "counter": 0,
                "timestamp_ns": 0,
                "state": "complete",
                "n_pending_tasks": 0,
                "metrics": metrics,
            }
        )
        errors_line = next(
            line for line in d.render().plain.splitlines() if "errors" in line
        )
        # Isolate the errors column: the value sits after its label.
        assert str(failed) in errors_line.split("errors", 1)[1]


@pytest.mark.unit
class TestEndOfRunFreshness:
    """The closing frame must show the authoritative terminal snapshot, never
    a stale mid-run one — regression: the dashboard froze on PERF_END while
    the LOADGEN panel still displayed a 20 s-old live snapshot."""

    def _snap(self, state: str, completed: int, dur_ns: int = 10**10) -> dict:
        return {
            "counter": 0,
            "timestamp_ns": 0,
            "state": state,
            "n_pending_tasks": 0,
            "metrics": [
                {"type": "counter", "name": "total_samples_issued", "value": 1000},
                {
                    "type": "counter",
                    "name": "total_samples_completed",
                    "value": completed,
                },
                {
                    "type": "counter",
                    "name": "tracked_samples_completed",
                    "value": completed,
                },
                {"type": "counter", "name": "tracked_duration_ns", "value": dur_ns},
            ],
        }

    @pytest.mark.parametrize("state", ["complete", "interrupted"])
    def test_terminal_snapshot_is_fresh_and_latched_without_force(
        self, state: str
    ) -> None:
        # The SUB path attaches without force=True; either terminal state must
        # register as terminal+done and tag "(final)", never stale. The latch
        # then refuses a later live frame from downgrading it.
        d = _dash()
        d.attach_loadgen_snapshot(self._snap(state, 1000))
        assert d.has_terminal_loadgen is True
        assert d.is_done is True
        completed_line = next(
            ln for ln in d.render().plain.splitlines() if "completed" in ln
        )
        assert "(final)" in d.render().plain
        assert "old)" not in d.render().plain and "unavailable" not in d.render().plain
        assert "1,000" in completed_line  # terminal completed count
        # A late live frame must NOT revert the authoritative terminal data:
        # the latch keeps both the tag AND the terminal count (1,000, not 1).
        d.attach_loadgen_snapshot(self._snap("live", 1))
        assert d.has_terminal_loadgen is True
        after = d.render().plain
        assert "(final)" in after
        assert "1,000" in next(ln for ln in after.splitlines() if "completed" in ln)

    def test_done_but_no_terminal_finalizes_then_flags_unavailable(self) -> None:
        # PERF_END fired (is_done) but the terminal snapshot has not arrived:
        # while the aggregator is still computing final stats this reads as
        # "finalizing…", and flips to an explicit failure only once the CLI
        # gives up waiting (mark_final_unavailable).
        d = _dash()
        d.attach_loadgen_snapshot(self._snap("live", 600), force=True)
        d._loadgen_snapshot_ts -= 20_000_000_000  # pretend 20 s old
        d.ingest_frames(_frame(Event.PERF_END, 0))  # run over → is_done
        assert d.is_done is True
        assert d.has_terminal_loadgen is False
        loadgen_line = next(
            line for line in d.render().plain.splitlines() if "LOADGEN" in line
        )
        assert "finalizing" in loadgen_line
        assert "unavailable" not in loadgen_line
        d.mark_final_unavailable()
        loadgen_line = next(
            line for line in d.render().plain.splitlines() if "LOADGEN" in line
        )
        assert "final snapshot unavailable" in loadgen_line

    def test_live_stale_snapshot_shows_age_when_not_done(self) -> None:
        # Run still live but the feed has lagged: show the plain age, not the
        # done-only "final unavailable" wording.
        d = _dash()
        d.attach_loadgen_snapshot(self._snap("live", 500), force=True)
        d._loadgen_snapshot_ts -= 9_000_000_000  # 9 s old, still running
        loadgen_line = next(
            line for line in d.render().plain.splitlines() if "LOADGEN" in line
        )
        assert "snapshot 9s old" in loadgen_line
        assert "unavailable" not in loadgen_line


@pytest.mark.unit
class TestBackpressureCause:
    """The cause tree (one leaf per worker-loop phase) hangs in the verdict,
    dropping straight down from the backpressure % value. No separate root
    line — the verdict's '[workers busy]' tag is the root."""

    def test_cause_tree_lines_name_all_phases(self) -> None:
        d = _dash()
        d._metrics["e2e"].add(100)
        d._metrics["backpressure"].add(50)  # 50% of E2E → is_backpressured
        lines = d._backpressure_cause_lines()
        assert len(lines) == len(d._BACKPRESSURE_PHASES)  # leaves only, no root
        for leaf, phase in zip(lines, d._BACKPRESSURE_PHASES, strict=False):
            assert leaf[0][0] == phase  # phase label
            assert leaf[1][0] in (" ─┤", " ─┘")  # right-hand spine connector

    def test_leaf_color_matches_its_stage_e2e_cell(self) -> None:
        d = _dash()
        d._metrics["e2e"].add(100)
        d._metrics["backpressure"].add(30)  # issue→conn 30% ≥ 25% → critical
        d._metrics["stream_gen"].add(5)  # 5% < 15% → server side colour
        leaf = {ln[0][0]: ln[0][1] for ln in d._backpressure_cause_lines()}
        # encode/tcp-acquire follow the backpressure stage's %E2E colour
        assert leaf["tcp-acquire"] == "critical"
        assert leaf["encode"] == "critical"
        # sse-decode follows stream_gen's heat (5% below heat → muted base)
        assert leaf["sse-decode"] == "muted"

    def test_heat_scale_boundaries(self) -> None:
        # Shared severity scale (pct 0-100): warn ≥15%, critical ≥25%.
        assert _heat(14.9) == ""
        assert _heat(15.0) == "warn"
        assert _heat(24.9) == "warn"
        assert _heat(25.0) == "critical"

    def test_verdict_row_has_uncolored_workers_busy_tag(self) -> None:
        d = _dash()
        d._metrics["e2e"].add(100)
        d._metrics["backpressure"].add(50)
        out = Text()
        d._render_verdict(out, 100.0)
        assert len(out.plain.splitlines()) == 1  # just the row; tree is beside the bar
        line = out.plain.splitlines()[0]
        assert "backpressure [workers busy]" in line  # tag in the label, left of value
        assert "worker loop busy" not in out.plain  # no separate root line
        # The tag is dim label text, never heat-colored.
        tag_idx = line.index("[workers busy]")
        tag_styles = [sp.style for sp in out.spans if sp.start <= tag_idx < sp.end]
        assert "warn" not in tag_styles and "critical" not in tag_styles

    def test_tree_drops_from_value_beside_bar(self) -> None:
        d = _dash()
        d._metrics["e2e"].add(100)
        d._metrics["backpressure"].add(50)
        out = Text()
        d._render_timeline(out, [("server", 100.0)])
        plain = out.plain
        assert "encode" in plain and "complete-ipc" in plain  # tree beside the bar
        # Every leaf's spine right-aligns to the backpressure value's column, so
        # the tree drops straight down from the % in the verdict above.
        verdict_right = 2 + 3 * 40 + 2 * 3
        spine_lines = [ln for ln in plain.splitlines() if "─┤" in ln or "─┘" in ln]
        assert len(spine_lines) == len(d._BACKPRESSURE_PHASES)
        assert all(len(ln) == verdict_right for ln in spine_lines)

    def test_backpressure_segment_uses_key_color_never_red(self) -> None:
        # The e2e bar uses ONLY the legend-key colors; severity/red lives in the
        # verdict + cause tree, never in the bar.
        d = _dash()
        out = Text()
        d._render_timeline(out, [("backpressure", 60.0), ("server", 40.0)])
        bp_styles = [
            sp.style for sp in out.spans if "▓" in out.plain[sp.start : sp.end]
        ]
        assert bp_styles and all(st == "warn" for st in bp_styles)
        assert "critical" not in bp_styles


@pytest.mark.unit
class TestEventLoopLagRender:
    """The EVENT LOOP LAG panel: fleet p99 + hot-worker count, main pinned
    first, workers sorted by max lag desc, and a top-N cap with an overflow
    note."""

    def test_empty_then_fleet_summary_sorted_with_overflow(self) -> None:
        d = _dash()
        # Precondition: nothing to show until a LOOP_LAG frame arrives.
        assert "(no LOOP_LAG events yet)" in d.render().plain

        # main + 18 workers with DISTINCT max lags so the sort order and which
        # workers get omitted are unambiguous. w0/w1 are hot (≥ 5 ms); w2..w17
        # descend from 3.2 ms to 0.2 ms.
        worker_lag = {0: 9_000_000, 1: 8_000_000}
        worker_lag.update({wid: (18 - wid) * 200_000 for wid in range(2, 18)})
        frames = [_frame(Event.LOOP_LAG, _loop_lag_sid(MAIN_PROC_LOOP_ID, 120_000))]
        frames += [
            _frame(Event.LOOP_LAG, _loop_lag_sid(wid, lag))
            for wid, lag in worker_lag.items()
        ]
        d.ingest_frames(b"".join(frames))

        text = d.render().plain
        assert "fleet p99" in text
        assert "hot workers (p99 ≥ 5 ms)  2/18" in text
        lag = text.split("EVENT LOOP LAG", 1)[1]
        # main pinned first; workers then ordered by max lag descending.
        assert lag.index("main") < lag.index("w0") < lag.index("w1") < lag.index("w2")
        # _LAG_TOP_N=16 → the two lowest-max workers (w16, w17) are dropped.
        assert "2 worker(s) with lower max lag not shown" in text
        assert "w16" not in lag and "w17" not in lag


@pytest.mark.unit
class TestTimeline:
    """The stacked e2e timeline bar must stay exactly _TIMELINE_W columns wide
    regardless of how the stage percentages drift (they need not sum to 100
    under frame drops), so the closing frame never reflows."""

    @pytest.mark.parametrize(
        "stage_pcts",
        [
            [("client", 0.3), ("server", 19.6), ("server", 78.7), ("client", 0.3)],
            [("client", 2.0), ("server", 18.0), ("server", 50.0)],  # sums 70
            [("server", 60.0), ("server", 70.0)],  # sums 130 (clamp drift)
            [("server", 42.0)],  # single stage
            [("client", 0.1), ("client", 0.1), ("server", 0.2)],  # all sub-column
            [],  # no data
        ],
    )
    def test_bar_is_fixed_width(self, stage_pcts: list[tuple[str, float]]) -> None:
        out = Text()
        _dash()._render_timeline(out, stage_pcts)
        bar = out.plain.splitlines()[0]
        assert len(bar.split("│")[1]) == Dashboard._TIMELINE_W
