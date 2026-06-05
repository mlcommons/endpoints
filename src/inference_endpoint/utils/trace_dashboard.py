# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard logic for the -vvv trace stream.

Pure aggregation + rendering — no I/O. The CLI entry point at
``scripts/trace_dashboard.py`` wires this up to the FIFO reader and
``rich.Live``. Tests target this module directly so the dashboard's
counts/lifecycle behaviour is verifiable in isolation.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

from hdrh.histogram import HdrHistogram
from rich.text import Text
from rich.theme import Theme

from inference_endpoint.utils.trace import (
    FRAME_SIZE,
    MAIN_PROC_LOOP_ID,
    PACKER,
    Event,
)

# Single source of truth for the dashboard palette. The CLI wrapper
# attaches this to its rich Console so style names below resolve at
# render time. Tones chosen to read cleanly on dark terminals without
# the bright-yellow / bold-cyan flash that earlier versions had.
DASHBOARD_THEME = Theme(
    {
        "rule": "grey39",
        "label": "grey62",
        "value": "white",
        "section": "cyan",
        "client_row": "grey78",
        "server_row": "deep_sky_blue3",
        "summary": "white",
        "warn": "orange3",
        "critical": "red3",
        "muted": "grey50",
        # ipc_2_worker / ipc_2_main tokens inside stage labels — call
        # out the IPC boundaries so the eye lands on them.
        "ipc_seg": "green3",
        # issued (in) vs completed (out) — paired so the eye reads the
        # backlog at a glance.
        "issued": "deep_sky_blue1",
        "completed": "green3",
    }
)

REFRESH_PERIOD_S = 0.3
REFRESH_HZ = 1.0 / REFRESH_PERIOD_S
WIDTH = 129
LABEL_W = 50
# Sized to one enlarged-pipe drain (1 MiB, see trace._KERNEL_PIPE_BUF) so
# each os.read pulls a full pipe's worth in one syscall instead of 16×
# 64 KiB reads, keeping the single reader ahead of ~240k frames/s.
READ_CHUNK = 1 << 20

# Defer folding by this much after COMPLETE arrives, so worker frames
# flushed slightly later than main's COMPLETE for the same sid have a
# chance to land before we pop the lifecycle.
_FOLD_DEFER_NS = int(REFRESH_PERIOD_S * 1_000_000_000)

# Evict partial lifecycles (no COMPLETE) older than this. Sized large
# enough to dwarf the worst-case request latency (long-streaming LLM
# completions, deep server queues) so legit in-flight requests are
# never collected before their COMPLETE arrives — losing the fold
# would silently zero out a stage row.
_LIFECYCLE_TTL_NS = 600_000_000_000  # 10 min

# Tail indicator threshold: if no ISSUED event has arrived for this
# long while in-flight > 0, the run is considered to be in the tail
# (draining; no new work being scheduled). Two refresh ticks is the
# minimum that filters out idle gaps between ingest batches.
_TAIL_QUIET_NS = int(2 * REFRESH_PERIOD_S * 1e9)

# End-of-run: no ISSUED or COMPLETE frame for this long ⇒ the run has
# stopped producing → freeze. Activity-based (not in-flight==0, which is
# unreliable when COMPLETE frames drop on the FIFO). Generous enough to
# never trip during a normal inter-completion gap at any real QPS.
_DONE_QUIET_NS = 3_000_000_000  # 3 s

# HDR Histogram requires a fixed trackable range at construction time.
# 1 hour cap with 3 sig figs ≈ 14k buckets per metric, ~100 KB each.
# min/avg/max are tracked exactly outside the histogram (see _Metric)
# so values past the cap still show their true magnitude in those
# slots; only p50/p99 read from HDR and clamp at the cap.
HDR_LOW = 1
HDR_HIGH = 3_600_000_000_000  # 1 hour in ns
HDR_SIG = 3


# -- data model -----------------------------------------------------------


class Stats(NamedTuple):
    """Per-metric summary. All durations in nanoseconds."""

    n: int
    avg: float
    min: float
    p50: float
    p99: float
    max: float


@dataclass(slots=True)
class _Lifecycle:
    """Per-request timing context keyed by sid. ``birth_ns`` is the
    monotonic time the first frame for this sid landed — used by the
    TTL eviction pass to drop partial lifecycles whose COMPLETE never
    arrived. The fold defer is owned by ``Dashboard._fold_queue``.
    """

    birth_ns: int = 0
    stages: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class _Metric:
    """Per-metric stats. ``min_ns`` / ``max_ns`` / ``sum_ns`` / ``total``
    are exact and uncapped — the dashboard's true min/avg/max always
    reflect real data even past the HDR range. ``hist`` is HDR-bounded
    (1 ns → 1 h) and only feeds the p50 / p99 columns; values past
    HDR_HIGH are pinned at the cap there."""

    total: int = 0
    sum_ns: float = 0.0
    min_ns: float = float("inf")
    max_ns: float = float("-inf")
    hist: HdrHistogram = field(
        default_factory=lambda: HdrHistogram(HDR_LOW, HDR_HIGH, HDR_SIG)
    )

    def add(self, ns: float) -> None:
        self.total += 1
        self.sum_ns += ns
        if ns < self.min_ns:
            self.min_ns = ns
        if ns > self.max_ns:
            self.max_ns = ns
        iv = int(ns)
        if iv < HDR_LOW:
            iv = HDR_LOW
        elif iv > HDR_HIGH:
            iv = HDR_HIGH
        self.hist.record_value(iv)


# -- stage definitions ----------------------------------------------------

_SIDE_CLIENT = "client"
_SIDE_SERVER = "server"

# (key, start_event, end_event) — every per-stage delta the fold computes.
# Superset of both layouts below; the render picks + labels a subset by
# mode, so we fold once and show it streaming-vs-offline. server_resp is
# headers→1st-chunk (streaming) or headers→full-body (offline); stream_gen
# + tail_stream only fold when RESPONSE_DONE is present (streaming).
_STAGE_FOLDS: tuple[tuple[str, Event, Event], ...] = (
    ("backpressure", Event.ISSUED, Event.CONN_ACQUIRED),
    ("socket_write", Event.CONN_ACQUIRED, Event.WRITTEN),
    ("server_headers", Event.WRITTEN, Event.RESPONSE_HEADERS),
    ("server_resp", Event.RESPONSE_HEADERS, Event.RESPONSE_BYTES),
    ("stream_gen", Event.RESPONSE_BYTES, Event.RESPONSE_DONE),
    ("tail_stream", Event.RESPONSE_DONE, Event.COMPLETE),
    ("tail_offline", Event.RESPONSE_BYTES, Event.COMPLETE),
)

# (side, label, key). Labels use ASCII '->' (some terminals render U+2192
# as two cells, shifting every row a column versus the header).
_LAYOUT_STREAMING: tuple[tuple[str, str, str], ...] = (
    (_SIDE_CLIENT, "issue -> ipc_2_worker -> conn acquired", "backpressure"),
    (_SIDE_CLIENT, "conn acquired -> payload written", "socket_write"),
    (_SIDE_SERVER, "payload written -> headers recvd", "server_headers"),
    (_SIDE_SERVER, "headers recvd -> 1st chunk", "server_resp"),
    (_SIDE_SERVER, "1st chunk -> last chunk", "stream_gen"),
    (_SIDE_CLIENT, "last chunk -> ipc_2_main -> complete", "tail_stream"),
)
_LAYOUT_OFFLINE: tuple[tuple[str, str, str], ...] = (
    (_SIDE_CLIENT, "issue -> ipc_2_worker -> conn acquired", "backpressure"),
    (_SIDE_CLIENT, "conn acquired -> payload written", "socket_write"),
    (_SIDE_SERVER, "payload written -> headers recvd", "server_headers"),
    (_SIDE_SERVER, "headers recvd -> response", "server_resp"),
    (_SIDE_CLIENT, "response -> ipc_2_main -> complete", "tail_offline"),
)

# All metric keys tracked by Dashboard. Per-stage keys come from
# _STAGE_FOLDS; the rest are summary / aggregate buckets for the verdict:
#   ipc_wait    = ISSUED → WORKER_RECEIVED   (worker-side pickup latency)
#   client_pre  = WORKER_RECEIVED → WRITTEN  (loadgen send-side work)
#   server_http = WRITTEN → last-body-byte   (server response, incl. token-gen)
#   client_post = last-body-byte → COMPLETE  (loadgen receive-side work)
# where last-body-byte = RESPONSE_DONE (streaming) or RESPONSE_BYTES (offline).
_METRIC_KEYS: tuple[str, ...] = tuple({k for k, _, _ in _STAGE_FOLDS}) + (
    "e2e",
    "ttft",
    "ipc_wait",
    "pool_wait",
    "client_pre",
    "server_http",
    "client_post",
)

# Backpressure thresholds: trigger the chip when either the worker-side
# pickup or the TCP-pool acquire takes ≥ this fraction of E2E.
_BACKPRESSURE_PCT = 0.20


# -- stats --------------------------------------------------------------


def _stats(m: _Metric) -> Stats:
    if m.total == 0:
        return Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    h = m.hist
    return Stats(
        n=m.total,
        avg=m.sum_ns / m.total,
        min=m.min_ns,  # exact, uncapped
        p50=float(h.get_value_at_percentile(50.0)),
        p99=float(h.get_value_at_percentile(99.0)),
        max=m.max_ns,  # exact, uncapped
    )


def _fmt_row(s: Stats, pct: float) -> str:
    ms = 1e6
    # Columns are 12 / 11×5 / 9 chars — sized so 6-digit ms values
    # (~hours of latency) still have a separator between cells.
    return (
        f"{s.n:>12,}{s.avg / ms:>11.2f}{s.min / ms:>11.2f}{s.p50 / ms:>11.2f}"
        f"{s.p99 / ms:>11.2f}{s.max / ms:>11.2f}{pct:>8.1f}%\n"
    )


_IPC_TOKEN_RE = re.compile(r"\bipc_\w+")


def _split_ipc_tokens(text: str) -> list[tuple[str, bool]]:
    """Yield ``(chunk, is_ipc)`` runs so the renderer can color the
    ``ipc_*`` tokens (e.g. ``ipc_2_worker``, ``ipc_2_main``) with
    a different style from the surrounding label."""
    out: list[tuple[str, bool]] = []
    last = 0
    for m in _IPC_TOKEN_RE.finditer(text):
        if m.start() > last:
            out.append((text[last : m.start()], False))
        out.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        out.append((text[last:], False))
    return out


# -- dashboard ----------------------------------------------------------


class Dashboard:
    """Aggregates trace frames; renders a rich :class:`Text`."""

    def __init__(
        self,
        *,
        fold_defer_ns: int = _FOLD_DEFER_NS,
        lifecycle_ttl_ns: int = _LIFECYCLE_TTL_NS,
    ) -> None:
        self._fold_defer_ns = fold_defer_ns
        self._lifecycle_ttl_ns = lifecycle_ttl_ns
        self._lifecycles: dict[int, _Lifecycle] = {}
        self._loop_lag: dict[int, _Metric] = {}
        self._metrics: dict[str, _Metric] = {k: _Metric() for k in _METRIC_KEYS}
        self._n_complete = 0
        self._start_ns = time.monotonic_ns()
        # Guards mutation of every aggregator field (lifecycles, fold/birth
        # queues, _metrics, _loop_lag, _dropped_bytes_by_proc, all _n_*
        # counters). The FIFO reader thread enters via ingest_frames; the
        # main thread enters via render(). Contention is bounded — render
        # ticks at REFRESH_HZ (~3 Hz), the reader holds the lock for at
        # most one frame batch at a time.
        self._lock = threading.Lock()
        # Per-process drop accounting: proc_id (worker_id or
        # MAIN_PROC_LOOP_ID) → total dropped bytes reported so far.
        self._dropped_bytes_by_proc: dict[int, int] = {}
        # Lifecycle counters maintained at ingest time. The reader thread
        # is the sole writer; the render thread does plain GIL-atomic
        # reads. issued/complete are main-proc; written is worker-proc
        # (WRITTEN = payload sent) and drives on-the-wire in-flight.
        self._n_issued = 0
        self._n_written = 0
        self._n_complete_seen = 0
        # Monotonic-ns time when ISSUED / COMPLETE last incremented.
        # _last_issued_ns drives the TAIL indicator; _last_complete_ns
        # is used to freeze the rate denominator once completions stop
        # arriving (prevents throughput from trending toward zero in tail).
        # _last_lifecycle_ns is the max of the two — updated only on real
        # request events (not LOOP_LAG/TRACE_DROPS) so the reader can
        # detect idle end-of-run even when LOOP_LAG frames keep the FIFO active.
        self._last_issued_ns = 0
        self._last_complete_ns = 0
        self._last_lifecycle_ns = 0
        # Fold queue: every COMPLETE event pushes (ts_seen, sid) here
        # at ingest time. finalize_completed pops from the front with a
        # time-based defer. This makes folding O(folds-per-render)
        # instead of O(lifecycles) and removes the previous
        # MAX_INFLIGHT-eviction footgun (where in-flight ISSUED entries
        # were popped before their COMPLETE could land, starving the
        # stage histograms).
        self._fold_queue: deque[tuple[int, int]] = deque()
        # Latest loadgen snapshot (parsed final_snapshot.json dict).
        # Populated by attach_loadgen_snapshot when available; the
        # comparison panel renders only if this is set.
        self._loadgen_snapshot: dict | None = None
        self._loadgen_snapshot_ts: int = 0  # monotonic_ns when data last changed
        self._loadgen_snapshot_sig: int = 0  # hash of the last seen total_duration_ns
        # Frozen Stats snapshot captured the first time is_done becomes
        # True. Stage rows render from this once set so late-arriving
        # straggler frames don't cause numbers to keep moving after the
        # run is logically complete.
        self._frozen_stats: dict[str, Stats] | None = None

    # ---- loadgen comparison hook ---------------------------------------

    def attach_loadgen_snapshot(self, snapshot: dict, *, force: bool = False) -> None:
        """Store the latest parsed snapshot dict for the comparison panel.

        ``force=True`` bypasses the staleness gate and always refreshes
        the timestamp — use for the end-of-run snapshot written by
        ``teardown()``, which may share the same ``tracked_samples_completed``
        value as the last live snapshot if the subscriber was stale.
        """
        self._loadgen_snapshot = snapshot
        if force:
            self._loadgen_snapshot_ts = time.monotonic_ns()
            return
        metrics = snapshot.get("metrics") or ()
        sig = next(
            (
                int(m.get("value") or 0)
                for m in metrics
                if m.get("name") == "tracked_samples_completed"
                and m.get("type") == "counter"
            ),
            0,
        )
        if sig != self._loadgen_snapshot_sig:
            self._loadgen_snapshot_sig = sig
            self._loadgen_snapshot_ts = time.monotonic_ns()

    # ---- observers (read-only; for tests & rendering) ------------------

    @property
    def n_issued(self) -> int:
        return self._n_issued

    @property
    def n_complete_seen(self) -> int:
        return self._n_complete_seen

    @property
    def elapsed_s(self) -> float:
        """Wall-clock seconds since PERF_START (or dashboard start)."""
        return max((time.monotonic_ns() - self._start_ns) / 1e9, 1e-9)

    @property
    def _active_elapsed_s(self) -> float:
        """Elapsed time capped at the last COMPLETE arrival.

        In the tail phase (no new completions) the wall clock grows
        but completions don't. Capping the denominator here keeps
        throughput rates from trending toward zero after the run drains.
        """
        end_ns = self._last_complete_ns or time.monotonic_ns()
        return max((end_ns - self._start_ns) / 1e9, 1e-9)

    @property
    def issuance_rate(self) -> float:
        """ISSUED events per second (main proc fire rate)."""
        return self._n_issued / self.elapsed_s

    @property
    def completion_rate(self) -> float:
        """COMPLETE events per second (effective server throughput)."""
        return self._n_complete_seen / self._active_elapsed_s

    @property
    def n_complete_folded(self) -> int:
        """Lifecycles that have been folded into the stage histograms."""
        return self._n_complete

    @property
    def _issuance_quiet(self) -> bool:
        """Main has stopped scheduling (no ISSUED for _TAIL_QUIET_NS)."""
        if self._last_issued_ns == 0:
            return False
        return (time.monotonic_ns() - self._last_issued_ns) >= _TAIL_QUIET_NS

    @property
    def is_tail(self) -> bool:
        """Issuance has gone quiet but completions are still arriving —
        the run is draining. Activity-based (independent of in-flight,
        which is lossy under FIFO drops)."""
        return self._issuance_quiet and not self.is_done

    @property
    def lifecycle_idle_s(self) -> float:
        """Seconds since the last ISSUED or COMPLETE frame.

        Zero until the first lifecycle event. Used by the reader to
        trigger an idle exit even when LOOP_LAG frames keep arriving.
        """
        if self._last_lifecycle_ns == 0:
            return 0.0
        return (time.monotonic_ns() - self._last_lifecycle_ns) / 1e9

    @property
    def is_done(self) -> bool:
        """Issuance quiet AND no ISSUED/COMPLETE for _DONE_QUIET_NS — the
        run has stopped producing events, so freeze. Activity-based, so it
        fires reliably even when dropped COMPLETE frames leave in-flight
        permanently > 0 (the old in-flight==0 test never fired under drops,
        which let straggler frames keep changing the display indefinitely).
        """
        if not self._issuance_quiet:
            return False
        return (time.monotonic_ns() - self._last_lifecycle_ns) >= _DONE_QUIET_NS

    @property
    def is_backpressured(self) -> bool:
        """True when the first lifecycle stage (ISSUED → CONN_ACQUIRED)
        takes ≥ _BACKPRESSURE_PCT of E2E — requests are backing up before
        the socket write. Triggered off the stage's folded end-points, so
        it survives intermediate-frame drops. Orthogonal to :attr:`is_tail`.
        """
        e2e_avg = _stats(self._metrics["e2e"]).avg
        if not e2e_avg:
            return False
        bp_avg = _stats(self._metrics["backpressure"]).avg
        return bp_avg / e2e_avg >= _BACKPRESSURE_PCT

    @property
    def in_flight(self) -> int:
        """On-the-wire requests = WRITTEN (payload sent) − COMPLETE.

        Counts requests actually sent to the server and awaiting their
        response — excludes the IPC backlog (issued but not yet written).
        WRITTEN is worker-proc and COMPLETE main-proc, both lossy over the
        FIFO, so the raw difference can momentarily exceed issued or go
        negative under heavy frame drop; clamp to [0, issued − complete].
        """
        written = min(self._n_written, self._n_issued)
        return max(0, written - self._n_complete_seen)

    @property
    def dropped_frames(self) -> int:
        return sum(self._dropped_bytes_by_proc.values()) // FRAME_SIZE

    def stage_n(self, key: str) -> int:
        """N for a stage metric (e.g. ``ipc_dispatch``, ``server_headers``).

        Returns 0 if the key is unknown.
        """
        m = self._metrics.get(key)
        return 0 if m is None else m.total

    def loop_lag_n(self, proc_id: int) -> int:
        m = self._loop_lag.get(proc_id)
        return 0 if m is None else m.total

    def lifecycle_count(self) -> int:
        """Number of sids still being tracked (pre-fold)."""
        return len(self._lifecycles)

    # ---- ingest ---------------------------------------------------------

    def ingest_frames(self, buf: bytes) -> None:
        n_whole = len(buf) // FRAME_SIZE
        if n_whole == 0:
            return
        # Decode all frames in C via iter_unpack rather than a per-frame
        # unpack_from loop — at ~240k frames/s across 24 worker pipes the
        # Python-level loop is the reader's bottleneck and the backpressure
        # that overflows the producers' pipes. iter_unpack requires an
        # exact frame-multiple, so slice off any trailing partial first
        # (the FIFO reader only hands us whole frames, but ingest is also
        # called directly with partials in tests).
        whole = buf if len(buf) == n_whole * FRAME_SIZE else buf[: n_whole * FRAME_SIZE]
        frames = PACKER.iter_unpack(whole)
        now_ns = time.monotonic_ns()
        # Reader thread enters here; serialise against the render thread
        # which may pop from the same queues / dicts inside render().
        with self._lock:
            for eb, sid, ts in frames:
                if eb == Event.LOOP_LAG:
                    self._record_loop_lag(sid)
                    continue
                if eb == Event.TRACE_DROPS:
                    self._record_drop(sid)
                    continue
                if eb == Event.PERF_START:
                    # Warmup done — drop everything seen so far so the
                    # LOADGEN vs TRACE comparison aligns with loadgen's
                    # tracked window.
                    self._reset_metrics(now_ns)
                    continue
                lc = self._lifecycles.get(sid)
                if lc is None:
                    lc = _Lifecycle(birth_ns=now_ns)
                    self._lifecycles[sid] = lc
                if eb == Event.ISSUED:
                    self._n_issued += 1
                    self._last_issued_ns = now_ns
                    self._last_lifecycle_ns = now_ns
                elif eb == Event.WRITTEN:
                    self._n_written += 1
                elif eb == Event.COMPLETE:
                    # Gate on ISSUED present: a warmup request whose ISSUED
                    # was cleared at PERF_START but whose COMPLETE lands
                    # afterward must not bleed into the perf window. Safe
                    # because COMPLETE and ISSUED are both main-proc events
                    # (same emitter, FIFO order) — a genuine perf COMPLETE
                    # always has its ISSUED already seen.
                    if Event.ISSUED in lc.stages:
                        self._n_complete_seen += 1
                        self._last_complete_ns = now_ns
                        self._last_lifecycle_ns = now_ns
                        # Enqueue for deferred fold; render thread will pop.
                        self._fold_queue.append((now_ns, sid))
                lc.stages[eb] = ts

    def _record_loop_lag(self, sid: int) -> None:
        worker_id = (sid >> 56) & 0xFF
        lag_ns = sid & ((1 << 56) - 1)
        m = self._loop_lag.get(worker_id)
        if m is None:
            m = _Metric()
            self._loop_lag[worker_id] = m
        m.add(float(lag_ns))

    def _record_drop(self, sid: int) -> None:
        # Payload is the producer's CUMULATIVE drop total, re-sent every
        # tick. Store the latest (max guards frame reorder) rather than
        # summing, so a lost TRACE_DROPS frame self-heals on the next one.
        proc_id = (sid >> 56) & 0xFF
        dropped = sid & ((1 << 56) - 1)
        prev = self._dropped_bytes_by_proc.get(proc_id, 0)
        if dropped > prev:
            self._dropped_bytes_by_proc[proc_id] = dropped

    def _reset_metrics(self, now_ns: int) -> None:
        """Drop warmup-phase state on PERF_START. Per-worker loop_lag
        and per-proc dropped-bytes counters are kept — they apply to
        the worker process, not the request stream."""
        self._lifecycles.clear()
        self._fold_queue.clear()
        for m in self._metrics.values():
            m.total = 0
            m.sum_ns = 0.0
            m.min_ns = float("inf")
            m.max_ns = float("-inf")
            m.hist.reset()
        self._n_issued = 0
        self._n_written = 0
        self._n_complete_seen = 0
        self._n_complete = 0
        self._last_issued_ns = 0
        self._last_complete_ns = 0
        self._last_lifecycle_ns = 0
        self._frozen_stats = None
        self._start_ns = now_ns  # uptime resets too — rate denominators

    # ---- finalize -------------------------------------------------------

    def flush_pending_folds(self) -> None:
        """Force-drain the fold queue ignoring the per-tick defer window.

        Called at FIFO EOF: any COMPLETE frames that arrived within the
        last ``_fold_defer_ns`` would otherwise sit in the queue past
        the final render and be lost. Acquires the same lock as
        ingest/render to stay consistent with the rest of the API.
        """
        with self._lock:
            self._finalize_completed_impl(fold_defer_ns=0)

    def finalize_completed(self) -> None:
        """Drain the fold queue (folds-since-last-tick) and the TTL queue
        (partial lifecycles too old to keep). Both pops are O(work
        done), not O(dict size).
        """
        self._finalize_completed_impl(fold_defer_ns=self._fold_defer_ns)

    def _finalize_completed_impl(self, *, fold_defer_ns: int) -> None:
        now_ns = time.monotonic_ns()
        fold_deadline = now_ns - fold_defer_ns
        while self._fold_queue and self._fold_queue[0][0] <= fold_deadline:
            _ts, sid = self._fold_queue.popleft()
            lc = self._lifecycles.pop(sid, None)
            if lc is None:
                # Already evicted by TTL or a previous duplicate COMPLETE.
                continue
            stages = lc.stages
            if Event.COMPLETE not in stages or Event.ISSUED not in stages:
                # Either ISSUED was never seen for this sid (e.g. its
                # producer started after we missed its first flush) or
                # COMPLETE was the only event for this sid. Skip — we
                # can't time anything.
                continue
            self._fold(stages)
        # TTL eviction: drop partial lifecycles (COMPLETE never landed)
        # older than the TTL. Python preserves dict insertion order so
        # we can scan from the oldest end and stop at the first entry
        # still inside the TTL window — no separate birth queue needed,
        # which is what kept this O(QPS × TTL) at high throughput.
        evict_deadline = now_ns - self._lifecycle_ttl_ns
        stale: list[int] = []
        for sid, lc in self._lifecycles.items():
            if lc.birth_ns > evict_deadline:
                break
            stale.append(sid)
        for sid in stale:
            del self._lifecycles[sid]

    def _fold(self, stages: dict[int, int]) -> None:
        issued = stages[Event.ISSUED]
        complete = stages[Event.COMPLETE]
        self._metrics["e2e"].add(complete - issued)
        recv_first = stages.get(Event.RECV_FIRST)
        if recv_first is not None:
            self._metrics["ttft"].add(recv_first - issued)
        for key, start_ev, end_ev in _STAGE_FOLDS:
            t0 = stages.get(start_ev)
            t1 = stages.get(end_ev)
            if t0 is not None and t1 is not None:
                self._metrics[key].add(t1 - t0)
        # Aggregate buckets for the verdict. client_pre measures
        # WORKER_RECEIVED → WRITTEN (true loadgen send-side work), NOT
        # ISSUED → WRITTEN — the latter folds in IPC queue wait, which is
        # back-pressure from server saturation and misleads the verdict.
        # body_done = last body byte: RESPONSE_DONE (streaming, last chunk)
        # or RESPONSE_BYTES (offline, full body) — so server_http captures
        # token-gen and client_post is the real client tail, both modes.
        worker_recv = stages.get(Event.WORKER_RECEIVED)
        conn_acq = stages.get(Event.CONN_ACQUIRED)
        written = stages.get(Event.WRITTEN)
        body_done = stages.get(Event.RESPONSE_DONE)
        if body_done is None:
            body_done = stages.get(Event.RESPONSE_BYTES)
        if worker_recv is not None:
            self._metrics["ipc_wait"].add(worker_recv - issued)
            if conn_acq is not None:
                self._metrics["pool_wait"].add(conn_acq - worker_recv)
            if written is not None:
                self._metrics["client_pre"].add(written - worker_recv)
                if body_done is not None:
                    self._metrics["server_http"].add(body_done - written)
        if body_done is not None:
            self._metrics["client_post"].add(complete - body_done)
        self._n_complete += 1

    # ---- render ---------------------------------------------------------

    def render(self) -> Text:
        # Held for the entire render so the reader thread can't mutate
        # the dicts / queues / histograms we are walking. finalize_completed
        # also mutates state (folds + evictions), so it must run inside
        # the same critical section.
        with self._lock:
            self.finalize_completed()
            # Freeze stage stats the first time is_done fires so that
            # late-arriving straggler frames don't cause the lifecycle
            # table to keep moving after the run is logically complete.
            if self.is_done and self._frozen_stats is None:
                self._frozen_stats = {k: _stats(v) for k, v in self._metrics.items()}
            stats = self._frozen_stats or None
            out = Text(no_wrap=True)
            self._render_header(out)
            out.append("\n")
            self._render_lifecycle(out, frozen_stats=stats)
            if self._loadgen_snapshot is not None:
                out.append("\n")
                self._render_loadgen_comparison(out)
            out.append("\n")
            self._render_loop_lag(out)
            return out

    def _render_header(self, out: Text) -> None:
        elapsed_s = self.elapsed_s
        qps = self.completion_rate
        dropped = self.dropped_frames
        out.append("═" * WIDTH + "\n", style="section")
        self._row(
            out,
            (
                ("uptime", f"{elapsed_s:>10.1f}s", ""),
                ("complete", f"{self._n_complete_seen:>10,}", "completed"),
                ("req/s", f"{qps:>10,.1f}", "completed"),
            ),
        )
        self._row(
            out,
            (
                ("issued", f"{self._n_issued:>10,}", "issued"),
                ("in-flight", f"{self.in_flight:>10,}", ""),
                ("", "", ""),
            ),
        )
        if dropped:
            drop_style = "critical" if dropped > 100 else "warn"
            self._row(
                out,
                (
                    ("dropped frames", f"{dropped:>10,}", drop_style),
                    ("", "", ""),
                    ("", "", ""),
                ),
            )
        bp = self.is_backpressured
        tail = self.is_tail
        done = self.is_done
        if done:
            self._row(
                out,
                (
                    ("status", f"{'DONE (report gen)':>24}", "warn"),
                    ("", "", ""),
                    ("", "", ""),
                ),
            )
        elif tail or bp:
            if tail and bp:
                chip, style = "TAIL + BACKPRESSURE", "critical"
            elif tail:
                chip, style = "TAIL", "warn"
            else:
                chip, style = "BACKPRESSURE", "critical"
            status_fields: tuple[tuple[str, str, str], ...] = (
                ("status", f"{chip:>24}", style),
                ("draining", f"{self.in_flight:>10,}", style) if tail else ("", "", ""),
                ("", "", ""),
            )
            self._row(out, status_fields)
        out.append("═" * WIDTH + "\n", style="section")

    @staticmethod
    def _row(
        out: Text,
        fields: tuple[tuple[str, str, str], ...],
        *,
        col_w: int = 40,
        col_gap: int = 3,
    ) -> None:
        """Render a row of (label, value, value_style) fields in equal-width
        columns. Labels are dim and left-aligned; values are right-aligned
        within their column. Empty (label, value) pairs render as blank
        space so column anchors stay consistent across rows.
        """
        out.append("  ")
        for i, (label, value, style) in enumerate(fields):
            if i > 0:
                out.append(" " * col_gap)
            pad = max(1, col_w - len(label) - 1 - len(value))
            if label:
                out.append(f"{label} ", style="rule")
                out.append(" " * pad)
                out.append(value, style=style)
            else:
                out.append(" " * col_w)
        out.append("\n")

    def _render_lifecycle(
        self, out: Text, frozen_stats: dict[str, Stats] | None = None
    ) -> None:
        def _get(key: str) -> Stats:
            if frozen_stats is not None:
                return frozen_stats.get(key, Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0))
            return _stats(self._metrics[key])

        e2e_avg = _get("e2e").avg or 1.0
        # Streaming if the 1st-chunk→last-chunk delta folded (RESPONSE_DONE
        # present); otherwise offline (single body read).
        streaming = _get("stream_gen").n > 0
        layout = _LAYOUT_STREAMING if streaming else _LAYOUT_OFFLINE
        section = "  REQUEST LIFECYCLE  (ms)"
        if frozen_stats is not None:
            section += "  [frozen]"
        out.append(section + "\n", style="section")
        out.append("─" * WIDTH + "\n", style="rule")
        out.append(
            f"  {'stage':<{LABEL_W}}"
            f"{'N':>12}{'avg':>11}{'min':>11}{'p50':>11}{'p99':>11}"
            f"{'max':>11}{'%E2E':>9}\n",
            style="label",
        )
        for side, label, key in layout:
            self._render_row_stats(out, side, label, _get(key), e2e_avg)
        e2e_stats = _get("e2e")
        self._render_summary_stats(
            out,
            "E2E TOTAL  issue -> complete",
            e2e_stats,
            e2e_avg,
            bold=True,
        )
        out.append("\n")
        self._render_verdict(out, e2e_avg)

    def _render_row(
        self, out: Text, side: str, label: str, m: _Metric, e2e_avg: float
    ) -> None:
        self._render_row_stats(out, side, label, _stats(m), e2e_avg)

    def _render_row_stats(
        self, out: Text, side: str, label: str, s: Stats, e2e_avg: float
    ) -> None:
        # Clamp at 100: a stage is a sub-interval of E2E so it cannot
        # exceed it per request. The raw ratio can top 100% because the
        # stage avg and the E2E avg are over different folded populations
        # (different N when intermediate frames drop) — the slow-request
        # subset that retained its frames biases the stage avg upward.
        pct = min(100.0, 100.0 * s.avg / e2e_avg) if e2e_avg and s.avg else 0.0
        side_style = "client_row" if side == _SIDE_CLIENT else "server_row"
        prefix = f"[{side}] {label}"
        Dashboard._append_label(out, prefix[:LABEL_W], LABEL_W, side_style, "  ")
        out.append(_fmt_row(s, pct), style=side_style)

    @staticmethod
    def _append_label(
        out: Text, text: str, width: int, base_style: str, leading: str = ""
    ) -> None:
        """Append a stage label with ``ipc_<...>`` tokens highlighted.
        Pads the whole label to ``width`` so columns stay aligned."""
        out.append(leading)
        consumed = 0
        for chunk, is_ipc in _split_ipc_tokens(text):
            out.append(chunk, style="ipc_seg" if is_ipc else base_style)
            consumed += len(chunk)
        pad = width - consumed
        if pad > 0:
            out.append(" " * pad, style=base_style)

    def _render_summary(
        self,
        out: Text,
        label: str,
        m: _Metric,
        e2e_avg: float,
        *,
        bold: bool = False,
    ) -> None:
        self._render_summary_stats(out, label, _stats(m), e2e_avg, bold=bold)

    def _render_summary_stats(
        self,
        out: Text,
        label: str,
        s: Stats,
        e2e_avg: float,
        *,
        bold: bool = False,
    ) -> None:
        pct = 100.0 * s.avg / e2e_avg if e2e_avg and s.avg else 0.0
        style = "summary" if bold else ""
        out.append(f"  {label[:LABEL_W]:<{LABEL_W}}", style=style)
        out.append(_fmt_row(s, pct), style=style)

    def _render_verdict(self, out: Text, e2e_avg: float) -> None:
        # Two plain rows; no section headers, no dividers, no footer.
        # Same 3-column grid as the header so everything anchors.
        client = (
            _stats(self._metrics["client_pre"]).avg
            + _stats(self._metrics["client_post"]).avg
        )
        server = _stats(self._metrics["server_http"]).avg
        queue = _stats(self._metrics["ipc_wait"]).avg
        c_pct = 100.0 * client / e2e_avg if e2e_avg else 0.0
        s_pct = 100.0 * server / e2e_avg if e2e_avg else 0.0
        q_pct = 100.0 * queue / e2e_avg if e2e_avg else 0.0
        c_style = "warn" if c_pct > 50 else ""
        q_style = "warn" if q_pct > 50 else ""

        self._row(
            out,
            (
                ("client work", f"{c_pct:>9.1f}%", c_style),
                ("server work", f"{s_pct:>9.1f}%", ""),
                ("backpressure", f"{q_pct:>9.1f}%", q_style),
            ),
        )
        if self.elapsed_s < 2.0:
            return
        iss = self.issuance_rate
        cmp = self.completion_rate
        backlog = iss - cmp
        bl_style = "warn" if backlog > max(cmp, 1) else ""
        self._row(
            out,
            (
                ("issued", f"{iss:>10,.0f}/s", "issued"),
                ("completed", f"{cmp:>10,.0f}/s", "completed"),
                ("backlog", f"{backlog:>+10,.0f}/s", bl_style),
            ),
        )

    def _render_loadgen_comparison(self, out: Text) -> None:
        """Side-by-side trace-vs-loadgen panel. Two sub-tables:
        counts/rates (with errors), and latency percentiles
        (min/p50/p99/max + Δmax). Skipped when no snapshot is attached.
        """
        snap = self._loadgen_snapshot
        if snap is None:
            return
        metrics = snap.get("metrics") or ()
        counters = {
            m.get("name"): m.get("value") for m in metrics if m.get("type") == "counter"
        }
        series = {m.get("name"): m for m in metrics if m.get("type") == "series"}
        lg_completed = int(counters.get("total_samples_completed") or 0) or None
        lg_failed = int(counters.get("tracked_samples_failed") or 0) or None
        lg_tracked = int(counters.get("tracked_samples_completed") or 0)
        # tracked_duration_ns may be 0 mid-run; fall back to total.
        lg_dur_ns = int(counters.get("tracked_duration_ns") or 0) or int(
            counters.get("total_duration_ns") or 0
        )
        lg_dur_s = lg_dur_ns / 1e9 if lg_dur_ns > 0 else 0.0
        lg_qps = (lg_tracked / lg_dur_s) if lg_dur_s and lg_tracked else None
        lg_osl_total = float((series.get("osl") or {}).get("total") or 0.0)
        lg_tps = (lg_osl_total / lg_dur_s) if lg_dur_s and lg_osl_total else None
        lg_e2e = series.get("sample_latency_ns") or {}
        lg_ttft = series.get("ttft_ns") or {}
        lg_tpot = series.get("tpot_ns") or {}

        trace_e2e = _stats(self._metrics["e2e"])
        trace_ttft = _stats(self._metrics["ttft"])
        trace_qps = self.completion_rate or None
        trace_completed = self._n_complete_seen or None

        snap_age_s = (time.monotonic_ns() - self._loadgen_snapshot_ts) / 1e9
        age_tag = f"  (snapshot {snap_age_s:.0f}s old)" if snap_age_s > 2.0 else ""
        out.append(f"  LOADGEN vs TRACE{age_tag}\n", style="section")
        out.append("─" * WIDTH + "\n", style="rule")

        # --- counts / rates table ---
        out.append(
            f"  {'counts / rates':<{self._CMP_LABEL_W}}"
            f"{'loadgen':>{self._CMP_VAL_W}}"
            f"{'trace':>{self._CMP_VAL_W}}"
            f"{'Δ':>{self._CMP_DELTA_W}}\n",
            style="label",
        )
        self._cmp_row(
            out, "samples completed", lg_completed, trace_completed, fmt=",.0f"
        )
        self._cmp_row(out, "errors", lg_failed, None, fmt=",.0f")
        self._cmp_row(out, "throughput (req/s)", lg_qps, trace_qps, fmt=",.1f")
        self._cmp_row(out, "throughput (tok/s)", lg_tps, None, fmt=",.1f")

        # --- latency percentiles table ---
        out.append("\n")
        out.append(
            f"  {'latency':<{self._LAT_LABEL_W}}"
            f"{'src':<{self._LAT_SRC_W}}"
            f"{'min':>{self._LAT_VAL_W}}"
            f"{'p50':>{self._LAT_VAL_W}}"
            f"{'p99':>{self._LAT_VAL_W}}"
            f"{'max':>{self._LAT_VAL_W}}"
            f"{'Δmax':>{self._LAT_DELTA_W}}\n",
            style="label",
        )
        # Label has no unit — _cmp_dist appends one auto-picked from
        # the observed max so sub-ms values (e.g. tpot on small local
        # models) don't all flatten to "0.00 ms".
        self._cmp_dist(out, "e2e", lg_e2e, trace_e2e)
        self._cmp_dist(out, "ttft", lg_ttft, trace_ttft)
        self._cmp_dist(out, "tpot", lg_tpot, None)

    _CMP_LABEL_W = 40
    _CMP_VAL_W = 22
    _CMP_DELTA_W = 14
    _CMP_WARN_PCT = 5.0
    _LAT_LABEL_W = 18
    _LAT_SRC_W = 10
    _LAT_VAL_W = 14
    _LAT_DELTA_W = 12

    @staticmethod
    def _cmp_row(
        out: Text,
        label: str,
        loadgen: float | None,
        trace: float | None,
        *,
        fmt: str = ",.1f",
    ) -> None:
        """Single-value row: skipped when both sides are empty; em-dash
        on the missing side; Δ em-dashed unless both sides populated."""
        if loadgen is None and trace is None:
            return
        val_w = Dashboard._CMP_VAL_W
        out.append(f"  {label:<{Dashboard._CMP_LABEL_W}}", style="label")
        if loadgen is None:
            out.append(f"{'—':>{val_w}}", style="muted")
        else:
            out.append(f"{loadgen:>{val_w}{fmt}}", style="")
        if trace is None:
            out.append(f"{'—':>{val_w}}", style="muted")
        else:
            out.append(f"{trace:>{val_w}{fmt}}", style="")
        delta_s, delta_style = Dashboard._delta(loadgen, trace)
        out.append(f"{delta_s:>{Dashboard._CMP_DELTA_W}}\n", style=delta_style)

    @staticmethod
    def _cmp_dist(
        out: Text,
        label: str,
        lg_series: dict,
        trace_stats: Stats | None,
    ) -> None:
        """Two-line distribution block: loadgen + trace rows for one
        metric. Auto-picks ns/µs/ms/s based on the observed max so
        sub-ms values keep precision. Skipped if neither side has data."""
        lg_pcts = (lg_series or {}).get("percentiles") or {}
        lg_count = (lg_series or {}).get("count") or 0
        has_lg = lg_count > 0
        has_tr = trace_stats is not None and trace_stats.n > 0
        if not has_lg and not has_tr:
            return

        # Pick the largest max across both sources to size the unit.
        lg_max_ns = float((lg_series or {}).get("max") or 0.0) if has_lg else 0.0
        tr_max_ns = (
            float(trace_stats.max) if has_tr and trace_stats is not None else 0.0
        )
        divisor, unit = Dashboard._pick_unit(max(lg_max_ns, tr_max_ns))

        def cells(
            min_: float | None, p50: float | None, p99: float | None, max_: float | None
        ) -> tuple[str, ...]:
            return tuple(
                f"{v:>{Dashboard._LAT_VAL_W},.2f}"
                if v is not None
                else f"{'—':>{Dashboard._LAT_VAL_W}}"
                for v in (min_, p50, p99, max_)
            )

        if has_lg:
            lg_min = float(lg_series.get("min") or 0.0) / divisor
            lg_max = lg_max_ns / divisor
            lg_p50 = float(lg_pcts.get("50.0") or 0.0) / divisor
            lg_p99 = float(lg_pcts.get("99.0") or 0.0) / divisor
        else:
            lg_min = lg_p50 = lg_p99 = lg_max = None  # type: ignore[assignment]
        out.append(
            f"  {label + ' (' + unit + ')':<{Dashboard._LAT_LABEL_W}}", style="label"
        )
        out.append(f"{'loadgen':<{Dashboard._LAT_SRC_W}}", style="label")
        for c in cells(lg_min, lg_p50, lg_p99, lg_max):
            out.append(c, style="" if has_lg else "muted")
        out.append("\n")

        if has_tr and trace_stats is not None:
            tr_min = trace_stats.min / divisor
            tr_max = tr_max_ns / divisor
            tr_p50 = trace_stats.p50 / divisor
            tr_p99 = trace_stats.p99 / divisor
        else:
            tr_min = tr_p50 = tr_p99 = tr_max = None  # type: ignore[assignment]
        out.append(f"  {'':<{Dashboard._LAT_LABEL_W}}")
        out.append(f"{'trace':<{Dashboard._LAT_SRC_W}}", style="label")
        for c in cells(tr_min, tr_p50, tr_p99, tr_max):
            out.append(c, style="" if has_tr else "muted")
        delta_s, delta_style = (
            Dashboard._delta(lg_max, tr_max) if has_lg and has_tr else ("—", "muted")
        )
        out.append(f"{delta_s:>{Dashboard._LAT_DELTA_W}}\n", style=delta_style)

    @staticmethod
    def _pick_unit(max_ns: float) -> tuple[float, str]:
        """Pick the most readable (divisor, suffix) for a metric whose
        observed max is ``max_ns`` nanoseconds. ms is the fallback so a
        block with no data still labels consistently."""
        if max_ns >= 1e9:
            return 1e9, "s"
        if max_ns >= 1e6:
            return 1e6, "ms"
        if max_ns >= 1e3:
            return 1e3, "µs"
        return 1e6, "ms"

    @staticmethod
    def _delta(loadgen: float | None, trace: float | None) -> tuple[str, str]:
        """Format the Δ cell: percentage change of trace vs loadgen.
        Em-dash when either side is missing or loadgen is zero."""
        if loadgen is None or trace is None or loadgen == 0:
            return ("—", "muted")
        delta_pct = 100.0 * (trace - loadgen) / loadgen
        style = "warn" if abs(delta_pct) > Dashboard._CMP_WARN_PCT else ""
        return (f"{delta_pct:+.1f}%", style)

    # Maximum worker rows shown (excl. main which is always first).
    _LAG_TOP_N = 16

    def _render_loop_lag(self, out: Text) -> None:
        out.append("  EVENT LOOP LAG  (ms)\n", style="section")
        out.append("─" * WIDTH + "\n", style="rule")
        if not self._loop_lag:
            out.append("  (no LOOP_LAG events yet)\n", style="muted italic")
            return

        # Separate main from workers; sort workers by max lag descending,
        # keep top _LAG_TOP_N worst offenders.
        main_entry = self._loop_lag.get(MAIN_PROC_LOOP_ID)
        all_worker_stats = [
            (wid, _stats(m))
            for wid, m in self._loop_lag.items()
            if wid != MAIN_PROC_LOOP_ID
        ]
        all_worker_stats.sort(key=lambda t: t[1].max, reverse=True)
        workers = all_worker_stats[: self._LAG_TOP_N]

        # Fleet summary: median p99 across all workers + hot-worker count.
        # "Hot" = p99 > 5 ms (GIL or syscall stall territory).
        _HOT_THRESH_NS = 5_000_000  # 5 ms
        all_p99s = [s.p99 for _, s in all_worker_stats]
        fleet_p99_ms = sorted(all_p99s)[len(all_p99s) // 2] / 1e6 if all_p99s else 0.0
        n_hot = sum(1 for p in all_p99s if p >= _HOT_THRESH_NS)
        n_workers = len(all_worker_stats)
        hot_style = "critical" if n_hot > n_workers // 2 else ("warn" if n_hot else "")
        out.append(
            f"  fleet p99 {fleet_p99_ms:.2f} ms   " f"hot workers (p99 ≥ 5 ms)  ",
            style="label",
        )
        out.append(f"{n_hot}/{n_workers}\n", style=hot_style or "label")

        def _emit(label: str, s: Stats, *, highlight: bool = False) -> None:
            mx_ms = s.max / 1e6
            p99_ms = s.p99 / 1e6
            mx_style = "critical" if mx_ms > 50 else ("warn" if mx_ms > 10 else "")
            p99_style = "critical" if p99_ms > 10 else ("warn" if p99_ms > 1 else "")
            row_style = "summary" if highlight else ""
            out.append(f"  {label:<10}", style=row_style)
            out.append(f"{s.n:>10,}", style=row_style)
            out.append(f"{s.min / 1e6:>10.2f}", style=row_style)
            out.append(f"{s.p50 / 1e6:>10.2f}", style=row_style)
            out.append(f"{p99_ms:>10.2f}", style=p99_style or row_style)
            out.append(f"{mx_ms:>10.2f}\n", style=mx_style or row_style)

        out.append(
            f"  {'worker':<10}{'#samples':>10}{'min':>10}{'p50':>10}"
            f"{'p99':>10}{'max':>10}\n",
            style="label",
        )

        # main always first, always highlighted
        if main_entry is not None:
            _emit("main", _stats(main_entry), highlight=True)

        for wid, s in workers:
            _emit(f"w{wid}", s)

        omitted = max(0, len(self._loop_lag) - 1 - self._LAG_TOP_N)
        if omitted:
            out.append(
                f"  … {omitted} worker(s) with lower max lag not shown\n",
                style="muted",
            )
