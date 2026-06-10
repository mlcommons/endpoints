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
import statistics
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
        # "default" = the terminal's own foreground, so these read on both
        # dark and light backgrounds (a literal "white"/light grey vanishes on
        # light bg). client rows use it so [client] vs [server] reads as
        # default-vs-blue either way.
        "value": "default",
        "section": "cyan",
        "client_row": "default",
        "server_row": "deep_sky_blue3",
        "summary": "default",
        "warn": "orange3",
        "critical": "red3",
        "muted": "grey50",
        # ipc_2_worker / ipc_2_main tokens inside stage labels — dim,
        # low-attention markers (just note the IPC boundary, don't shout).
        "ipc_seg": "grey50",
        # issued (in) → completed (out): two shades of one hue so they
        # read as the two ends of the same lifecycle (light = in, deep = out),
        # distinct from server (blue) / ipc (green) / section (cyan).
        "issued": "medium_purple1",
        "completed": "medium_purple3",
    }
)

REFRESH_PERIOD_S = 0.3
REFRESH_HZ = 1.0 / REFRESH_PERIOD_S
WIDTH = 129
LABEL_W = 50
# Per-os.read cap: 1 MiB pulls a large batch per syscall (vs many 64 KiB
# reads) to keep the single reader ahead of ~240k frames/s. Independent of
# the (best-effort, up to 512 MiB) pipe buffer — see trace._KERNEL_PIPE_BUF.
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
_SIDE_BACKPRESSURE = "backpressure"

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
    "client_work",
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


def _fmt_row(s: Stats) -> str:
    """Numeric columns N..max (no %E2E — caller appends it separately so
    the %E2E cell can be color-graded). 12 / 11×5 chars."""
    ms = 1e6
    return (
        f"{s.n:>12,}{s.avg / ms:>11.2f}{s.min / ms:>11.2f}{s.p50 / ms:>11.2f}"
        f"{s.p99 / ms:>11.2f}{s.max / ms:>11.2f}"
    )


# Numeric column header shared by every distribution table (lifecycle,
# loadgen, loop-lag) so the N/avg/min/p50/p99/max columns line up vertically
# down the whole dashboard. Widths match _fmt_row (12 / 11×5).
_DIST_NUM_HDR = f"{'N':>12}{'avg':>11}{'min':>11}{'p50':>11}{'p99':>11}{'max':>11}"


def _series_stats(s: dict) -> Stats:
    """Build a Stats (ns) from a loadgen snapshot series dict so loadgen
    latency rows render through the same _fmt_row path as the lifecycle
    table — identical widths, identical ms formatting. Empty series → n=0."""
    count = int(s.get("count") or 0)
    total = float(s.get("total") or 0.0)
    pcts = s.get("percentiles") or {}
    return Stats(
        count,
        (total / count) if count else 0.0,
        float(s.get("min") or 0.0),
        float(pcts.get("50.0") or 0.0),
        float(pcts.get("99.0") or 0.0),
        float(s.get("max") or 0.0),
    )


def _heat(pct: float) -> str:
    """Heatmap style for a share-of-E2E cell (pct in 0-100). One scale shared
    across the lifecycle %E2E column, the verdict, and the worker-loop tree:
    warn (orange) ≥ 15%, critical (red) ≥ 25%."""
    if pct >= 25.0:
        return "critical"
    if pct >= 15.0:
        return "warn"
    return ""


# Lifecycle endpoints + IPC hops highlighted inside stage labels:
#   issue → issued (blue), complete → completed (green), ipc_* → ipc_seg.
_LABEL_TOKEN_RE = re.compile(r"\bipc_\w+|\bissue\b|\bcomplete\b")


def _label_token_style(tok: str) -> str:
    if tok.startswith("ipc_"):
        return "ipc_seg"
    return "issued" if tok == "issue" else "completed"


def _split_label_tokens(text: str) -> list[tuple[str, str | None]]:
    """Yield ``(chunk, style|None)`` runs so the renderer can highlight
    the lifecycle endpoints (``issue``/``complete``) and IPC hops
    (``ipc_2_worker``/``ipc_2_main``) distinctly from the label body."""
    out: list[tuple[str, str | None]] = []
    last = 0
    for m in _LABEL_TOKEN_RE.finditer(text):
        if m.start() > last:
            out.append((text[last : m.start()], None))
        out.append((m.group(0), _label_token_style(m.group(0))))
        last = m.end()
    if last < len(text):
        out.append((text[last:], None))
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
        # Set by the PERF_END frame (main proc, run over) → is_done.
        self._ended = False
        # Monotonic ns when PERF_END first landed; drives the "finalizing… Ns"
        # age. The snapshot ts can't: the aggregator keeps ticking during the
        # drain, so the freshest-snapshot age stays ~0 the whole time.
        self._done_ns = 0
        # Set by the CLI once its end-of-run wait elapses without a terminal
        # snapshot: the aggregator never finalized (vs merely still finalizing).
        self._final_unavailable = False
        # Fold queue: every COMPLETE event pushes (ts_seen, sid) here at
        # ingest time; finalize_completed pops from the front with a
        # time-based defer, keeping folding O(folds-per-render) rather than
        # O(lifecycles). An ISSUED entry is retained until its COMPLETE lands
        # so the stage histograms see both endpoints.
        self._fold_queue: deque[tuple[int, int]] = deque()
        # Latest loadgen snapshot (parsed final_snapshot.json dict).
        # Populated by attach_loadgen_snapshot when available; the
        # comparison panel renders only if this is set.
        self._loadgen_snapshot: dict | None = None
        self._loadgen_snapshot_ts: int = 0  # monotonic_ns when data last changed
        self._loadgen_snapshot_sig: int = -1  # last seen publish counter (snap seq)
        self._loadgen_state: str | None = (
            None  # aggregator SessionState (authoritative done signal)
        )
        # Frozen Stats snapshot captured the first time is_done becomes
        # True. Stage rows render from this once set so late-arriving
        # straggler frames don't cause numbers to keep moving after the
        # run is logically complete.
        self._frozen_stats: dict[str, Stats] | None = None
        # Fleet-wide ESTABLISHED TCP connection count, sampled by the CLI
        # wrapper from /proc (observer-side; zero producer/hot-path cost).
        # -1 = no sample yet / probe unavailable → cell hidden.
        self._tcp_established = -1

    # ---- loadgen comparison hook ---------------------------------------

    def attach_loadgen_snapshot(self, snapshot: dict, *, force: bool = False) -> None:
        """Store the latest parsed snapshot dict for the LOADGEN panel.

        Thread-safe: this is called from both the main thread (sidecar
        fallback) and the metrics-SUB reader thread, and the fields it writes
        are read by ``render()`` — so the whole body runs under ``self._lock``
        (cheap; this is the dashboard process, off the benchmark hot path).

        ``force=True`` bypasses the staleness gate and refreshes the timestamp.
        A terminal snapshot (state complete / interrupted) is always treated
        as fresh and **latched**: once a terminal snapshot is in hand, a later
        non-terminal one (a late SUB delivery / out-of-order frame) is ignored,
        so the closing frame can never revert to a stale mid-run snapshot.
        """
        incoming_state = snapshot.get("state")
        incoming_terminal = (incoming_state or "").lower() in (
            "complete",
            "interrupted",
        )
        with self._lock:
            if self.has_terminal_loadgen and not incoming_terminal:
                return  # latched terminal — don't downgrade to a live frame
            self._loadgen_snapshot = snapshot
            self._loadgen_state = incoming_state
            if force or incoming_terminal:
                self._loadgen_snapshot_ts = time.monotonic_ns()
                return
            # Freshness keys off the publish counter (the snapshot sequence
            # number), which advances every aggregator tick — so the feed reads
            # as live even during warmup (before any completions) and as stale
            # only when the counter genuinely stops advancing.
            sig = int(snapshot.get("counter") or 0)
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
    def has_terminal_loadgen(self) -> bool:
        """True once a terminal (COMPLETE / INTERRUPTED) loadgen snapshot has
        been attached — the authoritative end-of-run data is in hand. The
        end-of-run wait in the CLI blocks on this so the closing frame shows
        final totals, never a stale mid-run snapshot."""
        return (self._loadgen_state or "").lower() in ("complete", "interrupted")

    @property
    def is_done(self) -> bool:
        """Run finished → freeze. The deterministic signal is PERF_END,
        emitted by the main proc the instant session.run() returns; it
        fires regardless of frame drops or a stale loadgen snapshot. The
        loadgen terminal state is a backup for the same intent.
        """
        return self._ended or self.has_terminal_loadgen

    def mark_final_unavailable(self) -> None:
        """Called by the CLI when its end-of-run wait elapsed without a
        terminal loadgen snapshot — the aggregator did not finalize. Flips the
        LOADGEN freshness tag from "finalizing…" to an explicit failure."""
        with self._lock:
            self._final_unavailable = True

    def set_tcp_established(self, n: int) -> None:
        """Attach the latest fleet-wide ESTABLISHED TCP conn count. Sampled
        outside this module (the CLI wrapper reads /proc) so the dashboard
        stays pure and the benchmark processes carry no collection logic."""
        with self._lock:
            self._tcp_established = n

    @property
    def is_backpressured(self) -> bool:
        """True when the first lifecycle stage (ISSUED → CONN_ACQUIRED)
        takes ≥ _BACKPRESSURE_PCT of E2E (mean share, same basis as the
        %E2E column) — requests are backing up before the socket write.
        Triggered off the stage's folded end-points, so it survives
        intermediate-frame drops. Orthogonal to :attr:`is_tail`.
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
        """N for a stage metric (e.g. ``backpressure``, ``server_headers``).

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
                    # perf-window stats align with loadgen's tracked window.
                    self._reset_metrics(now_ns)
                    continue
                if eb == Event.PERF_END:
                    if not self._ended:
                        self._done_ns = now_ns  # start the finalize clock once
                    self._ended = True  # benchmark over → freeze (is_done)
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
        """Drop warmup-phase state on PERF_START so every panel reflects only
        the perf window. Per-proc dropped-bytes counters are kept: the producer
        re-emits its *cumulative* total each tick (a window reset would just be
        overwritten) and it is a trace-fidelity gauge, not a request sample."""
        self._lifecycles.clear()
        self._fold_queue.clear()
        # Loop-lag is per-worker-process, but its warmup samples (cold caches,
        # connection ramp) skew p99/max — clear so the EVENT LOOP panel reads
        # only perf-window lag, like every other panel. Rows repopulate from
        # the first post-PERF_START tick (~0.1 s).
        self._loop_lag.clear()
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
        self._ended = False
        self._done_ns = 0
        self._final_unavailable = False
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
            # render() freezes _frozen_stats once, when is_done first fires —
            # which is BEFORE this end-of-run flush, so the completions folded
            # here (those still inside the defer window at freeze time) would
            # never reach the closing frame. Invalidate the freeze so the final
            # render re-captures the now-complete _metrics; otherwise the
            # LIFECYCLE/verdict/timeline panels undercount the last completions
            # and disagree with the authoritative LOADGEN panel.
            self._frozen_stats = None

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
                    # Per-request combined client work (pre + post): folded as
                    # one value so its percentiles are a true per-request
                    # distribution for the verdict's p99 share.
                    self._metrics["client_work"].add(
                        (written - worker_recv) + (complete - body_done)
                    )
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
                self._render_loadgen(out)
            out.append("\n")
            self._render_loop_lag(out)
            return out

    def _loadgen_view(self) -> tuple[int, float, float] | None:
        """(issued, req/s, tok/s) for the header from the loadgen snapshot,
        or None. Authoritative — the trace's own counts undercount under
        FIFO drops, so the header sources these from loadgen when a snapshot
        exists. (Completed/in-flight live in the LOADGEN panel, which reads
        the snapshot directly.)"""
        snap = self._loadgen_snapshot
        if snap is None:
            return None
        c = {
            m.get("name"): m.get("value")
            for m in (snap.get("metrics") or ())
            if m.get("type") == "counter"
        }
        issued = int(c.get("total_samples_issued") or 0)
        tracked = int(c.get("tracked_samples_completed") or 0)
        dur_ns = int(c.get("tracked_duration_ns") or 0) or int(
            c.get("total_duration_ns") or 0
        )
        dur_s = dur_ns / 1e9 if dur_ns > 0 else 0.0
        qps = (tracked / dur_s) if dur_s and tracked else 0.0
        osl = float(self._loadgen_series("osl").get("total") or 0.0)
        tps = (osl / dur_s) if dur_s and osl else 0.0
        return issued, qps, tps

    def _loadgen_series(self, name: str) -> dict:
        snap = self._loadgen_snapshot or {}
        for m in snap.get("metrics") or ():
            if m.get("type") == "series" and m.get("name") == name:
                return m
        return {}

    def _render_header(self, out: Text) -> None:
        # Fixed 4 lines (border + 2 grid rows + border) — height never
        # jumps. Middle column carries run health (status chip + dropped
        # frames); the request counts live in the LOADGEN panel below.
        elapsed_s = self.elapsed_s
        dropped = self.dropped_frames
        view = self._loadgen_view()
        if view is not None:
            issued, qps, tps = view
        else:
            issued, qps, tps = self._n_issued, self.completion_rate, 0.0

        if self.is_done:
            chip, chip_style = "DONE", "warn"
        elif self.is_tail and self.is_backpressured:
            chip, chip_style = "TAIL + BACKPRESSURE", "critical"
        elif self.is_tail:
            chip, chip_style = "TAIL", "warn"
        elif self.is_backpressured:
            chip, chip_style = "BACKPRESSURE", "critical"
        else:
            chip, chip_style = "LIVE", "completed"
        drop_style = ("critical" if dropped > 100 else "warn") if dropped else ""

        out.append("═" * WIDTH + "\n", style="section")
        self._row(
            out,
            (
                ("uptime", f"{elapsed_s:>10.1f}s", ""),
                ("status", f"{chip:>10}", chip_style),
                ("req/s", f"{qps:>10,.1f}", "completed"),
            ),
        )
        self._row(
            out,
            (
                ("issued", f"{issued:>10,}", "issued"),
                ("dropped frames", f"{dropped:>10,}", drop_style),
                ("tok/s", f"{tps:>10,.1f}", ""),
            ),
        )
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

    def _stat(self, key: str, frozen_stats: dict[str, Stats] | None) -> Stats:
        """Stats for ``key`` from the frozen snapshot when the run has ended
        (post-PERF_END), else live from ``_metrics``. Threading ``frozen_stats``
        through every panel keeps the verdict + cause tree consistent with the
        ``[frozen]`` lifecycle table instead of drifting on straggler frames."""
        if frozen_stats is not None:
            return frozen_stats.get(key, Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0))
        return _stats(self._metrics[key])

    def _render_lifecycle(
        self, out: Text, frozen_stats: dict[str, Stats] | None = None
    ) -> None:
        def _get(key: str) -> Stats:
            return self._stat(key, frozen_stats)

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
            f"  {'stage':<{LABEL_W}}{_DIST_NUM_HDR}{'%E2E':>9}\n",
            style="label",
        )
        stage_data: list[tuple[str, float]] = []
        for side, label, key in layout:
            s = _get(key)
            sub_rows: list[tuple[str, Stats]] = []
            if key == "backpressure":
                sub_rows = [
                    (lbl, ss)
                    for lbl, k in self._BACKPRESSURE_SUB_ROWS
                    if (ss := _get(k)).n > 0
                ]
            # When the breakdown renders, the parent's %E2E cell stays blank —
            # the share lives in the sub-rows instead of being double-shown.
            self._render_row_stats(out, side, label, s, e2e_avg, pct_cell=not sub_rows)
            pct = min(100.0, 100.0 * s.avg / e2e_avg) if e2e_avg and s.avg else 0.0
            # Backpressure (issue→conn-acquired) gets its own bar colour + key:
            # it's part of E2E but a distinct cause, not generic client work.
            bar_side = _SIDE_BACKPRESSURE if key == "backpressure" else side
            stage_data.append((bar_side, pct))
            for i, (lbl, ss) in enumerate(sub_rows):
                conn = "└" if i == len(sub_rows) - 1 else "├"
                spct = (
                    min(100.0, 100.0 * ss.avg / e2e_avg) if e2e_avg and ss.avg else 0.0
                )
                Dashboard._append_label(
                    out, f"   {conn} {lbl}"[:LABEL_W], LABEL_W, "client_row", "  "
                )
                out.append(_fmt_row(ss), style="client_row")
                out.append(f"{spct:>8.1f}%\n", style=_heat(spct) or "client_row")
        e2e_stats = _get("e2e")
        self._render_summary_stats(
            out,
            "E2E TOTAL  issue -> complete",
            e2e_stats,
            e2e_avg,
            bold=True,
        )
        out.append("\n")
        self._render_verdict(out, e2e_avg, frozen_stats)
        self._render_timeline(out, stage_data, frozen_stats)

    # Render-only breakdown of the backpressure row from events already on
    # the wire: inbox/pickup wait (ISSUED→WORKER_RECEIVED) vs encode + pool
    # acquire (WORKER_RECEIVED→CONN_ACQUIRED, fused — the encode/acquire
    # boundary has no event). Sub-rows hide when their frames never folded;
    # the parent row then keeps its own %E2E cell.
    _BACKPRESSURE_SUB_ROWS: tuple[tuple[str, str], ...] = (
        ("issue -> worker_pickup", "ipc_wait"),
        ("worker_pickup -> encode + conn acquired", "pool_wait"),
    )

    def _render_row_stats(
        self,
        out: Text,
        side: str,
        label: str,
        s: Stats,
        e2e_avg: float,
        *,
        pct_cell: bool = True,
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
        out.append(_fmt_row(s), style=side_style)
        if not pct_cell:
            out.append("\n")
            return
        # %E2E cell heat-graded so the hot stages pop (green/orange/red).
        out.append(f"{pct:>8.1f}%\n", style=_heat(pct) or side_style)

    @staticmethod
    def _append_label(
        out: Text, text: str, width: int, base_style: str, leading: str = ""
    ) -> None:
        """Append a stage label with lifecycle endpoints (issue/complete)
        and IPC hops highlighted. Pads to ``width`` so columns align."""
        out.append(leading)
        consumed = 0
        for chunk, tok_style in _split_label_tokens(text):
            out.append(chunk, style=tok_style or base_style)
            consumed += len(chunk)
        pad = width - consumed
        if pad > 0:
            out.append(" " * pad, style=base_style)

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
        out.append(_fmt_row(s), style=style)
        out.append(f"{pct:>8.1f}%\n", style=style)

    def _render_verdict(
        self, out: Text, e2e_avg: float, frozen_stats: dict[str, Stats] | None = None
    ) -> None:
        # Two plain rows; no section headers, no dividers, no footer.
        # Same 3-column grid as the header so everything anchors.
        # client_work is folded per request as (pre + post), so its stats are
        # a real per-request distribution rather than a sum of two series.
        client = self._stat("client_work", frozen_stats).avg
        server = self._stat("server_http", frozen_stats).avg
        # "backpressure" = the whole ISSUED→CONN_ACQUIRED wait (inbox queue +
        # pool acquire), the same metric the timeline ▓ segment and the cause
        # tree use, so all three colour off one number.
        queue = self._stat("backpressure", frozen_stats).avg
        # Clamp at 100: each bucket is a sub-interval of E2E. The raw ratio
        # can top 100% because each avg is over a different folded
        # population (different N when intermediate frames drop).
        c_pct = min(100.0, 100.0 * client / e2e_avg) if e2e_avg else 0.0
        s_pct = min(100.0, 100.0 * server / e2e_avg) if e2e_avg else 0.0
        q_pct = min(100.0, 100.0 * queue / e2e_avg) if e2e_avg else 0.0
        # Same shared heat scale (warn ≥15%, crit ≥25%). server work is left
        # uncolored — a high server share is healthy (low client overhead), not
        # a problem.
        c_style = _heat(c_pct)
        q_style = _heat(q_pct)

        # Same 3-col grid as _row. The backpressure cell carries a plain
        # "[workers busy]" tag inside its label (same dim style as the other
        # labels — it names the cause tree rendered beside the timeline below),
        # with the value right-aligned to the column edge so all three line up.
        out.append("  ")
        for i, (label, value, style) in enumerate(
            (
                ("client work", f"{c_pct:>9.1f}%", c_style),
                ("server work", f"{s_pct:>9.1f}%", ""),
                ("backpressure [workers busy]", f"{q_pct:>9.1f}%", q_style),
            )
        ):
            if i > 0:
                out.append(" " * 3)
            pad = max(1, 40 - len(label) - 1 - len(value))
            out.append(f"{label} ", style="rule")
            out.append(" " * pad)
            out.append(value, style=style)
        out.append("\n")

    # Worker-loop phases that occupy the event loop; when they pile up the loop
    # is slow to return to requests.recv() and queued queries wait in the ZMQ
    # inbox — exactly what backpressure (ISSUED→pickup) measures.
    _BACKPRESSURE_PHASES = (
        "encode",
        "tcp-acquire",
        "sse-decode",
        "final-decode",
        "complete-ipc",
    )

    # Each phase maps to the lifecycle stage it occurs in; the leaf takes that
    # stage's %E2E-column colour so the tree matches the REQUEST LIFECYCLE table
    # above. ("tail" resolves to tail_stream / tail_offline by mode.)
    _PHASE_STAGE: dict[str, str] = {
        "encode": "pool_wait",
        "tcp-acquire": "pool_wait",
        "sse-decode": "stream_gen",
        "final-decode": "tail",
        "complete-ipc": "tail",
    }

    def _backpressure_cause_lines(
        self, frozen_stats: dict[str, Stats] | None = None
    ) -> list[list[tuple[str, str]]]:
        """Reverse cause tree (right-aligned spine ``─┤`` / ``─┘``, dropping from
        the verdict's backpressure value above), one leaf per worker-loop phase.
        Every leaf starts at the muted base colour and takes its lifecycle
        stage's heat (warn ≥ 15% / critical ≥ 25% of E2E), so it lights up in
        step with that stage's %E2E cell in the table above."""
        e2e = self._stat("e2e", frozen_stats).avg or 1.0
        tail = (
            "tail_stream"
            if self._stat("stream_gen", frozen_stats).n > 0
            else "tail_offline"
        )
        lines: list[list[tuple[str, str]]] = []
        last = len(self._BACKPRESSURE_PHASES) - 1
        for i, phase in enumerate(self._BACKPRESSURE_PHASES):
            conn = " ─┘" if i == last else " ─┤"
            stage_key = self._PHASE_STAGE[phase]
            if stage_key == "tail":
                stage_key = tail
            pct = min(100.0, 100.0 * self._stat(stage_key, frozen_stats).avg / e2e)
            lines.append([(phase, _heat(pct) or "muted"), (conn, "rule")])
        return lines

    _TIMELINE_W = 80  # stacked-bar width in columns
    _TIMELINE_LABEL_W = 18

    def _render_timeline(
        self,
        out: Text,
        stage_data: list[tuple[str, float]],
        frozen_stats: dict[str, Stats] | None = None,
    ) -> None:
        """One stacked bar of where E2E goes, segmented by stage and colored
        █ server / ▒ client / ▓ backpressure — the visual companion to the
        verdict line directly above it.

        The bar is always exactly ``_TIMELINE_W`` columns wide: segments are
        allocated by cumulative-boundary rounding (each stage's share of the
        summed stage time), so the sum is invariant and the bar never grows
        or shrinks between frames as the percentages drift. Sub-column stages
        round to width 0 and drop out rather than padding the total.

        Segments use only the legend-key colors (▒ client / ▓ backpressure / █
        server) — never heat red. Severity lives in the verdict + cause tree.
        """
        w = self._TIMELINE_W
        total = sum(pct for _side, pct in stage_data)
        tree = self._backpressure_cause_lines(frozen_stats)
        bar_end = 2 + self._TIMELINE_LABEL_W + 1 + w + 1  # col after closing │
        legend_end = (
            2 + self._TIMELINE_LABEL_W + len("▒ client   ▓ backpressure   █ server")
        )
        verdict_right = 2 + 3 * 40 + 2 * 3  # backpressure value's right edge
        for row in range(max(len(tree), 2)):
            col = 0
            if row == 0:
                out.append(f"  {'e2e':<{self._TIMELINE_LABEL_W}}", style="label")
                out.append("│", style="rule")
                if total <= 0:
                    out.append("░" * w, style="rule")
                else:
                    prev = 0
                    cum = 0.0
                    for side, pct in stage_data:
                        cum += pct
                        colpx = round(cum / total * w)
                        seg = colpx - prev
                        prev = colpx
                        if seg <= 0:
                            continue
                        if side == _SIDE_SERVER:
                            glyph, gstyle = "█", "server_row"
                        elif side == _SIDE_BACKPRESSURE:
                            glyph, gstyle = "▓", "warn"  # fixed legend-key color
                        else:
                            glyph, gstyle = "▒", "client_row"
                        out.append(glyph * seg, style=gstyle)
                out.append("│", style="rule")
                col = bar_end
            elif row == 1:
                out.append(f"  {'':<{self._TIMELINE_LABEL_W}}", style="label")
                out.append("▒ client", style="client_row")
                out.append("   ", style="label")
                out.append("▓ backpressure", style="warn")
                out.append("   ", style="label")
                out.append("█ server", style="server_row")
                col = legend_end
            # Reverse cause tree beside the bar: right-align each line so its
            # spine drops under the backpressure value in the verdict above.
            if row < len(tree):
                line_w = sum(len(t) for t, _ in tree[row])
                out.append(" " * max(1, verdict_right - line_w - col))
                for text, style in tree[row]:
                    out.append(text, style=style)
            out.append("\n")

    # Loadgen latency rows, top-to-bottom: ttft, tpot, then e2e last.
    _LOADGEN_LATENCIES: tuple[tuple[str, str], ...] = (
        ("ttft", "ttft_ns"),
        ("tpot", "tpot_ns"),
        ("e2e", "sample_latency_ns"),
    )
    # Reason shown when a latency series is empty, so the row reads as
    # "no data, here's why" rather than a broken panel.
    _LATENCY_EMPTY_HINT: dict[str, str] = {
        "ttft": "no first-token timings yet",
        "tpot": "streaming-only; no streamed output tokens to time",
        "e2e": "no completed samples yet",
    }

    def _render_loadgen(self, out: Text) -> None:
        """Authoritative loadgen panel: issued/completed counts + rates and
        the ttft/tpot/e2e latency table. Drop-immune (the trace's own counts
        undercount under FIFO drops) and reported over the perf window
        (tracked_*) so the counts, the per-second rates, and the latency table
        all share one window — and agree with the benchmark's headline
        throughput. Skipped until a snapshot lands.
        """
        snap = self._loadgen_snapshot
        if snap is None:
            return
        c = {
            m.get("name"): m.get("value")
            for m in (snap.get("metrics") or ())
            if m.get("type") == "counter"
        }
        # Perf-window (tracked_*) throughout: counts, rates, and the latency
        # table below all share the tracked window, so issued/completed match
        # their per-second rates instead of dividing an all-run total count by
        # the perf-window duration (which over-stated issued/s under --warmup).
        issued = int(c.get("tracked_samples_issued") or 0)
        completed = int(c.get("tracked_samples_completed") or 0)
        failed = int(c.get("tracked_samples_failed") or 0)
        dur_ns = int(c.get("tracked_duration_ns") or 0) or int(
            c.get("total_duration_ns") or 0
        )
        dur_s = dur_ns / 1e9 if dur_ns > 0 else 0.0
        issued_s = (issued / dur_s) if dur_s else 0.0
        completed_s = (completed / dur_s) if dur_s else 0.0
        tok_s = (
            (float(self._loadgen_series("osl").get("total") or 0.0) / dur_s)
            if dur_s
            else 0.0
        )

        # Freshness tag. has_terminal → authoritative "(final)". After the run
        # ends (PERF_END) the aggregator is still computing exact final stats
        # in its own process, so a done-but-not-yet-terminal panel reads as
        # "finalizing…"; it becomes an explicit failure only once the CLI's
        # end-of-run wait elapses without a terminal snapshot. A still-live run
        # with a lagging feed shows its plain snapshot age.
        if self.has_terminal_loadgen:
            age_tag = "  (final)"
        elif self.is_done:
            if self._final_unavailable:
                age_tag = "  (final snapshot unavailable — aggregator did not finalize)"
            else:
                age_s = (
                    (time.monotonic_ns() - self._done_ns) / 1e9
                    if self._done_ns
                    else 0.0
                )
                age_tag = f"  (finalizing… {age_s:.0f}s)"
        else:
            age_s = (time.monotonic_ns() - self._loadgen_snapshot_ts) / 1e9
            age_tag = f"  (snapshot {age_s:.0f}s old)" if age_s > 2.0 else ""
        out.append(f"  LOADGEN{age_tag}\n", style="section")
        out.append("─" * WIDTH + "\n", style="rule")
        self._row(
            out,
            (
                ("issued", f"{issued:>10,}", "issued"),
                ("completed", f"{completed:>10,}", "completed"),
                ("errors", f"{failed:>10,}", "critical" if failed else ""),
            ),
        )
        self._row(
            out,
            (
                ("issued/s", f"{issued_s:>10,.1f}", "issued"),
                ("completed/s", f"{completed_s:>10,.1f}", "completed"),
                ("tok/s", f"{tok_s:>10,.1f}", ""),
            ),
        )
        out.append("\n")
        out.append(f"  {'latency (ms)':<{LABEL_W}}{_DIST_NUM_HDR}\n", style="label")
        for label, name in self._LOADGEN_LATENCIES:
            s = _series_stats(self._loadgen_series(name))
            if s.n <= 0:
                # Always show the row (incl. tpot, which is streaming-only)
                # but say *why* it is empty — a bare "—" reads as a broken
                # panel. TPOT in particular needs streamed output tokens, so
                # an empty tpot tracks an empty tok/s (no tokens to time).
                out.append(f"  {label:<{LABEL_W}}", style="default")
                out.append(f"{'—':>12}", style="default")
                out.append(
                    f"   ({self._LATENCY_EMPTY_HINT.get(label, 'no data')})\n",
                    style="default",
                )
                continue
            out.append(f"  {label:<{LABEL_W}}", style="default")
            out.append(_fmt_row(s), style="default")
            out.append("\n")

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
        fleet_p99_ms = statistics.median(all_p99s) / 1e6 if all_p99s else 0.0
        n_hot = sum(1 for p in all_p99s if p >= _HOT_THRESH_NS)
        n_workers = len(all_worker_stats)
        hot_style = "critical" if n_hot > n_workers // 2 else ("warn" if n_hot else "")
        out.append(
            f"  fleet p99 {fleet_p99_ms:.2f} ms   hot workers (p99 ≥ 5 ms)  ",
            style="label",
        )
        out.append(f"{n_hot}/{n_workers}", style=hot_style or "label")
        if self._tcp_established >= 0:
            out.append("   tcp conns  ", style="label")
            out.append(f"{self._tcp_established:,}", style="value")
        out.append("\n")

        def _emit(label: str, s: Stats, *, highlight: bool = False) -> None:
            mx_ms = s.max / 1e6
            p99_ms = s.p99 / 1e6
            mx_style = "critical" if mx_ms > 50 else ("warn" if mx_ms > 10 else "")
            p99_style = "critical" if p99_ms > 10 else ("warn" if p99_ms > 1 else "")
            row_style = "summary" if highlight else ""
            # Same geometry as _fmt_row (12 / 11×5) so the columns line up
            # with the lifecycle + loadgen tables; p99/max stay individually
            # heat-graded, so we can't route through _fmt_row wholesale.
            out.append(f"  {label:<{LABEL_W}}", style=row_style)
            out.append(f"{s.n:>12,}", style=row_style)
            out.append(f"{s.avg / 1e6:>11.2f}", style=row_style)
            out.append(f"{s.min / 1e6:>11.2f}", style=row_style)
            out.append(f"{s.p50 / 1e6:>11.2f}", style=row_style)
            out.append(f"{p99_ms:>11.2f}", style=p99_style or row_style)
            out.append(f"{mx_ms:>11.2f}\n", style=mx_style or row_style)

        out.append(f"  {'worker':<{LABEL_W}}{_DIST_NUM_HDR}\n", style="label")

        # main always first, always highlighted
        if main_entry is not None:
            _emit("main", _stats(main_entry), highlight=True)

        for wid, s in workers:
            _emit(f"w{wid}", s)

        # Subtract main only when it actually has a LOOP_LAG entry; otherwise
        # the "not shown" count under-reports by one.
        omitted = max(
            0,
            len(self._loop_lag)
            - (1 if main_entry is not None else 0)
            - self._LAG_TOP_N,
        )
        if omitted:
            out.append(
                f"  … {omitted} worker(s) with lower max lag not shown\n",
                style="muted",
            )
