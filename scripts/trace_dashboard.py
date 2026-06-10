#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for the -vvv trace dashboard.

Reads fixed-size binary frames from the FIFO opened by
:func:`inference_endpoint.utils.trace.bootstrap` and renders the
dashboard via ``rich.Live``. Dashboard aggregation lives in
:mod:`inference_endpoint.utils.trace_dashboard` so it can be unit
tested without standing up a TUI.

Linux only: timestamps are compared across processes and rely on
``CLOCK_MONOTONIC`` being system-wide (per ``man 7 time``).
"""

# ruff: noqa: I001
# The pre-commit ruff hook is pinned to v0.3.3 (see
# .pre-commit-config.yaml's "TODO: sync rev with ruff version"), which
# does not auto-detect `inference_endpoint` as a first-party package
# and therefore disagrees with the project's local ruff (v0.15.8) on
# import order in this file. File-level noqa keeps both versions quiet
# until the rev is synced.
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import threading
import time

import zmq
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshotCodec,
    snapshot_to_dict,
)
from inference_endpoint.core.record import BATCH_TOPIC, TOPIC_FRAME_SIZE
from inference_endpoint.utils.trace import (
    _F_SETPIPE_SZ,
    _KERNEL_PIPE_BUF,
    FRAME_SIZE,
    metrics_addr_path,
    snapshot_sidecar_path,
)
from inference_endpoint.utils.trace_dashboard import (
    DASHBOARD_THEME,
    READ_CHUNK,
    REFRESH_HZ,
    Dashboard,
)
from rich.console import Console
from rich.live import Live


def _try_load_snapshot(path: str) -> dict | None:
    """Best-effort read of the loadgen snapshot sidecar. Returns None
    if the file is missing or transiently mid-rename (atomic write may
    briefly produce a half-rename window that json.load tolerates)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _benchmark_pids(root_pid: int) -> list[int]:
    """The benchmark process tree whose TCP conns we count: the main proc
    plus its direct children (workers, aggregator, event logger), minus
    this dashboard process itself."""
    pids = [root_pid]
    try:
        with open(f"/proc/{root_pid}/task/{root_pid}/children") as f:
            pids += [int(p) for p in f.read().split()]
    except (OSError, ValueError):
        pass  # parent exiting / non-Linux — count what we have
    me = os.getpid()
    return [p for p in pids if p != me]


def _count_established_tcp(pids: list[int]) -> int:
    """ESTABLISHED TCP conns held by ``pids`` via /proc cross-reference:
    each pid's socket-fd inodes ∩ the ESTABLISHED inodes in /proc/net/tcp*.
    Pure observation from this (dashboard) process — the counted processes
    carry no collection logic and take no hot-path cost. Returns -1 when
    /proc/net is unreadable (non-Linux), which hides the dashboard cell."""
    inodes: set[str] = set()
    readable = False
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                next(f, None)  # header row
                for line in f:
                    parts = line.split()
                    # Column 3 is the socket state (01 = ESTABLISHED);
                    # column 9 is the socket inode.
                    if len(parts) > 9 and parts[3] == "01":
                        inodes.add(parts[9])
            readable = True
        except OSError:
            continue  # e.g. ipv6 disabled — the other table may still work
    if not readable:
        return -1
    n = 0
    for pid in pids:
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue  # process exited mid-scan
        for fd in fds:
            try:
                tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue  # fd closed mid-scan
            if tgt.startswith("socket:[") and tgt[8:-1] in inodes:
                n += 1
    return n


# End-of-run exit policy. The authoritative final snapshot (state=="complete")
# is written by the parent's trace.teardown() just before it closes its FIFO
# write fd; every worker closes its write fd at exit. The reader runs until true
# FIFO EOF (all writers closed) with NO time cap, and main() independently stops
# the moment the terminal snapshot lands on the sidecar — so the closing frame
# always reflects the real final report, and a wedged worker that never closes
# the FIFO cannot strand the dashboard (the sidecar terminal still exits it).


class _FrameReader(threading.Thread):
    """Blocking read loop; ingests whole frames into the Dashboard.

    Uses ``select`` with a short timeout so the loop wakes periodically while
    it reads to true EOF — the FIFO may not reach EOF until all 24+ worker
    processes have drained their ZMQ queues and exited, tens of seconds after
    the benchmark has finished.
    """

    def __init__(self, fd: int, dash: Dashboard, *, owns_fd: bool = False) -> None:
        super().__init__(daemon=True, name="trace-reader")
        self._fd = fd
        self._owns_fd = owns_fd
        self._dash = dash
        self._pending = bytearray()
        self._eof = threading.Event()

    @property
    def eof(self) -> bool:
        return self._eof.is_set()

    def run(self) -> None:
        import select

        try:
            while True:
                ready, _, _ = select.select([self._fd], [], [], 0.5)
                if ready:
                    try:
                        chunk = os.read(self._fd, READ_CHUNK)
                    except OSError:
                        return  # FIFO closed/errored under us — stop reading
                    if not chunk:
                        return  # true EOF — all writers closed
                    self._pending.extend(chunk)
                    whole = (len(self._pending) // FRAME_SIZE) * FRAME_SIZE
                    if whole:
                        self._dash.ingest_frames(bytes(self._pending[:whole]))
                        del self._pending[:whole]
                # No idle/time cap: read until true EOF (the parent closes the
                # FIFO in teardown, after the report is built). main()'s loop
                # exits independently on the terminal snapshot, so a wedged
                # worker that never closes the FIFO cannot hang the dashboard.
        finally:
            self._eof.set()
            # Close the FIFO read end we opened (never sys.stdin — owns_fd is
            # False on that path). This thread is the sole reader, so it holds
            # the last reference.
            if self._owns_fd:
                os.close(self._fd)


class _MetricsSubReader(threading.Thread):
    """Opens a SUB straight to the aggregator's metrics PUB and feeds the
    dashboard fresh LOADGEN snapshots.

    The aggregator is a separate process that publishes every tick (and the
    terminal COMPLETE frame), so it is immune to the main benchmark loop's
    saturation — unlike the sidecar, which is written from the main proc's
    starved in-process subscriber. Best-effort: any failure leaves the
    sidecar fallback in place. Sets ``delivered`` once a snapshot lands so
    main() can stop polling the sidecar.
    """

    def __init__(self, addr_path: str, dash: Dashboard) -> None:
        super().__init__(daemon=True, name="metrics-sub")
        self._addr_path = addr_path
        self._dash = dash
        # Not ``_stop``: that shadows threading.Thread._stop and breaks join().
        self._stopping = threading.Event()
        self.delivered = threading.Event()

    def stop(self) -> None:
        self._stopping.set()

    def _wait_for_addr(self) -> str | None:
        # The parent writes the addr only after the aggregator PUB binds,
        # which is well after the dashboard spawns — poll for it.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not self._stopping.is_set():
            try:
                with open(self._addr_path) as f:
                    addr = f.read().strip()
                if addr:
                    return addr
            except OSError:
                pass  # addr file not published yet — retry until the deadline
            time.sleep(0.2)
        return None

    def run(self) -> None:
        addr = self._wait_for_addr()
        if addr is None:
            return  # never published — main() keeps polling the sidecar
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)  # only the freshest snapshot
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt(zmq.LINGER, 0)
        codec = MetricsSnapshotCodec()
        try:
            sock.connect(addr)
            while not self._stopping.is_set():
                try:
                    raw = sock.recv()
                except zmq.Again:
                    continue  # no snapshot within RCVTIMEO — re-check stop, retry
                except zmq.ZMQError:
                    return  # socket closed/terminated — stop the SUB reader
                if raw[:TOPIC_FRAME_SIZE] == BATCH_TOPIC:
                    continue  # metrics snapshots are never batched
                payload = raw[TOPIC_FRAME_SIZE:] if len(raw) > TOPIC_FRAME_SIZE else raw
                try:
                    snap = snapshot_to_dict(codec.decode(payload))
                except Exception:  # noqa: BLE001 — telemetry, never crash
                    continue
                self._dash.attach_loadgen_snapshot(snap)
                self.delivered.set()
        finally:
            sock.close(0)


def _open_trace_input(pipe_path: str | None) -> tuple[int, bool]:
    """Returns (fd, owns_fd). owns_fd is True only when we opened a FIFO and
    are responsible for closing it; a None/empty path yields sys.stdin's
    fileno, which must never be closed."""
    if pipe_path:
        fd = os.open(pipe_path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, _F_SETPIPE_SZ, _KERNEL_PIPE_BUF)
        except OSError:
            pass  # F_SETPIPE_SZ is best-effort — keep the kernel default
        return fd, True
    return sys.stdin.fileno(), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-pipe",
        help="FIFO path to read binary trace frames from (default: stdin).",
    )
    args = parser.parse_args()

    dash = Dashboard()
    # Dashboard renders to stderr because the parent benchmark process
    # has redirected its own stdout/stderr to a log file (see trace.bootstrap).
    console = Console(file=sys.stderr, force_terminal=True, theme=DASHBOARD_THEME)
    trace_fd, owns_trace_fd = _open_trace_input(args.trace_pipe)
    reader = _FrameReader(trace_fd, dash, owns_fd=owns_trace_fd)
    reader.start()

    # Convention paths keyed on the parent's pid (the main proc).
    snap_path = snapshot_sidecar_path(os.getppid())
    # Primary LOADGEN feed: a SUB straight to the aggregator PUB (fresh).
    # Falls back to the main-proc-written sidecar if the SUB never connects.
    sub_reader = _MetricsSubReader(metrics_addr_path(os.getppid()), dash)
    sub_reader.start()

    # screen=True uses the alternate-screen buffer so updates redraw
    # cleanly without scrollback noise. When Live() exits the alt
    # screen is torn down — to keep the final frame visible we capture
    # it BEFORE leaving the context and print it to the normal buffer
    # afterward.
    final_frame = None
    last_tcp_sample = 0.0
    with Live(
        dash.render(),
        console=console,
        refresh_per_second=REFRESH_HZ,
        screen=True,
        transient=False,
    ) as live:
        # Render until the report is final — the terminal loadgen snapshot is
        # in hand (teardown writes it to the sidecar before closing the FIFO)
        # or the FIFO truly closes. NO time cap: the aggregator's end-of-run
        # finalize (exact percentiles + the tokenizer drain) can run far longer
        # than issuance, and we wait for it instead of quitting on a stale frame.
        while not reader.eof and not dash.has_terminal_loadgen:
            # During the run the SUB is the fresh primary; fall back to the
            # sidecar until it delivers. Once the run is winding down (is_done)
            # always poll the sidecar too — the SUB can miss the aggregator's
            # terminal frame, and the sidecar is its reliable carrier.
            if dash.is_done or not sub_reader.delivered.is_set():
                snap = _try_load_snapshot(snap_path)
                if snap is not None:
                    dash.attach_loadgen_snapshot(snap, force=dash.is_done)
            # TCP conn gauge: 1 Hz is plenty for a live gauge, and the /proc
            # scan cost scales with conn count — keep it off the render tick.
            now = time.monotonic()
            if now - last_tcp_sample >= 1.0:
                last_tcp_sample = now
                dash.set_tcp_established(
                    _count_established_tcp(_benchmark_pids(os.getppid()))
                )
            live.update(dash.render())
            time.sleep(1.0 / REFRESH_HZ)
        # Join the SUB before the final render so no late snapshot attaches.
        sub_reader.stop()
        sub_reader.join(timeout=1.0)
        # One last sidecar poll: if the reader hit FIFO EOF between refresh
        # ticks, teardown may have written the terminal snapshot just before
        # closing the FIFO — don't flag "unavailable" without re-checking.
        if not dash.has_terminal_loadgen:
            snap = _try_load_snapshot(snap_path)
            if snap is not None:
                dash.attach_loadgen_snapshot(snap, force=True)
        # Still no terminal snapshot → the FIFO closed before teardown wrote it
        # (main process SIGKILLed / OOM). The only genuine "did not finalize"
        # case; otherwise the panel reads "(final)".
        if not dash.has_terminal_loadgen:
            dash.mark_final_unavailable()
        # Bypass the per-tick fold-defer window so the final render
        # captures COMPLETE frames that landed within the last 300 ms
        # — otherwise they sit queued and the closing frame shows
        # stale stage histograms / verdict.
        dash.flush_pending_folds()
        final_frame = dash.render()
        live.update(final_frame)
    # Now we're back on the normal screen — print the last snapshot
    # so the user can see the totals + verdict after the run ends.
    if final_frame is not None:
        console.print(final_frame)
        console.print("[dim]── trace finished ──[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
