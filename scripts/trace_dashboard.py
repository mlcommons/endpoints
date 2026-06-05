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

from inference_endpoint.utils.trace import FRAME_SIZE, snapshot_sidecar_path
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


# End-of-run exit policy for the reader thread.
#
# The authoritative final snapshot (state=="complete") is written by the
# parent's trace.teardown() *just before* it closes its FIFO write fd, and
# every worker closes its write fd at process exit. So true FIFO EOF (all
# writers closed) is the guaranteed signal that the complete snapshot is
# already on disk — preferring EOF makes the FINAL panel deterministic.
#
# But workers can take tens of seconds to drain their ZMQ queues before
# exiting, so we can't wait for EOF unconditionally or the dashboard hangs
# long past the end of the run. Compromise: once lifecycle frames go quiet,
# keep reading — letting true EOF trigger an immediate clean exit — until
# this generous cap bounds a wedged-worker hang. After the reader exits,
# main() polls the sidecar for the complete snapshot.
_IDLE_EXIT_CAP_S = 30.0


class _FrameReader(threading.Thread):
    """Blocking read loop; ingests whole frames into the Dashboard.

    Uses ``select`` with a short timeout so the thread can check for an
    idle-exit condition — the FIFO may not reach EOF until all 24+ worker
    processes have drained their ZMQ queues and exited, which can be tens
    of seconds after the benchmark has finished.
    """

    def __init__(self, fd: int, dash: Dashboard) -> None:
        super().__init__(daemon=True, name="trace-reader")
        self._fd = fd
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
                        return
                    if not chunk:
                        return  # true EOF — all writers closed
                    self._pending.extend(chunk)
                    whole = (len(self._pending) // FRAME_SIZE) * FRAME_SIZE
                    if whole:
                        self._dash.ingest_frames(bytes(self._pending[:whole]))
                        del self._pending[:whole]
                # Check lifecycle idle time regardless of whether LOOP_LAG
                # frames are still arriving — workers emit LOOP_LAG every
                # 0.3 s even after the run ends, so frame-arrival time is
                # not a reliable proxy for end-of-run. We keep reading so
                # true EOF (the parent's post-teardown fd close) wins and
                # guarantees the complete snapshot is on disk; this cap only
                # bounds a wedged-worker hang where EOF never arrives.
                d = self._dash
                if (d.is_done or d.is_tail) and d.lifecycle_idle_s >= _IDLE_EXIT_CAP_S:
                    return
        finally:
            self._eof.set()


# Match the producers' write-fd request in trace.py. Capped by
# /proc/sys/fs/pipe-max-size; request fails above that → default kept.
_F_SETPIPE_SZ = getattr(fcntl, "F_SETPIPE_SZ", 1031)
_KERNEL_PIPE_BUF = 64 * 1024 * 1024


def _open_trace_input(pipe_path: str | None) -> int:
    if pipe_path:
        fd = os.open(pipe_path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, _F_SETPIPE_SZ, _KERNEL_PIPE_BUF)
        except OSError:
            pass
        return fd
    return sys.stdin.fileno()


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
    reader = _FrameReader(_open_trace_input(args.trace_pipe), dash)
    reader.start()

    # Snapshot sidecar path is convention-named after the parent's pid
    # (main proc that spawned us). The benchmark writes it periodically
    # under -vvv so we can live-update the LOADGEN vs TRACE panel.
    snap_path = snapshot_sidecar_path(os.getppid())

    # screen=True uses the alternate-screen buffer so updates redraw
    # cleanly without scrollback noise. When Live() exits the alt
    # screen is torn down — to keep the final frame visible we capture
    # it BEFORE leaving the context and print it to the normal buffer
    # afterward.
    final_frame = None
    with Live(
        dash.render(),
        console=console,
        refresh_per_second=REFRESH_HZ,
        screen=True,
        transient=False,
    ) as live:
        while not reader.eof:
            snap = _try_load_snapshot(snap_path)
            if snap is not None:
                dash.attach_loadgen_snapshot(snap)
            live.update(dash.render())
            time.sleep(1.0 / REFRESH_HZ)
        # After reader exits (FIFO EOF or idle timeout), the main
        # benchmark process may still be finalizing the metrics
        # aggregator. Poll the sidecar for up to _FINAL_SNAP_WAIT_S
        # seconds until we see a snapshot with state=="complete",
        # then use it for the final comparison panel.
        _FINAL_SNAP_WAIT_S = 20.0
        deadline = time.monotonic() + _FINAL_SNAP_WAIT_S
        best_snap: dict | None = None
        while time.monotonic() < deadline:
            s = _try_load_snapshot(snap_path)
            if s is not None:
                best_snap = s
                if s.get("state") in ("complete", "COMPLETE"):
                    break
            time.sleep(0.2)
        if best_snap is not None:
            dash.attach_loadgen_snapshot(best_snap, force=True)
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
