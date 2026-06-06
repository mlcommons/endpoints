# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary trace event channel for ``-vvv`` runs.

Design
------
Each process (main + each worker) packs 17-byte events into a 1 MiB
ring and flushes once per asyncio tick via :func:`emit_loop_lag`.
Producer and flusher share the loop thread, so the emit hot path is
lock-free pack-and-return.

Transport: a POSIX named pipe (``mkfifo``) at
``/tmp/endpoints_trace_<main_pid>/fifo`` (per-pid 0o700 dir;
``/tmp`` is shared with snapshot.json and logs.txt — ``/dev/shm`` is
reserved for execute.py's bulk artifacts). The ``scripts/trace_dashboard.py``
subprocess opens the read end; main + every worker open the write
end. FIFO over alternatives because:

* PIPE_BUF (4096 B) write atomicity — concurrent writers from N
  processes can't interleave inside an event since each ≤ 4080 B
  chunk is atomic. Unix SOCK_STREAM gives no such guarantee.
* Filesystem path = cross-process discovery without env vars or a
  listen/accept rendezvous; workers just open the path.
* Kernel-buffered (``F_SETPIPE_SZ`` to 1 MiB) so the dashboard can
  block briefly without dropping data.
* ``O_NONBLOCK`` + ``EAGAIN`` is the backpressure signal; we
  account dropped bytes and emit ``Event.TRACE_DROPS`` rather than
  stalling the loop.
* Reader EOF on parent exit is a natural shutdown signal.

Wire format (17 B / event): ``<BQQ`` = (event:u8, sid:u64, ts:u64).
``sid`` is the first 16 hex chars of the request UUID for lifecycle
events, or ``(worker_id << 56) | payload`` for loop-lag / drops.
"""

from __future__ import annotations

import asyncio
import enum
import errno
import fcntl
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "Event",
    "FRAME_SIZE",
    "MAIN_PROC_LOOP_ID",
    "PACKER",
    "bootstrap",
    "cleanup",
    "emit_loop_lag",
    "emit_trace",
    "emit_trace_id",
    "enable_tracing",
    "fifo_path",
    "is_enabled",
    "snapshot_sidecar_path",
    "start_lag_task",
    "start_snapshot_tap",
    "teardown",
]


# Sentinel worker_id used by the main process when spawning emit_loop_lag.
# Workers use 0..N-1; main is rendered as a distinct row in the dashboard.
MAIN_PROC_LOOP_ID = 255


class Event(enum.IntEnum):
    """All known trace events. Adding one = one line here + a row in the
    dashboard's :data:`_EV_NAMES`. The integer value goes on the wire,
    so changing existing numbers will break the dashboard — only ever
    append new ones with the next free value.
    """

    ISSUED = 1
    WORKER_RECEIVED = 2
    CONN_ACQUIRED = 3
    WRITTEN = 4
    RESPONSE_HEADERS = 5
    RESPONSE_BYTES = 6
    RECV_FIRST = 7
    MAIN_RECEIVED = 8
    COMPLETE = 9
    LOOP_LAG = 10
    # TRACE_DROPS sid encodes (proc_id << 56) | cumulative_dropped_bytes;
    # re-emitted by emit_loop_lag every tick while the counter is nonzero.
    TRACE_DROPS = 11
    # Emitted by main proc when the performance phase starts (after
    # warmup). Dashboard resets its in-flight counters + metrics so
    # the LOADGEN vs TRACE comparison only sees the same tracked
    # window the loadgen aggregator uses. sid=0 (no per-request id).
    PERF_START = 12
    # Worker: last body byte received. Streaming → end of the SSE stream
    # (the last chunk); non-streaming does NOT emit it (RESPONSE_BYTES is
    # already the full body). Lets the dashboard split server token-gen
    # (1st→last chunk) from the client tail (last chunk→complete).
    RESPONSE_DONE = 13


PACKER = struct.Struct("<BQQ")
FRAME_SIZE = PACKER.size  # 17


def fifo_path(pid: int) -> str:
    """Convention path for the trace FIFO (per-pid 0o700 dir)."""
    return f"/tmp/endpoints_trace_{pid}/fifo"


def snapshot_sidecar_path(pid: int) -> str:
    """Convention path for the live MetricsSnapshot JSON sidecar."""
    return f"/tmp/endpoints_trace_{pid}/snapshot.json"


# Buffer budget ≈ 4 GiB cap: 64 MiB ring × (1 main + N workers) + 64 MiB
# FIFO. Stays under 4 GiB to ~62 processes; lower _BUF_CAPACITY beyond.
# SPSC ring (producer + emit_loop_lag share the loop, no lock). 64 MiB ≈
# 3.9M frames/tick, so ring-overflow drops are effectively impossible.
_BUF_CAPACITY = 64 * 1024 * 1024  # 64 MiB ≈ 3.9M frames per 0.3 s tick
_WRITE_CHUNK = 4080  # FRAME-aligned ≤ PIPE_BUF so writes are atomic
# Best-effort from both ends. Capped by /proc/sys/fs/pipe-max-size
# (1 MiB unprivileged); request fails above that → kernel default kept.
_KERNEL_PIPE_BUF = 64 * 1024 * 1024  # 64 MiB best-effort F_SETPIPE_SZ
_F_SETPIPE_SZ = getattr(fcntl, "F_SETPIPE_SZ", 1031)  # Linux-only
_DASHBOARD_READY_S = 0.5  # grace after spawn before opening FIFO


def _sid_from_uuid(req_id: str) -> int:
    """Deterministic 64-bit sid from a UUID hex string.

    ``hash(str)`` is per-process randomised (PEP 456) and would break
    cross-process correlation.
    """
    return int(req_id[:16], 16)


class _TraceEmitter:
    """Buffered binary trace emitter. SPSC; one per process."""

    __slots__ = ("_buf", "_dead", "_dropped_bytes", "_fd", "_offset")

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buf = bytearray(_BUF_CAPACITY)
        self._offset = 0
        self._dead = False
        self._dropped_bytes = 0

    def emit(self, event: int, sid: int) -> None:
        if self._dead:
            return
        o = self._offset
        if o + FRAME_SIZE > _BUF_CAPACITY:
            # Buffer full this cycle; account the drop.
            self._dropped_bytes += FRAME_SIZE
            return
        PACKER.pack_into(self._buf, o, event, sid, time.monotonic_ns())
        self._offset = o + FRAME_SIZE

    def flush(self) -> None:
        if self._dead:
            return
        end = self._offset
        if end == 0:
            return
        self._offset = 0
        view = memoryview(self._buf)
        pos = 0
        while pos < end:
            n = min(_WRITE_CHUNK, end - pos)
            try:
                os.write(self._fd, view[pos : pos + n])
            except OSError as e:
                if e.errno == errno.EAGAIN:
                    # Reader is behind; PIPE_BUF atomicity means none
                    # of this chunk landed. Drop the rest, don't block.
                    self._dropped_bytes += end - pos
                    return
                self._die()
                return
            pos += n

    def dropped_bytes(self) -> int:
        """Cumulative dropped bytes since process start (never reset).

        emit_loop_lag re-emits this running total every tick, so a
        TRACE_DROPS frame that is itself dropped (ring/pipe full) is
        corrected on the next tick — the count is never lost.
        """
        return self._dropped_bytes

    def _die(self) -> None:
        global emit_trace, _active_emitter
        self._dead = True
        emit_trace = _noop
        _active_emitter = None  # is_enabled() → False after pipe death
        try:
            os.close(self._fd)
        except OSError:
            pass  # already closed (double-_die, BrokenPipe auto-close)


def _noop(event: int, sid: int) -> None:  # noqa: ARG001
    pass


def is_enabled() -> bool:
    """True after :func:`enable_tracing` has been called this process."""
    return _active_emitter is not None


# Default (disabled) bindings. Replaced by enable_tracing() under -vvv.
# Hot path is a single bound-method call: pack_into + offset bump, no
# branching past the dead check.
emit_trace = _noop

# Live emitter reference used by emit_loop_lag to drive periodic flushes
# on the same loop thread that owns emit_trace. None when tracing is off.
_active_emitter: _TraceEmitter | None = None


def emit_trace_id(event: int, req_id: str) -> None:
    """Emit by request UUID. No-op guard makes non-hex ids safe when
    tracing is off (existing tests pass ids like ``"q-1"``)."""
    if emit_trace is _noop:
        return
    emit_trace(event, _sid_from_uuid(req_id))


async def emit_loop_lag(worker_id: int, period_s: float = 0.3) -> None:
    """Per-process tick: emit loop-lag + drops, then drain the ring.

    Spawn once per process after :func:`enable_tracing`. Runs on the
    same loop as every ``emit_trace`` call, which is what makes the
    SPSC ring lock-free.
    """
    sid_high = (worker_id & 0xFF) << 56
    target_ns = int(period_s * 1e9)
    mask_low = (1 << 56) - 1
    while True:
        try:
            t0 = time.monotonic_ns()
            await asyncio.sleep(period_s)
            lag = (time.monotonic_ns() - t0) - target_ns
            if lag < 0:
                lag = 0
            emit_trace(Event.LOOP_LAG, sid_high | (lag & mask_low))
            if _active_emitter is not None:
                # Cumulative total, re-emitted every tick — self-heals if
                # this frame is dropped (see _TraceEmitter.dropped_bytes).
                dropped = _active_emitter.dropped_bytes()
                if dropped:
                    emit_trace(Event.TRACE_DROPS, sid_high | (dropped & mask_low))
                _active_emitter.flush()
        except asyncio.CancelledError:
            # Final tick: report drops accrued since the last loop, then
            # flush, so the closing snapshot's drop count is complete.
            if _active_emitter is not None:
                dropped = _active_emitter.dropped_bytes()
                if dropped:
                    emit_trace(Event.TRACE_DROPS, sid_high | (dropped & mask_low))
                _active_emitter.flush()
            return
        except Exception:
            # Don't let a transient fd error kill the flush task.
            logger.exception("emit_loop_lag tick failed; continuing")


def enable_tracing(pipe_path: str) -> None:
    """Install the emitter; idempotent. No-op if the FIFO is gone
    (dashboard exited between dispatch and open)."""
    global emit_trace
    if emit_trace is not _noop:
        return
    if not os.path.exists(pipe_path):
        return
    # Two-step open: blocking O_WRONLY first to synchronise with the
    # reader (O_NONBLOCK would ENXIO if reader isn't up yet), then
    # flip to non-blocking so subsequent writes never stall the loop.
    try:
        fd = os.open(pipe_path, os.O_WRONLY)
    except OSError:
        return  # pipe gone (dashboard exited before we opened)
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except OSError:
        pass  # rare; non-blocking is best-effort
    try:
        fcntl.fcntl(fd, _F_SETPIPE_SZ, _KERNEL_PIPE_BUF)
    except OSError:
        # F_SETPIPE_SZ requires CAP_SYS_RESOURCE above the kernel
        # default, or is unavailable on the running kernel — either
        # way the smaller default pipe buffer is fine, the user-space
        # ring buffer absorbs short spikes on its own.
        pass
    global _active_emitter
    emitter = _TraceEmitter(fd)
    emit_trace = emitter.emit
    _active_emitter = emitter


def bootstrap(verbose: int) -> str:
    """Resolve verbose count to a log-level. On ``-vvv`` create the
    FIFO, spawn the dashboard, redirect this process's stdout/stderr
    to ``logs.txt`` in the same per-pid dir, and install the emitter.

    Caller MUST invoke :func:`teardown` once the run is done — there
    is no atexit hook. Leaving cleanup explicit keeps shutdown order
    deterministic and avoids the "atexit fired during interpreter
    teardown" failure mode where ``os.write``/``os.unlink`` race
    finalised globals.
    """
    if verbose < 2:
        return "INFO"
    if verbose == 2:
        return "DEBUG"

    # PID reuse on long-lived hosts can leave a stale dir; Linux PIDs
    # are unique at any instant so an existing dir at our own PID is
    # always a dead previous occupant — wipe and recreate.
    path = fifo_path(os.getpid())
    trace_dir = os.path.dirname(path)
    try:
        os.mkdir(trace_dir, 0o700)
    except FileExistsError:
        shutil.rmtree(trace_dir, ignore_errors=True)
        try:
            os.mkdir(trace_dir, 0o700)
        except FileExistsError:
            # rmtree + mkdir lost a race; refuse rather than spin.
            raise RuntimeError(
                f"trace dir {trace_dir} reappeared after cleanup"
            ) from None
    os.mkfifo(path, 0o600)
    _state.fifo_path = path

    # Spawn dashboard BEFORE the stdout/stderr redirect so it inherits
    # the original terminal fds.
    dashboard_proc = _spawn_dashboard(path)
    if dashboard_proc is None:
        # Wheel install without scripts/. Don't enable_tracing — its
        # blocking O_WRONLY would deadlock with no reader.
        sys.stderr.write(
            "trace: scripts/trace_dashboard.py not found — open the FIFO "
            f"manually: 'python scripts/trace_dashboard.py --trace-pipe {path}'\n"
        )
        return "TRACE"

    # Grace window so a dashboard import error surfaces here instead
    # of deadlocking the blocking O_WRONLY below.
    time.sleep(_DASHBOARD_READY_S)
    if dashboard_proc.poll() is not None:
        sys.stderr.write(
            f"trace: dashboard exited rc={dashboard_proc.returncode}; "
            f"trace events disabled this run\n"
        )
        return "TRACE"

    # Log file lives inside the per-pid 0o700 dir; O_NOFOLLOW is
    # belt-and-suspenders against a symlink attack on /tmp.
    log_path = os.path.join(trace_dir, "logs.txt")
    orig_stderr_fd = os.dup(2)
    os.write(
        orig_stderr_fd,
        f"trace: dashboard active — logs piped to {log_path}\n".encode(),
    )
    log_fd = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
        0o600,
    )
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    _state.orig_stderr_fd = orig_stderr_fd
    _state.log_path = log_path

    enable_tracing(path)
    return "TRACE"


@dataclass
class _BootstrapState:
    """Bookkeeping for :func:`teardown`. Populated by :func:`bootstrap`."""

    fifo_path: str | None = None
    orig_stderr_fd: int | None = None
    log_path: str | None = None
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    # Daemon threads spawned by start_snapshot_tap. Not asyncio tasks;
    # they exit when _tap_stop is set.
    tap_threads: list[threading.Thread] = field(default_factory=list)
    tap_stop: threading.Event = field(default_factory=threading.Event)


_state = _BootstrapState()

SnapshotProvider = Callable[[], dict | None]


def start_lag_task(loop: asyncio.AbstractEventLoop) -> None:
    """Spawn :func:`emit_loop_lag` for the main proc on ``loop``."""
    if not is_enabled():
        return
    _state.tasks.append(loop.create_task(emit_loop_lag(MAIN_PROC_LOOP_ID)))


def start_snapshot_tap(
    loop: asyncio.AbstractEventLoop,
    provider: SnapshotProvider,
    *,
    period_s: float = 0.5,
) -> None:
    """Spawn the snapshot sidecar tap in a daemon thread.

    Running in a thread (not a coroutine) means the sidecar updates on
    a real-time clock even when the benchmark asyncio loop is saturated
    at 30 k+ req/s and cannot schedule coroutines. ``provider`` must be
    thread-safe; ``loop`` is accepted for API compat but unused.
    """
    if not is_enabled():
        return
    path = snapshot_sidecar_path(os.getpid())
    # Re-arm here (right before spawn), NOT in cleanup(): clearing in
    # cleanup() would un-stop a tap thread that outlived teardown's join
    # and let it overwrite the authoritative final snapshot. Any orphan
    # from a prior session is already dead by the next start.
    _state.tap_stop.clear()
    stop = _state.tap_stop

    def _tap_thread() -> None:
        while not stop.wait(timeout=period_s):
            try:
                snap = provider()
                if snap is not None:
                    _atomic_write_json(path, snap)
            except Exception:  # noqa: BLE001 — telemetry, never crash
                logger.debug("snapshot tap write failed", exc_info=True)

    t = threading.Thread(target=_tap_thread, daemon=True, name="snapshot-tap")
    t.start()
    _state.tap_threads.append(t)


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        logger.debug("snapshot sidecar write failed: %s", path, exc_info=True)


def cleanup() -> None:
    """Sync portion of teardown: flush emitter, close fd, unlink FIFO,
    print log-path reminder. Safe from any context (no loop required),
    idempotent, and the only cleanup that runs when bootstrap fired
    but the async event loop never started (e.g. CLI parse / config /
    endpoint setup raised before run_benchmark_async)."""
    global emit_trace, _active_emitter
    emitter = _active_emitter
    if emitter is not None:
        emitter.flush()
        try:
            os.close(emitter._fd)
        except OSError:
            pass
    emit_trace = _noop
    _active_emitter = None

    if _state.fifo_path is not None:
        try:
            os.unlink(_state.fifo_path)
        except OSError:
            pass  # dashboard or stale-dir wipe already removed it
        _state.fifo_path = None

    if _state.orig_stderr_fd is not None and _state.log_path is not None:
        try:
            os.write(
                _state.orig_stderr_fd,
                f"\ntrace: full run log → {_state.log_path}\n".encode(),
            )
        except OSError:
            pass  # terminal closed mid-run (nohup, detached)
        _state.orig_stderr_fd = None
        _state.log_path = None


async def teardown(*, final_snapshot: dict | None = None) -> None:
    """Async teardown: stops the tap thread, cancels lag tasks, writes
    the final snapshot, then delegates to :func:`cleanup`.  Idempotent."""
    # Stop the tap thread first so we can write the authoritative final
    # snapshot without a concurrent thread overwriting it.
    _state.tap_stop.set()
    for t in _state.tap_threads:
        t.join(timeout=2.0)
    _state.tap_threads.clear()

    # Write the final snapshot AFTER the tap thread is gone.
    if final_snapshot is not None and _state.fifo_path is not None:
        _atomic_write_json(snapshot_sidecar_path(os.getpid()), final_snapshot)

    for task in _state.tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass  # telemetry; let the bench keep tearing down
    _state.tasks.clear()
    cleanup()


def _spawn_dashboard(pipe_path: str) -> subprocess.Popen | None:
    """Spawn ``scripts/trace_dashboard.py`` to read from the FIFO.

    Returns the live ``subprocess.Popen`` on success, or ``None`` if the
    script isn't on disk (e.g. installed without the repo tree). The
    dashboard inherits the parent's stdout/stderr at spawn time, so its
    rich.Live render goes to the terminal — and the parent's stdout
    and stderr remain untouched (trace bytes ride the FIFO).
    """
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "trace_dashboard.py"
    if not script.exists():
        return None
    return subprocess.Popen(
        [sys.executable, str(script), "--trace-pipe", pipe_path],
        # Inherit stdout / stderr / stdin from parent — dashboard renders
        # to its own stderr (= terminal); trace bytes do NOT come through
        # stdin, they come through the FIFO.
    )
