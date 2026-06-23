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

"""``MetricsPublisher``: publish ``MetricsSnapshot`` over pub/sub + JSON file."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshot,
    MetricsSnapshotCodec,
    SessionState,
    snapshot_to_dict,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqMessagePublisher

logger = logging.getLogger(__name__)


class MetricsPublisher:
    """Periodic snapshot publisher: pub/sub for live, JSON file for final.

    The live tick task runs at ``publish_interval_s`` cadence and publishes
    a non-final snapshot over pub/sub each tick. ``publish_final``:

    1. Cancels the tick task (and awaits its exit).
    2. Atomically writes the final snapshot as pretty-printed JSON to
       ``final_snapshot_path`` — this is the **primary** delivery path
       and what the Report consumer reads.
    3. Publishes a (msgpack) terminal-state snapshot over pub/sub as a
       **TUI shutdown signal** — a future TUI can switch to "final view"
       on seeing this frame without polling the file. The Report consumer
       does NOT depend on this pub/sub send.

    Decoupling the file from pub/sub means ``conflate=True`` on the SUB
    side is unambiguously safe (a TUI that drops the COMPLETE frame just
    needs to notice the file appeared / the publisher socket dropped),
    and the file artifact is self-contained: ``cat final_snapshot.json``
    is the canonical source of truth for a finished run.

    Pub/sub publish and disk write are **independent** best-effort
    paths: a failure in one MUST NOT suppress the other.
    """

    def __init__(
        self,
        codec: MetricsSnapshotCodec,
        zmq_ctx: ManagedZMQContext,
        socket_name: str,
        loop: asyncio.AbstractEventLoop,
        final_snapshot_path: Path,
    ) -> None:
        # final_snapshot_path is the absolute path the JSON file is written
        # to. Injected (not derived from output_dir) so tests can place it
        # in tmp_path without recomputing extension/filename.
        self._publisher: ZmqMessagePublisher[MetricsSnapshot] = ZmqMessagePublisher(
            codec,
            socket_name,
            zmq_ctx,
            loop=loop,
            send_threshold=1,
            sndhwm=4,
            linger=10_000,
        )
        self._loop = loop
        self._final_snapshot_path = final_snapshot_path
        self._tick_task: asyncio.Task | None = None
        self._closed = False
        # publish_final is idempotent: the SIGTERM/SIGINT handler in
        # __main__.py and the aggregator's ENDED-driven path can both
        # call it; the second call must not re-publish or re-write.
        self._finalized = False

    # ------------------------------------------------------------------
    # Live tick task
    # ------------------------------------------------------------------

    def start(
        self,
        registry: MetricsRegistry,
        publish_interval_s: float,
        get_runtime_state: Callable[[], tuple[SessionState, int]],
    ) -> None:
        """Begin publishing live ticks every ``publish_interval_s`` seconds.

        ``get_runtime_state`` returns ``(state, n_pending_tasks)`` for the
        current moment: the aggregator's session state (``LIVE`` or
        ``DRAINING``) and the count of in-flight async tokenize tasks. The
        callable is invoked once per tick and the values are plumbed into
        the published snapshot. ``COMPLETE`` is emitted only by
        ``publish_final``, never by the tick task.

        Idempotent on the tick-task slot: a second call (e.g. from a
        spurious duplicate ``STARTED`` event or a buggy replay producer)
        is a no-op rather than orphaning the original task. The original
        task remains the one cancelled by ``publish_final`` / ``aclose``.
        """
        if self._tick_task is not None:
            logger.warning(
                "MetricsPublisher.start called again while tick task is "
                "still running (id=%r); ignoring the second start.",
                id(self._tick_task),
            )
            return
        if publish_interval_s <= 0:
            raise ValueError(
                f"publish_interval_s must be positive, got {publish_interval_s}"
            )

        async def _tick() -> None:
            while True:
                try:
                    await asyncio.sleep(publish_interval_s)
                    state, n_pending = get_runtime_state()
                    snap = registry.build_snapshot(
                        state=state, n_pending_tasks=n_pending
                    )
                    self._publisher.publish(snap)
                except asyncio.CancelledError:
                    # Graceful cancellation from publish_final/close.
                    return
                except Exception:  # noqa: BLE001 — keep ticking on transient errors.
                    logger.exception("metrics tick failed; continuing")

        self._tick_task = self._loop.create_task(_tick())

    # ------------------------------------------------------------------
    # Final delivery
    # ------------------------------------------------------------------

    async def publish_final(
        self,
        registry: MetricsRegistry,
        *,
        n_pending_tasks: int,
        interrupted: bool = False,
    ) -> None:
        """Write the final snapshot to disk and signal pub/sub consumers.

        ``n_pending_tasks`` is the count of in-flight async tokenize tasks
        at finalization time. Drain timeout is detected by Report consumers
        as ``state == COMPLETE and n_pending_tasks > 0``.

        ``interrupted=True`` is set by the signal handler in __main__.py
        when SIGTERM/SIGINT triggers shutdown before ``ENDED`` arrived;
        the resulting snapshot is tagged ``state=INTERRUPTED`` so Report
        can distinguish "user killed the run mid-execution" from a clean
        end. Stats in an INTERRUPTED snapshot are best-effort partial
        captures of whatever the aggregator had at signal time.

        Two delivery channels, independent best-effort:

        1. **JSON file at ``final_snapshot_path``** (primary). Atomic
           write (tmp + fsync(file) + rename + fsync(parent dir)) so the
           file is either fully present or absent — partial reads are
           impossible. Pretty-printed for ``cat`` / ``jq`` use. This is
           what the Report consumer reads.
        2. **msgpack pub/sub** (TUI signal). A future TUI uses this as
           the "run is over, switch to final view" cue without polling
           the file. The Report consumer does NOT read this channel.

        A failure in one channel MUST NOT suppress the other; each is
        wrapped in its own try/except.

        Awaits tick-task cancellation BEFORE building the snapshot so a
        late live tick cannot land after the terminal frame on the wire
        (which would let a conflate-mode TUI see the live tick instead
        of the terminal state as the last message).

        Idempotent: only the first call writes/publishes; subsequent
        calls early-return. The SIGTERM/SIGINT handler relies on this to
        race safely with the ENDED-driven path.
        """
        if self._finalized:
            return
        self._finalized = True
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                # Expected: we just cancelled it.
                pass
            self._tick_task = None

        terminal_state = (
            SessionState.INTERRUPTED if interrupted else SessionState.COMPLETE
        )
        snap = registry.build_snapshot(
            state=terminal_state, n_pending_tasks=n_pending_tasks
        )

        # Primary: atomic JSON file write. Run on a worker thread because
        # fsync(file) + fsync(parent dir) can block tens-to-hundreds of ms
        # on a busy host and would otherwise back-pressure any in-flight
        # event-record processing on the aggregator's event loop.
        try:
            # ``allow_nan=False`` makes a producer-side NaN/Inf leak a
            # hard error here rather than a silent ``NaN`` / ``Infinity``
            # token in the file (which strict JSON consumers reject).
            # ``snapshot_to_dict`` already scrubs non-finite floats to
            # ``None``, so the only way this raises is a genuine bug.
            payload = json.dumps(
                snapshot_to_dict(snap), indent=2, allow_nan=False
            ).encode("utf-8")
            await asyncio.to_thread(self._write_atomic_json, payload)
        except Exception:  # noqa: BLE001 — best-effort; pub/sub still needs to fire.
            logger.exception("metrics: final JSON snapshot write failed")

        # TUI signal: msgpack pub/sub send. Wrapped so a transport bug
        # doesn't suppress the file write above and so a SUB-side issue
        # doesn't crash the aggregator on shutdown. Also legitimately
        # covers the ENDED-vs-SIGTERM race: if a signal-driven
        # publish_final raced ahead and reached `aclose()` before this
        # publish call runs, the underlying ZMQ socket is already
        # closed and the send raises. Dropping the TUI frame in that
        # race is acceptable — the JSON file written above is the
        # authoritative Report source.
        try:
            self._publisher.publish(snap)
        except Exception:  # noqa: BLE001 — best-effort; file is the source of truth.
            logger.exception("metrics: pub/sub final signal failed")

    def _write_atomic_json(self, payload: bytes) -> None:
        """Write payload atomically to ``final_snapshot_path``.

        Sequence: write tmp + fsync(tmp) → rename → fsync(parent dir) so
        the rename itself is durable across crashes. The path either
        contains the new snapshot or contains the old contents (if any)
        — never partial bytes. The parent directory is the caller's
        responsibility — `__main__.py` validates it on startup so a
        missing directory surfaces in the subprocess's own context
        rather than as a 30 s parent-side launch timeout.
        """
        path = self._final_snapshot_path
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Cancel tick task and close the underlying publisher.

        Sync, best-effort. The tick task is cancelled but NOT awaited;
        if a live tick is mid-publish when this runs, it may print a
        CancelledError-during-shutdown trace before the loop tears down.
        Prefer :meth:`aclose` from async contexts to avoid that. This
        sync form exists for error-path / signal-handler fallbacks where
        no event loop is reasonably available to await on.

        ``ZmqMessagePublisher.close()`` drains pending frames; bounded by
        the ``linger=10s`` set at construction.
        """
        if self._closed:
            return
        self._closed = True
        if self._tick_task is not None:
            self._tick_task.cancel()
        self._publisher.close()

    async def aclose(self) -> None:
        """Async-aware close: cancel the tick task and await its exit.

        Preferred over :meth:`close` whenever the caller is running on
        an event loop. Eliminates the cancelled-tick-task-vs-publisher-
        close race that the sync :meth:`close` is exposed to.
        """
        if self._closed:
            return
        self._closed = True
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                # Expected: we just cancelled it.
                pass
            self._tick_task = None
        self._publisher.close()
