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

"""``MetricsPublisher``: publish ``MetricsSnapshot`` over pub/sub + disk fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

import msgspec
import msgspec.msgpack
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshot,
    MetricsSnapshotCodec,
    SessionState,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqMessagePublisher

logger = logging.getLogger(__name__)


class MetricsPublisher:
    """Periodic snapshot publisher with best-effort disk fallback.

    The live tick task runs at ``1/refresh_hz`` cadence and publishes a
    non-final snapshot each tick. ``publish_final`` cancels the tick task,
    publishes a final snapshot over pub/sub, and atomically writes a
    msgpack copy to ``fallback_path`` so a missed pub/sub final can still
    be reconstructed.

    Pub/sub publish and disk fallback are **independent** best-effort
    paths: a failure in one MUST NOT suppress the other.
    """

    def __init__(
        self,
        codec: MetricsSnapshotCodec,
        zmq_ctx: ManagedZMQContext,
        socket_name: str,
        loop: asyncio.AbstractEventLoop,
        fallback_path: Path,
    ) -> None:
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
        self._fallback_path = fallback_path
        self._tick_task: asyncio.Task | None = None
        self._encoder = msgspec.msgpack.Encoder()
        self._closed = False

    # ------------------------------------------------------------------
    # Live tick task
    # ------------------------------------------------------------------

    def start(
        self,
        registry: MetricsRegistry,
        refresh_hz: float,
        get_runtime_state: Callable[[], tuple[SessionState, int]],
    ) -> None:
        """Begin publishing live ticks at ``refresh_hz``.

        ``get_runtime_state`` returns ``(state, n_pending_tasks)`` for the
        current moment: the aggregator's session state (``LIVE`` or
        ``DRAINING``) and the count of in-flight async tokenize tasks. The
        callable is invoked once per tick and the values are plumbed into
        the published snapshot. ``COMPLETE`` is emitted only by
        ``publish_final``, never by the tick task.
        """
        if refresh_hz <= 0:
            raise ValueError(f"refresh_hz must be positive, got {refresh_hz}")
        period = 1.0 / refresh_hz

        async def _tick() -> None:
            while True:
                try:
                    await asyncio.sleep(period)
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
        self, registry: MetricsRegistry, *, n_pending_tasks: int
    ) -> None:
        """Publish the ``COMPLETE`` snapshot over pub/sub AND mirror to disk.

        ``n_pending_tasks`` is the count of in-flight async tokenize tasks
        at finalization time. Drain timeout is detected by consumers as
        ``state == COMPLETE and n_pending_tasks > 0``.

        Awaits tick-task cancellation BEFORE building/publishing so a late
        live tick cannot land after the COMPLETE frame on the wire (which
        would let a conflate-mode subscriber see the live tick as the
        latest message instead of COMPLETE).

        Pub/sub publish and disk fallback are independent best-effort
        paths, each wrapped in its own try/except.
        """
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                # Expected: we just cancelled it.
                pass
            self._tick_task = None
        snap = registry.build_snapshot(
            state=SessionState.COMPLETE, n_pending_tasks=n_pending_tasks
        )

        # Pub/sub first — buffer write, can't fail in normal operation.
        # Wrapped anyway so a transport bug doesn't suppress the disk
        # fallback below.
        try:
            self._publisher.publish(snap)
        except Exception:  # noqa: BLE001 — best-effort, must not block disk.
            logger.exception("metrics: pub/sub final publish failed")

        # Disk fallback — best-effort, must not affect pub/sub above.
        # The atomic write does synchronous f.flush + fsync(file) +
        # fsync(parent dir) + rename, which can block tens-to-hundreds of
        # ms on a busy host. Run it on a worker thread so it doesn't
        # back-pressure any in-flight event-record processing on the
        # aggregator's event loop.
        try:
            await asyncio.to_thread(
                self._write_atomic_fallback, self._encoder.encode(snap)
            )
        except Exception:  # noqa: BLE001 — best-effort.
            logger.exception("metrics: disk fallback write failed")

    def _write_atomic_fallback(self, payload: bytes) -> None:
        """Write payload atomically to ``fallback_path``.

        Sequence: write tmp + fsync(tmp) → rename → fsync(parent dir) so
        the rename itself is durable across crashes.
        """
        path = self._fallback_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # 1. Write payload to tmp + fsync the file.
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        # 2. Atomic rename.
        os.rename(tmp, path)
        # 3. fsync parent dir so the rename is durable across crash.
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

        ``ZmqMessagePublisher.close()`` drains pending frames; bounded by
        the ``linger=10s`` set at construction.
        """
        if self._closed:
            return
        self._closed = True
        if self._tick_task is not None:
            self._tick_task.cancel()
        self._publisher.close()
