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

"""Subscribe to ``MetricsSnapshot`` from the aggregator subprocess.

The main process uses ``MetricsSnapshotSubscriber`` to keep the latest
live snapshot, and to capture the snapshot whose ``state`` is
``SessionState.COMPLETE`` when it arrives. Mirrors the publisher on the
aggregator side.
"""

from __future__ import annotations

import asyncio
import logging

from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshot,
    MetricsSnapshotCodec,
    SessionState,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqMessageSubscriber

logger = logging.getLogger(__name__)


class MetricsSnapshotSubscriber(ZmqMessageSubscriber[MetricsSnapshot]):
    """Subscriber that tracks ``latest`` and the ``COMPLETE`` snapshot.

    ``latest`` is updated on every received snapshot regardless of state.
    ``complete`` is set the first time a snapshot with
    ``state == SessionState.COMPLETE`` arrives, and ``_complete_event`` is
    signaled so the main process can ``await`` it.
    """

    def __init__(
        self,
        path: str,
        zmq_ctx: ManagedZMQContext,
        loop: asyncio.AbstractEventLoop,
        *,
        conflate: bool = True,
    ) -> None:
        # conflate=True (default) keeps only the freshest snapshot in the SUB
        # queue — appropriate for a TUI and safe for the main process Report
        # consumer (the COMPLETE snapshot is the last message the publisher
        # emits, so it's never conflated away). Pass conflate=False if a
        # consumer needs every intermediate tick.
        super().__init__(
            MetricsSnapshotCodec(),
            path,
            zmq_ctx,
            loop,
            topics=None,
            conflate=conflate,
        )
        self.latest: MetricsSnapshot | None = None
        self.complete: MetricsSnapshot | None = None
        self._complete_event = asyncio.Event()

    async def wait_for_complete(self, timeout: float | None = None) -> bool:
        """Wait until a ``COMPLETE``-state snapshot arrives.

        Returns True iff received before ``timeout``.
        """
        try:
            await asyncio.wait_for(self._complete_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def process(self, items: list[MetricsSnapshot]) -> None:
        for snap in items:
            self.latest = snap
            if snap.state == SessionState.COMPLETE and self.complete is None:
                self.complete = snap
                self._complete_event.set()
                logger.info(
                    "Received COMPLETE metrics snapshot "
                    "(counter=%d, n_pending_tasks=%d)",
                    snap.counter,
                    snap.n_pending_tasks,
                )
