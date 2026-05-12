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

A live-state subscriber for TUI / dashboard consumers. Keeps the latest
snapshot in ``self.latest`` and updates it on every tick. Terminal
snapshots (``SessionState.COMPLETE`` / ``INTERRUPTED``) arrive over
pub/sub as a "run finished" signal for consumers that want to switch to
a final-view rendering on the wire event.
"""

from __future__ import annotations

import asyncio
import logging

from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshot,
    MetricsSnapshotCodec,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqMessageSubscriber

logger = logging.getLogger(__name__)


class MetricsSnapshotSubscriber(ZmqMessageSubscriber[MetricsSnapshot]):
    """Subscriber that tracks the latest ``MetricsSnapshot`` for live views.

    ``latest`` is updated on every received snapshot regardless of state.
    A consumer detects "run finished" by observing
    ``latest.state in {COMPLETE, INTERRUPTED}`` — both are terminal and
    no further snapshots will arrive.
    """

    def __init__(
        self,
        path: str,
        zmq_ctx: ManagedZMQContext,
        loop: asyncio.AbstractEventLoop,
        *,
        conflate: bool = True,
    ) -> None:
        # conflate=True (default) keeps only the freshest snapshot in the
        # SUB queue — the right shape for live consumers that render the
        # current state on a timer. Pass conflate=False if a consumer
        # needs every intermediate tick (no current callers do).
        super().__init__(
            MetricsSnapshotCodec(),
            path,
            zmq_ctx,
            loop,
            topics=None,
            conflate=conflate,
        )
        self.latest: MetricsSnapshot | None = None

    async def process(self, items: list[MetricsSnapshot]) -> None:
        for snap in items:
            self.latest = snap
