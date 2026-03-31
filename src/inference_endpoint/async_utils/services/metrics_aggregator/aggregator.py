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

"""MetricsAggregatorService: thin event router for real-time metrics."""

from __future__ import annotations

import asyncio

from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqEventRecordSubscriber,
)
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)

from .emitter import MetricEmitter
from .metrics_table import (
    ChunkDeltaTrigger,
    IslTrigger,
    MetricsTable,
    OslTrigger,
    RequestDurationTrigger,
    SampleLatencyTrigger,
    TpotTrigger,
    TtftTrigger,
)
from .token_metrics import TokenizePool

_TRACKED_SAMPLE_EVENTS = frozenset(
    {
        SampleEventType.ISSUED,
        SampleEventType.COMPLETE,
        SampleEventType.RECV_FIRST,
        SampleEventType.RECV_NON_FIRST,
        SampleEventType.CLIENT_SEND,
        SampleEventType.CLIENT_RESP_DONE,
    }
)


class MetricsAggregatorService(ZmqEventRecordSubscriber):
    """Subscribes to EventRecords and computes per-sample metrics in real time.

    The aggregator is a thin event router. All state management, trigger
    dispatch, and row lifecycle are handled by MetricsTable.
    """

    def __init__(
        self,
        *args,
        emitter: MetricEmitter,
        tokenize_pool: TokenizePool | None = None,
        streaming: bool = False,
        shutdown_event: asyncio.Event | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._emitter = emitter
        self._shutdown_event = shutdown_event
        self._shutdown_received = False

        self._table = MetricsTable()
        self._register_triggers(
            self._table, emitter, tokenize_pool, self.loop, streaming
        )

    @staticmethod
    def _register_triggers(
        table: MetricsTable,
        emitter: MetricEmitter,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
        streaming: bool,
    ) -> None:
        """Register metric triggers on the table.

        Streaming-only triggers (TTFT, chunk_delta, TPOT) are only registered
        when ``streaming=True``.
        """
        # Always registered
        table.add_trigger("issued_ns", IslTrigger(emitter, tokenize_pool, loop))
        table.add_trigger("client_resp_done_ns", RequestDurationTrigger(emitter))
        table.add_trigger("complete_ns", SampleLatencyTrigger(emitter))
        table.add_trigger("complete_ns", OslTrigger(emitter, tokenize_pool, loop))

        # Streaming-only
        if streaming:
            table.add_trigger("recv_first_ns", TtftTrigger(emitter))
            table.add_trigger("last_recv_ns", ChunkDeltaTrigger(emitter))
            table.add_trigger("complete_ns", TpotTrigger(emitter, tokenize_pool, loop))

    async def process(self, records: list[EventRecord]) -> None:
        saw_shutdown = False
        table = self._table

        for record in records:
            if self._shutdown_received:
                break

            ev = record.event_type

            # --- Session events ---
            if isinstance(ev, SessionEventType):
                if ev == SessionEventType.ENDED:
                    self._shutdown_received = True
                    saw_shutdown = True
                else:
                    table.handle_session_event(record)
                continue

            # --- Sample events ---
            if (
                not isinstance(ev, SampleEventType)
                or ev not in _TRACKED_SAMPLE_EVENTS
                or not record.sample_uuid
            ):
                continue

            uuid = record.sample_uuid
            ts = record.timestamp_ns

            if ev == SampleEventType.ISSUED:
                table.set_field(uuid, "issued_ns", ts, record)
            elif ev == SampleEventType.RECV_FIRST:
                table.set_field(uuid, "recv_first_ns", ts, record)
                table.set_field(uuid, "last_recv_ns", ts, record)
            elif ev == SampleEventType.RECV_NON_FIRST:
                table.set_field(uuid, "last_recv_ns", ts, record)
            elif ev == SampleEventType.CLIENT_SEND:
                table.set_field(uuid, "client_send_ns", ts, record)
            elif ev == SampleEventType.CLIENT_RESP_DONE:
                table.set_field(uuid, "client_resp_done_ns", ts, record)
            elif ev == SampleEventType.COMPLETE:
                table.set_field(uuid, "complete_ns", ts, record)

        if saw_shutdown:
            await table.drain_tasks()
            self._finalize()

    def _finalize(self) -> None:
        self.close()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        elif self.loop is not None and self.loop.is_running():
            self.loop.stop()

    def close(self) -> None:
        self._emitter.close()
        super().close()
