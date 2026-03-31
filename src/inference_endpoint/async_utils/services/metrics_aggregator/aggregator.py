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
import logging
from enum import Enum

from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqEventRecordSubscriber,
)
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)

from .kv_store import KVStore
from .metrics_table import (
    ChunkDeltaTrigger,
    IslTrigger,
    MetricsTable,
    OslTrigger,
    SampleField,
    SampleLatencyTrigger,
    TpotTrigger,
    TtftTrigger,
)
from .token_metrics import TokenizePool

logger = logging.getLogger(__name__)


class MetricCounterKey(str, Enum):
    """Counter metric keys tracked by the aggregator.

    Total counters include all samples (warmup + tracked).
    Tracked counters only include samples within performance tracking windows.
    """

    TOTAL_SAMPLES_ISSUED = "total_samples_issued"
    TOTAL_SAMPLES_COMPLETED = "total_samples_completed"
    TOTAL_SAMPLES_FAILED = "total_samples_failed"
    TRACKED_SAMPLES_ISSUED = "tracked_samples_issued"
    TRACKED_SAMPLES_COMPLETED = "tracked_samples_completed"
    TRACKED_DURATION_NS = "tracked_duration_ns"
    # Total wall-clock duration since session start. Updated on every event as
    # max(current, event_timestamp - session_start) to be defensive against
    # non-monotonic timestamps.
    #
    # An alternative design was considered: store session_start_ns once and
    # compute duration as (now - start) on read. This is infeasible because
    # time.monotonic_ns() has inconsistent epoch per process — a reader in
    # another process would get a meaningless value.
    TOTAL_DURATION_NS = "total_duration_ns"


_TRACKED_SAMPLE_EVENTS = frozenset(
    {
        SampleEventType.ISSUED,
        SampleEventType.COMPLETE,
        SampleEventType.RECV_FIRST,
        SampleEventType.RECV_NON_FIRST,
    }
)


class MetricsAggregatorService(ZmqEventRecordSubscriber):
    """Subscribes to EventRecords and computes per-sample metrics in real time.

    The aggregator is a thin event router. All state management, trigger
    dispatch, and row lifecycle are handled by MetricsTable. The KVStore
    is shared between the table (for series metrics via triggers) and the
    aggregator (for counter metrics like n_issued, n_completed, etc.).
    """

    def __init__(
        self,
        *args,
        kv_store: KVStore,
        tokenize_pool: TokenizePool | None = None,
        streaming: bool = False,
        shutdown_event: asyncio.Event | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._kv_store = kv_store
        self._tokenize_pool = tokenize_pool
        self._shutdown_event = shutdown_event
        self._shutdown_received = False

        for key in MetricCounterKey:
            kv_store.create_key(key.value, "counter")

        self._total_issued = 0
        self._total_completed = 0
        self._total_failed = 0
        self._tracked_issued = 0
        self._tracked_completed = 0
        self._session_start_ns: int | None = None
        self._total_duration_ns: float = 0.0

        self._table = MetricsTable(kv_store)
        self._register_triggers(streaming)

    def _register_triggers(self, streaming: bool) -> None:
        """Register metric triggers on the table.

        Streaming-only triggers (TTFT, chunk_delta, TPOT) are only registered
        when ``streaming=True``.
        """
        table = self._table
        store = self._kv_store
        pool = self._tokenize_pool
        loop = self.loop

        # Always registered
        table.add_trigger(SampleField.ISSUED_NS, IslTrigger(store, pool, loop))
        table.add_trigger(SampleField.COMPLETE_NS, SampleLatencyTrigger(store))
        table.add_trigger(SampleField.COMPLETE_NS, OslTrigger(store, pool, loop))

        # Streaming-only
        if streaming:
            table.add_trigger(SampleField.RECV_FIRST_NS, TtftTrigger(store))
            table.add_trigger(SampleField.LAST_RECV_NS, ChunkDeltaTrigger(store))
            table.add_trigger(SampleField.COMPLETE_NS, TpotTrigger(store, pool, loop))

    async def process(self, records: list[EventRecord]) -> None:
        saw_shutdown = False
        table = self._table
        store = self._kv_store

        for record in records:
            if self._shutdown_received:
                break

            ev = record.event_type

            # Update total_duration_ns on every event
            if self._session_start_ns is not None:
                elapsed = record.timestamp_ns - self._session_start_ns
                if elapsed > self._total_duration_ns:
                    self._total_duration_ns = elapsed
                    store.update(
                        MetricCounterKey.TOTAL_DURATION_NS.value,
                        self._total_duration_ns,
                    )

            # --- Session events ---
            if isinstance(ev, SessionEventType):
                if ev == SessionEventType.ENDED:
                    self._shutdown_received = True
                    saw_shutdown = True
                else:
                    if ev == SessionEventType.STARTED:
                        self._session_start_ns = record.timestamp_ns
                    table.handle_session_event(record)
                    if ev == SessionEventType.STOP_PERFORMANCE_TRACKING:
                        store.update(
                            MetricCounterKey.TRACKED_DURATION_NS.value,
                            table.total_tracked_duration_ns,
                        )
                logger.debug("Session event: %s", ev)
                continue

            # --- Error events ---
            if isinstance(ev, ErrorEventType):
                self._total_failed += 1
                store.update(
                    MetricCounterKey.TOTAL_SAMPLES_FAILED.value, self._total_failed
                )
                logger.debug("Error event: %s", record)
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
                table.set_field(uuid, SampleField.ISSUED_NS, ts, record)
                self._total_issued += 1
                store.update(
                    MetricCounterKey.TOTAL_SAMPLES_ISSUED.value, self._total_issued
                )
                if table.get_row(uuid) is not None:
                    self._tracked_issued += 1
                    store.update(
                        MetricCounterKey.TRACKED_SAMPLES_ISSUED.value,
                        self._tracked_issued,
                    )
            elif ev == SampleEventType.RECV_FIRST:
                table.set_field(uuid, SampleField.RECV_FIRST_NS, ts, record)
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
            elif ev == SampleEventType.RECV_NON_FIRST:
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
            elif ev == SampleEventType.COMPLETE:
                # Check if tracked before set_field (which removes the row)
                is_tracked = table.get_row(uuid) is not None
                table.set_field(uuid, SampleField.COMPLETE_NS, ts, record)
                self._total_completed += 1
                store.update(
                    MetricCounterKey.TOTAL_SAMPLES_COMPLETED.value,
                    self._total_completed,
                )
                if is_tracked:
                    self._tracked_completed += 1
                    store.update(
                        MetricCounterKey.TRACKED_SAMPLES_COMPLETED.value,
                        self._tracked_completed,
                    )

        if saw_shutdown:
            await table.drain_tasks()
            store.update(
                MetricCounterKey.TRACKED_DURATION_NS.value,
                table.total_tracked_duration_ns,
            )
            self._finalize()

    def _finalize(self) -> None:
        self.close()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        elif self.loop is not None and self.loop.is_running():
            self.loop.stop()

    def close(self) -> None:
        self._kv_store.close()
        super().close()
