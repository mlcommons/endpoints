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
from typing import Final

from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqMessageSubscriber,
)
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    EventRecordCodec,
    SampleEventType,
    SessionEventType,
)

from .metrics_table import (
    ChunkDeltaTrigger,
    IslTrigger,
    MetricSeriesKey,
    MetricsTable,
    OslTrigger,
    SampleField,
    SampleLatencyTrigger,
    TpotTrigger,
    TtftTrigger,
)
from .publisher import MetricsPublisher
from .registry import MetricsRegistry
from .snapshot import SessionState
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
    # Failed samples that were within a performance-tracking window.
    # Counted at ERROR-event time; correctness depends on
    # session.py:_handle_response emitting ERROR before COMPLETE so the
    # tracked row still exists when the aggregator sees the ERROR.
    TRACKED_SAMPLES_FAILED = "tracked_samples_failed"
    TRACKED_DURATION_NS = "tracked_duration_ns"
    # Total wall-clock duration since session start. Updated on every event as
    # max(current, event_timestamp - session_start). Stored as a counter
    # rather than computed from (now - start) at read time because
    # time.monotonic_ns() has a process-local epoch — a reader in another
    # process would get a meaningless value.
    TOTAL_DURATION_NS = "total_duration_ns"


_TRACKED_SAMPLE_EVENTS = frozenset(
    {
        SampleEventType.ISSUED,
        SampleEventType.COMPLETE,
        SampleEventType.RECV_FIRST,
        SampleEventType.RECV_NON_FIRST,
    }
)


# HDR bounds per series — chosen conservatively so realistic benchmark
# values cannot fall outside [low, high]. Values outside the range are
# clamped on insert and a warning is logged once per series.
_NS_HDR_LOW: Final[int] = 1
_NS_HDR_HIGH: Final[int] = 3_600_000_000_000  # 1 hour in ns
_TOKEN_HDR_LOW: Final[int] = 1
_TOKEN_HDR_HIGH: Final[int] = 10_000_000  # 10M tokens

_DRAIN_TIMEOUT_S: Final[float] = 30.0


class MetricsAggregatorService(ZmqMessageSubscriber[EventRecord]):
    """Subscribes to EventRecords and computes per-sample metrics in real time.

    The aggregator is a thin event router. All state management, trigger
    dispatch, and row lifecycle are handled by ``MetricsTable``. The
    ``MetricsRegistry`` holds counters and series; the ``MetricsPublisher``
    publishes ``MetricsSnapshot`` over pub/sub at a fixed cadence and
    mirrors the final snapshot to disk.
    """

    def __init__(
        self,
        *args,
        registry: MetricsRegistry,
        publisher: MetricsPublisher,
        refresh_hz: float,
        sig_figs: int,
        n_histogram_buckets: int,
        tokenize_pool: TokenizePool | None = None,
        streaming: bool = False,
        shutdown_event: asyncio.Event | None = None,
        **kwargs,
    ):
        super().__init__(EventRecordCodec(), *args, **kwargs)
        self._registry = registry
        self._publisher = publisher
        self._refresh_hz = refresh_hz
        self._tokenize_pool = tokenize_pool
        self._streaming = streaming
        self._shutdown_event = shutdown_event
        self._shutdown_received = False

        self._session_start_ns: int | None = None
        self._total_duration_ns: int = 0
        self._total_processed = 0
        self._last_log_count = 0
        # Tracks the run's lifecycle state, surfaced on the wire as
        # MetricsSnapshot.state. Transitions: LIVE → DRAINING (on ENDED) →
        # COMPLETE (set implicitly via publish_final).
        self._session_state: SessionState = SessionState.LIVE

        # Pre-register all metrics on the registry. Tests can introspect via
        # registry.has_counter / has_series.
        self._register_metrics(streaming, sig_figs, n_histogram_buckets)

        self._table = MetricsTable(self._registry)
        self._register_triggers(streaming)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register_metrics(
        self, streaming: bool, sig_figs: int, n_histogram_buckets: int
    ) -> None:
        """Register all counters and series on the registry."""
        for key in MetricCounterKey:
            self._registry.register_counter(key.value)

        # Always-present series
        self._registry.register_series(
            MetricSeriesKey.SAMPLE_LATENCY_NS.value,
            hdr_low=_NS_HDR_LOW,
            hdr_high=_NS_HDR_HIGH,
            sig_figs=sig_figs,
            n_histogram_buckets=n_histogram_buckets,
        )
        self._registry.register_series(
            MetricSeriesKey.ISL.value,
            hdr_low=_TOKEN_HDR_LOW,
            hdr_high=_TOKEN_HDR_HIGH,
            sig_figs=sig_figs,
            n_histogram_buckets=n_histogram_buckets,
        )
        self._registry.register_series(
            MetricSeriesKey.OSL.value,
            hdr_low=_TOKEN_HDR_LOW,
            hdr_high=_TOKEN_HDR_HIGH,
            sig_figs=sig_figs,
            n_histogram_buckets=n_histogram_buckets,
        )

        # Streaming-only series
        if streaming:
            self._registry.register_series(
                MetricSeriesKey.TTFT_NS.value,
                hdr_low=_NS_HDR_LOW,
                hdr_high=_NS_HDR_HIGH,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
            )
            self._registry.register_series(
                MetricSeriesKey.CHUNK_DELTA_NS.value,
                hdr_low=_NS_HDR_LOW,
                hdr_high=_NS_HDR_HIGH,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
            )
            self._registry.register_series(
                MetricSeriesKey.TPOT_NS.value,
                hdr_low=_NS_HDR_LOW,
                hdr_high=_NS_HDR_HIGH,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
                dtype=float,
            )

    def _register_triggers(self, streaming: bool) -> None:
        """Register metric triggers on the table.

        Streaming-only triggers (TTFT, chunk_delta, TPOT) are only registered
        when ``streaming=True``.
        """
        table = self._table
        registry = self._registry
        pool = self._tokenize_pool
        loop = self.loop

        # Always registered
        table.add_trigger(SampleField.ISSUED_NS, IslTrigger(registry, pool, loop))
        table.add_trigger(SampleField.COMPLETE_NS, SampleLatencyTrigger(registry))
        table.add_trigger(SampleField.COMPLETE_NS, OslTrigger(registry, pool, loop))

        # Streaming-only
        if streaming:
            table.add_trigger(SampleField.RECV_FIRST_NS, TtftTrigger(registry))
            table.add_trigger(SampleField.LAST_RECV_NS, ChunkDeltaTrigger(registry))
            table.add_trigger(
                SampleField.COMPLETE_NS, TpotTrigger(registry, pool, loop)
            )

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def process(self, records: list[EventRecord]) -> None:
        saw_shutdown = False
        table = self._table
        registry = self._registry

        self._total_processed += len(records)
        if self._total_processed - self._last_log_count >= 10000:
            logger.debug(
                "Aggregator processed %d records (%d in this batch)",
                self._total_processed,
                len(records),
            )
            self._last_log_count = self._total_processed

        for record in records:
            if self._shutdown_received:
                break

            ev = record.event_type

            # Update total_duration_ns on every event
            if self._session_start_ns is not None:
                elapsed = record.timestamp_ns - self._session_start_ns
                if elapsed > self._total_duration_ns:
                    self._total_duration_ns = elapsed
                    registry.set_counter(
                        MetricCounterKey.TOTAL_DURATION_NS.value,
                        self._total_duration_ns,
                    )

            # --- Session events ---
            if isinstance(ev, SessionEventType):
                if ev == SessionEventType.ENDED:
                    logger.info("ENDED event received, shutting down aggregator")
                    self._shutdown_received = True
                    saw_shutdown = True
                else:
                    if ev == SessionEventType.STARTED:
                        self._session_start_ns = record.timestamp_ns
                        # Now that we have an event loop running, start the
                        # publisher tick task. The callable is invoked once
                        # per tick to capture the live (state, n_pending_tasks)
                        # pair at each emit.
                        self._publisher.start(
                            registry,
                            self._refresh_hz,
                            get_runtime_state=lambda: (
                                self._session_state,
                                len(table._in_flight_tasks),
                            ),
                        )
                    table.handle_session_event(record)
                    if ev == SessionEventType.STOP_PERFORMANCE_TRACKING:
                        registry.set_counter(
                            MetricCounterKey.TRACKED_DURATION_NS.value,
                            table.total_tracked_duration_ns,
                        )
                logger.debug("Session event: %s", ev)
                continue

            # --- Error events ---
            # Counted BEFORE the COMPLETE event (session.py emits ERROR
            # first), so the tracked row still exists for tracked-failed
            # detection.
            if isinstance(ev, ErrorEventType):
                registry.increment(MetricCounterKey.TOTAL_SAMPLES_FAILED.value)
                if record.sample_uuid and table.get_row(record.sample_uuid) is not None:
                    registry.increment(MetricCounterKey.TRACKED_SAMPLES_FAILED.value)
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
                registry.increment(MetricCounterKey.TOTAL_SAMPLES_ISSUED.value)
                if table.get_row(uuid) is not None:
                    registry.increment(MetricCounterKey.TRACKED_SAMPLES_ISSUED.value)
            elif ev == SampleEventType.RECV_FIRST:
                table.set_field(uuid, SampleField.RECV_FIRST_NS, ts, record)
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
            elif ev == SampleEventType.RECV_NON_FIRST:
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
            elif ev == SampleEventType.COMPLETE:
                # Check if tracked before set_field (which removes the row)
                is_tracked = table.get_row(uuid) is not None
                table.set_field(uuid, SampleField.COMPLETE_NS, ts, record)
                registry.increment(MetricCounterKey.TOTAL_SAMPLES_COMPLETED.value)
                if is_tracked:
                    registry.increment(MetricCounterKey.TRACKED_SAMPLES_COMPLETED.value)

        if saw_shutdown:
            # ENDED has been observed; transition to DRAINING so any tick
            # that fires before publish_final reflects the new state.
            self._session_state = SessionState.DRAINING
            logger.info("Draining %d async tasks...", len(table._in_flight_tasks))
            try:
                await asyncio.wait_for(table.drain_tasks(), timeout=_DRAIN_TIMEOUT_S)
            except TimeoutError:
                logger.warning(
                    "drain_tasks timed out after %.1fs; some async metrics "
                    "may be incomplete",
                    _DRAIN_TIMEOUT_S,
                )
                for t in list(table._in_flight_tasks):
                    if not t.done():
                        t.cancel()
            n_pending = len(table._in_flight_tasks)
            logger.info(
                "Async tasks drained (n_pending_tasks=%d at finalize)", n_pending
            )
            registry.set_counter(
                MetricCounterKey.TRACKED_DURATION_NS.value,
                table.total_tracked_duration_ns,
            )
            await self._publisher.publish_final(registry, n_pending_tasks=n_pending)
            self._publisher.close()
            self._finalize()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        logger.info(
            "Aggregator finalized: %d total records processed", self._total_processed
        )
        self.close()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        elif self.loop is not None and self.loop.is_running():
            self.loop.stop()

    def close(self) -> None:
        try:
            self._publisher.close()
        except Exception:  # noqa: BLE001 — close is best-effort during shutdown.
            logger.exception("metrics: publisher close failed")
        super().close()
