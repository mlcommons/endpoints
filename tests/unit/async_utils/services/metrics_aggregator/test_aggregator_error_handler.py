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

"""Targeted regression tests for the aggregator's ERROR-event handler.

These tests cover the ``TRACKED_SAMPLES_FAILED`` increment path (design
v5 §3) without reviving the broader ``test_aggregator.py`` module. They
construct the aggregator with a mocked publisher and inject events
directly via ``process()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricCounterKey,
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    CounterStat,
    SessionState,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    ErrorData,
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)


def _counters(registry: MetricsRegistry) -> dict[str, int | float]:
    """Read all counter values via a snapshot. State/n_pending don't matter
    for counter inspection; we just need the snapshot to materialize values."""
    snap = registry.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
    return {m.name: m.value for m in snap.metrics if isinstance(m, CounterStat)}


def _make_aggregator(
    zmq_ctx: ManagedZMQContext,
    loop,
    socket_name: str,
    *,
    streaming: bool = False,
) -> tuple[MetricsAggregatorService, MetricsRegistry, MagicMock]:
    """Construct an aggregator with a real ZMQ subscriber and a mocked
    publisher. ``start()`` is intentionally NOT called — we don't want the
    socket reader added to the loop, since we'll inject events directly via
    ``process()``.

    ``zmq_ctx`` must have a ``socket_dir`` set (pass via ``ManagedZMQContext.
    scoped(socket_dir=...)``) since the aggregator's SUB socket connects on
    IPC.
    """
    registry = MetricsRegistry()
    publisher = MagicMock()
    agg = MetricsAggregatorService(
        socket_name,
        zmq_ctx,
        loop,
        registry=registry,
        publisher=publisher,
        refresh_hz=4.0,
        sig_figs=3,
        n_histogram_buckets=10,
        streaming=streaming,
    )
    return agg, registry, publisher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_event_increments_tracked_failed_when_row_exists(tmp_path):
    """ERROR for a tracked, in-flight sample increments BOTH total and
    tracked failure counters.

    Regression for design v5 §3: this only works because session.py emits
    ERROR before COMPLETE — if the order regresses, the row is removed by
    set_field(...COMPLETE...) before the ERROR handler runs and
    ``TRACKED_SAMPLES_FAILED`` silently stays at 0.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as zmq_ctx:
        agg, registry, _ = _make_aggregator(zmq_ctx, loop, "test_agg_err_in_flight")
        try:
            ts = 1_000_000_000
            uuid = "tracked-uuid-1"

            await agg.process(
                [
                    EventRecord(event_type=SessionEventType.STARTED, timestamp_ns=ts),
                    EventRecord(
                        event_type=SessionEventType.START_PERFORMANCE_TRACKING,
                        timestamp_ns=ts,
                    ),
                    EventRecord(
                        event_type=SampleEventType.ISSUED,
                        timestamp_ns=ts + 100,
                        sample_uuid=uuid,
                    ),
                ]
            )
            # Pre-condition: ISSUED while tracking creates a row.
            assert agg._table.get_row(uuid) is not None

            # ERROR arrives while the row is still in flight.
            await agg.process(
                [
                    EventRecord(
                        event_type=ErrorEventType.GENERIC,
                        timestamp_ns=ts + 200,
                        sample_uuid=uuid,
                        data=ErrorData(error_type="t", error_message="boom"),
                    )
                ]
            )

            counters = _counters(registry)
            assert counters[MetricCounterKey.TOTAL_SAMPLES_FAILED.value] == 1
            assert counters[MetricCounterKey.TRACKED_SAMPLES_FAILED.value] == 1
        finally:
            agg.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_after_complete_misses_tracked_failed(tmp_path):
    """If COMPLETE arrives before ERROR, the tracked row is gone and the
    aggregator cannot tell the failure was tracked. This documents the
    failure mode that motivated the session.py event-order swap.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as zmq_ctx:
        agg, registry, _ = _make_aggregator(
            zmq_ctx, loop, "test_agg_err_after_complete"
        )
        try:
            ts = 1_000_000_000
            uuid = "out-of-order-uuid"

            # Reverse-order delivery: COMPLETE then ERROR.
            await agg.process(
                [
                    EventRecord(event_type=SessionEventType.STARTED, timestamp_ns=ts),
                    EventRecord(
                        event_type=SessionEventType.START_PERFORMANCE_TRACKING,
                        timestamp_ns=ts,
                    ),
                    EventRecord(
                        event_type=SampleEventType.ISSUED,
                        timestamp_ns=ts + 100,
                        sample_uuid=uuid,
                    ),
                    EventRecord(
                        event_type=SampleEventType.COMPLETE,
                        timestamp_ns=ts + 200,
                        sample_uuid=uuid,
                    ),
                    EventRecord(
                        event_type=ErrorEventType.GENERIC,
                        timestamp_ns=ts + 201,
                        sample_uuid=uuid,
                        data=ErrorData(error_type="t", error_message="boom"),
                    ),
                ]
            )

            counters = _counters(registry)
            # Total still increments — the ERROR is observed.
            assert counters[MetricCounterKey.TOTAL_SAMPLES_FAILED.value] == 1
            # But tracked DOES NOT — the row was already gone. This is the
            # bug the session.py event-order swap was added to prevent.
            assert counters[MetricCounterKey.TRACKED_SAMPLES_FAILED.value] == 0
        finally:
            agg.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_for_untracked_sample_only_increments_total(tmp_path):
    """Sample issued outside a tracking window has no row. ERROR for it
    increments TOTAL but not TRACKED.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as zmq_ctx:
        agg, registry, _ = _make_aggregator(zmq_ctx, loop, "test_agg_err_untracked")
        try:
            ts = 1_000_000_000
            uuid = "untracked-uuid"

            await agg.process(
                [
                    EventRecord(event_type=SessionEventType.STARTED, timestamp_ns=ts),
                    # No START_PERFORMANCE_TRACKING — ISSUED creates no row.
                    EventRecord(
                        event_type=SampleEventType.ISSUED,
                        timestamp_ns=ts + 100,
                        sample_uuid=uuid,
                    ),
                ]
            )
            assert agg._table.get_row(uuid) is None

            await agg.process(
                [
                    EventRecord(
                        event_type=ErrorEventType.GENERIC,
                        timestamp_ns=ts + 200,
                        sample_uuid=uuid,
                        data=ErrorData(error_type="t", error_message="boom"),
                    )
                ]
            )

            counters = _counters(registry)
            assert counters[MetricCounterKey.TOTAL_SAMPLES_FAILED.value] == 1
            assert counters[MetricCounterKey.TRACKED_SAMPLES_FAILED.value] == 0
        finally:
            agg.close()
