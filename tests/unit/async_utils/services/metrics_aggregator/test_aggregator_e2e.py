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

"""End-to-end tests for MetricsAggregatorService with real ZMQ pub/sub.

These tests launch an EventPublisherService, connect a MetricsAggregatorService
over ZMQ IPC, publish EventRecords, and verify the aggregator computes and
emits the correct metrics into the KVStore.
"""

import asyncio
import time
from threading import Lock

import pytest
import zmq
from inference_endpoint.async_utils.event_publisher import EventPublisherService
from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)

from .conftest import InMemoryKVStore

# ---------------------------------------------------------------------------
# Signaling KVStore for e2e tests
# ---------------------------------------------------------------------------


class SignalingKVStore(InMemoryKVStore):
    """InMemoryKVStore that signals an asyncio.Event when a target series count is reached.

    This replaces the old CollectingEmitter.set_wait_target() pattern. Call
    set_wait_target(event, count) before publishing records; the event will be
    set once the total number of series values across all series keys reaches
    the target count.
    """

    def __init__(self) -> None:
        super().__init__()
        self._target_event: asyncio.Event | None = None
        self._target_count: int = 0
        self._lock = Lock()

    def set_wait_target(self, event: asyncio.Event, count: int) -> None:
        self._target_event = event
        self._target_count = count

    def update(self, key: str, value: float) -> None:
        super().update(key, value)
        with self._lock:
            if self._target_event is not None:
                total = sum(len(v) for v in self._series.values())
                if total >= self._target_count:
                    self._target_event.set()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WAIT_TIMEOUT = 3.0


@pytest.fixture
def zmq_context():
    with ManagedZMQContext.scoped() as ctx:
        yield ctx


@pytest.fixture
def publisher(zmq_context):
    try:
        service = EventPublisherService(zmq_context)
    except zmq.ZMQError as exc:
        pytest.skip(f"ZMQ IPC bind unavailable (sandboxed?): {exc}")
    yield service
    service.close()


@pytest.fixture
def aggregator_loop():
    manager = LoopManager()
    # Use unique name per test invocation to avoid loop reuse across tests
    name = f"test_metrics_agg_{id(object())}"
    return manager.create_loop(name)


@pytest.fixture
def signaling_store():
    return SignalingKVStore()


@pytest.fixture
def shutdown_event():
    return asyncio.Event()


@pytest.fixture
def aggregator(
    publisher, aggregator_loop, zmq_context, signaling_store, shutdown_event
):
    """MetricsAggregatorService connected to the publisher via ZMQ."""
    agg = MetricsAggregatorService(
        publisher.bind_path,
        zmq_context,
        aggregator_loop,
        topics=None,
        kv_store=signaling_store,
        tokenize_pool=None,
        streaming=True,
        shutdown_event=shutdown_event,
    )
    aggregator_loop.call_soon_threadsafe(agg.start)
    # Allow ZMQ slow-joiner to connect
    time.sleep(0.5)
    yield agg
    if not agg.is_closed:
        agg.close()


def _publish_and_sleep(publisher, record, delay=0.05):
    """Publish a record, flush, and sleep briefly to let the event loop drain."""
    publisher.publish(record)
    publisher.flush()
    time.sleep(delay)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAggregatorE2E:
    @pytest.mark.asyncio
    async def test_single_sample_timing_metrics(
        self, publisher, aggregator, signaling_store
    ):
        """Full streaming sample lifecycle over real ZMQ pub/sub."""
        done = asyncio.Event()
        # Expect: ttft_ns, chunk_delta_ns, sample_latency_ns = 3 series values
        signaling_store.set_wait_target(done, 3)

        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SessionEventType.START_PERFORMANCE_TRACKING,
                timestamp_ns=0,
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=1000,
                sample_uuid="s1",
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.RECV_FIRST,
                timestamp_ns=2000,
                sample_uuid="s1",
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.RECV_NON_FIRST,
                timestamp_ns=3000,
                sample_uuid="s1",
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                timestamp_ns=4000,
                sample_uuid="s1",
            ),
        )

        await asyncio.wait_for(done.wait(), timeout=_WAIT_TIMEOUT)

        assert 1000 in signaling_store.get_series_values("ttft_ns")
        assert 1000 in signaling_store.get_series_values("chunk_delta_ns")
        assert 3000 in signaling_store.get_series_values("sample_latency_ns")

    @pytest.mark.asyncio
    async def test_tracking_window_respected(
        self, publisher, aggregator, signaling_store
    ):
        """Samples issued before START_PERFORMANCE_TRACKING are not tracked."""
        done = asyncio.Event()
        # Only s2 should produce metrics (1 metric: sample_latency_ns)
        signaling_store.set_wait_target(done, 1)

        # Issue s1 before tracking starts -- should be ignored
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=100,
                sample_uuid="s1",
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SessionEventType.START_PERFORMANCE_TRACKING,
                timestamp_ns=200,
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=300,
                sample_uuid="s2",
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                timestamp_ns=500,
                sample_uuid="s1",
            ),
        )
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                timestamp_ns=600,
                sample_uuid="s2",
            ),
        )

        await asyncio.wait_for(done.wait(), timeout=_WAIT_TIMEOUT)

        assert 300 in signaling_store.get_series_values("sample_latency_ns")
        # s1 should not have produced any latency values besides s2's
        latencies = signaling_store.get_series_values("sample_latency_ns")
        assert len(latencies) == 1

    @pytest.mark.asyncio
    async def test_session_ended_triggers_shutdown(
        self, publisher, aggregator, signaling_store, shutdown_event
    ):
        """ENDED event causes store close and shutdown signal."""
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SessionEventType.ENDED,
                timestamp_ns=1000,
            ),
        )
        await asyncio.wait_for(shutdown_event.wait(), timeout=_WAIT_TIMEOUT)
        assert signaling_store.closed

    @pytest.mark.asyncio
    async def test_multiple_samples_concurrent(
        self, publisher, aggregator, signaling_store
    ):
        """Multiple samples in flight concurrently produce independent metrics."""
        done = asyncio.Event()
        # 2 samples x 2 metrics each (ttft_ns + sample_latency_ns) = 4
        signaling_store.set_wait_target(done, 4)

        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SessionEventType.START_PERFORMANCE_TRACKING,
                timestamp_ns=0,
            ),
        )
        for uuid, issued_ts, recv_ts, complete_ts in [
            ("a", 100, 200, 400),
            ("b", 150, 350, 500),
        ]:
            _publish_and_sleep(
                publisher,
                EventRecord(
                    event_type=SampleEventType.ISSUED,
                    timestamp_ns=issued_ts,
                    sample_uuid=uuid,
                ),
            )
            _publish_and_sleep(
                publisher,
                EventRecord(
                    event_type=SampleEventType.RECV_FIRST,
                    timestamp_ns=recv_ts,
                    sample_uuid=uuid,
                ),
            )
            _publish_and_sleep(
                publisher,
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    timestamp_ns=complete_ts,
                    sample_uuid=uuid,
                ),
            )

        await asyncio.wait_for(done.wait(), timeout=_WAIT_TIMEOUT)

        ttfts = signaling_store.get_series_values("ttft_ns")
        latencies = signaling_store.get_series_values("sample_latency_ns")
        assert 100 in ttfts  # a: 200 - 100
        assert 300 in latencies  # a: 400 - 100
        assert 200 in ttfts  # b: 350 - 150
        assert 350 in latencies  # b: 500 - 150
