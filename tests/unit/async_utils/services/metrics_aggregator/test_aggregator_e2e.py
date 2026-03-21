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
emits the correct metrics.
"""

import asyncio
import json
import time

import pytest
import zmq
from inference_endpoint.async_utils.event_publisher import EventPublisherService
from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.emitter import (
    JsonlMetricEmitter,
    MetricEmitter,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CollectingEmitter(MetricEmitter):
    """Thread-safe emitter that collects metrics and signals when a target count is reached."""

    def __init__(self):
        self.emitted: list[tuple[str, str, int | float]] = []
        self._target_event: asyncio.Event | None = None
        self._target_count: int = 0
        self.flushed = False
        self.closed = False

    def set_wait_target(self, event: asyncio.Event, count: int) -> None:
        self._target_event = event
        self._target_count = count

    def emit(self, sample_uuid: str, metric_name: str, value: int | float) -> None:
        self.emitted.append((sample_uuid, metric_name, value))
        if self._target_event is not None and len(self.emitted) >= self._target_count:
            self._target_event.set()

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True

    def get_metrics(self, sample_uuid: str) -> dict[str, int | float]:
        return {name: val for uuid, name, val in self.emitted if uuid == sample_uuid}


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
    EventPublisherService._instance = None
    try:
        service = EventPublisherService(zmq_context)
    except zmq.ZMQError as exc:
        EventPublisherService._instance = None
        pytest.skip(f"ZMQ IPC bind unavailable (sandboxed?): {exc}")
    yield service
    service.close()
    EventPublisherService._instance = None


@pytest.fixture
def aggregator_loop():
    manager = LoopManager()
    # Use unique name per test invocation to avoid loop reuse across tests
    name = f"test_metrics_agg_{id(object())}"
    return manager.create_loop(name)


@pytest.fixture
def collecting_emitter():
    return CollectingEmitter()


@pytest.fixture
def shutdown_event():
    return asyncio.Event()


@pytest.fixture
def aggregator(
    publisher, aggregator_loop, zmq_context, collecting_emitter, shutdown_event
):
    """MetricsAggregatorService connected to the publisher via ZMQ."""
    agg = MetricsAggregatorService(
        publisher.bind_address,
        zmq_context,
        aggregator_loop,
        topics=None,
        emitter=collecting_emitter,
        tokenize_pool=None,
        shutdown_event=shutdown_event,
    )
    aggregator_loop.call_soon_threadsafe(agg.start)
    # Allow ZMQ slow-joiner to connect
    time.sleep(0.5)
    yield agg
    if not agg.is_closed:
        agg.close()


def _publish_and_sleep(publisher, record, delay=0.05):
    """Publish a record and sleep briefly to let the event loop drain."""
    publisher.publish(record)
    time.sleep(delay)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAggregatorE2E:
    @pytest.mark.asyncio
    async def test_single_sample_timing_metrics(
        self, publisher, aggregator, collecting_emitter
    ):
        """Full streaming sample lifecycle over real ZMQ pub/sub."""
        done = asyncio.Event()
        # Expect: ttft_ns, chunk_delta_ns, sample_latency_ns = 3 metrics
        collecting_emitter.set_wait_target(done, 3)

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

        m = collecting_emitter.get_metrics("s1")
        assert m["ttft_ns"] == 1000
        assert m["chunk_delta_ns"] == 1000
        assert m["sample_latency_ns"] == 3000

    @pytest.mark.asyncio
    async def test_tracking_window_respected(
        self, publisher, aggregator, collecting_emitter
    ):
        """Samples issued before START_PERFORMANCE_TRACKING are not tracked."""
        done = asyncio.Event()
        # Only s2 should produce metrics (1 metric: sample_latency_ns)
        collecting_emitter.set_wait_target(done, 1)

        # Issue s1 before tracking starts — should be ignored
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

        assert collecting_emitter.get_metrics("s1") == {}
        assert collecting_emitter.get_metrics("s2")["sample_latency_ns"] == 300

    @pytest.mark.asyncio
    async def test_session_ended_triggers_shutdown(
        self, publisher, aggregator, collecting_emitter, shutdown_event
    ):
        """ENDED event causes emitter flush, aggregator close, and shutdown signal."""
        _publish_and_sleep(
            publisher,
            EventRecord(
                event_type=SessionEventType.ENDED,
                timestamp_ns=1000,
            ),
        )
        await asyncio.wait_for(shutdown_event.wait(), timeout=_WAIT_TIMEOUT)
        assert collecting_emitter.flushed
        assert collecting_emitter.closed

    @pytest.mark.asyncio
    async def test_multiple_samples_concurrent(
        self, publisher, aggregator, collecting_emitter
    ):
        """Multiple samples in flight concurrently produce independent metrics."""
        done = asyncio.Event()
        # 2 samples x 2 metrics each (ttft_ns + sample_latency_ns) = 4
        collecting_emitter.set_wait_target(done, 4)

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

        ma = collecting_emitter.get_metrics("a")
        mb = collecting_emitter.get_metrics("b")
        assert ma["ttft_ns"] == 100
        assert ma["sample_latency_ns"] == 300
        assert mb["ttft_ns"] == 200
        assert mb["sample_latency_ns"] == 350

    @pytest.mark.asyncio
    async def test_jsonl_emitter_e2e(
        self, publisher, aggregator_loop, zmq_context, tmp_path
    ):
        """Full pipeline with JsonlMetricEmitter writing to disk."""
        emitter = JsonlMetricEmitter(tmp_path / "metrics", flush_interval=1)
        agg = MetricsAggregatorService(
            publisher.bind_address,
            zmq_context,
            aggregator_loop,
            topics=None,
            emitter=emitter,
        )
        aggregator_loop.call_soon_threadsafe(agg.start)
        time.sleep(0.5)

        try:
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
                    sample_uuid="file-test",
                ),
            )
            _publish_and_sleep(
                publisher,
                EventRecord(
                    event_type=SampleEventType.RECV_FIRST,
                    timestamp_ns=2000,
                    sample_uuid="file-test",
                ),
            )
            _publish_and_sleep(
                publisher,
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    timestamp_ns=3000,
                    sample_uuid="file-test",
                ),
            )

            # Wait for metrics to be written
            for _ in range(30):
                try:
                    content = (tmp_path / "metrics.jsonl").read_text()
                    lines = [line for line in content.strip().split("\n") if line]
                    if len(lines) >= 2:
                        break
                except FileNotFoundError:
                    pass  # File not yet created by the emitter; retry.
                await asyncio.sleep(0.1)

            content = (tmp_path / "metrics.jsonl").read_text()
            lines = [line for line in content.strip().split("\n") if line]
            assert len(lines) >= 2

            records = [json.loads(line) for line in lines]
            metric_names = {r["metric_name"] for r in records}
            assert "ttft_ns" in metric_names
            assert "sample_latency_ns" in metric_names

            ttft = next(r for r in records if r["metric_name"] == "ttft_ns")
            assert ttft["value"] == 1000
            assert ttft["sample_uuid"] == "file-test"
        finally:
            if not agg.is_closed:
                agg.close()
