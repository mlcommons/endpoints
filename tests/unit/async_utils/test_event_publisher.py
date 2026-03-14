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

"""Unit tests for EventPublisherService and event publish/subscribe flow.

Uses a test subscriber that collects received EventRecords for assertions.
Follows the same flow as scripts/zmq_pubsub_async_utils_demo.py:
- Subscribers use their own event loop (not the publisher's), running in a separate thread.
- .start() is scheduled on the subscriber's loop via call_soon_threadsafe so the reader
  is registered on the correct loop.
- Allow time for subscribers to connect (ZMQ slow-joiner) before publishing.
- Publish EventRecord instances (same API as the demo).
- process() is async and scheduled via create_task. Wait conditions use asyncio.Event.
"""

import asyncio
import time

import pytest
import zmq
from inference_endpoint.async_utils.event_publisher import EventPublisherService
from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqEventRecordSubscriber
from inference_endpoint.core.record import (
    TOPIC_FRAME_SIZE,
    EventRecord,
    SampleEventType,
    SessionEventType,
    decode_event_record,
)
from inference_endpoint.core.types import TextModelOutput

# Default timeout when waiting for records in tests.
_WAIT_RECORDS_TIMEOUT = 1


# =============================================================================
# Test subscriber: collects received EventRecords (own loop, async process)
# =============================================================================


class CollectingEventSubscriber(ZmqEventRecordSubscriber):
    """Subscriber that appends all received EventRecords to a list for tests.

    Uses its own event loop (passed in). Call .start() to begin receiving.
    Supports event-based waiting: call set_wait_target(event, count) then
    await event.wait(); the event is set when len(received) >= count.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received: list[EventRecord] = []
        self._wait_event: asyncio.Event | None = None
        self._wait_count: int | None = None

    def set_wait_target(self, event: asyncio.Event, count: int) -> None:
        """Set an event to be set when at least `count` records have been received."""
        self._wait_event = event
        self._wait_count = count
        if len(self.received) >= count:
            event.set()

    async def process(self, records: list[EventRecord]) -> None:
        """Append received records and set wait event if target reached."""
        self.received.extend(records)
        if (
            self._wait_event is not None
            and self._wait_count is not None
            and len(self.received) >= self._wait_count
        ):
            self._wait_event.set()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ev_pub_zmq_context():
    """Scoped ZMQ context for EventPublisherService; lifecycle tied to tests that use it."""
    with ManagedZMQContext.scoped() as zmq_ctx:
        yield zmq_ctx


@pytest.fixture
def event_publisher_service(ev_pub_zmq_context):
    """Create EventPublisherService; socket directory comes from zmq_context."""
    EventPublisherService._instance = None
    service = EventPublisherService(ev_pub_zmq_context)
    yield service
    service.close()
    EventPublisherService._instance = None


@pytest.fixture
def subscriber_loop():
    """Dedicated event loop for the test subscriber (not shared with publisher)."""
    manager = LoopManager()
    return manager.create_loop("test_event_pub_sub")


@pytest.fixture
def collecting_subscriber(event_publisher_service, subscriber_loop, ev_pub_zmq_context):
    """Create a subscriber with its own loop; schedule .start() on that loop."""
    subscriber = CollectingEventSubscriber(
        connect_address=event_publisher_service.bind_address,
        zmq_context=ev_pub_zmq_context,
        loop=subscriber_loop,
        topics=None,
    )
    # Schedule start on the subscriber's loop (runs in another thread) so add_reader is correct
    subscriber_loop.call_soon_threadsafe(subscriber.start)
    # Allow the subscriber thread to register the reader and ZMQ to establish connection
    time.sleep(0.5)
    yield subscriber
    subscriber.close()


# =============================================================================
# EventPublisherService singleton and publish
# =============================================================================


class TestEventPublisherService:
    """Tests for EventPublisherService."""

    @pytest.mark.asyncio
    async def test_publish_sends_data_on_ipc_socket(
        self, event_publisher_service, ev_pub_zmq_context
    ):
        """Manual ZMQ SUB socket reader verifies data is sent when publish() is called."""
        sub = ev_pub_zmq_context.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVTIMEO, int(_WAIT_RECORDS_TIMEOUT * 1000))
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.connect(event_publisher_service.bind_address)
        # Allow ZMQ slow-joiner to establish connection
        await asyncio.sleep(0.05)
        record = EventRecord(
            event_type=SessionEventType.STARTED,
            sample_uuid="",
        )
        event_publisher_service.publish(record)
        # Yield so the publisher's event loop can drain the send buffer
        await asyncio.sleep(0.1)
        loop = asyncio.get_event_loop()
        # Publisher sends a single frame: padded_topic (TOPIC_FRAME_SIZE bytes) + payload
        frame = await asyncio.wait_for(
            loop.run_in_executor(None, sub.recv),
            timeout=_WAIT_RECORDS_TIMEOUT,
        )
        assert len(frame) > TOPIC_FRAME_SIZE, "Expected single frame (topic + payload)"
        topic_bytes = frame[:TOPIC_FRAME_SIZE].rstrip(b"\x00")
        payload = frame[TOPIC_FRAME_SIZE:]
        assert topic_bytes == b"session.started"
        rec = decode_event_record(bytes(payload))
        assert rec.event_type.value == SessionEventType.STARTED.value
        assert rec.data is None
        # Socket is closed by ManagedZMQContext.cleanup() in ev_pub_zmq_context fixture teardown.

    def test_singleton_returns_same_instance(
        self, event_publisher_service, ev_pub_zmq_context
    ):
        """Multiple EventPublisherService(ctx) calls return the same instance."""
        p1 = event_publisher_service
        p2 = EventPublisherService(ev_pub_zmq_context)
        assert p1 is p2
        assert p1 is event_publisher_service

    @pytest.mark.asyncio
    async def test_publish_session_event_received_by_subscriber(
        self, event_publisher_service, collecting_subscriber
    ):
        """Publishing a session event is received by the collecting subscriber."""
        received_event = asyncio.Event()
        collecting_subscriber.set_wait_target(received_event, 1)
        record = EventRecord(
            event_type=SessionEventType.STARTED,
            sample_uuid="",
        )
        event_publisher_service.publish(record)
        await asyncio.sleep(0.05)  # Let publisher drain send buffer
        await asyncio.wait_for(received_event.wait(), timeout=_WAIT_RECORDS_TIMEOUT)
        assert len(collecting_subscriber.received) == 1
        rec = collecting_subscriber.received[0]
        assert rec.event_type.value == SessionEventType.STARTED.value
        assert rec.data is None

    @pytest.mark.asyncio
    async def test_publish_sample_event_received_by_subscriber(
        self, event_publisher_service, collecting_subscriber
    ):
        """Publishing a sample event is received by the collecting subscriber."""
        received_event = asyncio.Event()
        collecting_subscriber.set_wait_target(received_event, 1)
        data = TextModelOutput(output="sample output")
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="sample-1",
            data=data,
        )
        event_publisher_service.publish(record)
        await asyncio.sleep(0.05)  # Let publisher drain send buffer
        await asyncio.wait_for(received_event.wait(), timeout=_WAIT_RECORDS_TIMEOUT)
        assert len(collecting_subscriber.received) == 1
        rec = collecting_subscriber.received[0]
        assert rec.event_type.value == SampleEventType.COMPLETE.value
        assert rec.sample_uuid == "sample-1"
        assert rec.data == data

    @pytest.mark.asyncio
    async def test_multiple_events_received_in_order(
        self, event_publisher_service, collecting_subscriber
    ):
        """Multiple published events are received in order."""
        received_event = asyncio.Event()
        collecting_subscriber.set_wait_target(received_event, 3)
        for i in range(3):
            record = EventRecord(
                event_type=SampleEventType.ISSUED,
                sample_uuid=f"sample-{i}",
            )
            event_publisher_service.publish(record)
            await asyncio.sleep(0.05)  # Small delay between events (demo-style)
        await asyncio.wait_for(received_event.wait(), timeout=_WAIT_RECORDS_TIMEOUT)
        assert len(collecting_subscriber.received) == 3
        for i in range(3):
            assert collecting_subscriber.received[i].sample_uuid == f"sample-{i}"
            assert collecting_subscriber.received[i].data is None
