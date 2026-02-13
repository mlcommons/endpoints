#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ZMQ Pub-Sub Demo using async_utils (EventRecord, EventPublisherService, etc.)

Demonstrates the intended control flow:
- Publisher auto-starts upon importing async_utils.autoinit (EVENT_PUBLISHER).
- Each subscriber has its own event loop (LoopManager.create_loop); init does NOT start processing.
- When ready, .start() is called on each subscriber to add the reader and begin receiving.
- process(records) is async and scheduled via create_task so it does not block the socket.
- Cleanup: .close() on subscribers when the session has ended.

Same logical behavior as zmq_pubsub_simple_demo.py (console log, file output, duration stats)
but using the async_utils APIs and no extra queuing layer.

Usage:
    python scripts/zmq_pubsub_async_utils_demo.py
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path

# Auto-start publisher and loop manager so EVENT_PUBLISHER.bind_address is available.
from inference_endpoint.async_utils import autoinit  # noqa: F401
from inference_endpoint.async_utils.transport.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqEventRecordSubscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Subscribers: each has its own loop and implements async process()
# =============================================================================


class ConsoleSubscriber(ZmqEventRecordSubscriber):
    """Logs events to console. process() is async and runs when records are received."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_count = 0

    async def process(self, records: list[EventRecord]) -> None:
        for event in records:
            if event.event_type == SessionEventType.ENDED:
                logger.info("[Console] Received shutdown signal (session.ended)")
            self.event_count += 1
            sample_id = event.sample_uuid[:8] if event.sample_uuid else "N/A"
            logger.info(
                f"[Console] {event.event_type.topic} | {event.event_type.name} | "
                f"sample={sample_id} | data={event.data}"
            )


class FileSubscriber(ZmqEventRecordSubscriber):
    """Writes events to a file. process() is async."""

    def __init__(self, output_file: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file = output_file
        self.event_count = 0
        self._file = open(output_file, "w")
        self._file.write("timestamp_ns,topic,event_type,sample_uuid,data\n")

    async def process(self, records: list[EventRecord]) -> None:
        for event in records:
            if event.event_type == SessionEventType.ENDED:
                logger.info("[File] Received shutdown signal (session.ended)")
            self.event_count += 1
            data_str = str(event.data) if event.data else ""
            self._file.write(
                f"{event.timestamp_ns},{event.event_type.topic},"
                f"{event.event_type.name},{event.sample_uuid},{data_str}\n"
            )
        self._file.flush()

    def close(self) -> None:
        if not self.is_closed and hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except OSError:
                # File may already be closed or I/O error on close (e.g. disk full).
                pass
            self._file = None
        super().close()


class DurationSubscriber(ZmqEventRecordSubscriber):
    """Tracks sample durations from ISSUED to COMPLETE. process() is async."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_times: dict[str, int] = {}
        self.durations: dict[str, int] = {}
        self.event_count = 0

    async def process(self, records: list[EventRecord]) -> None:
        for event in records:
            if event.event_type == SessionEventType.ENDED:
                logger.info("[Duration] Received shutdown signal (session.ended)")
            self.event_count += 1
            if event.event_type == SampleEventType.ISSUED:
                self.start_times[event.sample_uuid] = event.timestamp_ns
            elif event.event_type == SampleEventType.COMPLETE:
                if event.sample_uuid in self.start_times:
                    start_ns = self.start_times[event.sample_uuid]
                    duration_ns = event.timestamp_ns - start_ns
                    self.durations[event.sample_uuid] = duration_ns
                    sample_id = event.sample_uuid[:8]
                    logger.info(
                        f"[Duration] Sample {sample_id} completed in {duration_ns}ns"
                    )

    def close(self) -> None:
        if self.durations:
            durations_ns = list(self.durations.values())
            avg_ns = sum(durations_ns) / len(durations_ns)
            min_ns = min(durations_ns)
            max_ns = max(durations_ns)
            logger.info(
                f"Duration stats: avg={avg_ns:.0f}ns, min={min_ns}ns, max={max_ns}ns"
            )
        super().close()


# =============================================================================
# Publish test events (same sequence as simple demo)
# =============================================================================


async def publish_test_events() -> None:
    """Publish hard-coded events using EVENT_PUBLISHER and EventRecord."""
    publisher = autoinit.EVENT_PUBLISHER
    logger.info("Waiting for subscribers to connect...")
    await asyncio.sleep(0.5)

    logger.info("Publishing test events...")
    uuid1 = uuid.uuid4().hex
    uuid2 = uuid.uuid4().hex
    uuid3 = uuid.uuid4().hex

    events: list[EventRecord] = [
        EventRecord(
            event_type=SampleEventType.ISSUED,
            timestamp_ns=10000,
            sample_uuid=uuid1,
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED,
            timestamp_ns=10003,
            sample_uuid=uuid2,
        ),
        EventRecord(
            event_type=SampleEventType.RECV_FIRST,
            timestamp_ns=10010,
            sample_uuid=uuid1,
            data={"ttft_ms": 10.0},
        ),
        EventRecord(
            event_type=SampleEventType.RECV_FIRST,
            timestamp_ns=10190,
            sample_uuid=uuid2,
            data={"ttft_ms": 187.0},
        ),
        EventRecord(
            event_type=SampleEventType.RECV_NON_FIRST,
            timestamp_ns=10201,
            sample_uuid=uuid1,
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED,
            timestamp_ns=10202,
            sample_uuid=uuid3,
        ),
        EventRecord(
            event_type=SampleEventType.RECV_NON_FIRST,
            timestamp_ns=10203,
            sample_uuid=uuid1,
        ),
        EventRecord(
            event_type=SampleEventType.RECV_NON_FIRST,
            timestamp_ns=10210,
            sample_uuid=uuid2,
        ),
        EventRecord(
            event_type=SampleEventType.RECV_NON_FIRST,
            timestamp_ns=10211,
            sample_uuid=uuid1,
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=10211,
            sample_uuid=uuid1,
            data={"tokens": 50},
        ),
        EventRecord(
            event_type=SampleEventType.RECV_NON_FIRST,
            timestamp_ns=10214,
            sample_uuid=uuid2,
        ),
        EventRecord(
            event_type=SampleEventType.RECV_NON_FIRST,
            timestamp_ns=10217,
            sample_uuid=uuid2,
        ),
        EventRecord(
            event_type=SampleEventType.RECV_NON_FIRST,
            timestamp_ns=10219,
            sample_uuid=uuid2,
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=10219,
            sample_uuid=uuid2,
            data={"tokens": 75},
        ),
    ]

    logger.info(f"Generated {len(events)} events for 3 samples")
    logger.info(f"Sample UUIDs: {uuid1[:8]}, {uuid2[:8]}, {uuid3[:8]}")

    for i, event in enumerate(events, 1):
        publisher.publish(event)
        logger.info(f"Published event {i}/{len(events)}: {event.event_type.topic}")
        await asyncio.sleep(0.05)

    shutdown_event = EventRecord(
        event_type=SessionEventType.ENDED,
        timestamp_ns=time.monotonic_ns(),
        sample_uuid="",
    )
    publisher.publish(shutdown_event)
    logger.info("Sending shutdown signal (session.ended)...")
    await asyncio.sleep(0.2)
    logger.info("All events published")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    logger.info("=" * 80)
    logger.info("ZMQ Pub-Sub Demo (async_utils)")
    logger.info("=" * 80)

    output_file = Path("/tmp/zmq_events_async_utils_output.csv")
    manager = autoinit.LOOP_MANAGER
    publisher = autoinit.EVENT_PUBLISHER
    connect_address = publisher.bind_address

    # Each subscriber has its own event loop (not shared with the publisher).
    console_loop = manager.create_loop("demo_console")
    file_loop = manager.create_loop("demo_file")
    duration_loop = manager.create_loop("demo_duration")

    logger.info("Creating subscribers (init does NOT start processing)...")
    console_sub = ConsoleSubscriber(
        connect_address=connect_address,
        loop=console_loop,
        topics=None,
    )
    file_sub = FileSubscriber(
        output_file,
        connect_address=connect_address,
        loop=file_loop,
        topics=None,
    )
    duration_sub = DurationSubscriber(
        connect_address=connect_address,
        loop=duration_loop,
        topics=[
            SampleEventType.ISSUED.topic,
            SampleEventType.COMPLETE.topic,
            SessionEventType.ENDED.topic,
        ],
    )
    logger.info("Subscribers created")

    # Start listening (add reader to each loop).
    logger.info("Starting subscribers (.start())...")
    console_loop.call_soon_threadsafe(console_sub.start)
    file_loop.call_soon_threadsafe(file_sub.start)
    duration_loop.call_soon_threadsafe(duration_sub.start)

    try:
        await publish_test_events()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Cleaning up (closing subscribers)...")
        console_sub.close()
        file_sub.close()
        duration_sub.close()
        logger.info("=" * 80)
        logger.info(f"Output file written to: {output_file}")
        logger.info("Demo complete")
        logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
