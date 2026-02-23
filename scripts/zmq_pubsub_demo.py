#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ZMQ Pub-Sub Demo using async_utils (EventRecord, EventPublisherService, etc.)

Demonstrates the intended control flow:
- Publisher should be created within a ManagedZMQContext.scoped() context manager.
- Each subscriber has its own event loop (LoopManager.create_loop); init does NOT start processing.
- When ready, .start() is called on each subscriber to add the reader and begin receiving.
- process(records) is async and scheduled via create_task so it does not block the socket.
- Cleanup: .close() on subscribers when the session has ended.

The demo runs the event_logger with both JSONL and SQL writers, then opens the SQLite
database and prints its contents to verify SQLWriter worked.

Usage:
    python scripts/zmq_pubsub_demo.py
"""

import asyncio
import logging
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

from inference_endpoint.async_utils.event_publisher import EventPublisherService
from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqEventRecordSubscriber
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Subscribers: each has its own loop and implements async process()
# =============================================================================


def _is_error_event(event: EventRecord) -> bool:
    """True if the event is an error (should not be dropped after ENDED)."""
    return isinstance(event.event_type, ErrorEventType)


class ConsoleSubscriber(ZmqEventRecordSubscriber):
    """Logs events to console. Stops and cleans up on SessionEventType.ENDED."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_count = 0
        self._shutdown_received = False

    async def process(self, records: list[EventRecord]) -> None:
        for event in records:
            if self._shutdown_received and not _is_error_event(event):
                continue
            if event.event_type == SessionEventType.ENDED:
                self._shutdown_received = True
                logger.info("[Console] Received session ended signal (session.ended)")
                self.loop.call_soon_threadsafe(self.close)
            self.event_count += 1
            sample_id = event.sample_uuid[:8] if event.sample_uuid else "N/A"
            logger.info(
                f"[Console] {event.event_type.topic} | {event.event_type.name} | "
                f"sample={sample_id} | data={event.data}"
            )


class DurationSubscriber(ZmqEventRecordSubscriber):
    """Tracks sample durations from ISSUED to COMPLETE. Stops on SessionEventType.ENDED."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_times: dict[str, int] = {}
        self.durations: dict[str, int] = {}
        self.event_count = 0
        self._shutdown_received = False

    async def process(self, records: list[EventRecord]) -> None:
        for event in records:
            if self._shutdown_received and not _is_error_event(event):
                continue
            if event.event_type == SessionEventType.ENDED:
                self._shutdown_received = True
                logger.info("[Duration] Received session ended signal (session.ended)")
                self.loop.call_soon_threadsafe(self.close)
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


async def publish_test_events(publisher) -> None:
    """Publish hard-coded events using EventPublisherService and EventRecord."""
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
    logger.info("Sending session ended signal (session.ended)...")
    await asyncio.sleep(0.2)
    logger.info("All events published")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    logger.info("=" * 80)
    logger.info("ZMQ Pub-Sub Demo (async_utils)")
    logger.info("=" * 80)

    event_log_dir = Path("/tmp/zmq_demo_event_logger")
    event_log_dir.mkdir(parents=True, exist_ok=True)

    with ManagedZMQContext.scoped() as zmq_ctx:
        publisher = EventPublisherService(zmq_ctx)
        connect_address = publisher.bind_address

        # Start event_logger as a subprocess (logs to both JSONL and SQL under event_log_dir).
        event_logger_cmd = [
            sys.executable,
            "-m",
            "inference_endpoint.async_utils.services.event_logger",
            "--log-dir",
            str(event_log_dir),
            "--socket-address",
            connect_address,
            "--writers",
            "jsonl",
            "sql",
        ]
        logger.info("Starting event_logger subprocess: %s", " ".join(event_logger_cmd))
        event_logger_proc = subprocess.Popen(
            event_logger_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Each in-process subscriber has its own event loop (not shared with the publisher).
        loop_manager = LoopManager()
        console_loop = loop_manager.create_loop("demo_console")
        duration_loop = loop_manager.create_loop("demo_duration")

        logger.info("Creating subscribers (init does NOT start processing)...")
        console_sub = ConsoleSubscriber(
            connect_address=connect_address,
            zmq_context=zmq_ctx,
            loop=console_loop,
            topics=None,
        )
        duration_sub = DurationSubscriber(
            connect_address=connect_address,
            zmq_context=zmq_ctx,
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
        duration_loop.call_soon_threadsafe(duration_sub.start)

        try:
            await publish_test_events(publisher)
            # Give event_logger time to receive ENDED and exit.
            try:
                event_logger_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("event_logger did not exit within 5s, terminating")
                event_logger_proc.terminate()
                event_logger_proc.wait(timeout=2)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info("Cleaning up (closing subscribers)...")
            console_sub.close()
            duration_sub.close()
            if event_logger_proc.poll() is None:
                event_logger_proc.terminate()
                event_logger_proc.wait(timeout=2)

            # Verify SQLWriter: open the SQLite DB and show contents.
            events_db = event_log_dir / "events.db"
            if events_db.exists():
                logger.info("=" * 80)
                logger.info("SQLWriter verification: contents of %s", events_db)
                logger.info("=" * 80)
                conn = sqlite3.connect(str(events_db))
                try:
                    cur = conn.execute(
                        "SELECT id, sample_uuid, event_type, timestamp_ns, data FROM events ORDER BY id"
                    )
                    rows = cur.fetchall()
                    for row in rows:
                        row_id, sample_uuid, event_type, timestamp_ns, data = row
                        sample_short = (
                            (sample_uuid[:8] + "..") if sample_uuid else "N/A"
                        )
                        data_preview = (
                            data[:60] + b"..."
                            if data and len(data) > 60
                            else data or b""
                        ) or b""
                        logger.info(
                            "  id=%s | event_type=%s | sample=%s | timestamp_ns=%s | data=%s",
                            row_id,
                            event_type,
                            sample_short,
                            timestamp_ns,
                            data_preview.decode("utf-8", errors="replace"),
                        )
                    logger.info("Total rows: %s", len(rows))
                finally:
                    conn.close()
            else:
                logger.warning(
                    "SQL DB not found at %s (SQL writer may not have been used)",
                    events_db,
                )

            logger.info("=" * 80)
            logger.info("Event log directory: %s", event_log_dir)
            logger.info("Demo complete")
            logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
