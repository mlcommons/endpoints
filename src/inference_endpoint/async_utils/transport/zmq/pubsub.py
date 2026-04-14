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
# See the for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
from collections import deque
from urllib.parse import urlparse

import msgspec.msgpack
import zmq

from inference_endpoint.async_utils.transport.protocol import (
    EventRecordPublisher,
    EventRecordSubscriber,
)
from inference_endpoint.core.record import BATCH_TOPIC, TOPIC_FRAME_SIZE

from .context import ManagedZMQContext

logger = logging.getLogger(__name__)

_batch_encoder = msgspec.msgpack.Encoder()
_batch_decoder = msgspec.msgpack.Decoder(type=list[bytes])


class ZmqEventRecordPublisher(EventRecordPublisher):
    """ZMQ PUB socket publisher with batched sending.

    Records are buffered in memory and flushed as a single msgpack-encoded
    batch when the buffer reaches ``send_threshold``. This reduces syscalls
    from one per record to one per batch (~19x throughput, ~29% smaller).

    The ``send_threshold`` is the *minimum* number of records in the buffer
    before an automatic flush is triggered. There is no maximum — records
    accumulate until the threshold is reached or ``flush()``/``close()``
    is called explicitly. Callers that need immediate delivery (e.g.,
    session control events) should call ``flush()`` after publishing.

    Batching protocol:
      - Batched messages use ``BATCH_TOPIC`` as the ZMQ routing prefix.
      - The payload is ``msgpack(list[bytes])`` where each element is a
        pre-encoded record payload (no per-record topic prefix).
      - Subscribers unpack the list and yield payloads in insertion order.
      - Per-record topics are omitted because EventRecord already contains
        event_type for dispatching.
      - Single-record flushes use the record's own topic (no batch overhead).
    """

    def __init__(
        self,
        path: str,
        zmq_context: ManagedZMQContext,
        loop: asyncio.AbstractEventLoop | None = None,
        scheme: str = "ipc",
        send_threshold: int = 1000,
    ):
        self._socket = zmq_context.socket(zmq.PUB)

        # Guarantee delivery: unlimited send buffer, wait on close.
        self._socket.setsockopt(zmq.SNDHWM, 0)
        self._socket.setsockopt(zmq.LINGER, -1)
        self._socket.setsockopt(zmq.IMMEDIATE, 1)

        bind_address = zmq_context.bind(self._socket, path, scheme)
        super().__init__(bind_address, loop)
        self.bind_path = path
        logger.info(f"Publisher bound to {self.bind_address}")

        self._fd = self._socket.getsockopt(zmq.FD)
        self._send_threshold = send_threshold
        self._batch_buffer: list[bytes] = []
        self._last_topic: bytes = b""
        self._pending: deque[bytes] = deque()
        self._writing = False

    def send(self, topic: bytes, payload: bytes) -> None:
        """Buffer a record for batched sending.

        Only the payload is buffered — topics are not stored per-record
        since the EventRecord already contains event_type for dispatching.
        When the buffer reaches ``send_threshold``, payloads are encoded
        as a single msgpack list and sent with BATCH_TOPIC. For a single
        record, a direct send with the record's own topic is used instead.
        """
        self._last_topic = topic
        self._batch_buffer.append(payload)

        if len(self._batch_buffer) >= self._send_threshold:
            self._flush_batch()

    def flush(self) -> None:
        """Force-send any buffered records, regardless of threshold.

        Uses direct per-record send when only 1 record is buffered
        (avoids batch encoding overhead for single records like ENDED).
        """
        if self._batch_buffer:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Encode and send the buffered payloads."""
        buf = self._batch_buffer
        self._batch_buffer = []

        if len(buf) == 1:
            # Single record: send with its own topic (no batch overhead).
            # _last_topic is the topic from the most recent send() call.
            self._send_frame(self._last_topic + buf[0])
        else:
            # Multiple records: encode payloads as msgpack list[bytes],
            # prefix with BATCH_TOPIC for routing. Individual topics are
            # not included — subscribers decode EventRecord.event_type
            # from the payload for dispatching.
            batch_payload = _batch_encoder.encode(buf)
            self._send_frame(BATCH_TOPIC + batch_payload)

    def _send_frame(self, frame: bytes) -> None:
        """Attempt direct send; fall back to pending queue + writer."""
        if not self._pending:
            mode = zmq.NOBLOCK if self.loop else 0
            try:
                self._socket.send(frame, flags=mode, copy=False, track=False)
                return
            except zmq.Again:
                # Socket would block; fall through to queue and async writer.
                pass

        if self.loop is None:
            raise RuntimeError(
                "Failed direct send, but publisher is set to eager-only mode."
            )

        self._pending.append(frame)
        if not self._writing:
            self._writing = True
            self.loop.add_writer(self._fd, self._on_writable)

    def _on_writable(self) -> None:
        """Drain pending frames when socket becomes writable."""
        if self.is_closed:
            return
        self._drain_pending(force=False)
        if not self._pending:
            self._stop_writer()

    def _drain_pending(self, force: bool = False) -> None:
        try:
            while self._pending:
                frame = self._pending[0]
                mode = 0 if force else zmq.NOBLOCK
                self._socket.send(frame, flags=mode, copy=False, track=False)
                self._pending.popleft()
        except zmq.Again:
            # Socket would block; remaining items stay in queue for next writable callback.
            return

    def _stop_writer(self) -> None:
        if self._writing:
            self._writing = False
            if self.loop is not None and self._fd is not None:
                try:
                    self.loop.remove_writer(self._fd)
                except (ValueError, OSError):
                    # Writer already removed or fd invalid (e.g. during shutdown).
                    pass

    def close(self) -> None:
        if self.is_closed:
            return

        # Flush buffered records before marking closed so that any
        # concurrent publish() calls that arrive during flush are still
        # accepted into the buffer rather than silently dropped.
        if self._batch_buffer:
            self._flush_batch()

        self.is_closed = True

        if self.loop:
            self._stop_writer()
            if self._pending:
                logger.warning("Closing publisher with pending frames. Draining...")
                self._drain_pending(force=True)
                self._pending.clear()

        # Cleanup IPC socket file.
        parsed = urlparse(self.bind_address)
        if parsed.scheme == "ipc" and parsed.path:
            try:
                if os.path.exists(parsed.path):
                    os.unlink(parsed.path)
            except OSError:
                # IPC socket file already removed or unlink failed (e.g. permissions).
                pass


class ZmqEventRecordSubscriber(EventRecordSubscriber):
    """ZMQ SUB socket subscriber that handles both single and batched messages.

    Automatically subscribes to BATCH_TOPIC in addition to any explicit
    topic subscriptions. Batched messages are unpacked into individual
    records and yielded in order via ``receive()``.

    Note on topic filtering with batches: batched messages contain records
    of mixed event types. Subscribers with specific topic filters will
    receive ALL event types from batches, not just their filtered topics.
    Per-record filtering must be done in application code (e.g., checking
    ``EventRecord.event_type`` after decode). This is acceptable because
    the decode cost (~0.6us/record) is negligible compared to processing.
    """

    def __init__(
        self,
        path: str,
        zmq_context: ManagedZMQContext,
        loop: asyncio.AbstractEventLoop,
        topics: list[str] | None = None,
        scheme: str = "ipc",
    ):
        self._socket = zmq_context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 0)

        if not topics:
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            for topic in topics:
                self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
            # Always subscribe to batch topic so batched messages are received
            self._socket.setsockopt(zmq.SUBSCRIBE, BATCH_TOPIC)

        connect_address = zmq_context.connect(self._socket, path, scheme)
        super().__init__(connect_address, loop, topics)
        self.connect_path = path
        logger.info(f"Subscriber connected to {self.connect_address}")

        self._fd = self._socket.getsockopt(zmq.FD)
        self._buffer: deque[bytes] = deque()

        # Reader is added in .start(); do not add here.

    def receive(self) -> bytes | None:
        """Receive a single record payload.

        If a batched message was received, individual payloads are buffered
        and returned one at a time in insertion order.
        """
        if self.is_closed:
            return None

        # Return buffered payloads first (from a previous batch)
        if self._buffer:
            return self._buffer.popleft()

        try:
            raw = self._socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again as e:
            raise StopIteration from e

        # Batch message: BATCH_TOPIC prefix + msgpack list[bytes] of payloads.
        # Individual payloads do not have topic prefixes — EventRecord.event_type
        # is used for dispatching instead.
        if raw[:TOPIC_FRAME_SIZE] == BATCH_TOPIC:
            batch_data = raw[TOPIC_FRAME_SIZE:]
            try:
                payloads = _batch_decoder.decode(batch_data)
            except (msgspec.DecodeError, ValueError):
                # Corrupt batch. On IPC this should never happen (ZMQ delivers
                # complete messages atomically). Possible causes: encoder bug,
                # ZMQ library bug, or memory corruption. Log enough detail to
                # diagnose, but there is no recovery path — the publisher's
                # buffer is already gone.
                logger.error(
                    "Failed to decode batch message (%d bytes), dropping. "
                    "This indicates a bug — IPC messages should never be corrupt.",
                    len(batch_data),
                )
                return None

            for payload in payloads:
                if payload:
                    self._buffer.append(payload)

            if self._buffer:
                return self._buffer.popleft()
            return None

        # Single-record message: topic prefix + payload
        if len(raw) > TOPIC_FRAME_SIZE:
            return raw[TOPIC_FRAME_SIZE:]
        return None

    def close(self) -> None:
        """Close the subscriber. Idempotent."""
        if self.is_closed:
            return
        self.is_closed = True
        super().close()
