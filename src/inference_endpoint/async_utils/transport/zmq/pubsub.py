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

import zmq

from inference_endpoint.async_utils.transport.protocol import (
    EventRecordPublisher,
    EventRecordSubscriber,
)
from inference_endpoint.core.record import TOPIC_FRAME_SIZE

from .context import ManagedZMQContext

logger = logging.getLogger(__name__)


class ZmqEventRecordPublisher(EventRecordPublisher):
    def __init__(
        self,
        bind_address: str,
        zmq_context: ManagedZMQContext,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(bind_address, loop)

        # Validate IPC path length
        if bind_address.startswith("ipc://"):
            if len(bind_address) > zmq.IPC_PATH_MAX_LEN:
                raise ValueError(
                    f"IPC path too long ({len(bind_address)} > {zmq.IPC_PATH_MAX_LEN})"
                )

        self._socket = zmq_context.socket(zmq.PUB)

        # One of the guarantees of event records is that if it is published,
        # it must be eventually received by all live subscribers.
        self._socket.setsockopt(zmq.SNDHWM, 0)  # Unlimited send buffer
        self._socket.setsockopt(
            zmq.LINGER, -1
        )  # Wait indefinitely on close() to send pending messages
        self._socket.setsockopt(zmq.IMMEDIATE, 1)

        self._socket.bind(self.bind_address)
        logger.info(f"Publisher bound to {self.bind_address}")

        self._fd = self._socket.getsockopt(zmq.FD)
        self._buffer: deque[bytes] = deque()
        self._writing = False

    def send(self, topic: bytes, payload: bytes) -> None:
        """Send the message via zmq.

        Args:
            topic: The topic of the message.
            payload: The payload of the message.
        """
        # Combine into a single frame to avoid overhead of .send_multipart()
        frame = topic + payload

        # Attempt direct send:
        if not self._buffer:
            mode = zmq.NOBLOCK if self.loop else 0
            try:
                self._socket.send(
                    frame,
                    flags=mode,
                    copy=False,
                    track=False,
                )
                return
            except zmq.Again:
                # Socket would block; fall through to buffer and use writer.
                pass

        if self.loop is None:
            # This should never be reached, since in eager mode, the send_multipart will block and
            # should always succeed, but just in case, this guard will raise an error
            raise RuntimeError(
                "Failed direct send, but publisher is set to eager-only mode."
            )

        # Add to buffer since socket is blocked.
        self._buffer.append(frame)
        if not self._writing:
            # Add writer callback to asyncio loop to drain the buffer when writable.
            self._writing = True
            self.loop.add_writer(self._fd, self._on_writable)

    def _on_writable(self) -> None:
        """Drains buffer when socket becomes writable. Used as an asyncio writer callback."""
        if self.is_closed:
            return

        self._drain_buffer(force=False)

        if not self._buffer:
            self._stop_writer()

    def _drain_buffer(self, force: bool = False) -> None:
        """Drains the buffer.

        Args:
            force (bool): If True, will use blocking sends to drain the buffer to ensure
                that when this method returns, the buffer is empty.
        """
        try:
            while self._buffer:
                # Do not pre-emptively pop in case of errors
                frame = self._buffer[0]
                mode = 0 if force else zmq.NOBLOCK
                self._socket.send(
                    frame,
                    flags=mode,
                    copy=False,
                    track=False,
                )
                self._buffer.popleft()
        except zmq.Again:
            return

    def _stop_writer(self) -> None:
        """Stops the writer callback."""
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

        self.is_closed = True

        if self.loop:
            # Remove writer callback if present
            self._stop_writer()

            # Drain the buffer since we should not drop messages.
            if self._buffer:
                logger.warning(
                    "Closing publisher with pending messages. Draining buffer..."
                )
                self._drain_buffer(force=True)
                self._buffer.clear()  # This should be a no-op, but just in case.

        # Socket is closed by ManagedZMQContext.cleanup() when the context scope exits.

        # Cleanup IPC socket file
        if self.bind_address.startswith("ipc://"):
            socket_path = self.bind_address[len("ipc://") :]
            try:
                if os.path.exists(socket_path):
                    os.unlink(socket_path)
            except OSError:
                # IPC path already removed or unlink failed (e.g. permissions).
                pass


class ZmqEventRecordSubscriber(EventRecordSubscriber):
    def __init__(
        self,
        connect_address: str,
        zmq_context: ManagedZMQContext,
        loop: asyncio.AbstractEventLoop,
        topics: list[str] | None = None,
    ):
        super().__init__(connect_address, loop, topics)

        self._socket = zmq_context.socket(zmq.SUB)

        self._socket.setsockopt(zmq.RCVHWM, 0)

        # Subscribe to topics
        if not self.topics:
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            for topic in self.topics:
                self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))

        self._socket.connect(self.connect_address)
        logger.info(f"Subscriber connected to {self.connect_address}")

        self._fd = self._socket.getsockopt(zmq.FD)
        self._buffer: deque[bytes] = deque()

        # Reader is added in .start(); do not add here.

    def receive(self) -> bytes | None:
        """Receive a message from the socket"""
        if self.is_closed:
            return None

        try:
            frame = self._socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again as e:
            raise StopIteration from e

        if len(frame) > TOPIC_FRAME_SIZE:
            # Should be (padded_topic + payload). Return the payload bytes.
            return frame[TOPIC_FRAME_SIZE:]
        return None

    def close(self) -> None:
        """Close the subscriber and remove the loop reader. Idempotent; safe to call multiple times.
        Socket is closed by ManagedZMQContext.cleanup() when the context scope exits.
        """
        if self.is_closed:
            return
        self.is_closed = True

        super().close()
