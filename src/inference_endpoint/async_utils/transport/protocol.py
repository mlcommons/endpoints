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

"""Transport protocol definitions for worker IPC.

Defines the protocols and base types for transport abstraction, allowing the
Worker to be completely agnostic of the transport backend (ZMQ, shared memory, etc.).
"""

from __future__ import annotations

import asyncio
import builtins
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import msgspec
from pydantic import BaseModel, ConfigDict, Field

from inference_endpoint.core.types import Query, QueryResult, StreamChunk

T = TypeVar("T")


class TransportConfig(BaseModel, ABC):
    """Base transport configuration. Subclassed per transport backend.

    Each subclass must:
    - Set ``type`` to a unique Literal string for discriminated union dispatch
    - Implement the ``transport_class`` property to return the transport's
      ``WorkerPoolTransport`` implementation class
    """

    type: str = Field(description="Transport backend (currently: zmq)")  # noqa: A003
    recv_buffer_size: int = Field(
        default=4 * 4 * 1024 * 1024,
        ge=1,
        description="IPC receive buffer size in bytes (default 16MB). Increase for multimodal payloads.",
    )
    send_buffer_size: int = Field(
        default=4 * 4 * 1024 * 1024,
        ge=1,
        description="IPC send buffer size in bytes (default 16MB). Increase for multimodal payloads.",
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def create_default(cls) -> TransportConfig:
        """Create the default transport config."""
        from .zmq import ZMQTransportConfig

        return ZMQTransportConfig()  # type: ignore[return-value]

    @property
    @abstractmethod
    def transport_class(self) -> builtins.type[WorkerPoolTransport]:
        """The WorkerPoolTransport implementation for this backend."""
        ...


@runtime_checkable
class ReceiverTransport(Protocol):
    """Protocol for receiving messages from a transport."""

    async def recv(self) -> Any | None:
        """Receive a message from the transport (async, blocking).

        Returns:
            The received message, or None when transport is closed.
        """
        pass

    def poll(self) -> Any | None:
        """Non-blocking receive.

        Returns:
            The received message if available, None otherwise.
        """
        pass

    def close(self) -> None:
        """Close the transport and release resources.

        After close(), recv() returns None immediately.
        """
        pass


@runtime_checkable
class SenderTransport(Protocol):
    """Protocol for sending messages through a transport."""

    def send(self, data: Any) -> None:
        """Send a message through the transport.

        Args:
            data: The message to send.
        """
        pass

    def close(self) -> None:
        """Close the transport and release resources."""
        pass


class WorkerConnector(Protocol):
    """Picklable connector passed to pass to child processes.

    Yields (Send, Recv) Transport for child <-> main communication.
    """

    @asynccontextmanager
    async def connect(
        self, worker_id: int
    ) -> AsyncIterator[tuple[ReceiverTransport, SenderTransport]]:
        """Connect worker transports and signal readiness.

        Creates request receiver and response sender, signals readiness
        to main process, then yields transports. Cleans up on exit.
        Transport-specific context (e.g. ZMQ) is managed internally by
        the connector implementation.

        Args:
            worker_id: Unique identifier for this worker.

        Yields:
            Tuple of (request_receiver, response_sender) transports.
            - request_receiver: Receives Query objects from main
            - response_sender: Sends QueryResult/StreamChunk to main
        """
        yield  # type: ignore[misc]


@runtime_checkable
class WorkerPoolTransport(Protocol):
    """
    Transport for endpoint-child child-process (workers) pool communication.
    Provides fan-out (send to workers) and fan-in (receive from workers).

    Context and pool creation are managed by WorkerManager from TransportConfig.

    Usage:
        pool.send(worker_id, query)
        result = pool.poll()        # Non-blocking
        result = await pool.recv()  # Blocking
        pool.cleanup()
    """

    @classmethod
    def create(
        cls,
        loop: asyncio.AbstractEventLoop,
        num_workers: int,
        config: TransportConfig | None = None,
    ) -> WorkerPoolTransport:
        """Factory to create a worker pool transport.

        Transport implementations manage their own context internally.

        Args:
            loop: Event loop for transport registration.
            num_workers: Number of workers (required).
            config: Transport configuration. Defaults per implementation.

        Returns:
            Configured WorkerPoolTransport instance.
        """
        pass

    @property
    def worker_connector(self) -> WorkerConnector:
        """Connector to pass to worker processes."""
        pass

    def send(self, worker_id: int, query: Query) -> None:
        """Send request to specific worker.

        Args:
            worker_id: Target worker ID.
            query: Query to send.
        """
        pass

    def poll(self) -> QueryResult | StreamChunk | None:
        """Non-blocking poll for response.

        Returns:
            QueryResult or StreamChunk if available, None otherwise.
        """
        pass

    async def recv(self) -> QueryResult | StreamChunk | None:
        """Blocking receive. Waits for next response.

        Returns:
            QueryResult or StreamChunk from a worker, or None when closed.
        """
        pass

    async def wait_for_workers_ready(self, timeout: float | None = None) -> None:
        """Block until all workers signal readiness.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.

        Raises:
            TimeoutError: If workers don't signal in time (only if timeout set).
        """
        pass

    def cleanup(self) -> None:
        """Close all transports and release resources. Idempotent.

        Implementations should clean up any resources they created,
        including temporary directories for IPC sockets.
        """
        pass


class MessageCodec(Protocol[T]):
    """Encode/decode policy for a single message type on the pub/sub layer.

    The codec is the only type-specific surface in the pub/sub stack. All
    transport machinery (ZmqMessagePublisher / ZmqMessageSubscriber) operates
    on (topic_bytes, payload_bytes); the codec is what binds those bytes to
    a concrete Python type T.
    """

    def encode(self, item: T) -> tuple[bytes, bytes]:
        """Return (topic, payload). topic must be exactly TOPIC_FRAME_SIZE bytes."""
        ...

    def decode(self, payload: bytes) -> T:
        """Decode payload back to T. May raise; the caller routes failures
        through on_decode_error."""
        ...

    def on_decode_error(self, payload: bytes, exc: Exception) -> T | None:
        """Fallback for malformed payloads. Return a sentinel item or None
        to drop the message."""
        ...


class MessagePublisher(ABC, Generic[T]):
    """Abstract base for publishing typed messages over a transport.

    Subclasses implement send(topic, payload) and close(). publish() is
    generic over T via the codec.
    """

    def __init__(
        self,
        codec: MessageCodec[T],
        bind_address: str,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Creates a new MessagePublisher.

        Args:
            codec: Encode policy. Required because turning T into wire bytes
                is the only type-specific operation; injecting it is the
                whole point of generalization.
            bind_address: IPC or TCP socket address to bind to.
            loop: Event loop to register async writes on. If None, send is
                eager/blocking — used by callers that publish before a loop
                is running (e.g. service startup).
        """
        self._codec = codec
        self.bind_address = bind_address
        self.loop = loop
        self.is_closed: bool = False

    def publish(self, item: T) -> None:
        """Encode item via the codec and send."""
        if self.is_closed:
            return
        topic, payload = self._codec.encode(item)
        self.send(topic, payload)

    @abstractmethod
    def send(self, topic: bytes, payload: bytes) -> None:
        """Send raw frame via the implemented transport layer."""
        raise NotImplementedError

    def flush(self) -> None:  # noqa: B027 — intentionally non-abstract
        """Force-send any buffered records.

        Unbuffered implementations need no override. Buffered subclasses
        (e.g. ZmqMessagePublisher) override this to drain their buffer.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the publisher and release resources.

        Implementations must flush any buffered records before closing.
        """
        raise NotImplementedError


class MessageSubscriber(ABC, Generic[T]):
    """Abstract base for subscribing to typed messages over a transport.

    Subclasses implement receive() (raw bytes from socket) and process()
    (handle decoded items). _on_readable wires them together using the
    codec.
    """

    def __init__(
        self,
        codec: MessageCodec[T],
        connect_address: str,
        loop: asyncio.AbstractEventLoop,
        topics: list[str] | None = None,
    ):
        """Creates a new MessageSubscriber.

        Initializing does NOT start processing — call .start() to add the
        socket reader to the loop. Subclasses must set ``self._fd`` to the
        socket file descriptor before .start() is called.

        Args:
            codec: Decode policy. Required for the same reason as in
                MessagePublisher.
            connect_address: IPC or TCP socket address to connect to.
            loop: Dedicated loop for this subscriber (typically from
                LoopManager — not shared with the publisher).
            topics: Topics to subscribe to. None means subscribe to all.
        """
        self._codec = codec
        self.connect_address = connect_address
        self.topics = topics
        self.loop = loop
        self.is_closed: bool = False

        self._fd: int | None = None

    @abstractmethod
    def receive(self) -> bytes | None:
        """Receive a single payload (no topic prefix) from the transport.

        Returns None for malformed-but-recognized frames. Raises
        StopIteration when the transport has nothing more to deliver right
        now (EAGAIN).
        """
        raise NotImplementedError

    @abstractmethod
    async def process(self, items: list[T]) -> None:
        """Handle a batch of decoded items. Called as an asyncio task so
        heavy work does not block the socket read path."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the subscriber. Idempotent."""
        if self.loop is not None and self._fd is not None:
            try:
                self.loop.remove_reader(self._fd)
            except (ValueError, OSError):
                # Reader already removed or fd invalid (e.g. during shutdown).
                pass

    def _on_readable(self) -> None:
        """Drain socket, decode via codec, and schedule process()."""
        if self.is_closed:
            return

        items: list[T] = []
        try:
            while True:
                payload = self.receive()
                if payload is None:
                    continue
                try:
                    items.append(self._codec.decode(payload))
                except msgspec.DecodeError as e:
                    fallback = self._codec.on_decode_error(payload, e)
                    if fallback is not None:
                        items.append(fallback)
        except StopIteration:
            pass
        finally:
            if items:
                self.loop.create_task(self.process(items))

    def start(self) -> None:
        """Add the socket reader to the loop and begin processing."""
        if self._fd is None:
            raise ValueError("Subscriber not initialized with a file descriptor")
        self.loop.add_reader(self._fd, self._on_readable)


__all__ = [
    "TransportConfig",
    "ReceiverTransport",
    "SenderTransport",
    "WorkerConnector",
    "WorkerPoolTransport",
    "MessageCodec",
    "MessagePublisher",
    "MessageSubscriber",
]
