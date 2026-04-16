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
from typing import Any, Protocol, runtime_checkable

import msgspec
from pydantic import BaseModel, ConfigDict, Field

from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    decode_event_record,
    encode_event_record,
)
from inference_endpoint.core.types import ErrorData, Query, QueryResult, StreamChunk


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


class EventRecordPublisher(ABC):
    """Abstract base class for publishing event records over a transport."""

    def __init__(
        self,
        bind_address: str,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Creates a new EventRecordPublisher.

        Args:
            bind_address: The address to bind the publisher to. This can be an IPC or TCP socket address.
            loop: The event loop to use for the publisher. If not provided, it is assumed that the publisher
                should always execute eagerly and will be blocking. This means that the call to `.publish()`
                will always be called immediately and the current loop and thread will block until the message
                is sent.
        """
        self.bind_address = bind_address
        self.loop = loop
        self.is_closed: bool = False

    def publish(self, event_record: EventRecord) -> None:
        """Publish the event record on the bound address.

        Args:
            event_record: The event record to publish.
        """
        if self.is_closed:
            return

        topic, payload = encode_event_record(event_record)
        self.send(topic, payload)

    @abstractmethod
    def send(self, topic: bytes, payload: bytes) -> None:
        """Send the message via the implemented transport layer.

        Args:
            topic: The topic of the message.
            payload: The payload of the message.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def flush(self) -> None:  # noqa: B027 — intentionally non-abstract
        """Force-send any buffered records.

        Unbuffered implementations need no override. Buffered subclasses
        (e.g., ZmqEventRecordPublisher) override this to drain their buffer.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the publisher and release resources.

        Implementations must flush any buffered records before closing.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class EventRecordSubscriber(ABC):
    """Abstract base class for subscribing to event records over a transport."""

    def __init__(
        self,
        connect_address: str,
        loop: asyncio.AbstractEventLoop,
        topics: list[str] | None = None,
    ):
        """Creates a new EventRecordSubscriber.

        Initializing the subscriber does NOT start processing. The subscriber connects
        to the address and subscribes to topics, but the socket reader is only added
        when .start() is called. This allows bookkeeping or other setup before
        listening. Each subscriber should use its own event loop (e.g. from LoopManager),
        not shared with the publisher.

        It is mandatory for subscriber implementations to set the `_fd` attribute to the file
        descriptor of the socket to add an asyncio reader to the event loop.

        Args:
            connect_address: The address to connect the subscriber to. This can be an IPC or TCP socket address.
            loop: The event loop to use for the subscriber (typically a dedicated loop per subscriber).
            topics: The topics to subscribe to. If not provided, it is assumed that the subscriber should subscribe to all topics.
        """
        self.connect_address = connect_address
        self.topics = topics
        self.loop = loop
        self.is_closed: bool = False

        self._fd: int | None = None

    @abstractmethod
    def receive(self) -> bytes | None:
        """Receive data from the transport.

        Should receive data from the socket and return a bytes object that should be able
        to be decoded into an EventRecord.

        If the received data is malformed, this method should return None.

        For the specific case that the transport is not readable or the underlying socket is busy
        (such as when an EAGAIN error is raised), this method should raise a StopIteration exception.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def process(self, records: list[EventRecord]) -> None:
        """Process a list of EventRecords.

        Called asynchronously (scheduled via create_task) so that heavy work does not
        block the socket read path. Implementations should be async.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def close(self) -> None:
        """Close the subscriber and release resources (e.g. remove reader, close socket).
        Should be idempotent; safe to call multiple times. Call when the session has ended.
        """
        if self.loop is not None and self._fd is not None:
            try:
                self.loop.remove_reader(self._fd)
            except (ValueError, OSError):
                # Reader already removed or fd invalid (e.g. during shutdown).
                pass

    def _on_readable(self) -> None:
        """Drain socket, decode records, and schedule process() as an async task."""
        if self.is_closed:
            return

        records: list[EventRecord] = []
        try:
            while True:
                payload = self.receive()
                if payload is None:
                    continue

                # Attempt decode
                try:
                    event_record = decode_event_record(payload)
                except msgspec.DecodeError as e:
                    event_record = EventRecord(
                        event_type=ErrorEventType.GENERIC,
                        data=ErrorData(
                            error_type="msgspec.DecodeError",
                            error_message=str(e),
                        ),
                    )
                records.append(event_record)
        except StopIteration:
            # No more messages to receive right now
            pass
        finally:
            if records:
                # Schedule process() so it does not block the socket read path
                self.loop.create_task(self.process(records))

    def start(self) -> None:
        """Start the subscriber: add the socket reader to the loop and begin processing.

        Call this after any setup (e.g. when the session is about to start). Before
        start() is called, no messages are received.
        """
        if self._fd is None:
            raise ValueError("Subscriber not initialized with a file descriptor")

        self.loop.add_reader(self._fd, self._on_readable)


__all__ = [
    "TransportConfig",
    "ReceiverTransport",
    "SenderTransport",
    "WorkerConnector",
    "WorkerPoolTransport",
    "EventRecordPublisher",
    "EventRecordSubscriber",
]
