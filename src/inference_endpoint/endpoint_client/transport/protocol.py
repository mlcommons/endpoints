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

Defines the protocols for transport abstraction, allowing the Worker to be
completely agnostic of the transport backend (ZMQ, shared memory, etc.).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from inference_endpoint.core.types import Query, QueryResult, StreamChunk


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

    Usage:
        pool = ZmqWorkerPoolTransport.create(loop, num_workers=4)

        # Spawn workers with connector
        for i in range(4):
            spawn_worker(i, pool.worker_connector, ...)

        # Wait for workers
        await pool.wait_for_workers_ready(timeout=30)

        # Use
        pool.send(worker_id, query)
        result = pool.poll()        # Non-blocking
        result = await pool.recv()  # Blocking

        # Cleanup
        pool.cleanup()
    """

    @classmethod
    def create(
        cls,
        loop: asyncio.AbstractEventLoop,
        num_workers: int,
        **overrides: Any,
    ) -> WorkerPoolTransport:
        """Factory to create a worker pool transport.

        Args:
            loop: Event loop for transport registration.
            num_workers: Number of workers.
            **overrides: Transport-specific config overrides.

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


__all__ = [
    "ReceiverTransport",
    "SenderTransport",
    "WorkerConnector",
    "WorkerPoolTransport",
]
