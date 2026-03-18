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

"""HTTP endpoint client implementation."""

import asyncio
import logging
import uuid
from itertools import cycle

from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport import ZmqWorkerPoolTransport
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.worker_manager import WorkerManager

logger = logging.getLogger(__name__)


class HTTPEndpointClient:
    """
    HTTP client for LLM inference.

    Architecture:
    - Main process: Accepts requests, distributes to workers, handles responses
    - Worker processes: Make actual HTTP requests to the endpoint
    - Requests are distributed to workers round-robin

    When config uses ZmqWorkerPoolTransport, the caller must scope the client
    lifetime within ManagedZMQContext.scoped() and pass the context in.
    The client does not create or own the ZMQ context.

    Usage:
        with ManagedZMQContext.scoped() as zmq_ctx:
            client = HTTPEndpointClient(config, zmq_context=zmq_ctx)
            client.issue(query)
            response = client.poll()        # Non-blocking, returns None if nothing ready
            responses = client.drain()      # Drain all available responses
            # response = await client.recv()  # Blocking; only if caller provides its own loop
            client.shutdown()               # Blocks until workers stop
    """

    def __init__(
        self,
        config: HTTPClientConfig,
        loop: asyncio.AbstractEventLoop | None = None,
        zmq_context: ManagedZMQContext | None = None,
    ):
        self.client_id = uuid.uuid4().hex[:8]
        self.config = config
        self._worker_cycle = cycle(range(self.config.num_workers))

        # TODO(vir): make context setup/teardown part of transport protocol
        if config.worker_pool_transport is ZmqWorkerPoolTransport:
            if zmq_context is None:
                raise ValueError(
                    "zmq_context is required when using ZmqWorkerPoolTransport; "
                    "use ManagedZMQContext.scoped() and pass the context in."
                )
        self._zmq_context = zmq_context

        # Use provided loop or create one via LoopManager (uvloop + eager task factory)
        self._owns_loop = loop is None
        self._loop_name: str | None = None
        if loop is None:
            self._loop_name = f"HttpClient-{self.client_id}"
            self.loop = LoopManager().create_loop(
                name=self._loop_name,
                backend="uvloop",
                task_factory_mode="eager",
            )
        else:
            self.loop = loop
        assert self.loop is not None

        # Initialize on event loop
        asyncio.run_coroutine_threadsafe(self._initialize(), self.loop).result()

        logger.info(
            f"EndpointClient initialized with num_workers={self.config.num_workers}, "
            f"endpoints={self.config.endpoint_urls}, "
            f"adapter={self.config.adapter.__name__}, "
            f"accumulator={self.config.accumulator.__name__}, "
            f"pool_transport={self.config.worker_pool_transport.__name__}"
        )

    async def _initialize(self) -> None:
        """Initialize worker manager and transports."""
        self._shutdown: bool = False
        self._dropped_requests: int = 0

        assert self.loop is not None
        additional_args = []
        if self.config.worker_pool_transport is ZmqWorkerPoolTransport:
            assert self._zmq_context is not None
            additional_args.append(self._zmq_context)
        self.worker_manager = WorkerManager(self.config, self.loop, *additional_args)
        await self.worker_manager.initialize()
        self.pool = self.worker_manager.pool_transport

    def issue(self, query: Query) -> None:
        """
        Issue query to endpoint (round-robin to workers).
        Non-blocking - buffers if socket would block.

        Thread-safe: schedules the send on the event loop thread via
        call_soon_threadsafe, since the underlying ZMQ sockets and send
        buffers are not thread-safe and belong to the event loop thread.
        """
        if self._shutdown:
            # NOTE(vir): drop requests during shutdown
            self._dropped_requests += 1
        else:
            worker_id = next(self._worker_cycle)
            self.loop.call_soon_threadsafe(self.pool.send, worker_id, query)

    def poll(self) -> QueryResult | StreamChunk | None:
        """Non-blocking. Returns response if available, None otherwise."""
        return self.pool.poll()

    async def recv(self) -> QueryResult | StreamChunk | None:
        """Blocking. Waits for next response. Returns None when closed."""
        return await self.pool.recv()

    def drain(self) -> list[QueryResult | StreamChunk]:
        """Non-blocking. Returns all available responses."""
        return list(iter(self.poll, None))

    def shutdown(self) -> None:
        """Gracefully shutdown client. Synchronous — blocks the caller until complete."""
        if self._shutdown:  # Already shutdown, no-op
            return
        asyncio.run_coroutine_threadsafe(self._shutdown_async(), self.loop).result()

    async def _shutdown_async(self) -> None:
        """Async shutdown internals - must be called on the event loop."""
        self._shutdown = True

        logger.info(f"[{self.client_id}] Shutting down...")

        # Shutdown workers
        await self.worker_manager.shutdown()

        # Stop event loop if we own it
        if self._owns_loop and self._loop_name is not None:
            LoopManager().stop_loop(self._loop_name)
            self._loop_name = None

        if self._dropped_requests > 0:
            logger.info(
                f"[{self.client_id}] Dropped {self._dropped_requests} requests during shutdown"
            )
        logger.info(f"[{self.client_id}] Shutdown complete.")
