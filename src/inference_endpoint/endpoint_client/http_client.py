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
import threading
import uuid
from itertools import cycle

import uvloop

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.worker_manager import WorkerManager

logger = logging.getLogger(__name__)


class AsyncHttpEndpointClient:
    """
    Async HTTP client for LLM inference.

    Architecture:
    - Main process: Accepts requests, distributes to workers, handles responses
    - Worker processes: Make actual HTTP requests to the endpoint
    - Requests are distributed to workers round-robin

    Usage:
        client = AsyncHttpEndpointClient(config, aiohttp_config)
        client.issue(query)
        response = client.poll()        # Non-blocking
        response = await client.recv()  # Blocking
        responses = client.drain()      # Drain all available
    """

    def __init__(
        self,
        config: HTTPClientConfig,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.client_id = uuid.uuid4().hex[:8]
        self.config = config
        self._worker_cycle = cycle(range(self.config.num_workers))

        # Use provided loop or create own
        if loop is None:
            self.loop = uvloop.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self.loop.run_forever,
                daemon=True,
                name=f"HttpClient-{self.client_id}",
            )
            self._loop_thread.start()
        else:
            self.loop = loop
            self._loop_thread = None

        # Use eager task factory for immediate coroutine execution
        # Tasks start executing synchronously until first await
        #
        # NOTE(vir):
        # CRITICAL for http-client performance
        # ensures issue() does not get starved by other threads under load
        self.loop.set_task_factory(asyncio.eager_task_factory)

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
        # CPython GIL provides atomic boolean writes, no need for asyncio.Event()
        self._shutdown: bool = False

        # WorkerManager creates and owns all transports
        self.worker_manager = WorkerManager(self.config, self.loop)
        await self.worker_manager.initialize()
        self.pool = self.worker_manager.pool_transport

    def issue(self, query: Query) -> None:
        """
        Issue query to endpoint (round-robin to workers).
        Non-blocking - buffers if socket would block.
        """
        if not self._shutdown:
            # NOTE(vir): silently drop requests during shutdown
            self.pool.send(next(self._worker_cycle), query)

    def poll(self) -> QueryResult | StreamChunk | None:
        """Non-blocking. Returns response if available, None otherwise."""
        return self.pool.poll()

    async def recv(self) -> QueryResult | StreamChunk | None:
        """Blocking. Waits for next response. Returns None when closed."""
        return await self.pool.recv()

    def drain(self) -> list[QueryResult | StreamChunk]:
        """Non-blocking. Returns all available responses."""
        results: list[QueryResult | StreamChunk] = []
        while (r := self.poll()) is not None:
            results.append(r)
        return results

    async def shutdown(self) -> None:
        """Gracefully shutdown client."""
        logger.info(f"[{self.client_id}] Shutting down...")
        self._shutdown = True

        # Shutdown workers
        await self.worker_manager.shutdown()

        # Stop event loop if we own it
        if self._loop_thread is not None:
            self.loop.call_soon(self.loop.stop)

        logger.info(f"[{self.client_id}] Shutdown complete.")


class HTTPEndpointClient(AsyncHttpEndpointClient):
    """
    Sync HTTP client for LLM inference.
    Inherits from AsyncHttpEndpointClient and provides sync interface.

    Usage:
        client = HTTPEndpointClient(config)
        client.issue(query)
    """

    def issue(self, query: Query) -> None:  # type: ignore[override]
        """Issue query."""
        # Schedule on event loop thread
        self.loop.call_soon_threadsafe(
            lambda: super(HTTPEndpointClient, self).issue(query)
        )

    def shutdown(self) -> None:  # type: ignore[override]
        """Sync shutdown."""
        asyncio.run_coroutine_threadsafe(super().shutdown(), self.loop).result()
