# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import itertools
import logging
import threading
import uuid

import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker_manager import WorkerManager
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket
from inference_endpoint.utils.asyncio import create_eager_loop

logger = logging.getLogger(__name__)


class AsyncHttpEndpointClient:
    """
    Async HTTP client for LLM inference.

    Architecture:
    - Main process: Accepts requests, distributes to workers, handles responses
    - Worker processes: Make actual HTTP requests to the endpoint
    - requests are distributed to workers ROUND-ROBIN

    Usage:
        client = AsyncHttpEndpointClient(config, aiohttp_config, zmq_config)
        await client.issue_query(query)
        response_if_any = await client.try_receive()
    """

    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.client_id = uuid.uuid4().hex[:8]
        self.config = config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config

        # Use provided loop or create own
        if loop is None:
            self.loop = create_eager_loop()
            self._loop_thread: threading.Thread | None = threading.Thread(
                target=self.loop.run_forever,
                daemon=True,  # no need for explicit join
                name=f"HttpClient-{self.client_id}",
            )
            self._loop_thread.start()
        else:
            self.loop = loop
            self._loop_thread = None

        # Initialize on event loop
        asyncio.run_coroutine_threadsafe(self._initialize(), self.loop).result()

        logger.info(
            f"HTTPEndpointClient[{self.config.adapter.__name__}] initialized with num_workers={self.config.num_workers}"
        )

    async def _initialize(self) -> None:
        """Initialize ZMQ context, sockets, and start workers."""
        self._shutdown_event = asyncio.Event()

        self.zmq_context = zmq.asyncio.Context(
            io_threads=self.zmq_config.zmq_io_threads
        )

        self._worker_push_sockets = [
            ZMQPushSocket(
                self.zmq_context,
                f"{self.zmq_config.zmq_request_queue_prefix}_{i}_requests",
                self.zmq_config,
            )
            for i in range(self.config.num_workers)
        ]
        self._worker_cycle = itertools.cycle(self._worker_push_sockets)

        self.worker_manager = WorkerManager(
            self.config, self.aiohttp_config, self.zmq_config, self.zmq_context
        )
        await self.worker_manager.initialize()

        self._response_socket = ZMQPullSocket(
            self.zmq_context,
            self.zmq_config.zmq_response_queue_addr,
            self.zmq_config,
            bind=True,
            decoder_type=QueryResult | StreamChunk,
        )

    async def issue_query(self, query: Query) -> None:
        """
        Issue query to endpoint.
        Query is assigned to a worker to process in round-robin fashion.
        """
        assert (
            not self._shutdown_event.is_set()
        ), "Cannot issue query: client is shutting down"
        await next(self._worker_cycle).send(query)

    async def try_receive(self) -> QueryResult | StreamChunk | None:
        """Receive next ready response if available, else return None."""
        return await self._response_socket.receive()

    async def shutdown(self) -> None:
        """Gracefully shutdown client."""
        logger.info(f"[{self.client_id}] Shutting down...")
        self._shutdown_event.set()

        self._response_socket.close()
        for socket in self._worker_push_sockets:
            socket.close()

        await self.worker_manager.shutdown()
        self.zmq_context.destroy(linger=0)

        # Stop event loop if we own it (scheduled to run after this coroutine completes)
        if self._loop_thread is not None:
            self.loop.call_soon(self.loop.stop)

        logger.info(f"[{self.client_id}] Shutdown complete.")


class HTTPEndpointClient(AsyncHttpEndpointClient):
    """
    Sync HTTP client for LLM inference.
    Inherits from AsyncHttpEndpointClient and provides sync interface.

    TODO(vir): sync recv. API is not required/implemented yet

    Usage:
        client = HTTPEndpointClient(config, aiohttp_config, zmq_config)
        client.issue_query(query)
        response = await client.try_receive()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop.set_exception_handler(self._exception_handler)

    def _exception_handler(
        self, loop: asyncio.AbstractEventLoop, context: dict
    ) -> None:
        """Supress errors post shutdown."""
        if self._shutdown_event.is_set():
            return  # Suppress all errors during shutdown
        # Default handling for non-shutdown errors
        loop.default_exception_handler(context)

    def issue_query(self, query: Query) -> None:  # type: ignore[override]
        """Issue query."""
        coro = super().issue_query(query)

        # NOTE(vir):
        # asyncio.run_coroutine_threadsafe wraps callback with unnecessary future,
        # use loop.call_soon_threadsafe directly since we have a fire-and-forget pattern
        #
        # TODO(vir): does this need create_eager_task?
        self.loop.call_soon_threadsafe(self.loop.create_task, coro)

    def shutdown(self) -> None:  # type: ignore[override]
        """Sync shutdown wrapper - blocks until base class async shutdown completes."""
        asyncio.run_coroutine_threadsafe(super().shutdown(), self.loop).result()
