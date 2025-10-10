"""HTTP endpoint client implementation with multiprocessing and ZMQ."""

import asyncio
import logging
import threading
import uuid

import zmq
import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.asyncio_utils import new_event_loop
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import WorkerManager
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

logger = logging.getLogger(__name__)


class HTTPEndpointClient:
    """
    HTTP endpoint client with multiprocessing workers and ZMQ communication.

    This client provides high-performance HTTP request handling by:
    - Using multiple worker processes to parallelize running concurrent HTTP requests
    - ZMQ for inter-process communication for efficient message passing
    - Round-robin load balancing across workers

    Architecture:
    - Main process: Accepts requests, distributes to workers, handles responses
    - Worker processes: Make actual HTTP requests to the endpoint

    See README.md for detailed usage examples and configuration options.
    """

    def __init__(
        self,
        config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
    ):
        """
        Initialize HTTP endpoint client.

        Args:
            config: HTTP client configuration
            aiohttp_config: aiohttp configuration
            zmq_config: ZMQ configuration
        """
        # Generate unique ID to avoid conflicts between multiple client instances
        self.client_id = uuid.uuid4().hex[:8]

        self.config = config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config

        self.loop: asyncio.AbstractEventLoop | None = None
        self.loop_thread: threading.Thread | None = None

        self.zmq_context: zmq.asyncio.Context | None = None
        self.worker_push_sockets: list[ZMQPushSocket] = []
        self.worker_manager: WorkerManager | None = None
        self.current_worker_idx = 0

        self._shutdown_event: asyncio.Event | None = None
        self._response_socket: ZMQPullSocket | None = None
        self._concurrency_semaphore: asyncio.Semaphore | None = None

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start event loop thread and initialize client."""
        try:
            self.loop = uvloop.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self.loop_thread = threading.Thread(
                target=self.loop.run_forever,
                daemon=True,
                name=f"HttpClient-EventLoop-{self.client_id}",
            )
            self.loop_thread.start()

            asyncio.run_coroutine_threadsafe(self.async_start(), self.loop).result()
        except Exception as e:
            logger.error(f"Failed to start HTTP endpoint client: {e}")
            raise e

    async def async_start(self):
        """Initialize ZMQ, workers, and sockets."""
        self.zmq_context = zmq.asyncio.Context(
            io_threads=self.zmq_config.zmq_io_threads
        )
        self._shutdown_event = asyncio.Event()

        if self.config.max_concurrency > 0:
            self._concurrency_semaphore = asyncio.Semaphore(self.config.max_concurrency)

        for i in range(self.config.num_workers):
            address = f"{self.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(self.zmq_context, address, self.zmq_config)
            self.worker_push_sockets.append(push_socket)

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

    def issue_query(self, query: Query) -> None:
        """
        Issue query to endpoint.
        Synchronous wrapper for issue_query_async.

        Args:
            query: Query object containing request details
        """
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self.issue_query_async(query))
        )

    async def issue_query_async(self, query: Query) -> None:
        """
        Issue query to endpoint.
        Issue query to worker via ZMQ (non-blocking)

        Args:
            query: Query object containing request details
        """
        if self._concurrency_semaphore:
            async with self._concurrency_semaphore:
                await self._send_to_worker(query)
        else:
            await self._send_to_worker(query)

    def get_ready_responses(self) -> QueryResult | StreamChunk | None:
        """
        Get next ready response from workers (synchronous wrapper).
        Blocks until a response is available or timeout occurs.

        Returns:
            QueryResult or StreamChunk from worker, or None on timeout
        """
        future = asyncio.run_coroutine_threadsafe(
            self.get_ready_responses_async(), self.loop
        )
        return future.result()

    async def get_ready_responses_async(self) -> QueryResult | StreamChunk | None:
        """
        Get next ready response from any worker.

        Returns:
            QueryResult or StreamChunk from worker, or None on timeout
        """
        assert self._response_socket is not None
        return await self._response_socket.receive()

    async def _send_to_worker(self, query: Query) -> None:
        """Send query to worker via ZMQ."""
        worker_idx = self.current_worker_idx
        self.current_worker_idx = (self.current_worker_idx + 1) % len(
            self.worker_push_sockets
        )
        await self.worker_push_sockets[worker_idx].send(query)

    def shutdown(self):
        """Shutdown client, stop event loop, and cleanup."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.async_shutdown(), self.loop).result()
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.loop_thread:
            self.loop_thread.join(timeout=0.1)

    async def async_shutdown(self):
        """Shutdown async components."""
        logger.info("Shutting down HTTP endpoint client...")
        if self._shutdown_event:
            self._shutdown_event.set()

        if self._response_socket:
            self._response_socket.close()

        for socket in self.worker_push_sockets:
            socket.close()

        if self.worker_manager:
            await self.worker_manager.shutdown()

        if self.zmq_context:
            self.zmq_context.destroy(linger=0)
        logger.info("HTTP endpoint client shutdown complete.")
