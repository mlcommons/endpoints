"""HTTP endpoint client implementation with multiprocessing and ZMQ."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import zmq
import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import WorkerManager
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

logger = logging.getLogger(__name__)


class StreamingFuture(asyncio.Future):
    """Future that also exposes first chunk for streaming responses."""

    def __init__(self):
        super().__init__()
        self._first = asyncio.Future()

    @property
    def first(self):
        """First chunk future - can be awaited or checked."""
        return self._first

    def _set_first_chunk(self, chunk: str):
        """Set first chunk."""
        assert not self._first.done(), "First chunk already set"
        self._first.set_result(chunk)


class HTTPEndpointClient:
    """
    HTTP endpoint client with multiprocessing workers and ZMQ communication.

    This client provides high-performance HTTP request handling by:
    - Using multiple worker processes to parallelize running concurrent HTTP requests
    - ZMQ for inter-process communication for efficient message passing
    - Both future-based and callback-based response handling
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
        complete_callback: Callable[[Any], None] | None = None,
    ):
        """
        Initialize HTTP endpoint client.

        Args:
            config: HTTP client configuration
            aiohttp_config: aiohttp configuration
            zmq_config: ZMQ configuration
            complete_callback: Optional synchronous callback for completed requests
        """
        self.config = config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.complete_callback = complete_callback
        self.zmq_context = zmq.asyncio.Context(io_threads=zmq_config.zmq_io_threads)
        self.worker_push_sockets: list[ZMQPushSocket] = []

        self.worker_manager: WorkerManager | None = None
        self.current_worker_idx = 0

        self._shutdown_event = asyncio.Event()
        self._response_handler_task: asyncio.Task | None = None
        self._pending_futures: dict[str, asyncio.Future] = {}

        # Create concurrency semaphore if configured
        self._concurrency_semaphore = None
        if config.max_concurrency > 0:
            self._concurrency_semaphore = asyncio.Semaphore(config.max_concurrency)

    def issue_query(
        self, query: Query
    ) -> asyncio.Future[QueryResult] | StreamingFuture:
        """
        Send a query to the endpoint and return a future for the response.

        The returned future can be:
        - Awaited directly: `result = await client.issue_query(query)`
        - Checked for completion: `if future.done(): result = future.result()`
        - Used with asyncio utilities: `done, pending = await asyncio.wait([future])`

        For streaming queries, returns a StreamingFuture which also exposes:
        - `await future.first` to get the first chunk as soon as available

        Args:
            query: Query object containing request details

        Returns:
            StreamingFuture for streaming queries, asyncio.Future otherwise
        """
        # Create appropriate future type
        future = (
            StreamingFuture()
            if query.stream
            else asyncio.get_event_loop().create_future()
        )
        self._pending_futures[query.id] = future

        # Schedule the actual send
        asyncio.create_task(self._issue_query_impl(query))

        return future

    async def _issue_query_impl(self, query: Query) -> None:
        """Internal implementation of send request."""
        try:
            # Apply concurrency limit if configured
            if self._concurrency_semaphore:
                async with self._concurrency_semaphore:
                    await self._send_to_worker(query)
            else:
                await self._send_to_worker(query)
        except Exception as e:
            # If sending fails, complete the future with error
            future = self._pending_futures.get(query.id)
            if future and not future.done():
                future.set_exception(e)

                # Also set exception on first chunk future for streaming
                if isinstance(future, StreamingFuture) and not future.first.done():
                    future.first.set_exception(e)

    async def _send_to_worker(self, query: Query) -> None:
        """Send query to worker via ZMQ."""
        # Round-robin to next worker
        worker_idx = self.current_worker_idx
        self.current_worker_idx = (self.current_worker_idx + 1) % len(
            self.worker_push_sockets
        )

        # Send query directly to worker's queue
        await self.worker_push_sockets[worker_idx].send(query)

    async def start(self) -> None:
        """Initialize client and start worker manager."""
        # Initialize worker push sockets
        for i in range(self.config.num_workers):
            address = f"{self.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(self.zmq_context, address, self.zmq_config)
            self.worker_push_sockets.append(push_socket)

        # Start worker manager
        self.worker_manager = WorkerManager(
            self.config, self.aiohttp_config, self.zmq_config, self.zmq_context
        )
        await self.worker_manager.initialize()

        # Start response handler
        self._response_handler_task = asyncio.create_task(self._handle_responses())

    async def _handle_responses(self) -> None:
        """Handle responses from workers."""
        response_socket = ZMQPullSocket(
            self.zmq_context,
            self.zmq_config.zmq_response_queue_addr,
            self.zmq_config,
            bind=True,
        )

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Blocking receive with timeout for shutdown check
                    message = await asyncio.wait_for(
                        response_socket.receive(),
                        timeout=self.config.response_handler_timeout,
                    )

                    # Future must exist - we created it when issuing query
                    future = self._pending_futures[message.query_id]
                    assert future is not None, f"No future for {message.query_id}"
                    assert not future.done(), f"Double response for {message.query_id}"

                    # Handle by message type
                    match message:
                        case StreamChunk():
                            # Only streaming queries get StreamChunk
                            assert isinstance(
                                future, StreamingFuture
                            ), "StreamChunk for non-streaming query"
                            future._set_first_chunk(message.response_chunk)

                            if message.is_complete:
                                # Single chunk complete - create QueryResult and complete
                                result = QueryResult(
                                    query_id=message.query_id,
                                    response_output=message.response_chunk,
                                )
                                future.set_result(result)
                                self._pending_futures.pop(message.query_id)

                        case QueryResult():
                            # Complete the future
                            if message.error:
                                exception = Exception(message.error)
                                future.set_exception(exception)

                                # Set exception on first chunk if streaming and not set yet
                                if (
                                    isinstance(future, StreamingFuture)
                                    and not future.first.done()
                                ):
                                    future.first.set_exception(exception)
                            else:
                                # Set first chunk for empty streaming responses
                                if isinstance(future, StreamingFuture):
                                    if (
                                        message.metadata.get("first_chunk")
                                        and not future.first.done()
                                    ):
                                        future._set_first_chunk("")  # Empty response

                                future.set_result(message)

                            self._pending_futures.pop(message.query_id)

                    # Call callback after future is resolved
                    # NOTE(vir):
                    # We call the callback after future resolution to ensure that
                    # even if the callback raises an exception, the future is still properly
                    # resolved and the caller can await it. This prevents hangs in user code.
                    if self.complete_callback:
                        self.complete_callback(message)

                except TimeoutError:
                    # Check shutdown and continue
                    continue
                except Exception as e:
                    logger.error(f"Error handling response: {e}")

        finally:
            response_socket.close()

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        self._shutdown_event.set()

        # Cancel all pending futures
        for future in self._pending_futures.values():
            if not future.done():
                future.cancel()
        self._pending_futures.clear()

        # Cancel response handler
        if self._response_handler_task:
            self._response_handler_task.cancel()
            try:
                await self._response_handler_task
            except asyncio.CancelledError:
                pass

        # Close push sockets
        for socket in self.worker_push_sockets:
            socket.close()

        # Shutdown worker manager
        if self.worker_manager:
            await self.worker_manager.shutdown()

        # Close ZMQ context
        self.zmq_context.term()
