"""Futures-based wrapper for HTTPEndpointClient."""

import asyncio
import logging

import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.worker import WorkerManager
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

logger = logging.getLogger(__name__)


class StreamingFuture(asyncio.Future):
    """Future that also exposes first chunk for streaming responses."""

    def __init__(self):
        super().__init__()
        self.first = asyncio.Future()


class FuturesHttpClient(HTTPEndpointClient):
    """
    HTTP client with futures-based API for async contexts.
    FuturesHttpClient will run on the current event loop.

    complete_callback is called with on all responses (chunks and final result).
    Final result is also available via the future object.
    """

    def __init__(self, *args, complete_callback=None, **kwargs):
        """Initialize FuturesHttpClient.

        Args:
            complete_callback: Optional callback function that receives responses.
                              For streaming: called with first StreamChunk, then final QueryResult.
                              For non-streaming: called with QueryResult.
            *args: Passed to HTTPEndpointClient.
            **kwargs: Passed to HTTPEndpointClient.
        """
        super().__init__(*args, **kwargs)
        self._pending_futures: dict[str | int, asyncio.Future] = {}
        self._response_handler_task: asyncio.Task | None = None
        self.complete_callback = complete_callback

    async def async_start(self):
        """Start HTTP client and response handler."""
        # Set loop to current running loop
        self.loop = asyncio.get_running_loop()

        # Initialize ZMQ, workers, and sockets (parent's async_start without loop creation)
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
        )

        # Schedule response handler in current loop
        self._response_handler_task = asyncio.create_task(self._handle_responses())

    async def issue_query(self, query: Query) -> asyncio.Future:
        """Issue query and return future for response."""
        # Create appropriate future type based on streaming flag
        future = (
            StreamingFuture() if query.data.get("stream", False) else asyncio.Future()
        )
        self._pending_futures[query.id] = future

        # Issue query via base class with error handling
        try:
            await super().issue_query_async(query)
        except Exception as e:
            # If send fails, set exception on future and clean up
            logger.exception(f"Failed to send query {query.id}: {e}")
            self._set_future_exception(future, e)
            self._pending_futures.pop(query.id, None)
            raise

        return future

    def _set_future_exception(self, future: asyncio.Future, exception: Exception):
        """Set exception on future and streaming first chunk if applicable."""
        if not future.done():
            future.set_exception(exception)
        if isinstance(future, StreamingFuture) and not future.first.done():
            future.first.set_exception(exception)

    async def _handle_responses(self):
        """Handle responses and complete futures."""
        while True:
            try:
                response = await self.get_ready_responses_async()

                # Handle timeout (no response available)
                if response is None:
                    continue

                future = self._pending_futures.get(response.id)
                if not future:
                    logger.warning(
                        f"Received response for unknown query: {response.id}"
                    )
                    continue

                if future.done():
                    logger.warning(
                        f"Received duplicate response for query: {response.id}"
                    )
                    continue

                # Handle different response types
                match response:
                    case StreamChunk(response_chunk=chunk, is_complete=False):
                        future.first.set_result(chunk)
                        if self.complete_callback:
                            self.complete_callback(response)

                    case StreamChunk(is_complete=True):
                        raise NotImplementedError(
                            "StreamChunk(is_complete=True) should not be received, QueryResult is expected instead"
                        )

                    case QueryResult(error=err) if err:
                        self._set_future_exception(future, Exception(err))
                        self._pending_futures.pop(response.id)

                    case QueryResult():
                        future.set_result(response)
                        if (
                            isinstance(future, StreamingFuture)
                            and not future.first.done()
                        ):
                            # For streaming futures with no first chunk, set first chunk to empty string
                            future.first.set_result("")

                        if self.complete_callback:
                            self.complete_callback(response)

                        self._pending_futures.pop(response.id)

                    case _:
                        logger.error(f"Unexpected response type: {type(response)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in response handler: {e}")

    async def async_shutdown(self):
        """Async shutdown for external loop usage."""
        # Cancel response handler task first
        if self._response_handler_task:
            self._response_handler_task.cancel()
            try:
                await asyncio.wait_for(self._response_handler_task, timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                pass

        # Cancel any pending futures
        for future in self._pending_futures.values():
            if not future.done():
                future.cancel()
        self._pending_futures.clear()

        # Call parent's shutdown to handle all cleanup
        await super().async_shutdown()

    def start(self):
        """Synchronous start is not supported for FuturesHttpClient.

        Raises:
            RuntimeError: Always raised to prevent improper usage.

        Use async_start() instead:
            await client.async_start()
        """
        raise RuntimeError(
            "FuturesHttpClient does not support synchronous start(). "
            "Use 'await client.async_start()' instead."
        )

    def shutdown(self):
        """Synchronous shutdown is not supported for FuturesHttpClient.

        Raises:
            RuntimeError: Always raised to prevent improper usage.

        Use async_shutdown() instead:
            await client.async_shutdown()
        """
        raise RuntimeError(
            "FuturesHttpClient does not support synchronous shutdown(). "
            "Use 'await client.async_shutdown()' instead."
        )
