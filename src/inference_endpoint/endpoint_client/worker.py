"""Worker process implementation for HTTP endpoint client."""

import asyncio
import logging
import os
import signal
from collections.abc import AsyncGenerator
from multiprocessing import Process
from typing import Any

import aiohttp
import orjson
import zmq
import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

logger = logging.getLogger(__name__)


def worker_main(
    worker_id: int,
    http_config: HTTPClientConfig,
    aiohttp_config: AioHttpConfig,
    zmq_config: ZMQConfig,
    request_queue_addr: str,
    response_queue_addr: str,
    readiness_queue_addr: str,
):
    """Entry point for worker process."""
    # Install uvloop which also enables it
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        logger.info("uvloop not available, using default event loop")

    # Create and run worker
    worker = Worker(
        worker_id=worker_id,
        http_config=http_config,
        aiohttp_config=aiohttp_config,
        zmq_config=zmq_config,
        request_socket_addr=request_queue_addr,
        response_socket_addr=response_queue_addr,
        readiness_socket_addr=readiness_queue_addr,
    )

    # Run event loop
    asyncio.run(worker.run())


class Worker:
    """Worker process that performs actual HTTP requests."""

    def __init__(
        self,
        worker_id: int,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        request_socket_addr: str,
        response_socket_addr: str,
        readiness_socket_addr: str,
    ):
        """Initialize worker with configurations and ZMQ addresses."""
        self.worker_id = worker_id
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.request_socket_addr = request_socket_addr
        self.response_socket_addr = response_socket_addr
        self.readiness_socket_addr = readiness_socket_addr
        self._shutdown = False
        self._session: aiohttp.ClientSession | None = None
        self._zmq_context: zmq.asyncio.Context | None = None
        self._request_socket: ZMQPullSocket | None = None
        self._response_socket: ZMQPushSocket | None = None
        self._readiness_socket: ZMQPushSocket | None = None
        self.tcp_connector: aiohttp.TCPConnector | None = None

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses."""
        # Initialize ZMQ context and sockets
        self._zmq_context = zmq.asyncio.Context()
        self._request_socket = ZMQPullSocket(
            self._zmq_context, self.request_socket_addr, self.zmq_config, bind=True
        )
        self._response_socket = ZMQPushSocket(
            self._zmq_context, self.response_socket_addr, self.zmq_config
        )
        self._readiness_socket = ZMQPushSocket(
            self._zmq_context, self.readiness_socket_addr, self.zmq_config
        )

        # Create TCP connector
        self.tcp_connector = self.aiohttp_config.create_tcp_connector()

        # Create aiohttp session with TCP connector
        self._session = aiohttp.ClientSession(
            connector=self.tcp_connector,
            timeout=aiohttp.ClientTimeout(
                total=self.aiohttp_config.client_timeout_total,
                connect=self.aiohttp_config.client_timeout_connect,
                sock_read=self.aiohttp_config.client_timeout_sock_read,
            ),
            connector_owner=self.aiohttp_config.client_session_connector_owner,
            skip_auto_headers=self.aiohttp_config.skip_auto_headers,
        )

        try:
            # Signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

            # Send readiness signal
            await self._readiness_socket.send(self.worker_id)
            logger.info(f"Worker {self.worker_id} started and ready")

            # Main processing loop
            while not self._shutdown:
                try:
                    # Pull query from queue with timeout
                    query = await asyncio.wait_for(
                        self._request_socket.receive(),
                        timeout=self.http_config.worker_request_timeout,
                    )

                    # Process query asynchronously
                    asyncio.create_task(self._process_request(query))

                except TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error: {e}")

        finally:
            # Cleanup
            await self._cleanup()

    async def _handle_error(self, query_id: str, error: Exception | str) -> None:
        """Send error response for a query."""
        error_message = str(error) if isinstance(error, Exception) else error
        error_response = QueryResult(
            query_id=query_id,
            response_output=None,
            error=error_message,
        )
        await self._response_socket.send(error_response)

    async def _make_http_request(
        self, query: Query
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """
        Common HTTP request setup and execution.

        Yields the response object if status is 200.
        Handles error cases and sends error responses.
        """
        url = self.http_config.endpoint_url
        headers = query.headers if hasattr(query, "headers") else {}

        async with self._session.post(
            url,
            json=query.to_json(),
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                await self._handle_error(
                    query.id, f"HTTP {response.status}: {error_text}"
                )
                return

            yield response

    async def _process_request(self, query: Query) -> None:
        """Process a single query."""
        try:
            if query.stream:
                await self._handle_streaming_request(query)
            else:
                await self._handle_non_streaming_request(query)

        except Exception as e:
            await self._handle_error(query.id, e)

    async def _handle_streaming_request(self, query: Query) -> None:
        """Handle streaming response."""
        async for response in self._make_http_request(query):
            accumulated_content = []
            first_chunk_sent = False

            # Process SSE stream
            async for line_bytes in response.content:
                lines = line_bytes.decode("utf-8").strip().split("\n")

                for line in lines:
                    # Skip empty lines and non-SSE data
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    # Parse JSON and extract content
                    try:
                        chunk_data = orjson.loads(data_str)
                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        is_final = delta.get("finish_reason") is not None

                        if not content:
                            continue

                        accumulated_content.append(content)

                        # Send first chunk with metadata
                        if not first_chunk_sent:
                            stream_chunk = StreamChunk(
                                query_id=query.id,
                                response_chunk=content,
                                is_complete=is_final,
                                metadata={"first_chunk": True, "final_chunk": is_final},
                            )
                            await self._response_socket.send(stream_chunk)
                            first_chunk_sent = True

                    except (ValueError, TypeError, KeyError):
                        continue

            # Send final complete response
            final_response = QueryResult(
                query_id=query.id,
                response_output="".join(accumulated_content),
                metadata={"first_chunk": not first_chunk_sent, "final_chunk": True},
            )
            await self._response_socket.send(final_response)

    async def _handle_non_streaming_request(self, query: Query) -> None:
        """Handle non-streaming response."""
        async for response in self._make_http_request(query):
            response_text = await response.text()

            # Parse JSON response
            try:
                response_data = orjson.loads(response_text)
                response_obj = QueryResult.from_json(response_data)
            except (ValueError, TypeError) as e:
                await self._handle_error(
                    query.id, f"Failed to parse response: {str(e)}"
                )
                return

            await self._response_socket.send(response_obj)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}")
        self._shutdown = True

    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info(f"Worker {self.worker_id} shutting down...")

        # Close TCP connector
        if self.tcp_connector:
            await self.tcp_connector.close()
            self.tcp_connector = None

        # Close aiohttp session
        if self._session:
            await self._session.close()

        # Close ZMQ sockets
        if self._request_socket:
            self._request_socket.close()
        if self._response_socket:
            self._response_socket.close()
        if self._readiness_socket:
            self._readiness_socket.close()

        # Terminate ZMQ context
        if self._zmq_context:
            self._zmq_context.term()


class WorkerManager:
    """Manages the lifecycle of worker processes."""

    def __init__(
        self,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        zmq_context: zmq.asyncio.Context,
    ):
        """Initialize worker manager."""
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.zmq_context = zmq_context
        self.workers: list[Process] = []
        self.worker_pids: dict[int, int] = {}  # worker_id -> pid
        self._shutdown_event = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize workers and ZMQ infrastructure."""
        # Create readiness pull socket to receive worker ready signals
        readiness_socket = ZMQPullSocket(
            self.zmq_context,
            self.zmq_config.zmq_readiness_queue_addr,
            self.zmq_config,
            bind=True,
        )

        try:
            # Spawn worker processes
            for i in range(self.http_config.num_workers):
                worker = self._spawn_worker(i)
                self.workers.append(worker)
                self.worker_pids[i] = worker.pid

            # Wait for all workers to signal readiness
            ready_workers = set()
            start_time = asyncio.get_event_loop().time()
            timeout = self.http_config.worker_initialization_timeout

            while len(ready_workers) < self.http_config.num_workers:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Workers failed to initialize within {timeout} seconds. "
                        f"Only {len(ready_workers)}/{self.http_config.num_workers} workers ready."
                    )

                try:
                    # Wait for readiness signal with remaining timeout
                    remaining_timeout = timeout - elapsed
                    worker_id = await asyncio.wait_for(
                        readiness_socket.receive(), timeout=remaining_timeout
                    )
                    ready_workers.add(worker_id)
                    logger.info(
                        f"Worker {worker_id} is ready ({len(ready_workers)}/{self.http_config.num_workers})"
                    )
                except TimeoutError:
                    continue

            logger.info(f"All {self.http_config.num_workers} workers are ready")

            # Start monitoring task
            self._monitor_task = asyncio.create_task(self._monitor_workers())

        finally:
            # Close readiness socket
            readiness_socket.close()

    def _spawn_worker(self, worker_id: int) -> Process:
        """Spawn a single worker process."""
        request_queue_addr = (
            f"{self.zmq_config.zmq_request_queue_prefix}_{worker_id}_requests"
        )
        response_queue_addr = self.zmq_config.zmq_response_queue_addr
        readiness_queue_addr = self.zmq_config.zmq_readiness_queue_addr

        # Create worker process
        process = Process(
            target=worker_main,
            args=(
                worker_id,
                self.http_config,
                self.aiohttp_config,
                self.zmq_config,
                request_queue_addr,
                response_queue_addr,
                readiness_queue_addr,
            ),
            daemon=False,
        )
        process.start()
        return process

    async def _monitor_workers(self) -> None:
        """Monitor worker health and restart if needed."""
        while not self._shutdown_event.is_set():
            for i, worker in enumerate(self.workers):
                if not worker.is_alive():
                    logger.warning(f"Worker {i} died, restarting...")
                    # Terminate zombie process
                    if worker.pid:
                        try:
                            os.kill(worker.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

                    # Spawn new worker
                    new_worker = self._spawn_worker(i)
                    self.workers[i] = new_worker
                    self.worker_pids[i] = new_worker.pid

            await asyncio.sleep(self.http_config.worker_health_check_interval)

    async def shutdown(self) -> None:
        """Graceful shutdown of all workers."""
        self._shutdown_event.set()

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Send SIGTERM to all workers
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()

        # Wait for graceful shutdown
        await asyncio.sleep(self.http_config.worker_graceful_shutdown_wait)

        # Force kill any remaining workers
        for worker in self.workers:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=self.http_config.worker_force_kill_timeout)
