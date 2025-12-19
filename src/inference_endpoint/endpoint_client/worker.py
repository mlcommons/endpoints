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

"""Worker process implementation for HTTP endpoint client."""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback
from collections.abc import AsyncGenerator
from multiprocessing import Process
from typing import Any

import aiohttp
import zmq
import zmq.asyncio

from inference_endpoint.core.types import (
    Query,
    QueryResult,
    StreamChunk,
)
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.metrics.recorder import EventRecorder
from inference_endpoint.metrics.reporter import MetricsReporter
from inference_endpoint.profiling import profile

logger = logging.getLogger(__name__)


# Configure multiprocessing to use 'spawn' method for worker creation
# - 'spawn' starts a fresh Python interpreter for each worker (clean slate)
# - Slower startup (re-import modules) vs fork's copy-on-write
# - Requires pickling (can't use local functions in worker_main)
# - This is the recommended approach for async + multiprocessing applications
# - uvloop requires use of 'spawn'
try:
    multiprocessing.set_start_method("spawn", force=False)
except RuntimeError:
    # Already set, which is fine (likely in tests or when importing multiple times)
    pass


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
    # Configure logging for worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker-{worker_id}-{os.getpid()}] %(levelname)-5s - %(message)s (%(module)s:%(lineno)d)",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Install uvloop which also enables it
    import uvloop

    uvloop.install()

    # Create and run worker
    try:
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
        uvloop.run(worker.run())

    except Exception as e:
        logger.error(
            f"Worker {worker_id} crashed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )
        sys.exit(1)


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

        self._zmq_context: zmq.asyncio.Context | None = None
        self._request_socket: ZMQPullSocket | None = None
        self._response_socket: ZMQPushSocket | None = None
        self._readiness_socket: ZMQPushSocket | None = None

        self._session: aiohttp.ClientSession | None = None
        self.tcp_connector: aiohttp.TCPConnector | None = None

        # Track active request tasks
        self._active_tasks: set[asyncio.Task] = set()

        # Use adapter type from config
        self._adapter = self.http_config.adapter

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses."""
        try:
            # Initialize ZMQ context and sockets
            self._zmq_context = zmq.asyncio.Context()
            self._request_socket = ZMQPullSocket(
                self._zmq_context,
                self.request_socket_addr,
                self.zmq_config,
                bind=True,
                decoder_type=Query,
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
                timeout=aiohttp.ClientTimeout(
                    total=self.aiohttp_config.client_timeout_total,
                    connect=self.aiohttp_config.client_timeout_connect,
                    sock_read=self.aiohttp_config.client_timeout_sock_read,
                ),
                connector=self.tcp_connector,
                connector_owner=False,  # owned by Worker
                skip_auto_headers=self.aiohttp_config.skip_auto_headers,
            )

            # Signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self.shutdown)
            signal.signal(signal.SIGINT, self.shutdown)

            # Send readiness signal only after successful initialization
            await self._readiness_socket.send(self.worker_id)
            logger.info(f"Worker {self.worker_id} started and ready")

        except Exception as e:
            logger.error(
                f"Worker {self.worker_id} failed to initialize: {type(e).__name__}: {str(e)}"
            )
            # Exit with error code to signal failure to parent process
            sys.exit(1)
        finally:
            if self._readiness_socket:
                self._readiness_socket.close(linger_ms=1000)

        try:
            # Run main processing loop
            if self.http_config.record_worker_events:
                pid = os.getpid()
                worker_db_name = f"worker_report_{self.worker_id}_{pid}"
                report_path = self.http_config.event_logs_dir / f"{worker_db_name}.csv"
                logger.info(f"About to generate report {self.worker_id}")
                with EventRecorder(session_id=worker_db_name) as event_recorder:
                    await self._main_loop()
                    event_recorder.wait_for_writes(force_commit=True)
                    with MetricsReporter(event_recorder.connection_name) as reporter:
                        logger.info(f"About to dump worker report to {report_path}")
                        reporter.dump_all_to_csv(report_path)
                        logger.info(f"Worker report dumped to {report_path}")
            else:
                # No logging, just run the main loop
                await self._main_loop()
        except Exception as e:
            logger.error(
                f"Error in worker {self.worker_id}: {type(e).__name__}: {str(e)}"
            )

        finally:
            # Cleanup
            await self._cleanup()

    @profile
    async def _main_loop(self) -> None:
        """Main processing loop - continuously pull and process requests."""
        while not self._shutdown:
            try:
                # Pull query from queue with timeout
                query = await self._request_socket.receive()

                # Handle timeout (no query available)
                if query is None:
                    continue

                if self.http_config.record_worker_events:
                    EventRecorder.record_event(
                        SampleEvent.ZMQ_REQUEST_RECEIVED,
                        time.monotonic_ns(),
                        sample_uuid=query.id,
                        assert_active=True,
                    )
                # Process query asynchronously and track the task
                task = asyncio.create_task(self._process_request(query))
                self._active_tasks.add(task)

                # Remove task from active set when it completes
                task.add_done_callback(self._active_tasks.discard)

            except asyncio.CancelledError:
                break

            except Exception as e:
                # Don't exit on errors in the main loop, just log and continue
                logger.error(
                    f"Worker {self.worker_id} error in main loop: {type(e).__name__}: {str(e)}"
                )

    async def _handle_error(self, query_id: str, error: Exception | str) -> None:
        """Send error response for a query."""

        # Skip if we're shutting down or response socket is not available
        if self._shutdown or not self._response_socket:
            return

        error_message = str(error) if isinstance(error, Exception) else error
        error_response = QueryResult(
            id=query_id,
            response_output=None,
            error=error_message,
        )
        await self._response_socket.send(error_response)
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=query_id,
                assert_active=True,
            )

    @profile
    async def _make_http_request(self, query: Query):
        """
        Common HTTP request setup and execution as async generator.

        Yields the response object if status is 200.
        Handles error cases and sends error responses.
        Auto-closes response when context exits.

        NOTE:
        Does not use @asynccontextmanager (which allows "async with" syntax)
        This is done to avoid context manager overhead.

        Usage:
            async for response in self._make_http_request(query):
                # use response
        """
        # Check if we're shutting down or session is closed
        if self._shutdown or not self._session:
            await self._handle_error(query.id, "Worker is shutting down")
            return

        url = self.http_config.endpoint_url
        headers = (
            query.headers
            if hasattr(query, "headers") and len(query.headers) > 0
            else {"content-type": "application/json"}
        )

        logging.debug(
            f"Making HTTP request to {url} with query: {query} and headers: {headers}"
        )

        # Encode query to bytes using adapter
        payload_bytes = self._adapter.encode_query(query)

        # Issue the request with pre-encoded bytes
        # TODO replace with debug mode recorder
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.HTTP_REQUEST_ISSUED,
                time.monotonic_ns(),
                sample_uuid=query.id,
                assert_active=True,
            )

        async with self._session.post(
            url, data=payload_bytes, headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                await self._handle_error(
                    query.id, f"HTTP {response.status}: {error_text}"
                )
                logger.error(f"Request {query.id} failed with HTTP Error: {error_text}")
                return
            logger.debug(f"HTTP Response: {response}")
            yield response

        # TODO replace with debug mode recorder
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.HTTP_RESPONSE_COMPLETED,
                time.monotonic_ns(),
                sample_uuid=query.id,
                assert_active=True,
            )

    async def _process_request(self, query: Query) -> None:
        """Process a single query."""
        try:
            if query.data.get("stream", False):
                await self._handle_streaming_request(query)
            else:
                await self._handle_non_streaming_request(query)

        except Exception as e:
            await self._handle_error(query.id, e)

    @profile
    async def _iter_sse_lines(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[list[str], None]:
        """
        Iterate over complete SSE chunks (events) from response stream.

        SSE events are delimited by double newlines (\\n\\n).
        Handles incomplete chunks at boundaries by buffering until
        a complete event is encountered.

        Yields all complete chunks from a single network read as a batch,
        with content extracted from each SSE event, to reduce async
        suspend/resume overhead.
        """
        incomplete_chunk = b""

        async for chunk_bytes in response.content.iter_any():
            # Prepend incomplete chunk and find last complete event boundary
            buffer = incomplete_chunk + chunk_bytes
            last_delimiter = buffer.rfind(b"\n\n")

            if last_delimiter == -1:
                # No complete events yet, buffer everything
                incomplete_chunk = buffer
                continue

            # Save incomplete chunk for next iteration (+2 skips "\n\n")
            incomplete_chunk = buffer[last_delimiter + 2 :]

            # Yield batch if any content found
            if parsed_contents := self._adapter.parse_sse_chunk(buffer, last_delimiter):
                yield parsed_contents

        # After stream ends, parse any remaining incomplete chunk
        if incomplete_chunk:
            if parsed_contents := self._adapter.parse_sse_chunk(
                incomplete_chunk, len(incomplete_chunk)
            ):
                yield parsed_contents

    @profile
    async def _handle_streaming_request(self, query: Query) -> None:
        """Handle streaming response."""
        async for response in self._make_http_request(query):
            output_chunks = []
            reasoning_chunks = []
            first_chunk_sent = False

            # Process SSE stream - yields batches of chunks
            async for chunk_batch in self._iter_sse_lines(response):
                output_delta = []
                reasoning_delta = []
                for delta in chunk_batch:
                    if delta.content:
                        output_delta.append(delta.content)
                    elif delta.reasoning:
                        reasoning_delta.append(delta.reasoning)
                    else:
                        logger.debug("empty SSE delta")
                        continue

                for delta_batch, accumulator in (
                    (reasoning_delta, reasoning_chunks),
                    (output_delta, output_chunks),
                ):
                    if not delta_batch:
                        continue
                    accumulator.extend(delta_batch)

                    # Determine which chunks to send: all or just first
                    chunks_to_send = (
                        delta_batch
                        if self.http_config.stream_all_chunks
                        else delta_batch[:1]
                        if not first_chunk_sent
                        else []
                    )

                    # Send chunks
                    for content in chunks_to_send:
                        await self._response_socket.send(
                            StreamChunk(
                                id=query.id,
                                response_chunk=content,
                                is_complete=False,
                                metadata={
                                    "first_chunk": not first_chunk_sent,
                                    "final_chunk": False,
                                },
                            )
                        )
                        first_chunk_sent = True
                        if self.http_config.record_worker_events:
                            EventRecorder.record_event(
                                SampleEvent.ZMQ_RESPONSE_SENT,
                                time.monotonic_ns(),
                                sample_uuid=query.id,
                                assert_active=True,
                            )

            # Send final complete response
            if reasoning_chunks:
                # If there are reasoning chunks, then the first chunk received
                # is the first reasoning chunk. The rest of the reasoning chunks,
                # as well as the output chunks can be joined together.
                resp_reasoning = [reasoning_chunks[0]]
                if len(reasoning_chunks) > 1:
                    resp_reasoning.append("".join(reasoning_chunks[1:]))
                response_output = {
                    "output": "".join(output_chunks),
                    "reasoning": resp_reasoning,
                }
            elif output_chunks:
                # If there are only output chunks, the first chunk is the used for
                # TTFT calculations. The rest are joined together.
                resp_output = [output_chunks[0]]
                if len(output_chunks) > 1:
                    resp_output.append("".join(output_chunks[1:]))
                response_output = {"output": resp_output}
            else:
                response_output = {"output": []}

            await self._response_socket.send(
                QueryResult(
                    id=query.id,
                    response_output=response_output,
                    metadata={"first_chunk": not first_chunk_sent, "final_chunk": True},
                )
            )
            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.ZMQ_RESPONSE_SENT,
                    time.monotonic_ns(),
                    sample_uuid=query.id,
                    assert_active=True,
                )

    @profile
    async def _handle_non_streaming_request(self, query: Query) -> None:
        """Handle non-streaming response."""
        async for response in self._make_http_request(query):
            response_bytes = await response.read()
            result = self._adapter.decode_response(response_bytes, query.id)
            await self._response_socket.send(result)
            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.ZMQ_RESPONSE_SENT,
                    time.monotonic_ns(),
                    sample_uuid=query.id,
                    assert_active=True,
                )

    def shutdown(self, signum: int | None = None, frame: Any | None = None) -> None:
        """Trigger shutdown of worker process."""
        self._shutdown = True  # will invoke _cleanup() from main loop

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Close aiohttp session
        if self._session:
            await self._session.close()

        # Cancel and clear active tasks
        logger.info(
            f"Worker {self.worker_id} will cancel {len(self._active_tasks)} tasks and cleanup."
        )
        for task in self._active_tasks:
            task.cancel()
        self._active_tasks.clear()

        # Close TCP connector
        if self.tcp_connector:
            await self.tcp_connector.close()

        # Close ZMQ sockets (readiness socket already closed after initialization)
        for socket in (self._request_socket, self._response_socket):
            if socket:
                socket.close()

        # Terminate ZMQ context
        if self._zmq_context:
            self._zmq_context.destroy(linger=0)


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
            logger.info(f"Starting {self.http_config.num_workers} worker processes")

            # Spawn worker processes
            for i in range(self.http_config.num_workers):
                worker = self._spawn_worker(i)
                self.workers.append(worker)
                self.worker_pids[i] = worker.pid

            # Wait for all workers to signal readiness
            async def wait_for_all_workers():
                ready_count = 0
                # Keep trying until we get all N readiness signals

                while ready_count < self.http_config.num_workers:
                    worker_id = await readiness_socket.receive()
                    if worker_id is not None:
                        ready_count += 1
                        logger.debug(
                            f"Worker {worker_id} is ready ({ready_count}/{self.http_config.num_workers})"
                        )

                return ready_count

            ready_count = await asyncio.wait_for(
                wait_for_all_workers(),
                timeout=self.http_config.worker_initialization_timeout,
            )
            logger.info(f"{ready_count}/{self.http_config.num_workers} workers ready")
        except TimeoutError as e:
            raise TimeoutError(
                f"Workers failed to initialize within {self.http_config.worker_initialization_timeout} seconds."
            ) from e
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

    async def shutdown(self) -> None:
        """
        Graceful shutdown of all workers with proper zombie reaping.

        Zombie processes occur when a child dies but parent hasn't called join().
        Without proper reaping, dead workers remain in the process table as zombies
        with state 'Z'.
        """
        self._shutdown_event.set()

        # Send SIGTERM to alive workers
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()  # Send SIGTERM for graceful shutdown
            else:
                # Worker is already dead (possibly zombie) - reap immediately
                worker.join(timeout=0.1)

        # Wait for graceful shutdown to complete
        await asyncio.sleep(self.http_config.worker_graceful_shutdown_wait)

        # Force kill any remaining workers and ensure all are reaped
        loop = asyncio.get_event_loop()
        for worker in self.workers:
            if worker.is_alive():
                # Worker didn't respond to SIGTERM - force kill with SIGKILL
                worker.kill()
                # Run blocking join() in thread pool to avoid blocking event loop
                # This reaps the forcefully killed worker
                await loop.run_in_executor(
                    None, worker.join, self.http_config.worker_force_kill_timeout
                )
            else:
                # Worker died during graceful shutdown - reap to prevent zombie
                worker.join(timeout=0.1)
