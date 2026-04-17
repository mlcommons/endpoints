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

from inference_endpoint.config.schema import APIType
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
from inference_endpoint.openai.types import SSEDelta as OpenAISSEDelta
from inference_endpoint.profiling import profile
from inference_endpoint.sglang.types import SGLangSSEDelta
from inference_endpoint.utils.cpu_affinity import (
    AVAILABLE_CPUS,
    get_cpus_sorted_by_numa_preference,
    set_cpu_affinity,
)
from inference_endpoint.utils.logging import setup_logging

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
):
    """Entry point for worker process."""
    worker_log_format = f"%(asctime)s - %(name)s[W{worker_id}/%(process)d] - %(funcName)s - %(levelname)s - %(message)s"
    setup_logging(level=http_config.log_level, format_string=worker_log_format)

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
        )

        # Run event loop
        uvloop.run(worker.run())

    except Exception as e:
        logger.error(f"Crashed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


class OpenAISSEAccumulator:
    def __init__(self, query_id: str, stream_all_chunks: bool):
        self.output_chunks = []
        self.reasoning_chunks = []

        self.first_chunk_sent = False
        self.query_id = query_id
        self.stream_all_chunks = stream_all_chunks

    def add_chunk(self, delta: OpenAISSEDelta) -> StreamChunk | None:
        if not isinstance(delta, OpenAISSEDelta):
            return None

        content = None
        if delta.content:
            self.output_chunks.append(delta.content)
            content = delta.content
        elif delta.reasoning:
            self.reasoning_chunks.append(delta.reasoning)
            content = delta.reasoning
        else:
            logger.debug("empty SSE delta")
            return None

        if content is not None and (
            self.stream_all_chunks or not self.first_chunk_sent
        ):
            return StreamChunk(
                id=self.query_id,
                response_chunk=content,
                is_complete=False,
                metadata={
                    "first_chunk": not self.first_chunk_sent,
                    "final_chunk": False,
                },
            )
        else:
            return None

    def get_final_output(self) -> QueryResult:
        response_output = {"output": []}
        if self.reasoning_chunks:
            # If there are reasoning chunks, then the first chunk received
            # is the first reasoning chunk. The rest of the reasoning chunks,
            # as well as the output chunks can be joined together.
            resp_reasoning = [self.reasoning_chunks[0]]
            if len(self.reasoning_chunks) > 1:
                resp_reasoning.append("".join(self.reasoning_chunks[1:]))

            response_output = {
                "output": "".join(self.output_chunks),
                "reasoning": resp_reasoning,
            }
        elif self.output_chunks:
            # If there are only output chunks, the first chunk is the used for
            # TTFT calculations. The rest are joined together.
            resp_output = [self.output_chunks[0]]
            if len(self.output_chunks) > 1:
                resp_output.append("".join(self.output_chunks[1:]))

            response_output = {
                "output": resp_output,
            }
        return QueryResult(
            id=self.query_id,
            response_output=response_output,
            metadata={
                "first_chunk": not self.first_chunk_sent,
                "final_chunk": True,
            },
        )


class SGLangSSEAccumulator:
    def __init__(self, query_id: str, stream_all_chunks: bool):
        self.text = ""
        self.token_ids = []
        self.total_tokens = 0
        self.retraction_occurred = False

        self.first_chunk_sent = False
        self.query_id = query_id
        self.stream_all_chunks = stream_all_chunks

    def add_chunk(self, delta: SGLangSSEDelta) -> StreamChunk | None:
        if not isinstance(delta, SGLangSSEDelta):
            return None

        if delta.total_completion_tokens == self.total_tokens:
            return None

        # In SGLang /generate, the .text field is the total accumulated text, not
        # a difference, so we'll need to compute the diff for the StreamChunk
        content_diff = ""
        if len(delta.text) > (start_idx := len(self.text)):
            content_diff = delta.text[start_idx:]
        self.text = delta.text
        self.token_ids.extend(delta.token_delta)
        self.total_tokens = delta.total_completion_tokens
        if delta.has_retractions:
            # For now, we won't be handling retractions if they occur, but we will
            # report it as part of the metadata if it does happen.
            self.retraction_occurred = True

        if content_diff and (self.stream_all_chunks or not self.first_chunk_sent):
            metadata = {
                "first_chunk": not self.first_chunk_sent,
                "final_chunk": False,
                "retraction_occurred": delta.has_retractions,
                "n_tokens": len(delta.token_delta),
            }
            return StreamChunk(
                id=self.query_id,
                response_chunk=content_diff,
                is_complete=False,
                metadata=metadata,
            )
        else:
            return None

    def get_final_output(self) -> QueryResult:
        return QueryResult(
            id=self.query_id,
            response_output=self.text,
            metadata={
                "first_chunk": not self.first_chunk_sent,
                "final_chunk": True,
                "retraction_occurred": self.retraction_occurred,
                "n_tokens": self.total_tokens,
                "token_ids": self.token_ids,
            },
        )


class Worker:
    """Worker process that performs actual HTTP requests."""

    def __init__(
        self,
        worker_id: int,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
    ):
        """Initialize worker with configurations."""
        self.worker_id = worker_id
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
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
            # Derive socket addresses from zmq_config
            request_socket_addr = (
                f"{self.zmq_config.zmq_request_queue_prefix}_{self.worker_id}_requests"
            )
            logger.debug("Request Socket Addr: %s", request_socket_addr)
            response_socket_addr = self.zmq_config.zmq_response_queue_addr
            logger.debug("Response Socket Addr: %s", response_socket_addr)
            readiness_socket_addr = self.zmq_config.zmq_readiness_queue_addr
            logger.debug("Readiness Socket Addr: %s", readiness_socket_addr)

            # Initialize ZMQ context and sockets
            self._zmq_context = zmq.asyncio.Context()
            self._request_socket = ZMQPullSocket(
                self._zmq_context,
                request_socket_addr,
                self.zmq_config,
                bind=True,
                decoder_type=Query,
            )
            self._response_socket = ZMQPushSocket(
                self._zmq_context, response_socket_addr, self.zmq_config
            )
            self._readiness_socket = ZMQPushSocket(
                self._zmq_context, readiness_socket_addr, self.zmq_config
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
            logger.debug("Started and ready")

        except Exception as e:
            logger.error(f"Failed to initialize: {type(e).__name__}: {str(e)}")
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
                logger.debug("About to generate report")
                pg_storage = None
                if self.http_config.db_backend == "postgres":
                    from inference_endpoint.storage.db import PostgresBackend

                    pg_storage = PostgresBackend(conninfo=self.http_config.db_conninfo)

                with EventRecorder(
                    session_id=worker_db_name,
                    backend=self.http_config.db_backend,
                    storage=pg_storage,
                ) as event_recorder:
                    await self._main_loop()
                    print("calling run wait_for_writes")

                    event_recorder.wait_for_writes(force_commit=True)

                    if self.http_config.db_backend == "postgres":
                        reporter_client = "postgres"
                        reporter_table = event_recorder.table_name
                        from inference_endpoint.storage.db import PostgresBackend

                        reporter_storage = PostgresBackend(
                            conninfo=self.http_config.db_conninfo
                        )
                    else:
                        reporter_client = "sqlite"
                        reporter_table = "events"
                        reporter_storage = None

                    with MetricsReporter(
                        event_recorder.connection_name,
                        client_type=reporter_client,
                        table_name=reporter_table,
                        storage=reporter_storage,
                    ) as reporter:
                        logger.debug(f"About to dump report to {report_path}")
                        reporter.dump_all_to_csv(report_path)
                        logger.debug(f"Report dumped to {report_path}")
            else:
                # No logging, just run the main loop
                await self._main_loop()
        except Exception as e:
            logger.error(f"Error: {type(e).__name__}: {str(e)}")

        finally:
            # Cleanup
            await self._cleanup()

    @profile
    async def _main_loop(self) -> None:
        """Main processing loop - continuously pull and process requests."""
        while not self._shutdown:
            try:
                # TODO(vir): re-do work-consumer loop to leverage built-in zmq load-balancing
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

                # Create async-task to process request concurrently
                task = asyncio.create_task(self._process_request(query))

                # Record task to prevent garbage collection during execution
                # Task removes itself from set in _process_request's finally block
                self._active_tasks.add(task)

            except asyncio.CancelledError:
                break

            except Exception as e:
                # Don't exit on errors in the main loop, just log and continue
                logger.error(f"Error in main loop: {type(e).__name__}: {str(e)}")

    async def _handle_error(self, query_id: str, error: Exception | str) -> None:
        """Send error response for a query."""

        # Skip if we're shutting down or response socket is not available
        if self._shutdown or not self._response_socket:
            return

        error_message = repr(error) if isinstance(error, Exception) else error
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
        logger.debug(
            f"Making HTTP request to {url} with query: {query} and headers: {query.headers}"
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
            url, data=payload_bytes, headers=query.headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                await self._handle_error(
                    query.id, f"HTTP {response.status}: {error_text}"
                )
                logger.error(f"Request {query.id} failed: HTTP {error_text}")
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

        finally:
            # Clean up task reference to prevent memory leak (~850 MB per 1M tasks).
            # This is faster than using add_done_callback to remove from set
            # since it avoids an extra yield -> function-call per task
            self._active_tasks.discard(asyncio.current_task())

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
        if self.http_config.api_type == APIType.SGLANG:
            accumulator_type = SGLangSSEAccumulator
        else:
            # Default to OpenAI compatible adapter
            accumulator_type = OpenAISSEAccumulator

        async for response in self._make_http_request(query):
            accumulator = accumulator_type(query.id, self.http_config.stream_all_chunks)

            # Process SSE stream - yields batches of chunks
            async for chunk_batch in self._iter_sse_lines(response):
                for delta in chunk_batch:
                    if stream_chunk := accumulator.add_chunk(delta):
                        await self._response_socket.send(stream_chunk)
                        accumulator.first_chunk_sent = True

                        if self.http_config.record_worker_events:
                            EventRecorder.record_event(
                                SampleEvent.ZMQ_RESPONSE_SENT,
                                time.monotonic_ns(),
                                sample_uuid=query.id,
                                assert_active=True,
                            )

            await self._response_socket.send(accumulator.get_final_output())
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
        # Cancel pending tasks to drop HTTP requests
        if not_done := len(self._active_tasks):
            [task.cancel() for task in self._active_tasks]
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()
            logger.debug(f"Cancelled {not_done} pending requests.")

        # Close aiohttp session
        if self._session:
            await self._session.close()

        # Close TCP connector
        if self.tcp_connector:
            await self.tcp_connector.close()

        # Close ZMQ sockets
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

        initialization_succeeded = False
        try:
            logger.debug(f"Starting {self.http_config.num_workers} worker processes")

            # Spawn worker processes
            for i in range(self.http_config.num_workers):
                worker = self._spawn_worker(i)
                self.workers.append(worker)
                self.worker_pids[i] = worker.pid

            # Apply CPU affinity after all workers are started
            self._pin_workers()

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
            logger.debug(f"{ready_count}/{self.http_config.num_workers} workers ready")
            initialization_succeeded = True
        except TimeoutError as e:
            raise TimeoutError(
                f"Workers failed to initialize within {self.http_config.worker_initialization_timeout} seconds."
            ) from e
        finally:
            # Close readiness socket
            readiness_socket.close()
            # Shutdown any spawned workers on error/interrupt
            if not initialization_succeeded and self.workers:
                await self.shutdown()

    def _spawn_worker(self, worker_id: int) -> Process:
        """Spawn a single worker process."""
        # Create worker process as daemon.
        # 1. Automatic Termination: No need to manually join/reap worker processes
        # 2. Zombie Prevention: Relies on multiprocessing's internal atexit handler
        process = Process(
            target=worker_main,
            args=(
                worker_id,
                self.http_config,
                self.aiohttp_config,
                self.zmq_config,
            ),
            daemon=True,
        )
        process.start()
        return process

    def _pin_workers(self) -> None:
        """
        Pin workers to CPU cores based on config:
         - "auto": distribute workers across available CPUs
         - list[int]: pin workers to specified cores (round-robin)
         - None or falsy: disable CPU affinity override
        """
        if not self.http_config.cpu_affinity:
            return

        match self.http_config.cpu_affinity:
            case "auto":
                cpu_list = get_cpus_sorted_by_numa_preference()
            case list():
                cpu_list = sorted(set(self.http_config.cpu_affinity) & AVAILABLE_CPUS)
            case _:
                return

        # assign CPU affinity round-robin among available CPUs
        if not cpu_list:
            logger.warning("No available CPUs for worker pinning")
            return

        for worker_id, pid in self.worker_pids.items():
            cpus = {cpu_list[worker_id % len(cpu_list)]}
            set_cpu_affinity(pid=pid, cpus=cpus)

    async def shutdown(self) -> None:
        """Graceful shutdown of all workers."""
        self._shutdown_event.set()

        # Send SIGTERM to alive workers for graceful shutdown
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()

        # Wait for graceful shutdown
        await asyncio.sleep(self.http_config.worker_graceful_shutdown_wait)

        # Force kill any remaining workers
        for worker in self.workers:
            if worker.is_alive():
                worker.kill()

        # Join all workers to ensure termination
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    worker.join, timeout=self.http_config.worker_force_kill_timeout
                )
                for worker in self.workers
            )
        )
