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
from functools import partial
from multiprocessing import Process
from typing import Any

import aiohttp
import zmq
import zmq.asyncio
from aiohttp import hdrs
from aiohttp.client_reqrep import ClientRequest, ClientResponse
from yarl import URL

from inference_endpoint.core.types import (
    Query,
    QueryResult,
    StreamChunk,
)
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter
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
from inference_endpoint.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class PreparedRequest:
    """Track request state and timing."""

    __slots__ = ("query_id", "client_request", "timing_ctx", "process", "response")

    def __init__(
        self,
        query_id: str,
        client_request: ClientRequest,
        timing_ctx: dict,
        process,
    ):
        self.query_id = query_id
        self.client_request = client_request
        self.timing_ctx = timing_ctx
        self.process = process
        self.response: ClientResponse | None = None

    def log_timing(self) -> None:
        """Log request lifecycle timing."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        ctx = self.timing_ctx
        t_recv = ctx["t_recv"]
        t_prepare = ctx["t_prepare"]
        t_http = ctx["t_http"]
        t_headers = ctx["t_headers"]
        t_first_chunk = ctx.get("t_first_chunk")
        t_response = ctx["t_response"]
        t_zmq_sent = ctx["t_zmq_sent"]

        d_recv_to_prepare = (t_prepare - t_recv) / 1000.0
        d_prepare_to_http = (t_http - t_prepare) / 1000.0
        d_http_to_headers = (t_headers - t_http) / 1000.0
        d_response_to_zmq = (t_zmq_sent - t_response) / 1000.0
        d_end_to_end = (t_zmq_sent - t_recv) / 1000.0
        d_pre_overhead = (t_http - t_recv) / 1000.0
        d_post_overhead = (t_zmq_sent - t_response) / 1000.0

        parts = [
            f"d_recv_to_prepare={d_recv_to_prepare:.1f}us",
            f"d_prepare_to_http={d_prepare_to_http:.1f}us",
            f"d_http_to_headers={d_http_to_headers:.1f}us",
        ]

        if t_first_chunk:
            d_headers_to_first = (t_first_chunk - t_headers) / 1000.0
            d_first_to_last = (t_response - t_first_chunk) / 1000.0
            parts.append(f"d_headers_to_first={d_headers_to_first:.1f}us")
            parts.append(f"d_first_to_last={d_first_to_last:.1f}us")

        parts.append(f"d_response_to_zmq={d_response_to_zmq:.1f}us")
        parts.append(f"d_pre_overhead={d_pre_overhead:.1f}us")
        parts.append(f"d_post_overhead={d_post_overhead:.1f}us")
        parts.append(f"d_end_to_end={d_end_to_end:.1f}us")

        logger.debug(f"[{self.query_id}] timing: {', '.join(parts)}")


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
        self._timeout: aiohttp.ClientTimeout | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._response_params: dict | None = None  # cached set_response_params kwargs

        # Track active request tasks
        self._active_tasks: set[asyncio.Task] = set()

        self._url: URL = URL(self.http_config.endpoint_url)
        self._adapter: type[HttpRequestAdapter] = self.http_config.adapter

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses."""
        try:
            # Cache event loop reference (avoid get_running_loop() in hot path)
            self._loop = asyncio.get_running_loop()

            # Derive socket addresses from zmq_config
            request_socket_addr = (
                f"{self.zmq_config.zmq_request_queue_prefix}_{self.worker_id}_requests"
            )
            response_socket_addr = self.zmq_config.zmq_response_queue_addr
            readiness_socket_addr = self.zmq_config.zmq_readiness_queue_addr
            logger.debug("Request Socket Addr: %s", request_socket_addr)
            logger.debug("Response Socket Addr: %s", response_socket_addr)
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

            # Create HTTP timeout config (shared across requests)
            self._timeout = aiohttp.ClientTimeout(
                total=self.aiohttp_config.client_timeout_total,
                connect=self.aiohttp_config.client_timeout_connect,
                sock_read=self.aiohttp_config.client_timeout_sock_read,
            )

            # Cache response parsing params
            self._response_params = {
                "timer": None,
                "skip_payload": False,
                "read_until_eof": True,
                "auto_decompress": True,
                "read_timeout": self._timeout.sock_read,
                "read_bufsize": 2**16,  # 64KB, TODO(vir): make this a config
                "timeout_ceil_threshold": 5,
                "max_line_size": 8190,  # 8KB, TODO(vir): make this a config
                "max_field_size": 8190,  # 8KB, TODO(vir): make this a config
            }

            # Create aiohttp session
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
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
            logger.error(f"Error: {type(e).__name__}: {str(e)}")

        finally:
            # Cleanup
            await self._cleanup()

    @profile
    async def _main_loop(self) -> None:
        """Worker main loop: receive requests, fire HTTP, send responses."""
        while not self._shutdown:
            try:
                # TODO(vir): re-do work-consumer loop to leverage built-in zmq load-balancing
                query = await self._request_socket.receive()
                t_recv = time.perf_counter_ns()

                if query is None:
                    continue

                if self.http_config.record_worker_events:
                    EventRecorder.record_event(
                        SampleEvent.ZMQ_REQUEST_RECEIVED,
                        time.monotonic_ns(),
                        sample_uuid=query.id,
                        assert_active=True,
                    )

                # Prepare payload and make POST request
                prepared = self._prepare_request(query, t_recv)
                if not await self._fire_request(prepared):
                    continue

                # Process response asynchronously
                task = asyncio.create_task(self._process_response(prepared))
                self._active_tasks.add(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {type(e).__name__}: {str(e)}")

    async def _process_response(self, prepared: PreparedRequest) -> None:
        """Process HTTP response and send result via ZMQ."""
        try:
            try:
                await prepared.process()
            finally:
                if prepared.response:
                    prepared.response.close()
                if self.http_config.record_worker_events:
                    EventRecorder.record_event(
                        SampleEvent.HTTP_RESPONSE_COMPLETED,
                        time.monotonic_ns(),
                        sample_uuid=prepared.query_id,
                        assert_active=True,
                    )
        except Exception as e:
            await self._handle_error(prepared.query_id, e)
        finally:
            self._active_tasks.discard(asyncio.current_task())

    @profile
    async def _fire_request(self, prepared: PreparedRequest) -> bool:
        """Fire HTTP request. Returns True on success (response stored in prepared)."""
        if self._shutdown:
            await self._handle_error(prepared.query_id, "Worker is shutting down")
            return False

        try:
            conn = await self.tcp_connector.connect(
                prepared.client_request, traces=[], timeout=self._timeout
            )
            conn.protocol.set_response_params(**self._response_params)

            # Record time just-before request is issued
            prepared.timing_ctx["t_http"] = time.perf_counter_ns()
            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.HTTP_REQUEST_ISSUED,
                    time.monotonic_ns(),
                    sample_uuid=prepared.query_id,
                    assert_active=True,
                )

            # Issue post request
            resp = await prepared.client_request.send(conn)

            # Await for response headers
            await resp.start(conn)
            prepared.timing_ctx["t_headers"] = time.perf_counter_ns()

            # Check response status
            if resp.status != 200:
                error_text = await resp.text()
                await self._handle_error(
                    prepared.query_id, f"HTTP {resp.status}: {error_text}"
                )
                logger.error(
                    f"Request {prepared.query_id} failed: HTTP {resp.status}: {error_text}"
                )
                resp.close()
                return False

            # Store response in prepared for later processing
            prepared.response = resp
            return True

        except Exception as e:
            await self._handle_error(prepared.query_id, e)
            logger.error(f"Request {prepared.query_id} failed: {type(e).__name__}: {e}")
            return False

    @profile
    async def _handle_non_streaming_response(self, prepared: PreparedRequest) -> None:
        """Handle non-streaming HTTP response."""
        response_bytes = await prepared.response.read()
        prepared.timing_ctx["t_response"] = time.perf_counter_ns()

        result = self._adapter.decode_response(response_bytes, prepared.query_id)
        await self._response_socket.send(result)

        prepared.timing_ctx["t_zmq_sent"] = time.perf_counter_ns()
        prepared.log_timing()

        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=prepared.query_id,
                assert_active=True,
            )

    @profile
    async def _handle_streaming_response(self, prepared: PreparedRequest) -> None:
        """Handle streaming SSE HTTP response."""
        output_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        first_chunk_sent = False
        first_chunk_received = False
        query_id = prepared.query_id

        async for chunk_batch in self._iter_sse_lines(prepared.response):
            if not first_chunk_received:
                prepared.timing_ctx["t_first_chunk"] = time.perf_counter_ns()
                first_chunk_received = True

            output_delta: list[str] = []
            reasoning_delta: list[str] = []
            for delta in chunk_batch:
                if delta.content:
                    output_delta.append(delta.content)
                elif delta.reasoning:
                    reasoning_delta.append(delta.reasoning)

            for delta_batch, accumulator in (
                (reasoning_delta, reasoning_chunks),
                (output_delta, output_chunks),
            ):
                if not delta_batch:
                    continue
                accumulator.extend(delta_batch)

                chunks_to_send = (
                    delta_batch
                    if self.http_config.stream_all_chunks
                    else delta_batch[:1]
                    if not first_chunk_sent
                    else []
                )

                for content in chunks_to_send:
                    await self._response_socket.send(
                        StreamChunk(
                            id=query_id,
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
                            sample_uuid=query_id,
                            assert_active=True,
                        )

        # All chunks received
        prepared.timing_ctx["t_response"] = time.perf_counter_ns()

        # Build response: [first_token, rest_joined] format
        match (bool(reasoning_chunks), bool(output_chunks)):
            case (True, _):
                response_output = {
                    "output": "".join(output_chunks),
                    "reasoning": [reasoning_chunks[0], "".join(reasoning_chunks[1:])]
                    if len(reasoning_chunks) > 1
                    else reasoning_chunks,
                }
            case (False, True):
                response_output = {
                    "output": [output_chunks[0], "".join(output_chunks[1:])]
                    if len(output_chunks) > 1
                    else output_chunks
                }
            case _:
                response_output = {"output": []}

        await self._response_socket.send(
            QueryResult(
                id=query_id,
                response_output=response_output,
                metadata={"first_chunk": not first_chunk_sent, "final_chunk": True},
            )
        )

        prepared.timing_ctx["t_zmq_sent"] = time.perf_counter_ns()
        prepared.log_timing()

        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=query_id,
                assert_active=True,
            )

    async def _handle_error(self, query_id: str, error: Exception | str) -> None:
        """Send error response for a query via ZMQ."""

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

    @profile
    def _prepare_request(self, query: Query, t_recv: int) -> PreparedRequest:
        """Build PreparedRequest from Query."""

        # Encode Query into HTTP payload bytes
        payload_bytes = self._adapter.encode_query(query)

        # Build aiohttp ClientRequest (unique payload_bytes)
        client_request = ClientRequest(
            method=hdrs.METH_POST,
            url=self._url,
            headers=query.headers,
            data=payload_bytes,
            loop=self._loop,
            response_class=ClientResponse,
            timer=None,
            session=self._session,
            ssl=False,  # TODO(vir): checkme
        )

        timing_ctx = {"t_recv": t_recv, "t_prepare": time.perf_counter_ns()}

        handler = (
            self._handle_streaming_response
            if query.data.get("stream", False)
            else self._handle_non_streaming_response
        )

        prepared = PreparedRequest(
            query_id=query.id,
            client_request=client_request,
            timing_ctx=timing_ctx,
            process=None,
        )
        prepared.process = partial(handler, prepared)

        return prepared

    @profile
    async def _iter_sse_lines(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[list[str], None]:
        """Iterate SSE events, yielding batches of parsed chunks."""
        incomplete_chunk = b""

        async for chunk_bytes in response.content.iter_any():
            buffer = incomplete_chunk + chunk_bytes
            last_delimiter = buffer.rfind(b"\n\n")

            if last_delimiter == -1:
                incomplete_chunk = buffer
                continue

            incomplete_chunk = buffer[last_delimiter + 2 :]

            if parsed_contents := self._adapter.parse_sse_chunk(buffer, last_delimiter):
                yield parsed_contents

        if incomplete_chunk:
            if parsed_contents := self._adapter.parse_sse_chunk(
                incomplete_chunk, len(incomplete_chunk)
            ):
                yield parsed_contents

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
