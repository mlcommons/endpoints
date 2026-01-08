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
import gc
import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback
from collections.abc import AsyncGenerator
from functools import partial
from typing import Any
from urllib.parse import urlparse

import zmq
import zmq.asyncio

from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.timing_context import (
    BufferPrinter,
    DisabledPrinter,
    LogPrinter,
    RequestTimingContext,
    format_timing_log,
)
from inference_endpoint.endpoint_client.types import (
    ConnectionPool,
    HttpRequestTemplate,
    PooledConnection,
    PreparedRequest,
)
from inference_endpoint.endpoint_client.utils import get_ephemeral_port_limit
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.metrics.recorder import EventRecorder
from inference_endpoint.metrics.reporter import MetricsReporter
from inference_endpoint.profiling import profile
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

    match http_config.worker_gc_mode:
        case "disabled":
            gc.disable()
            logger.debug("GC fully disabled")
        case "relaxed":
            # Relaxed thresholds: 50x higher than default (700, 10, 10)
            gc_relaxed_thresholds = (35000, 500, 500)
            gc.set_threshold(*gc_relaxed_thresholds)
            logger.debug(f"GC thresholds relaxed to {gc_relaxed_thresholds}")
        case "system" | _:
            logger.debug("GC using default Python thresholds")

    # Install uvloop which also enables it
    import uvloop

    uvloop.install()

    # Create timing printer
    if http_config.event_logs_dir is not None:
        timing_printer = BufferPrinter(
            http_config.event_logs_dir / f"timing_worker_{worker_id}.jsonl"
        )
    else:
        timing_printer = LogPrinter(formatter=format_timing_log)

    # Create and run worker
    try:
        uvloop.run(
            Worker(
                worker_id=worker_id,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
                timing_printer=timing_printer,
            ).run()
        )
    except Exception as e:
        logger.error(f"Crashed: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        timing_printer.flush()


class Worker:
    """Worker process that performs actual HTTP requests."""

    def __init__(
        self,
        worker_id: int,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        timing_printer: BufferPrinter | LogPrinter | DisabledPrinter | None = None,
    ):
        """Initialize worker with configurations."""
        self.worker_id = worker_id
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self._timing_printer = timing_printer

        self._shutdown = False
        if self._timing_printer is None:
            self._timing_printer = DisabledPrinter()

        self._zmq_context: zmq.asyncio.Context | None = None
        self._request_socket: ZMQPullSocket | None = None
        self._response_socket: ZMQPushSocket | None = None
        self._readiness_socket: ZMQPushSocket | None = None

        self._pool: ConnectionPool | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Parse endpoint URL into components
        parsed = urlparse(self.http_config.endpoint_url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._path = parsed.path or "/"

        # Track active request tasks
        self._active_tasks: set[asyncio.Task] = set()

        self._adapter: type[HttpRequestAdapter] = self.http_config.adapter

        # Pre-computed HTTP request components
        self._http_template: HttpRequestTemplate | None = None

    async def run(self) -> None:
        """Initialize worker and launch main loop."""
        try:
            # Cache event loop reference
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

            # Initialize HTTP template from URL components
            self._http_template = HttpRequestTemplate.from_url(
                self._host, self._port, self._path
            )
            logger.debug(
                f"HTTP template initialized: path={self._path}, "
                f"host={self._host}:{self._port}"
            )

            # Create connection pool
            self._pool = ConnectionPool(
                host=self._host,
                port=self._port,
                loop=self._loop,
                socket_config=self.aiohttp_config.socket_defaults,
                max_connections=self.aiohttp_config.tcp_connector_limit,
                keepalive_timeout=self.aiohttp_config.tcp_connector_keepalive_timeout,
            )

            signal.signal(signal.SIGTERM, self.shutdown)
            signal.signal(signal.SIGINT, self.shutdown)

            # Warmup TCP connections before signaling readiness
            await self._warmup_connections()

            # Send readiness signal
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

    async def _warmup_connections(self) -> None:
        """
        Establish TCP connections to avoid cold-start connection delay.

        During benchmarking, the first batch of requests can experience
        significantly higher latency due to TCP connection establishment
        overhead (DNS lookup, TCP handshake, TLS negotiation if applicable).
        This "cold-start" penalty skews latency metrics and reduces
        throughput during the initial phase.

        This method pre-establishes TCP connections and adds them to
        the connection pool before the worker signals readiness.
        Subsequent requests can then reuse these pooled connections,
        avoiding the connection establishment overhead entirely.

        The number of connections to warm up is determined by:
        - "auto": Divides system ephemeral port limit by number of workers
        - "auto-min": Uses 10% of "auto" value (minimum 10 connections)
        - int: Uses the specified number directly
        - Other values: Disables warmup
        """
        match self.http_config.warmup_connections:
            case "auto":
                num_connections = max(
                    get_ephemeral_port_limit() // self.http_config.num_workers, 1
                )
            case "auto-min":
                num_connections = max(
                    int(
                        get_ephemeral_port_limit()
                        // self.http_config.num_workers
                        * 0.10
                    ),
                    10,
                )
            case int(n) if n > 0:
                num_connections = n
            case _:
                logger.debug("TCP connection warmup disabled")
                return

        logger.debug(f"Warming up {num_connections} TCP connections")
        pooled = await self._pool.warmup(num_connections)
        logger.debug(f"Warmup complete: {pooled}/{num_connections} connections pooled")

    @profile
    async def _main_loop(self) -> None:
        """
        Main processing loop:
        1. recv Query from zmq (pushed by SampleIssuer)
        2. prepare request (Query -> payload bytes)
        3. fire request (POST request)
        4. create async task to process further (ie. response headers, content)

        Why create_task is needed?
        -------------------------------------------------
        Without create_task, the main loop would block on each request's full
        lifecycle (send request -> wait for headers -> read response body).
        This serial execution would limit throughput to one request at a time.

        The pattern is:
        - Main loop: ZMQ recv -> prepare -> fire (TCP connect + send POST)
        - Background task: await headers -> read body -> send result via ZMQ

        This separation allows the main loop to stay "hot" on ZMQ consumption,
        while slow operations (server processing time, response streaming) are
        handled concurrently in background tasks tracked by _active_tasks.
        """
        while not self._shutdown:
            try:
                # TODO(vir): re-do work-consumer loop to leverage built-in zmq load-balancing
                query = await self._request_socket.receive()
                t_recv = time.monotonic_ns()

                if query is None:
                    continue

                if self.http_config.record_worker_events:
                    EventRecorder.record_event(
                        SampleEvent.ZMQ_REQUEST_RECEIVED,
                        t_recv,
                        sample_uuid=query.id,
                        assert_active=True,
                    )

                # Prepare payload and make POST request
                prepared = self._prepare_request(query, t_recv)
                if not await self._fire_request(prepared):
                    continue

                # Process response asynchronously
                prepared.timing["t_task_created"] = time.monotonic_ns()
                task = asyncio.Task(
                    self._process_response(prepared),
                    loop=self._loop,
                    eager_start=True,
                )

                # Keep task alive to prevent GC
                # Cleaned up in _process_response finally block
                self._active_tasks.add(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {type(e).__name__}: {str(e)}")

    @profile
    def _prepare_request(self, query: Query, t_recv: int) -> PreparedRequest:
        """Build PreparedRequest with pre-built HTTP bytes."""
        body_bytes = self._adapter.encode_query(query)
        t_encode = time.monotonic_ns()

        # Encode Query into HTTP payload bytes
        http_bytes = self._http_template.build_request(body_bytes, query.headers)
        is_streaming = query.data.get("stream", False)

        # Create timing context with initial timestamps
        timing = RequestTimingContext(id=query.id)
        timing["t_recv"] = t_recv
        timing["t_encode"] = t_encode

        # Setup state for processing
        prepared = PreparedRequest(
            query_id=query.id,
            http_bytes=http_bytes,
            timing=timing,
            is_streaming=is_streaming,
        )

        prepared.process = partial(
            self._handle_streaming_response
            if is_streaming
            else self._handle_non_streaming_response,
            prepared,
        )

        timing["t_prepare"] = time.monotonic_ns()
        return prepared

    @profile
    async def _fire_request(self, prepared: PreparedRequest) -> bool:
        """
        Fire HTTP POST request:
        1. establish / acquire TCP connection (from connection pool)
        2. send POST request bytes
        3. save conn for process_response task to process concurrently

        Returns True on success.
        """
        if self._shutdown:
            await self._handle_error(prepared.query_id, "Worker is shutting down")
            return False

        try:
            # Acquire connection from pool
            prepared.timing["t_conn_start"] = time.monotonic_ns()
            conn = await self._pool.acquire()
            prepared.timing["t_conn_end"] = time.monotonic_ns()

            # Record time just-before request is issued
            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.HTTP_REQUEST_ISSUED,
                    time.monotonic_ns(),
                    sample_uuid=prepared.query_id,
                    assert_active=True,
                )

            # Write request bytes directly to transport
            conn.protocol.write(prepared.http_bytes)
            prepared.timing["t_http"] = time.monotonic_ns()

            # Store connection for _process_response to use
            prepared.connection = conn

            # Emit pre-send timing metrics
            metrics = prepared.timing.compute_pre_overheads()
            self._timing_printer.write(
                {"query_id": prepared.timing.id, "phase": "pre", "metrics": metrics}
            )
            return True

        except Exception as e:
            await self._handle_error(prepared.query_id, e)
            logger.error(f"Request {prepared.query_id} failed: {type(e).__name__}: {e}")
            return False

    @profile
    async def _process_response(self, prepared: PreparedRequest) -> None:
        """
        Process HTTP response.
        1. process response header/status/errors
        2. process response data/errors
        3. release TCP socket once response completed/errors

        Header is awaited here instead of in _fire_request to overlap
        asyncio create_task overhead with TCP/HTTP POST/ACK communication.
        """
        conn = prepared.connection
        try:
            prepared.timing["t_task_awake"] = time.monotonic_ns()

            # Read headers using httptools protocol
            status, _ = await conn.protocol.read_headers()

            # Handle error response code
            if status != 200:
                error_body = await conn.protocol.read_body()
                error_text = error_body.decode("utf-8", errors="replace")
                await self._handle_error(
                    prepared.query_id, f"HTTP {status}: {error_text}"
                )
                logger.error(
                    f"Request {prepared.query_id} failed: HTTP {status}: {error_text}"
                )
                return

            # Mark headers received
            prepared.timing["t_headers"] = time.monotonic_ns()

            # Process using the pre-bound handler (streaming or non-streaming)
            await prepared.process()

        except Exception as e:
            await self._handle_error(prepared.query_id, e)
        finally:
            # Release connection back to pool
            self._pool.release(conn)

            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.HTTP_RESPONSE_COMPLETED,
                    time.monotonic_ns(),
                    sample_uuid=prepared.query_id,
                    assert_active=True,
                )

            self._active_tasks.discard(asyncio.current_task())

    @profile
    async def _handle_non_streaming_response(self, prepared: PreparedRequest) -> None:
        """Handle non-streaming HTTP response."""
        conn = prepared.connection

        # Await response bytes
        response_bytes = await conn.protocol.read_body()
        t_response = time.monotonic_ns()
        # For non-streaming, set t_first_chunk = t_response (whole response arrives at once)
        prepared.timing["t_response"] = t_response
        prepared.timing["t_first_chunk"] = t_response

        # Decode response into QueryResult
        result = self._adapter.decode_response(response_bytes, prepared.query_id)
        await self._response_socket.send(result)

        prepared.timing["t_zmq_sent"] = time.monotonic_ns()
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=prepared.query_id,
                assert_active=True,
            )

        # Emit post-receive timing metrics
        metrics = prepared.timing.compute_post_overheads()
        self._timing_printer.write(
            {"query_id": prepared.query_id, "phase": "post", "metrics": metrics}
        )

    @profile
    async def _handle_streaming_response(self, prepared: PreparedRequest) -> None:
        """Handle streaming SSE HTTP response."""
        conn = prepared.connection
        query_id = prepared.query_id
        first_chunk_received = False

        # Instantiate accumulator instance
        accumulator = self.http_config.accumulator(
            query_id, self.http_config.stream_all_chunks
        )
        async for chunk_batch in self._iter_sse_lines(conn):
            if not first_chunk_received:
                prepared.timing["t_first_chunk"] = time.monotonic_ns()
                first_chunk_received = True

            for delta in chunk_batch:
                if stream_chunk := accumulator.add_chunk(delta):
                    await self._response_socket.send(stream_chunk)
                    accumulator.first_chunk_sent = True

                    if self.http_config.record_worker_events:
                        EventRecorder.record_event(
                            SampleEvent.ZMQ_RESPONSE_SENT,
                            time.monotonic_ns(),
                            sample_uuid=query_id,
                            assert_active=True,
                        )

        # All chunks received
        prepared.timing["t_response"] = time.monotonic_ns()

        # Send final complete response
        await self._response_socket.send(accumulator.get_final_output())

        prepared.timing["t_zmq_sent"] = time.monotonic_ns()
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=query_id,
                assert_active=True,
            )

        # Emit post-receive timing metrics
        metrics = prepared.timing.compute_post_overheads()
        self._timing_printer.write(
            {"query_id": query_id, "phase": "post", "metrics": metrics}
        )

    async def _handle_error(self, query_id: str, error: Exception | str) -> None:
        """Report error for Query."""

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
    async def _iter_sse_lines(
        self, conn: PooledConnection
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

        async for chunk_bytes in conn.protocol.iter_body():
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

        # Close connection pool
        if self._pool:
            await self._pool.close()

        # Close ZMQ sockets
        for socket in (self._request_socket, self._response_socket):
            if socket:
                socket.close()

        # Terminate ZMQ context
        if self._zmq_context:
            self._zmq_context.destroy(linger=0)
