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
from typing import Any

import aiohttp
import zmq
import zmq.asyncio
from aiohttp import hdrs
from aiohttp.client_reqrep import ClientRequest, ClientResponse
from yarl import URL

from inference_endpoint.config.schema import APIType
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
from inference_endpoint.endpoint_client.types import FileTimingPrinter, PreparedRequest
from inference_endpoint.endpoint_client.utils import get_ephemeral_port_limit
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.metrics.recorder import EventRecorder
from inference_endpoint.metrics.reporter import MetricsReporter
from inference_endpoint.openai.types import SSEDelta as OpenAISSEDelta
from inference_endpoint.profiling import profile
from inference_endpoint.sglang.types import SGLangSSEDelta
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

    # Create and run worker
    try:
        # FileTimingPrinter context manager handles setup/teardown of timing output
        with FileTimingPrinter.configure(http_config.event_logs_dir, worker_id):
            uvloop.run(
                Worker(
                    worker_id=worker_id,
                    http_config=http_config,
                    aiohttp_config=aiohttp_config,
                    zmq_config=zmq_config,
                ).run()
            )
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
            # logger.debug("empty SSE delta")
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
                "n_tokens": len(delta.token_ids),
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
        self._timeout: aiohttp.ClientTimeout | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._response_params: dict | None = None  # cached set_response_params kwargs

        # Track active request tasks
        self._active_tasks: set[asyncio.Task] = set()

        self._url: URL = URL(self.http_config.endpoint_url)
        self._adapter: type[HttpRequestAdapter] = self.http_config.adapter

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

    async def _warmup_connections(self) -> None:
        """
        Establish TCP connections to avoid cold-start connection delay.

        During benchmarking, the first batch of requests can experience
        significantly higher latency due to TCP connection establishment
        overhead (DNS lookup, TCP handshake, TLS negotiation if applicable).
        This "cold-start" penalty skews latency metrics and reduces
        throughput during the initial phase.

        This method pre-establishes TCP connections and adds them to
        aiohttp's connection pool before the worker signals readiness.
        Subsequent requests can then reuse these pooled connections,
        avoiding the connection establishment overhead entirely.

        The number of connections to warm up is determined by:
        - "auto": Divides system ephemeral port limit by number of workers
        - "auto-min": Uses 10% of "auto" value (minimum 10 connections)
        - int: Uses the specified number directly
        - Other values: Disables warmup
        """
        warmup_cfg = self.http_config.warmup_connections

        # Determine number of connections to warm up
        match warmup_cfg:
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

        # Create a dummy request just for connection establishment
        dummy_request = ClientRequest(
            method=hdrs.METH_GET,
            url=self._url,
            loop=self._loop,
            response_class=ClientResponse,
            session=self._session,
            ssl=False,
        )

        # Establish all connections first, then release together (ensures no reuse)
        connections: list = []

        async def warmup_one():
            conn = await self.tcp_connector.connect(
                dummy_request, traces=[], timeout=self._timeout
            )
            connections.append(conn)

        # Establish all connections concurrently
        await asyncio.gather(
            *[warmup_one() for _ in range(num_connections)],
            return_exceptions=True,
        )

        # Release all connections to pool
        for conn in connections:
            conn.release()

        idle_conns = sum(len(conns) for conns in self.tcp_connector._conns.values())
        logger.debug(
            f"Warmup Complete: {idle_conns}/{num_connections} connections pooled"
        )

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
                prepared.timing_ctx["t_task_created"] = time.monotonic_ns()
                task = asyncio.create_task(self._process_response(prepared))

                # Keep task alive to prevent GC
                # Cleaned up in _process_response finally block
                self._active_tasks.add(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {type(e).__name__}: {str(e)}")

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

        # Select response handler
        handler = (
            self._handle_streaming_response
            if query.data.get("stream", False)
            else self._handle_non_streaming_response
        )

        # Build PreparedRequest via factory
        prepared_request = PreparedRequest.create(
            query_id=query.id,
            client_request=client_request,
            timing_ctx={"t_recv": t_recv},
            handler=handler,
        )
        prepared_request.timing_ctx["t_prepare"] = time.monotonic_ns()
        return prepared_request

    @profile
    async def _fire_request(self, prepared: PreparedRequest) -> bool:
        """
        Fire HTTP POST request:
        1. establish / acquire TCP connection (from connection pool)
        2. send POST request bytes
        3. save resp, conn for process_response task to process concurrently

        Returns True on success.
        """
        if self._shutdown:
            await self._handle_error(prepared.query_id, "Worker is shutting down")
            return False

        try:
            # Establish TCP connection (try-reuse connections from pool)
            prepared.timing_ctx["t_conn_start"] = time.monotonic_ns()
            conn = await self.tcp_connector.connect(
                prepared.client_request, traces=[], timeout=self._timeout
            )
            # NOTE(vir): need to re-do this every request to recreate HttpResponseParser
            conn.protocol.set_response_params(**self._response_params)
            prepared.timing_ctx["t_conn_end"] = time.monotonic_ns()

            # Record time just-before request is issued
            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.HTTP_REQUEST_ISSUED,
                    time.monotonic_ns(),
                    sample_uuid=prepared.query_id,
                    assert_active=True,
                )

            # Issue post request
            resp = await prepared.client_request.send(conn)
            prepared.timing_ctx["t_http"] = time.monotonic_ns()

            # Store response to resume processing in asyncio task
            prepared.response = resp
            prepared.connection = conn
            prepared.log_timing_pre()
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
        try:
            # Record task awake time (asyncio scheduling delay measurement)
            prepared.timing_ctx["t_task_awake"] = time.monotonic_ns()

            # Await response headers (deferred from main loop to overlap with scheduling)
            await prepared.response.start(prepared.connection)
            prepared.timing_ctx["t_headers"] = time.monotonic_ns()

            # Check response status
            if prepared.response.status != 200:
                error_text = await prepared.response.text()
                await self._handle_error(
                    prepared.query_id,
                    f"HTTP {prepared.response.status}: {error_text}",
                )
                logger.error(
                    f"Request {prepared.query_id} failed: "
                    f"HTTP {prepared.response.status}: {error_text}"
                )
                return

            await prepared.process()
        except Exception as e:
            await self._handle_error(prepared.query_id, e)
        finally:
            # release connection back to TCP pool
            prepared.response.close()

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
        # Await response bytes
        response_bytes = await prepared.response.read()
        prepared.timing_ctx["t_response"] = time.monotonic_ns()

        # Decode response into QueryResult
        result = self._adapter.decode_response(response_bytes, prepared.query_id)
        await self._response_socket.send(result)

        prepared.timing_ctx["t_zmq_sent"] = time.monotonic_ns()
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=prepared.query_id,
                assert_active=True,
            )
        prepared.log_timing_post()

    @profile
    async def _handle_streaming_response(self, prepared: PreparedRequest) -> None:
        """Handle streaming SSE HTTP response."""
        query_id = prepared.query_id
        first_chunk_received = False

        # Select accumulator based on API type
        match self.http_config.api_type:
            case APIType.SGLANG:
                accumulator_type = SGLangSSEAccumulator
            case _:  # Default to OpenAI compatible accumulator
                accumulator_type = OpenAISSEAccumulator

        accumulator = accumulator_type(query_id, self.http_config.stream_all_chunks)
        async for chunk_batch in self._iter_sse_lines(prepared.response):
            if not first_chunk_received:
                prepared.timing_ctx["t_first_chunk"] = time.monotonic_ns()
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
        prepared.timing_ctx["t_response"] = time.monotonic_ns()

        # Send final complete response
        await self._response_socket.send(accumulator.get_final_output())

        prepared.timing_ctx["t_zmq_sent"] = time.monotonic_ns()
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=query_id,
                assert_active=True,
            )
        prepared.log_timing_post()

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
