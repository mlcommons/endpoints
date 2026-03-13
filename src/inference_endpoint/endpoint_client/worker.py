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
import gc
import logging
import multiprocessing
import os
import signal
import ssl
import sys
import time
import traceback
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import urlparse

from inference_endpoint.async_utils.transport import (
    ReceiverTransport,
    SenderTransport,
    WorkerConnector,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http import (
    ConnectionPool,
    HttpRequestTemplate,
    InFlightRequest,
    PooledConnection,
)
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
except RuntimeError:  # pragma: no cover
    # Already set, which is fine (likely in tests or when importing multiple times)
    pass


def worker_main(
    worker_id: int,
    connector: WorkerConnector,
    http_config: HTTPClientConfig,
):
    """Entry point for worker process.

    Args:
        worker_id: Unique identifier for this worker.
        connector: Transport connector for IPC (ZMQ, shared memory, etc.).
        http_config: HTTP client configuration.
    """
    worker_log_format = f"%(asctime)s - %(name)s[W{worker_id}/%(process)d] - %(funcName)s - %(levelname)s - %(message)s"
    setup_logging(level=http_config.log_level, format_string=worker_log_format)

    # Configure GC based on worker_gc_mode
    match http_config.worker_gc_mode:
        case "disabled":
            gc.disable()
            logger.debug("GC fully disabled")
        case "relaxed":
            # NOTE(vir):
            # gc.set_threshold(gen0, gen1, gen2) default (700, 10, 10) means:
            # GC on Gen0 triggers when (allocations-deallocations) >= (700)
            # GC on Gen0+Gen1 triggers when (10) x Gen0 collections have occurred
            # GC on all generations triggers when (10) x Gen1 collections have occurred
            #
            # since worker has optimized hot-path (main-loop):
            #   - relax 100x for gen0,gen1 since request-lifecycle objects are "small"
            #   - relax 1000x for gen2 since worker is just about the event-loop
            gc_relaxed_thresholds = (70000, 10, 100)
            gc.set_threshold(*gc_relaxed_thresholds)
            logger.debug(f"GC thresholds relaxed to {gc_relaxed_thresholds}")
        case "system":
            logger.debug("GC using default Python thresholds")

    # Install uvloop which also enables it
    import uvloop

    uvloop.install()

    # Create and run worker
    try:
        worker = Worker(
            worker_id=worker_id,
            connector=connector,
            http_config=http_config,
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
        connector: WorkerConnector,
        http_config: HTTPClientConfig,
    ):
        """Initialize worker with configurations.

        Args:
            worker_id: Unique identifier for this worker.
            connector: Worker connector for IPC.
            http_config: HTTP client configuration.
        """
        self.worker_id = worker_id
        self._connector = connector
        self.http_config = http_config
        self._shutdown = False

        # Round-robin workers across endpoints
        endpoint_urls = self.http_config.endpoint_urls
        endpoint_url = endpoint_urls[worker_id % len(endpoint_urls)]

        # Parse endpoint URL into components
        parsed = urlparse(endpoint_url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._path = parsed.path or "/"
        self._scheme = parsed.scheme
        self._ssl_context = None

        if self._scheme == "https":
            self._ssl_context = ssl.create_default_context()

        # HTTP components (initialized in run())
        self._pool: ConnectionPool = None  # type: ignore[assignment]
        self._http_template: HttpRequestTemplate = None  # type: ignore[assignment]
        self._loop: asyncio.AbstractEventLoop = None  # type: ignore[assignment]

        # IPC transports (initialized in run())
        self._requests: ReceiverTransport = None  # type: ignore[assignment]
        self._responses: SenderTransport = None  # type: ignore[assignment]

        # Track active request tasks
        self._active_tasks: set[asyncio.Task] = set()

        # Use adapter type from config
        self._adapter: type[HttpRequestAdapter] = self.http_config.adapter

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses."""
        try:
            # Cache event loop reference
            self._loop = asyncio.get_running_loop()

            # Use eager task factory for immediate coroutine execution
            # Tasks start executing synchronously until first await
            # NOTE(vir): CRITICAL for minimizing TFB/TTFT
            self._loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]

            # Initialize HTTP template from URL components
            self._http_template = HttpRequestTemplate.from_url(
                self._host, self._port, self._path
            )
            if self.http_config.api_key:
                self._http_template.cache_headers(
                    {"Authorization": "Bearer " + self.http_config.api_key}
                )

            logger.debug(
                f"HTTP template initialized: path={self._path}, "
                f"host={self._host}:{self._port}"
            )

            # Create connection pool
            # Naively divide max connections among workers
            connections_per_worker = max(
                1, self.http_config.max_connections // self.http_config.num_workers
            )
            self._pool = ConnectionPool(
                host=self._host,
                port=self._port,
                loop=self._loop,
                max_connections=connections_per_worker,
                max_idle_time=self.http_config.max_idle_time,
                ssl_context=self._ssl_context,
            )

            # Signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self.shutdown)
            signal.signal(signal.SIGINT, self.shutdown)

            # Warmup connection pool if enabled
            warmup_cfg = self.http_config.warmup_connections
            if warmup_cfg != 0:
                if warmup_cfg == -1:
                    # Auto: 50% of pool (safe default)
                    warmup_count = connections_per_worker // 2
                else:
                    # Explicit total count split across workers
                    warmup_count = warmup_cfg // self.http_config.num_workers
                warmup_count = max(1, warmup_count)
                warmed = await self._pool.warmup(count=warmup_count)
                logger.debug(f"Warmed up {warmed} connections")

                # Error if 0 connections warmed up
                if warmed == 0:
                    msg = "Warmup: failed to establish connection to endpoint. Consider closing background TCP connections."
                    if self.http_config.min_required_connections == 0:
                        # log error but continue if disabled check
                        logger.error(msg)
                    else:
                        # NOTE(vir):
                        # 0 warmup connections is always fatal in practice,
                        # user needs to explicitly disable check to proceed
                        logger.error(
                            f"{msg} [ skip-check with --min_required_connections=0 ]"
                        )
                        sys.exit(1)

                # Warn if below min_required_connections threshold (skip if 0 = disabled)
                elif self.http_config.min_required_connections > 0:
                    min_required_per_worker = (
                        self.http_config.min_required_connections
                        // self.http_config.num_workers
                    )
                    if warmed < min_required_per_worker:
                        logger.warning(
                            f"Warmup: this worker has {warmed} connections, need {min_required_per_worker}. "
                            "Consider closing background TCP connections or adjusting --min_required_connections."
                        )

            # TODO(vir):
            # record_worker_events has high overhead - slows down the worker 100x
            # replace with fine-grained metrics, always captured/dumped per worker

            # Run main processing loop
            if self.http_config.record_worker_events:
                pid = os.getpid()
                worker_db_name = f"worker_report_{self.worker_id}_{pid}"
                assert (
                    self.http_config.event_logs_dir is not None
                ), "event_logs_dir must be set if record_worker_events is enabled"
                report_path = self.http_config.event_logs_dir / f"{worker_db_name}.csv"

                with EventRecorder(session_id=worker_db_name) as event_recorder:
                    await self._run_main_loop()
                    event_recorder.wait_for_writes(force_commit=True)

                    with MetricsReporter(event_recorder.connection_name) as reporter:
                        logger.debug(f"About to dump report to {report_path}")
                        reporter.dump_all_to_csv(report_path)
                        logger.debug(f"Report dumped to {report_path}")
            else:
                await self._run_main_loop()

        except Exception as e:
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            raise
        finally:
            await self._cleanup()

    @profile
    async def _run_main_loop(self) -> None:
        """Main processing loop - continuously pull and process requests."""

        # Reclaim any garbage before connecting/signaling readiness
        gc.collect(2)

        # Connect and signal readiness as we enter recv() loop. Scope ZMQ context
        # for this process so transports are cleaned up when the block exits.
        with ManagedZMQContext.scoped() as zmq_ctx:
            async with self._connector.connect(self.worker_id, zmq_ctx) as (
                requests,
                responses,
            ):
                self._requests = requests
                self._responses = responses
                logger.debug("Connected and ready")

                # TODO(vir):
                # batch-poll transport before await to reduce event loop yields under burst traffic.
                # Use requests.poll() in a while loop to drain all available queries synchronously,
                # only falling back to await requests.recv() when queue is empty.
                # Similar pattern to iter_body() sync drain optimization.
                while not self._shutdown:
                    try:
                        # Pull query from queue (blocks until message or transport closed)
                        query = await requests.recv()

                        # Transport closed (shutdown called)
                        if query is None:
                            break

                        if self.http_config.record_worker_events:
                            EventRecorder.record_event(
                                SampleEvent.ZMQ_REQUEST_RECEIVED,
                                time.monotonic_ns(),
                                sample_uuid=query.id,
                                assert_active=True,
                            )

                        # Prepare and fire request
                        req = self._prepare_request(query)
                        if not await self._fire_request(req):
                            continue

                        # Process response asynchronously
                        task = self._loop.create_task(self._process_response(req))

                        # Keep task alive to prevent GC
                        # Cleaned up in _process_response finally block
                        self._active_tasks.add(task)

                    except asyncio.CancelledError:
                        break

                    except Exception as e:
                        # Don't exit on errors in the main loop, just log and continue
                        logger.error(
                            f"Error in main loop: {type(e).__name__}: {str(e)}"
                        )

    @profile
    def _prepare_request(self, query: Query) -> InFlightRequest:
        """Build InFlightRequest with serialized HTTP bytes."""
        # Encode Query into HTTP payload bytes using adapter
        body_bytes = self._adapter.encode_query(query)
        is_streaming = query.data.get("stream", False)

        # Build complete HTTP request bytes
        http_bytes = self._http_template.build_request(
            body_bytes,
            is_streaming,
            extra_headers=query.headers,
        )

        # Create request context
        req = InFlightRequest(
            query_id=query.id,
            http_bytes=http_bytes,
            is_streaming=is_streaming,
        )

        return req

    @profile
    async def _fire_request(self, req: InFlightRequest) -> bool:
        """
        Fire HTTP POST request:
        1. Acquire TCP connection from pool
        2. Send POST request bytes

        Returns True on success, False on failure (error response sent).
        """
        if self._shutdown:
            await self._handle_error(req.query_id, "Worker is shutting down")
            return False

        try:
            # Acquire connection from pool
            conn = await self._pool.acquire()

            # Write request bytes directly to transport
            conn.protocol.write(req.http_bytes)

            # Store connection on req for response processing
            req.connection = conn

            return True

        except Exception as e:
            await self._handle_error(req.query_id, e)
            logger.error(f"Request {req.query_id} failed: {type(e).__name__}: {e}")
            return False

    @profile
    async def _process_response(self, req: InFlightRequest) -> None:
        """Process response for a fired request."""
        conn = req.connection

        try:
            # Await headers and handle error status
            status_code, _ = await conn.protocol.read_headers()
            if status_code != 200:
                error_body = await conn.protocol.read_body()
                self._pool.release(conn)
                await self._handle_error(
                    req.query_id,
                    f"HTTP {status_code}: {error_body.decode('utf-8', errors='replace')}",
                )
                return

            # Handle response body
            if req.is_streaming:
                await self._handle_streaming_body(req)
            else:
                await self._handle_non_streaming_body(req)

        except Exception as e:
            await self._handle_error(req.query_id, e)
            logger.warning(f"Request {req.query_id} failed: {type(e).__name__}: {e}")

        finally:
            # Release connection back to pool if not already
            self._pool.release(conn)

            # Record completion event
            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.HTTP_RESPONSE_COMPLETED,
                    time.monotonic_ns(),
                    sample_uuid=req.query_id,
                    assert_active=True,
                )

            # Clean up task reference
            current_task = asyncio.current_task()
            if current_task is not None:
                self._active_tasks.discard(current_task)

    @profile
    async def _handle_streaming_body(self, req: InFlightRequest) -> None:
        """Handle streaming (SSE) response body."""
        query_id = req.query_id
        conn = req.connection

        # Create accumulator for streaming response
        accumulator = self.http_config.accumulator(
            query_id, self.http_config.stream_all_chunks
        )

        # Process SSE stream - yields batches of chunks
        async for chunk_batch in self._iter_sse_lines(conn):
            for delta in chunk_batch:
                if stream_chunk := accumulator.add_chunk(delta):
                    self._responses.send(stream_chunk)

                    if self.http_config.record_worker_events:
                        EventRecorder.record_event(
                            SampleEvent.ZMQ_RESPONSE_SENT,
                            time.monotonic_ns(),
                            sample_uuid=query_id,
                            assert_active=True,
                        )

        # Release connection early - done with socket I/O (idempotent)
        self._pool.release(conn)

        # Send final complete back to main rank
        self._responses.send(accumulator.get_final_output())
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=query_id,
                assert_active=True,
            )

    @profile
    async def _handle_non_streaming_body(self, req: InFlightRequest) -> None:
        """Handle non-streaming response body."""
        query_id = req.query_id
        conn = req.connection

        # Read entire response body
        response_bytes = await conn.protocol.read_body()

        # Release connection early - done with socket I/O (idempotent)
        self._pool.release(conn)

        # Decode using adapter
        result = self._adapter.decode_response(response_bytes, query_id)

        # Send result back to main rank
        self._responses.send(result)
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=query_id,
                assert_active=True,
            )

    async def _handle_error(self, query_id: str, error: Exception | str) -> None:
        """Send error response for a query."""
        # Skip if we're shutting down or response socket is not available
        if self._shutdown or not self._responses:
            return

        error_message = repr(error) if isinstance(error, Exception) else error
        error_response = QueryResult(
            id=query_id,
            response_output=None,
            error=error_message,
        )
        self._responses.send(error_response)
        if self.http_config.record_worker_events:
            EventRecorder.record_event(
                SampleEvent.ZMQ_RESPONSE_SENT,
                time.monotonic_ns(),
                sample_uuid=query_id,
                assert_active=True,
            )

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

        async for chunk_list in conn.protocol.iter_body():
            # Join chunks (single chunk = no copy, multiple = join)
            chunk_data = chunk_list[0] if len(chunk_list) == 1 else b"".join(chunk_list)
            # Prepend incomplete data from previous iteration
            buffer = incomplete_chunk + chunk_data

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
        self._shutdown = True

        # Manually close request transport
        # unblock any pending recv() - it will return None
        if self._requests is not None:
            self._requests.close()

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
