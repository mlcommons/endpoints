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
from typing import Any

import aiohttp

from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.configs import AioHttpConfig, HTTPClientConfig
from inference_endpoint.endpoint_client.transport import (
    ReceiverTransport,
    SenderTransport,
    WorkerConnector,
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
except RuntimeError:
    # Already set, which is fine (likely in tests or when importing multiple times)
    pass


def worker_main(
    worker_id: int,
    connector: WorkerConnector,
    http_config: HTTPClientConfig,
    aiohttp_config: AioHttpConfig,
):
    """Entry point for worker process.

    Args:
        worker_id: Unique identifier for this worker.
        connector: Transport connector for IPC (ZMQ, shared memory, etc.).
        http_config: HTTP client configuration.
        aiohttp_config: aiohttp session configuration.
    """
    worker_log_format = f"%(asctime)s - %(name)s[W{worker_id}/%(process)d] - %(funcName)s - %(levelname)s - %(message)s"
    setup_logging(level=http_config.log_level, format_string=worker_log_format)

    # Install uvloop which also enables it
    import uvloop

    uvloop.install()

    # Create and run worker
    try:
        worker = Worker(
            worker_id=worker_id,
            connector=connector,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
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
        aiohttp_config: AioHttpConfig,
    ):
        """Initialize worker with configurations.

        Args:
            worker_id: Unique identifier for this worker.
            connector: Worker connector for IPC.
            http_config: HTTP client configuration.
            aiohttp_config: aiohttp session configuration.
        """
        self.worker_id = worker_id
        self._connector = connector
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self._shutdown = False

        self._session: aiohttp.ClientSession | None = None
        self.tcp_connector: aiohttp.TCPConnector | None = None
        self._requests: ReceiverTransport | None = None
        self._responses: SenderTransport | None = None

        # Track active request tasks
        self._active_tasks: set[asyncio.Task] = set()

        # Use adapter type from config
        self._adapter = self.http_config.adapter

    async def run(self) -> None:
        """Main worker loop - pull requests, execute, push responses."""
        try:
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

            # Connect transports via the injected connector
            # The connector handles transport setup and readiness signaling
            async with self._connector.connect(self.worker_id) as (requests, responses):
                logger.debug("Started and ready")

                # Run main processing loop
                if self.http_config.record_worker_events:
                    worker_db_name = f"worker_report_{self.worker_id}_{os.getpid()}"
                    report_path = (
                        self.http_config.event_logs_dir / f"{worker_db_name}.csv"
                    )
                    logger.debug("About to generate report")

                    with EventRecorder(session_id=worker_db_name) as event_recorder:
                        await self._run_main_loop(requests, responses)
                        event_recorder.wait_for_writes(force_commit=True)

                        with MetricsReporter(
                            event_recorder.connection_name
                        ) as reporter:
                            logger.debug(f"About to dump report to {report_path}")
                            reporter.dump_all_to_csv(report_path)
                            logger.debug(f"Report dumped to {report_path}")
                else:
                    await self._run_main_loop(requests, responses)

        except Exception as e:
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            raise
        finally:
            await self._cleanup()

    @profile
    async def _run_main_loop(
        self,
        requests: ReceiverTransport,
        responses: SenderTransport,
    ) -> None:
        """Main processing loop - continuously pull and process requests.

        Args:
            requests: Transport for receiving Query requests.
            responses: Transport for sending QueryResult/StreamChunk responses.
        """
        # Store reference while running
        self._requests = requests
        self._responses = responses

        while not self._shutdown:
            try:
                # TODO(vir): re-do work-consumer loop to leverage built-in zmq load-balancing
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

        url = self.http_config.endpoint_urls[
            self.worker_id % len(self.http_config.endpoint_urls)
        ]
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
        async for response in self._make_http_request(query):
            accumulator = self.http_config.accumulator(
                query.id, self.http_config.stream_all_chunks
            )

            # Process SSE stream - yields batches of chunks
            async for chunk_batch in self._iter_sse_lines(response):
                for delta in chunk_batch:
                    if stream_chunk := accumulator.add_chunk(delta):
                        self._responses.send(stream_chunk)

                        if self.http_config.record_worker_events:
                            EventRecorder.record_event(
                                SampleEvent.ZMQ_RESPONSE_SENT,
                                time.monotonic_ns(),
                                sample_uuid=query.id,
                                assert_active=True,
                            )

            self._responses.send(accumulator.get_final_output())
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
            self._responses.send(result)
            if self.http_config.record_worker_events:
                EventRecorder.record_event(
                    SampleEvent.ZMQ_RESPONSE_SENT,
                    time.monotonic_ns(),
                    sample_uuid=query.id,
                    assert_active=True,
                )

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

        # Close aiohttp session
        if self._session:
            await self._session.close()

        # Close TCP connector
        if self.tcp_connector:
            await self.tcp_connector.close()
