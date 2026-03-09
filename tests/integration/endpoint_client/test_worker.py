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

"""Integration tests for HttpClient worker process core functionality."""

import asyncio
import signal
import tempfile
from pathlib import Path

import pytest
from inference_endpoint.async_utils.transport import ZmqWorkerPoolTransport
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.worker import Worker


def test_set_start_method_already_set():
    """set_start_method handles already-set case.

    The module-level code catches RuntimeError when the
    multiprocessing start method is already set. In test environments, it
    is always already set (by pytest or another import). We verify the
    module imported successfully, which means the except block executed.
    """
    import multiprocessing

    from inference_endpoint.endpoint_client import worker

    assert worker is not None
    # The start method should be set (either by the module or by pytest)
    assert multiprocessing.get_start_method() is not None


class TestWorkerBasicFunctionality:
    """Test basic Worker functionality for request/response handling."""

    @pytest.fixture
    def worker_config(self, mock_http_echo_server):
        """Create worker configuration with echo server URL."""
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
        )
        return http_config

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "requests",
        [
            # Mixed streaming and non-streaming requests
            # Format: (prompt, stream, expected_output)
            # For streaming: expected_output is tuple (first_chunk, full_output_dict)
            #   where full_output_dict = {"output": (first_chunk, joined_rest)}
            # For non-streaming: expected_output is just the string
            [
                ("Non-streaming first", False, "Non-streaming first"),
                (
                    "Streaming second",
                    True,
                    ("Streaming", {"output": ("Streaming", " second")}),
                ),
                ("Non-streaming third", False, "Non-streaming third"),
            ],
            # Empty prompts for both streaming and non-streaming
            [
                ("", False, ""),
                ("", True, ("", {"output": ()})),
            ],
        ],
        ids=[
            "mixed_streaming_non_streaming",
            "empty_prompts",
        ],
    )
    async def test_worker_request_handling(self, worker_config, requests):
        """Test worker handling various request patterns including streaming, non-streaming, and multiple requests."""
        http_config = worker_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Send all queries
                for i, (prompt, stream, _) in enumerate(requests):
                    query = Query(
                        id=f"test-{i}",
                        data={
                            "prompt": prompt,
                            "model": "gpt-3.5-turbo",
                            "stream": stream,
                        },
                    )
                    pool.send(0, query)

                # Collect responses
                final_responses: dict[str, QueryResult] = {}
                streaming_chunks: dict[str, list[StreamChunk]] = {}

                # Receive all responses (streaming queries produce multiple messages)
                while len(final_responses) < len(requests):
                    response = await pool.recv()

                    if response is None:
                        break

                    if isinstance(response, StreamChunk):
                        # Intermediate streaming chunk
                        if response.id not in streaming_chunks:
                            streaming_chunks[response.id] = []
                        streaming_chunks[response.id].append(response)
                    elif isinstance(response, QueryResult):
                        if response.metadata.get("final_chunk", False):
                            # Final streaming response
                            final_responses[response.id] = response
                        else:
                            # Non-streaming response
                            final_responses[response.id] = response

                # Verify all responses received
                assert len(final_responses) == len(
                    requests
                ), f"Expected {len(requests)} responses, got {len(final_responses)}"

                # Verify each response
                for i, (_, stream, expected) in enumerate(requests):
                    query_id = f"test-{i}"

                    assert query_id in final_responses
                    response = final_responses[query_id]
                    assert response.error is None

                    if stream:
                        # Streaming response - expected is (first_chunk_content, full_output_tuple)
                        expected_first_chunk, expected_output = expected
                        assert response.metadata.get("final_chunk") is True
                        assert response.response_output == expected_output

                        # Verify first chunk metadata and content
                        if query_id in streaming_chunks and streaming_chunks[query_id]:
                            first_chunk = streaming_chunks[query_id][0]
                            assert first_chunk.metadata.get("first_chunk") is True
                            assert first_chunk.response_chunk == expected_first_chunk
                    else:
                        # Non-streaming response
                        assert response.response_output == expected

            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sig",
        [
            signal.SIGTERM,
            signal.SIGINT,
            signal.SIGHUP,
            signal.SIGQUIT,
        ],
        ids=["SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"],
    )
    async def test_worker_signal_handling(self, worker_config, sig):
        """Test worker responds to various signals correctly.

        Tests graceful shutdown handling for common signals:
        - SIGTERM: Standard termination signal
        - SIGINT: Interrupt signal (Ctrl+C)
        - SIGHUP: Hangup signal (terminal closed)
        - SIGQUIT: Quit signal (Ctrl+\\)
        """
        http_config = worker_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Verify worker is running
                assert not worker._shutdown

                # Send signal via shutdown method (simulates signal handler)
                worker.shutdown(sig, None)

                # Verify shutdown flag is set
                assert worker._shutdown

                # Await worker exit
                await asyncio.gather(worker_task, return_exceptions=True)

            finally:
                if not worker_task.done():
                    worker.shutdown()
                    await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config_modifier,test_id",
        [
            (
                {"warmup_connections": -1},
                "auto_warmup",
            ),
            (
                {"warmup_connections": 4},
                "explicit_warmup",
            ),
            (
                {"api_key": "test-key-123", "warmup_connections": 0},
                "api_key",
            ),
        ],
        ids=["auto_warmup", "explicit_warmup", "api_key"],
    )
    async def test_worker_config_variants(
        self, mock_http_echo_server, config_modifier, test_id
    ):
        """Worker handles various config options: warmup modes and API key."""
        base_config = {
            "endpoint_urls": [f"{mock_http_echo_server.url}/v1/chat/completions"],
            "num_workers": 1,
            "max_connections": 10,
        }
        base_config.update(config_modifier)
        http_config = HTTPClientConfig(**base_config)

        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )
            worker_task = asyncio.create_task(worker.run())
            try:
                await pool.wait_for_workers_ready(timeout=5)
                query = Query(
                    id=f"config-variant-{test_id}",
                    data={
                        "prompt": "hello",
                        "model": "test",
                        "stream": False,
                    },
                )
                pool.send(0, query)
                response = await pool.recv()
                assert isinstance(response, QueryResult)
                assert response.error is None
            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    async def test_worker_https_ssl_context(self):
        """Worker creates SSL context for HTTPS endpoints."""
        http_config = HTTPClientConfig(
            endpoint_urls=["https://localhost:443/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )
            assert worker._ssl_context is not None
            pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gc_mode",
        ["disabled", "relaxed", "system"],
        ids=["gc_disabled", "gc_relaxed", "gc_system"],
    )
    async def test_worker_main_gc_modes(self, mock_http_echo_server, gc_mode):
        """worker_main runs correctly as a subprocess via WorkerManager for each GC mode."""
        from inference_endpoint.endpoint_client.worker_manager import WorkerManager

        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
            worker_gc_mode=gc_mode,
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            manager = WorkerManager(http_config, loop, zmq_ctx)
            try:
                await manager.initialize()
                # Worker is running as subprocess -- worker_main was called
                assert len(manager.workers) == 1
                assert manager.workers[0].is_alive()

                # Send a query through the transport to verify the worker functions
                manager.pool_transport.send(
                    0,
                    Query(
                        id=f"gc-{gc_mode}",
                        data={
                            "prompt": "hello",
                            "model": "test",
                            "stream": False,
                        },
                    ),
                )
                response = await asyncio.wait_for(
                    manager.pool_transport.recv(), timeout=5
                )
                assert isinstance(response, QueryResult)
                assert response.error is None
            finally:
                await manager.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_event_recording(self, mock_http_echo_server, stream):
        """Event recording creates CSV report on shutdown for both streaming and non-streaming."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            http_config = HTTPClientConfig(
                endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
                num_workers=1,
                max_connections=10,
                warmup_connections=0,
                record_worker_events=True,
                event_logs_dir=Path(tmp_dir),
            )

            loop = asyncio.get_running_loop()
            with ManagedZMQContext.scoped() as zmq_ctx:
                pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

                worker = Worker(
                    worker_id=0,
                    connector=pool.worker_connector,
                    http_config=http_config,
                )

                worker_task = asyncio.create_task(worker.run())

                try:
                    await pool.wait_for_workers_ready(timeout=5)

                    query = Query(
                        id=f"event-recording-{'stream' if stream else 'non-stream'}",
                        data={
                            "prompt": "Event recording test",
                            "model": "gpt-3.5-turbo",
                            "stream": stream,
                        },
                    )
                    pool.send(0, query)

                    # Collect responses until final QueryResult
                    while True:
                        response = await asyncio.wait_for(pool.recv(), timeout=2)
                        if isinstance(response, QueryResult):
                            assert response.error is None
                            break

                finally:
                    worker.shutdown()
                    await asyncio.gather(worker_task, return_exceptions=True)
                    pool.cleanup()

            # Verify CSV file was created in tmp_dir
            csv_files = list(Path(tmp_dir).glob("worker_report_*.csv"))
            assert len(csv_files) > 0, (
                f"Expected worker_report_*.csv in {tmp_dir}, "
                f"found: {list(Path(tmp_dir).iterdir())}"
            )


class TestWorkerErrorHandling:
    """Test Worker error handling for various failure scenarios."""

    @pytest.fixture
    def worker_config(self, mock_http_echo_server):
        """Create worker configuration with echo server URL."""
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
        )
        return http_config

    @pytest.fixture
    def error_config(self):
        """Create configuration with invalid endpoint for error tests."""
        http_config = HTTPClientConfig(
            endpoint_urls=["http://localhost:59999/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
        )
        return http_config

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_connection_error_handling(self, error_config, stream):
        """Test worker error handling with invalid endpoint for both streaming and non-streaming."""
        http_config = error_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Send query
                query_id = f"test-connection-error-{'streaming' if stream else 'non-streaming'}"
                query = Query(
                    id=query_id,
                    data={
                        "prompt": "This should fail",
                        "model": "gpt-3.5-turbo",
                        "stream": stream,
                    },
                )
                pool.send(0, query)

                # Receive error response
                response = await pool.recv()

                # Verify error response
                assert isinstance(response, QueryResult)
                assert response.id == query_id
                assert response.error is not None
                # Check for connection error indicators
                error_lower = response.error.lower()
                assert (
                    ("connect" in error_lower and "failed" in error_lower)
                    or ("connection" in error_lower and "refused" in error_lower)
                    or ("cannot connect" in error_lower)
                ), f"Unexpected error message: {response.error}"

            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_exception_handling(self, worker_config, stream):
        """Test worker handles exceptions in body handler for both streaming and non-streaming."""
        http_config = worker_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Mock the appropriate body handler to raise an exception
            error_msg = f"Simulated {'streaming' if stream else 'non-streaming'} processing error"

            async def mock_handle_body(prepared):
                raise RuntimeError(error_msg)

            if stream:
                worker._handle_streaming_body = mock_handle_body
            else:
                worker._handle_non_streaming_body = mock_handle_body

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Send query
                query_id = (
                    f"test-exception-{'streaming' if stream else 'non-streaming'}"
                )
                query = Query(
                    id=query_id,
                    data={
                        "prompt": "Test exception handling",
                        "model": "gpt-3.5-turbo",
                        "stream": stream,
                    },
                )
                pool.send(0, query)

                # Receive error response
                response = await pool.recv()

                # Verify error response
                assert isinstance(response, QueryResult)
                assert response.id == query_id
                assert response.error is not None
                assert error_msg in response.error
                assert response.response_output is None

            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_http_error_status(self, stream):
        """Test worker propagates HTTP 500 as a QueryResult with error containing status code and body."""
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def error_handler(request):
            """Handler that returns HTTP 500 with a JSON error body."""
            return web.Response(
                status=500,
                body=b'{"error": "internal server error"}',
                content_type="application/json",
            )

        app = web.Application()
        app.router.add_post("/error", error_handler)

        # Start test server
        server = TestServer(app)
        await server.start_server()

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            # Create config with test server URL
            http_config = HTTPClientConfig(
                endpoint_urls=[f"http://localhost:{server.port}/error"],
                num_workers=1,
                max_connections=10,
                warmup_connections=False,
            )

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Send query
                query_id = f"test-http-500-{'streaming' if stream else 'non-streaming'}"
                query = Query(
                    id=query_id,
                    data={
                        "prompt": "Test HTTP 500 error",
                        "model": "gpt-3.5-turbo",
                        "stream": stream,
                    },
                )
                pool.send(0, query)

                # Receive response
                response = await pool.recv()

                # Verify error response
                assert isinstance(response, QueryResult)
                assert response.id == query_id
                assert response.error is not None
                assert "500" in response.error
                assert "internal server error" in response.error

            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()
        await server.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_malformed_json(self, stream):
        """Test worker handling malformed JSON in response."""
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def malformed_json_streaming_handler(request):
            """Handler that returns malformed JSON in streaming format."""
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/plain"
            await response.prepare(request)

            # Send malformed JSON chunks
            await response.write(b'data: {"invalid": json}\n\n')
            await response.write(b"data: [DONE]\n\n")

            return response

        async def malformed_json_non_streaming_handler(request):
            """Handler that returns malformed JSON in non-streaming format."""
            return web.Response(
                body=b'{"invalid": json}',
                content_type="application/json",
            )

        app = web.Application()
        if stream:
            app.router.add_post("/malformed", malformed_json_streaming_handler)
        else:
            app.router.add_post("/malformed", malformed_json_non_streaming_handler)

        # Start test server
        server = TestServer(app)
        await server.start_server()

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            # Create config with test server URL
            http_config = HTTPClientConfig(
                endpoint_urls=[f"http://localhost:{server.port}/malformed"],
                num_workers=1,
                max_connections=10,
                warmup_connections=0,
            )

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Send query
                query_id = (
                    f"test-malformed-json-{'streaming' if stream else 'non-streaming'}"
                )
                query = Query(
                    id=query_id,
                    data={
                        "prompt": "Test malformed JSON",
                        "model": "gpt-3.5-turbo",
                        "stream": stream,
                    },
                )
                pool.send(0, query)

                # Receive response
                response = await pool.recv()

                # Verify we get a response
                assert isinstance(response, QueryResult)
                assert response.id == query_id

                if stream:
                    # Streaming: malformed JSON is skipped, so we get an empty response
                    # (see _parse_sse_chunk which catches exceptions for non-content SSE messages)
                    assert response.error is None
                    assert response.response_output == {"output": ()}
                else:
                    # Non-streaming: malformed JSON causes a decode error
                    assert response.error is not None
                    assert (
                        "decode" in response.error.lower()
                        or "json" in response.error.lower()
                    )

            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()
        await server.close()


class TestWorkerEdgeCases:
    """Test Worker edge cases for SSE boundary handling and cleanup with pending tasks."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "handler_id",
        ["split_chunks", "trailing_content"],
        ids=["split_chunks", "trailing_content"],
    )
    async def test_worker_sse_boundary_handling(self, handler_id):
        """Test SSE boundary handling: split chunks across TCP reads and trailing content chunks."""
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def split_sse_handler(request):
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/event-stream"
            response.headers["Transfer-Encoding"] = "chunked"
            await response.prepare(request)

            # Send first half of SSE event without \n\n (forces buffering)
            await response.write(b'data: {"choices":[{"delta":{"content":"hel')
            await asyncio.sleep(0.01)  # Force separate TCP reads

            # Complete the SSE event
            await response.write(b'lo"}}]}\n\n')
            await asyncio.sleep(0.01)

            # Send trailing data without \n\n terminator
            await response.write(b"data: [DONE]\n")

            return response

        async def trailing_content_handler(request):
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/event-stream"
            response.headers["Transfer-Encoding"] = "chunked"
            await response.prepare(request)
            # Send a complete event
            await response.write(
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            )
            await asyncio.sleep(0.01)
            # Send final content chunk without trailing \n\n so it becomes
            # the incomplete_chunk after the stream ends.
            await response.write(b'data: {"choices":[{"delta":{"content":"!"}}]}\n')
            return response

        handlers = {
            "split_chunks": split_sse_handler,
            "trailing_content": trailing_content_handler,
        }

        app = web.Application()
        app.router.add_post("/sse-test", handlers[handler_id])

        # Start test server
        server = TestServer(app)
        await server.start_server()

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            # Create config with test server URL
            http_config = HTTPClientConfig(
                endpoint_urls=[f"http://localhost:{server.port}/sse-test"],
                num_workers=1,
                max_connections=10,
                warmup_connections=False,
            )

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Send a streaming query
                query = Query(
                    id=f"test-sse-{handler_id}",
                    data={
                        "prompt": "Test SSE boundary",
                        "model": "gpt-3.5-turbo",
                        "stream": True,
                    },
                )
                pool.send(0, query)

                # Collect all responses
                stream_chunks: list[StreamChunk] = []
                final_result: QueryResult | None = None

                while final_result is None:
                    response = await asyncio.wait_for(pool.recv(), timeout=2)

                    if response is None:
                        break

                    if isinstance(response, StreamChunk):
                        stream_chunks.append(response)
                    elif isinstance(response, QueryResult):
                        final_result = response

                # Verify the worker successfully processed the SSE data
                assert final_result is not None
                assert final_result.id == f"test-sse-{handler_id}"
                assert final_result.error is None

            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()
        await server.close()

    @pytest.mark.asyncio
    async def test_worker_cleanup_with_pending_tasks(self):
        """Test cleanup path -- shutdown while requests are in-flight."""
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def slow_handler(request):
            await asyncio.sleep(5)  # Very slow -- will be interrupted
            return web.Response(
                body=b'{"choices":[{"message":{"content":"done"}}]}',
                content_type="application/json",
            )

        app = web.Application()
        app.router.add_post("/slow", slow_handler)

        # Start test server
        server = TestServer(app)
        await server.start_server()

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            # Create config with test server URL
            http_config = HTTPClientConfig(
                endpoint_urls=[f"http://localhost:{server.port}/slow"],
                num_workers=1,
                max_connections=10,
                warmup_connections=False,
            )

            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker readiness
                await pool.wait_for_workers_ready(timeout=5)

                # Send multiple non-streaming queries
                for i in range(5):
                    query = Query(
                        id=f"test-pending-{i}",
                        data={
                            "prompt": f"Slow request {i}",
                            "model": "gpt-3.5-turbo",
                            "stream": False,
                        },
                    )
                    pool.send(0, query)

                # Give the worker a moment to dispatch the requests
                await asyncio.sleep(0.05)

                # Immediately shutdown the worker (don't wait for responses)
                worker.shutdown()

                # Await the worker task -- verify it completes without error
                await asyncio.wait_for(worker_task, timeout=5.0)

            finally:
                if not worker_task.done():
                    worker.shutdown()
                    await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()
        await server.close()

    @pytest.mark.asyncio
    async def test_worker_fire_request_during_shutdown(self):
        """Worker handles queries that arrive during shutdown gracefully."""
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def slow_handler(request):
            await asyncio.sleep(0.1)
            return web.Response(
                body=b'{"choices":[{"message":{"content":"ok"}}]}',
                content_type="application/json",
            )

        app = web.Application()
        app.router.add_post("/slow", slow_handler)
        server = TestServer(app)
        await server.start_server()

        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            http_config = HTTPClientConfig(
                endpoint_urls=[f"http://localhost:{server.port}/slow"],
                num_workers=1,
                max_connections=10,
                warmup_connections=0,
            )
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )
            worker_task = asyncio.create_task(worker.run())
            try:
                await pool.wait_for_workers_ready(timeout=5)
                # Send query then immediately shutdown
                pool.send(
                    0,
                    Query(
                        id="shutdown-race",
                        data={
                            "prompt": "test",
                            "model": "test",
                            "stream": False,
                        },
                    ),
                )
                await asyncio.sleep(0.01)
                worker.shutdown()
                # Send another query AFTER shutdown flag is set
                pool.send(
                    0,
                    Query(
                        id="post-shutdown",
                        data={
                            "prompt": "test2",
                            "model": "test",
                            "stream": False,
                        },
                    ),
                )
            finally:
                await asyncio.wait_for(
                    asyncio.gather(worker_task, return_exceptions=True),
                    timeout=5,
                )
                pool.cleanup()
        await server.close()

    @pytest.mark.asyncio
    async def test_worker_warmup_failure_continues(self):
        """Worker continues when warmup fails with min_required_connections=0."""
        http_config = HTTPClientConfig(
            endpoint_urls=["http://localhost:1/unreachable"],  # port 1 = unreachable
            num_workers=1,
            max_connections=10,
            warmup_connections=-1,  # auto: 50% of pool = 5
            min_required_connections=0,  # disable fatal check
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )
            worker_task = asyncio.create_task(worker.run())
            try:
                await pool.wait_for_workers_ready(timeout=5)
                # Worker should be running (warmup failed but continued)
                # Send a query -- it will fail since endpoint is unreachable
                pool.send(
                    0,
                    Query(
                        id="warmup-fail",
                        data={
                            "prompt": "test",
                            "model": "test",
                            "stream": False,
                        },
                    ),
                )
                response = await asyncio.wait_for(pool.recv(), timeout=5)
                assert isinstance(response, QueryResult)
                assert response.error is not None  # Connection refused
            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    async def test_worker_warmup_fatal_exit(self):
        """Worker exits when warmup fails and min_required_connections > 0.

        When warmup establishes 0 connections and min_required_connections is
        greater than 0, the worker calls sys.exit(1). When run as a subprocess
        via WorkerManager, this causes the worker process to die during init.
        """
        from inference_endpoint.endpoint_client.worker_manager import WorkerManager

        http_config = HTTPClientConfig(
            endpoint_urls=["http://localhost:1/unreachable"],
            num_workers=1,
            max_connections=10,
            warmup_connections=-1,  # auto: will attempt warmup
            min_required_connections=10,  # > 0, so warmup failure is fatal
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            manager = WorkerManager(http_config, loop, zmq_ctx)
            # Worker should die during init (sys.exit(1) in warmup)
            with pytest.raises((RuntimeError, TimeoutError)):
                await asyncio.wait_for(manager.initialize(), timeout=15)
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_worker_event_recording_with_error(self):
        """Event recording captures events even for error responses.

        When record_worker_events is enabled and a request results in an HTTP
        error, the _handle_error path should still record the ZMQ_RESPONSE_SENT
        event.
        """
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def error_handler(request):
            return web.Response(
                status=500,
                body=b'{"error": "internal server error"}',
                content_type="application/json",
            )

        app = web.Application()
        app.router.add_post("/error", error_handler)
        server = TestServer(app)
        await server.start_server()

        with tempfile.TemporaryDirectory() as tmp_dir:
            loop = asyncio.get_running_loop()
            with ManagedZMQContext.scoped() as zmq_ctx:
                pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
                http_config = HTTPClientConfig(
                    endpoint_urls=[f"http://localhost:{server.port}/error"],
                    num_workers=1,
                    max_connections=10,
                    warmup_connections=0,
                    record_worker_events=True,
                    event_logs_dir=Path(tmp_dir),
                )
                worker = Worker(
                    worker_id=0,
                    connector=pool.worker_connector,
                    http_config=http_config,
                )
                worker_task = asyncio.create_task(worker.run())
                try:
                    await pool.wait_for_workers_ready(timeout=2)
                    pool.send(
                        0,
                        Query(
                            id="event-err",
                            data={
                                "prompt": "test",
                                "model": "test",
                                "stream": False,
                            },
                        ),
                    )
                    response = await asyncio.wait_for(pool.recv(), timeout=5)
                    assert isinstance(response, QueryResult)
                    assert response.error is not None
                    assert "500" in response.error
                finally:
                    worker.shutdown()
                    await asyncio.gather(worker_task, return_exceptions=True)
                    pool.cleanup()

            # Verify CSV file was created in tmp_dir
            csv_files = list(Path(tmp_dir).glob("worker_report_*.csv"))
            assert len(csv_files) > 0, (
                f"Expected worker_report_*.csv in {tmp_dir}, "
                f"found: {list(Path(tmp_dir).iterdir())}"
            )
        await server.close()

    @pytest.mark.asyncio
    async def test_worker_warmup_below_min_required(self, mock_http_echo_server):
        """Worker warns when warmed connections are below min_required_connections."""
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=4,
            warmup_connections=-1,  # auto: 50% of 4 = 2
            min_required_connections=100,  # much higher than what warmup achieves
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )
            worker_task = asyncio.create_task(worker.run())
            try:
                await pool.wait_for_workers_ready(timeout=2)
                query = Query(
                    id="min-req-test",
                    data={
                        "prompt": "hello",
                        "model": "test",
                        "stream": False,
                    },
                )
                pool.send(0, query)
                response = await asyncio.wait_for(pool.recv(), timeout=2)
                assert isinstance(response, QueryResult)
                assert response.error is None
            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    async def test_worker_run_exception_handling(self, mock_http_echo_server):
        """Worker.run() catches and re-raises exceptions from _run_main_loop.

        When _run_main_loop raises an unexpected exception, the outer except
        block in run() logs the error and re-raises it. The finally block
        still runs _cleanup().
        """
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            # Patch _run_main_loop to raise, triggering the outer except in run()
            async def failing_main_loop():
                raise RuntimeError("injected failure in run")

            worker._run_main_loop = failing_main_loop

            with pytest.raises(RuntimeError, match="injected failure in run"):
                await worker.run()

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_worker_main_loop_exception_continues(self, mock_http_echo_server):
        """Worker continues processing after non-fatal error in main loop.

        When _prepare_request raises a non-CancelledError exception, the main
        loop catches it, logs, and continues to the next iteration. The worker
        should still process subsequent queries.
        """
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            worker_task = asyncio.create_task(worker.run())
            try:
                await pool.wait_for_workers_ready(timeout=5)

                # Monkey-patch _prepare_request to fail on first call only
                call_count = 0
                original_prepare = worker._prepare_request

                def failing_then_working_prepare(query):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise RuntimeError("transient error in prepare")
                    return original_prepare(query)

                worker._prepare_request = failing_then_working_prepare

                # First query triggers exception path (logged, continues)
                pool.send(
                    0,
                    Query(
                        id="fail-1",
                        data={
                            "prompt": "trigger error",
                            "model": "test",
                            "stream": False,
                        },
                    ),
                )
                # Give worker time to process and hit the exception
                await asyncio.sleep(0.1)

                # Second query should succeed (worker continued after error)
                pool.send(
                    0,
                    Query(
                        id="succeed-2",
                        data={
                            "prompt": "should work",
                            "model": "test",
                            "stream": False,
                        },
                    ),
                )

                response = await asyncio.wait_for(pool.recv(), timeout=5)
                assert isinstance(response, QueryResult)
                assert response.id == "succeed-2"
                assert response.error is None
            finally:
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    @pytest.mark.asyncio
    async def test_worker_fire_request_shutdown_early_return(
        self, mock_http_echo_server
    ):
        """_fire_request returns False when shutdown is set."""
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=0,
        )
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            worker = Worker(
                worker_id=0,
                connector=pool.worker_connector,
                http_config=http_config,
            )

            worker_task = asyncio.create_task(worker.run())
            try:
                await pool.wait_for_workers_ready(timeout=5)

                # Monkey-patch _prepare_request to set _shutdown after preparing
                # so _fire_request sees shutdown=True
                original_prepare = worker._prepare_request

                def prepare_then_shutdown(query):
                    result = original_prepare(query)
                    worker._shutdown = True
                    return result

                worker._prepare_request = prepare_then_shutdown

                # Send query -- _prepare_request will succeed, then set _shutdown,
                # so _fire_request sees shutdown=True and returns False.
                # The worker then exits the main loop cleanly.
                pool.send(
                    0,
                    Query(
                        id="during-shutdown",
                        data={
                            "prompt": "test",
                            "model": "test",
                            "stream": False,
                        },
                    ),
                )

                # Worker should exit cleanly since _shutdown is True
                await asyncio.wait_for(worker_task, timeout=5)

            finally:
                if not worker_task.done():
                    worker.shutdown()
                    await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()
