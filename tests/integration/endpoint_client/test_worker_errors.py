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

"""Integration tests for Worker error handling and edge cases."""

import asyncio

import pytest
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import Worker
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

from ...test_helpers import get_test_socket_path, setup_worker_test


class TestWorkerErrorHandling:
    """Test Worker error handling for various failure scenarios."""

    @pytest.fixture
    def basic_config(self, tmp_path):
        """Create basic configuration for error handling tests."""
        # Use invalid port to trigger connection errors
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/v1/chat/completions",
            num_workers=1,
        )
        aiohttp_config = AioHttpConfig()
        # Use tmp_path for unique socket paths per test
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_error", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_error", "_resp"
            ),
        )
        return http_config, aiohttp_config, zmq_config

    @pytest.mark.asyncio
    async def test_worker_error_handling(self, basic_config):
        """Test worker error handling with invalid endpoint."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Modify config to use invalid endpoint (localhost with invalid port for fast failure)
        http_config.endpoint_url = "http://localhost:99999/v1/chat/completions"
        aiohttp_config.client_timeout_total = 2.0  # Short timeout
        aiohttp_config.client_timeout_connect = 1.0  # Connect timeout

        async with setup_worker_test(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        ) as ctx:
            # Send query
            query = Query(
                id="test-error",
                data={
                    "prompt": "This should fail",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )
            await ctx.request_push.send(query)

            # Receive error response
            response = await asyncio.wait_for(ctx.response_pull.receive(), timeout=2.0)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-error"
            assert response.error is not None
            assert (
                (
                    "connection" in response.error.lower()
                    and "refused" in response.error.lower()
                )
                or "cannot connect" in response.error.lower()
                or "99999" in response.error
            )

    @pytest.mark.asyncio
    async def test_worker_streaming_http_error_handling(self, basic_config):
        """Test worker handling HTTP errors in streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use invalid endpoint to trigger connection error
        http_config.endpoint_url = "http://localhost:99999/invalid"
        aiohttp_config.client_timeout_total = 1.0  # Short timeout

        async with setup_worker_test(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        ) as ctx:
            # Send streaming query
            query = Query(
                id="test-streaming-error",
                data={
                    "prompt": "This should fail",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            await ctx.request_push.send(query)

            # Receive error response
            response = await ctx.response_pull.receive()

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-streaming-error"
            assert response.error is not None
            # Should get url back as error
            assert "http://localhost:99999/invalid" in response.error

    @pytest.mark.asyncio
    async def test_worker_non_streaming_exception_handling(self, basic_config):
        """Test worker handles exceptions in _process_request for non-streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use ZMQ utils for cleaner socket management
        request_addr = f"{zmq_config.zmq_request_queue_prefix}_0_requests"

        async with (
            ZMQPushSocket(request_addr, zmq_config) as request_socket,
            ZMQPullSocket(
                zmq_config.zmq_response_queue_addr,
                zmq_config,
                bind=True,
                decoder_type=QueryResult | StreamChunk,
            ) as response_pull,
            ZMQPullSocket(
                zmq_config.zmq_readiness_queue_addr,
                zmq_config,
                bind=True,
                decoder_type=int,
            ) as readiness_pull,
        ):
            await request_socket.initialize()
            await response_pull.initialize()
            await readiness_pull.initialize()

            worker = Worker(
                worker_id=0,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
                request_socket_addr=request_addr,
                response_socket_addr=zmq_config.zmq_response_queue_addr,
                readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
            )

            # Mock _handle_non_streaming_request to raise an exception immediately
            exception_raised = asyncio.Event()

            async def mock_handle_request(query):
                exception_raised.set()
                raise RuntimeError("Simulated processing error")

            worker._handle_non_streaming_request = mock_handle_request

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker to be ready
                await readiness_pull.receive()
                await asyncio.sleep(0.5)

                # Send non-streaming query
                query = Query(
                    id="test-exception-non-streaming",
                    data={
                        "prompt": "Test exception handling",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                    },
                )
                await request_socket.send(query)

                # Wait for exception to be raised
                try:
                    await asyncio.wait_for(exception_raised.wait(), timeout=3.0)
                except TimeoutError:
                    pytest.fail("Exception was not raised within timeout")

                # Receive error response (automatic decoding)
                response = await asyncio.wait_for(response_pull.receive(), timeout=3.0)

                # Verify error response
                assert isinstance(response, QueryResult)
                assert response.id == "test-exception-non-streaming"
                assert response.error is not None
                assert "Simulated processing error" in response.error
                assert response.response_output is None

            finally:
                # Proper cleanup
                if not worker._shutdown:
                    worker._shutdown = True
                    try:
                        await asyncio.wait_for(worker_task, timeout=2.0)
                    except (TimeoutError, asyncio.CancelledError):
                        pass

    @pytest.mark.asyncio
    async def test_worker_streaming_exception_handling(self, basic_config):
        """Test worker handles exceptions in _process_request for streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        request_addr = f"{zmq_config.zmq_request_queue_prefix}_0_requests"

        # Use ZMQ utils for cleaner socket management
        async with (
            ZMQPushSocket(request_addr, zmq_config) as request_push,
            ZMQPullSocket(
                zmq_config.zmq_response_queue_addr,
                zmq_config,
                bind=True,
                decoder_type=QueryResult | StreamChunk,
            ) as response_pull,
            ZMQPullSocket(
                zmq_config.zmq_readiness_queue_addr,
                zmq_config,
                bind=True,
                decoder_type=int,
            ) as readiness_pull,
        ):
            await request_push.initialize()
            await response_pull.initialize()
            await readiness_pull.initialize()

            worker = Worker(
                worker_id=0,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
                request_socket_addr=request_addr,
                response_socket_addr=zmq_config.zmq_response_queue_addr,
                readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
            )

            # Mock _handle_streaming_request to raise an exception
            async def mock_handle_request(query):
                raise RuntimeError("Simulated streaming processing error")

            worker._handle_streaming_request = mock_handle_request

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            try:
                # Wait for worker to be ready
                await readiness_pull.receive()
                await asyncio.sleep(0.5)

                # Send streaming query
                query = Query(
                    id="test-exception-streaming",
                    data={
                        "prompt": "Test streaming exception handling",
                        "model": "gpt-3.5-turbo",
                        "stream": True,
                    },
                )
                await request_push.send(query)

                # Receive error response (automatic decoding)
                response = await asyncio.wait_for(response_pull.receive(), timeout=2.0)

                # Verify error response
                assert isinstance(response, QueryResult)
                assert response.id == "test-exception-streaming"
                assert response.error is not None
                assert "Simulated streaming processing error" in response.error
                assert response.response_output is None

            finally:
                # Proper cleanup
                if not worker._shutdown:
                    worker._shutdown = True
                    try:
                        await asyncio.wait_for(worker_task, timeout=2.0)
                    except (TimeoutError, asyncio.CancelledError):
                        pass

    @pytest.mark.asyncio
    async def test_worker_non_streaming_connection_error(self, basic_config):
        """Test worker handles connection errors in non-streaming responses."""
        http_config, aiohttp_config, zmq_config = basic_config

        async with setup_worker_test(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        ) as ctx:
            # Send query
            query = Query(
                id="test-connection-error",
                data={
                    "prompt": "Test connection error",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )
            await ctx.request_push.send(query)

            # Should receive error response
            response = await ctx.response_pull.receive()

            assert isinstance(response, QueryResult)
            assert response.id == "test-connection-error"
            assert response.error is not None
            assert "99999" in response.error or "Cannot connect" in response.error

    @pytest.mark.asyncio
    async def test_worker_streaming_http_404_error(
        self, mock_http_echo_server, basic_config
    ):
        """Test worker handling HTTP 404 error in streaming request."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use echo server but with invalid endpoint to get 404
        http_config.endpoint_url = f"{mock_http_echo_server.url}/nonexistent"

        async with setup_worker_test(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        ) as ctx:
            # Send streaming query
            query = Query(
                id="test-streaming-404",
                data={
                    "prompt": "This should get 404",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            await ctx.request_push.send(query)

            # Receive error response
            response = await ctx.response_pull.receive()

            # Verify HTTP error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-streaming-404"
            assert response.error is not None
            assert "HTTP 404" in response.error

    @pytest.mark.asyncio
    async def test_non_streaming_http_error_early_return(self, basic_config):
        """Test non-streaming request with HTTP error status."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create a custom HTTP server that returns HTTP 500
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def http_500_handler(request):
            return web.json_response({"error": "Internal Server Error"}, status=500)

        # Create a test server
        app = web.Application()
        app.router.add_post("/error-500", http_500_handler)
        server = TestServer(app)

        await server.start_server()

        try:
            # Update config to use the error endpoint
            http_config.endpoint_url = f"http://localhost:{server.port}/error-500"

            async with setup_worker_test(
                worker_id=0,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
            ) as ctx:
                # Send query that will get HTTP error
                query = Query(
                    id="test-http-500",
                    data={
                        "prompt": "This will fail",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                    },
                )
                await ctx.request_push.send(query)

                # Verify error response was sent
                response = await asyncio.wait_for(
                    ctx.response_pull.receive(), timeout=1.0
                )

                assert isinstance(response, QueryResult)
                assert response.id == "test-http-500"
                assert "HTTP 500" in response.error
                assert "Internal Server Error" in response.error

        finally:
            await server.close()

    @pytest.mark.asyncio
    async def test_worker_streaming_malformed_json(self, basic_config):
        """Test worker handling malformed JSON in streaming response."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create a mock server that returns malformed JSON
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def malformed_json_handler(request):
            """Handler that returns malformed JSON in streaming format."""
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/plain"
            await response.prepare(request)

            # Send malformed JSON chunks
            await response.write(b'data: {"invalid": json}\n\n')
            await response.write(b"data: [DONE]\n\n")

            return response

        app = web.Application()
        app.router.add_post("/streaming", malformed_json_handler)
        server = TestServer(app)

        try:
            await server.start_server()

            # Update config with test server URL
            http_config.endpoint_url = f"http://localhost:{server.port}/streaming"

            async with setup_worker_test(
                worker_id=0,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
            ) as ctx:
                # Send streaming query
                query = Query(
                    id="test-malformed-json",
                    data={
                        "prompt": "Test malformed JSON",
                        "model": "gpt-3.5-turbo",
                        "stream": True,
                    },
                )
                await ctx.request_push.send(query)

                # Worker should handle malformed JSON gracefully by skipping invalid chunks
                # (see _parse_sse_chunk which catches exceptions for non-content SSE messages)
                response = await ctx.response_pull.receive()

                # Verify we get a response (worker handles malformed JSON gracefully)
                assert isinstance(response, QueryResult)
                assert response.id == "test-malformed-json"

                # Malformed JSON is skipped, so we get an empty response, not an error
                assert response.error is None
                assert response.response_output == ""

        finally:
            await server.close()

    @pytest.mark.asyncio
    async def test_worker_zmq_socket_binding_error(self, basic_config):
        """Test worker handling ZMQ socket binding errors."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create worker with invalid socket address
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr="invalid://socket/address",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Worker run should handle the error gracefully
        with pytest.raises(
            SystemExit
        ):  # Worker exits with code 1 on initialization failure
            await worker.run()

    @pytest.mark.asyncio
    async def test_worker_zmq_socket_error(self, basic_config):
        """Test worker handling ZMQ socket errors."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Use async context manager for socket
        async with ZMQPushSocket(
            zmq_config.zmq_response_queue_addr, zmq_config
        ) as response_socket:
            # Initialize worker resources manually for testing
            worker._response_socket = response_socket

            # Create query
            query = Query(
                id="test-zmq-error",
                data={
                    "prompt": "Test ZMQ error",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            # Close the response socket to simulate error
            response_socket.close()

            # Try to send response - should handle the error gracefully
            response = QueryResult(
                id=query.id,
                response_output="test",
                error="ZMQ socket closed",
            )

            # This should not raise an exception (socket is closed so send will fail)
            try:
                await response_socket.send(response)
            except Exception:
                # Expected - socket is closed
                pass

            # Worker should handle this gracefully without crashing

    @pytest.mark.asyncio
    async def test_worker_concurrent_error_handling(self, tmp_path):
        """Test multiple workers handling errors concurrently."""
        # Use the same pattern as single-worker tests - just run 3 in parallel
        num_workers = 3

        # Create separate configs for each worker to avoid socket conflicts
        workers_contexts = []

        for i in range(num_workers):
            http_config = HTTPClientConfig(
                endpoint_url="http://localhost:99999/api",  # Invalid endpoint
                num_workers=1,
            )
            aiohttp_config = AioHttpConfig(
                client_timeout_total=2.0,
                client_timeout_connect=1.0,
            )
            # Each worker gets unique socket paths
            zmq_config = ZMQConfig(
                zmq_request_queue_prefix=get_test_socket_path(
                    tmp_path, f"worker{i}", "_req"
                ),
                zmq_response_queue_addr=get_test_socket_path(
                    tmp_path, f"worker{i}", "_resp"
                ),
                zmq_readiness_queue_addr=get_test_socket_path(
                    tmp_path, f"worker{i}", "_ready"
                ),
            )

            workers_contexts.append((http_config, aiohttp_config, zmq_config))

        # Use setup_worker_test for each worker - this pattern already works
        async def run_worker_test(worker_id, http_config, aiohttp_config, zmq_config):
            async with setup_worker_test(
                worker_id=worker_id,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
            ) as ctx:
                # Send query
                query = Query(
                    id=f"test-concurrent-error-{worker_id}",
                    data={
                        "prompt": f"Worker {worker_id} error test",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                    },
                )
                await ctx.request_push.send(query)

                # Receive response
                response = await ctx.response_pull.receive()
                return response

        # Run all workers concurrently
        tasks = [run_worker_test(i, *workers_contexts[i]) for i in range(num_workers)]
        responses = await asyncio.gather(*tasks)

        # Verify all workers handled errors
        assert len(responses) == num_workers
        for i, response in enumerate(responses):
            assert isinstance(response, QueryResult)
            assert response.id == f"test-concurrent-error-{i}"
            assert response.error is not None
            assert any(
                [
                    "connection" in response.error.lower()
                    and "refused" in response.error.lower(),
                    "cannot connect" in response.error.lower(),
                    "99999" in response.error,
                ]
            )

    @pytest.mark.asyncio
    async def test_non_streaming_invalid_json_early_return(self, basic_config):
        """Test non-streaming request with invalid JSON response."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create a custom HTTP server that returns invalid JSON
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        async def invalid_json_handler(request):
            return web.Response(
                text="not valid json at all",
                content_type="application/json",
                status=200,
            )

        # Create a test server
        app = web.Application()
        app.router.add_post("/invalid-json", invalid_json_handler)
        server = TestServer(app)

        await server.start_server()

        try:
            # Update config to use the invalid JSON endpoint
            http_config.endpoint_url = f"http://localhost:{server.port}/invalid-json"

            async with setup_worker_test(
                worker_id=0,
                http_config=http_config,
                aiohttp_config=aiohttp_config,
                zmq_config=zmq_config,
            ) as ctx:
                # Send query that will get invalid JSON
                query = Query(
                    id="test-bad-json",
                    data={
                        "prompt": "Will get bad JSON",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                    },
                )
                await ctx.request_push.send(query)

                # Verify error response was sent
                response = await asyncio.wait_for(
                    ctx.response_pull.receive(), timeout=1.0
                )

                assert isinstance(response, QueryResult)
                assert response.id == "test-bad-json"
                assert response.error is not None
                assert (
                    "invalid literal" in response.error
                    or "JSONDecodeError" in response.error
                    or "DecodeError" in response.error
                )

        finally:
            await server.close()
