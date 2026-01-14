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

"""Core functionality tests for the HTTP endpoint client."""

import asyncio
import time

import pytest
import zmq
import zmq.asyncio
from inference_endpoint.core.types import Query

from .conftest import create_futures_client


class TestHttpEndpointClientScaleOut:
    """Test scale out capabilities of the HTTP endpoint client."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_massive_concurrency_non_streaming(self, futures_http_client):
        """Test high concurrent requests with proper connection management in non-streaming mode."""
        num_requests = 10000

        # Collect futures
        start_time = time.time()
        futures = []
        for i in range(num_requests):
            query = Query(
                id=f"massive-{i}",
                data={
                    "prompt": f"Request {i}",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )
            future = futures_http_client.issue(query)
            futures.append(future)

        # Wait for all futures to complete
        results = await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
        end_time = time.time()

        # Verify results
        assert len(results) == num_requests
        result_ids = {r.id for r in results}
        expected_ids = {f"massive-{i}" for i in range(num_requests)}
        assert result_ids == expected_ids

        # Print performance metrics
        duration = end_time - start_time
        rps = num_requests / duration
        print(
            f"\nNon-streaming mode performance: {num_requests} requests in {duration:.2f}s = {rps:.0f} RPS"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_massive_concurrency_streaming(self, futures_http_client):
        """Test high concurrent requests with proper connection management in streaming mode."""
        num_requests = 10000

        # Collect futures
        start_time = time.time()
        futures = []
        for i in range(num_requests):
            query = Query(
                id=f"massive-streaming-{i}",
                data={
                    "prompt": f"Streaming request {i}",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            future = futures_http_client.issue(query)
            futures.append(future)

        # Wait for all futures to complete
        results = await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
        end_time = time.time()

        # Verify results
        assert len(results) == num_requests
        result_ids = {r.id for r in results}
        expected_ids = {f"massive-streaming-{i}" for i in range(num_requests)}
        assert result_ids == expected_ids

        # Print performance metrics
        duration = end_time - start_time
        rps = num_requests / duration
        print(
            f"\nStreaming mode performance: {num_requests} requests in {duration:.2f}s = {rps:.0f} RPS"
        )

    @pytest.mark.asyncio
    async def test_massive_payloads(self, futures_http_client):
        """Test handling very large payloads."""
        # Create payloads of different sizes
        payload_sizes = [
            ("small", 128),  # 128 bytes
            ("medium", 1024),  # 1KB
            ("large", 1024 * 10),  # 10KB
            ("xlarge", 1024 * 100),  # 100KB
        ]

        futures = []

        for name, size in payload_sizes:
            # Create large prompt
            large_prompt = "x" * size
            query = Query(
                id=f"payload-{name}",
                data={
                    "prompt": large_prompt,
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 2000,
                },
            )
            future = futures_http_client.issue(query)
            futures.append((name, size, future))

        # Wait for all payloads
        for name, size, future in futures:
            result = await asyncio.wrap_future(future)
            assert result.id == f"payload-{name}"
            assert len(result.response_output) == size
            print(f"\nSuccessfully processed {name} payload ({size} bytes)")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_many_workers(self, mock_http_echo_server):
        """Test with many workers."""
        actual_max_concurrency = 1000
        worker_counts = [16, 32]

        for num_workers in worker_counts:
            print(f"\nTesting with {num_workers} workers...")

            client = create_futures_client(
                f"{mock_http_echo_server.url}/v1/chat/completions",
                num_workers=num_workers,
            )

            try:
                num_requests = actual_max_concurrency
                futures = []

                start_time = time.time()
                for i in range(num_requests):
                    query = Query(
                        id=f"worker-test-{i}",
                        data={
                            "prompt": f"Testing {num_workers} workers - request {i}",
                            "model": "gpt-3.5-turbo",
                        },
                    )
                    future = client.issue(query)
                    futures.append(future)

                # Wait for all
                results = await asyncio.gather(
                    *[asyncio.wrap_future(f) for f in futures]
                )
                duration = time.time() - start_time

                # Verify
                assert len(results) == num_requests
                print(
                    f"  Completed {num_requests} requests in {duration:.2f}s "
                    f"({num_requests / duration:.0f} req/s)"
                )

            finally:
                client.shutdown()


class TestHTTPEndpointClientFunctionality:
    """Test core functionality of the HTTP endpoint client."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_requests(self):
        """Test that shutdown properly cancels in-flight HTTP requests.

        This test verifies that when shutdown is called:
        1. Client-side futures are cancelled
        2. Server-side HTTP connections are terminated (via connection tracking)
        3. No requests complete after shutdown is initiated
        """
        from aiohttp import web

        # Track active connections on the server side
        active_connections: set[asyncio.Future] = set()
        connection_received = asyncio.Event()
        all_connections_closed = asyncio.Event()

        async def slow_handler(request):
            """Handler that blocks until cancelled, tracking active connections."""
            wait_forever = asyncio.get_event_loop().create_future()
            active_connections.add(wait_forever)
            connection_received.set()  # Signal that a connection arrived
            try:
                await wait_forever  # Block until cancelled
                return web.Response(text="should not reach here")
            except asyncio.CancelledError:
                raise
            finally:
                active_connections.discard(wait_forever)
                if len(active_connections) == 0:
                    all_connections_closed.set()

        # Create and start test server
        app = web.Application()
        app.router.add_post("/v1/chat/completions", slow_handler)

        from aiohttp.test_utils import TestServer

        server = TestServer(app)
        await server.start_server()

        try:
            # Create client pointing to slow server
            # NOTE(vir):
            # Using single worker to avoid timing issues with async TestServer
            # which will spawn its own even loop
            client = create_futures_client(
                f"http://localhost:{server.port}/v1/chat/completions",
                num_workers=1,
            )

            # Issue requests that will block on the server
            num_requests = 10
            futures = []
            for i in range(num_requests):
                query = Query(
                    id=f"cancel-test-{i}",
                    data={
                        "prompt": f"This request will be cancelled {i}",
                        "model": "gpt-3.5-turbo",
                    },
                )
                future = client.issue(query)
                futures.append(future)

            # Wait for at least one request to reach the server
            await connection_received.wait()
            connections_before_shutdown = len(active_connections)

            # Shutdown the client
            client.shutdown()

            # Verify client-side: all futures should be cancelled
            cancelled_count = sum(1 for f in futures if f.cancelled())
            assert (
                cancelled_count == num_requests
            ), f"Expected all {num_requests} futures cancelled, got {cancelled_count}"

            # Verify server-side: wait for all connections to close
            await all_connections_closed.wait()
            assert len(active_connections) == 0, (
                f"Expected 0 active connections after shutdown, "
                f"got {len(active_connections)}"
            )

            print(
                f"\nShutdown cancellation test: "
                f"{cancelled_count}/{num_requests} futures cancelled, "
                f"server connections: {connections_before_shutdown} -> 0"
            )

        finally:
            await server.close()

    @pytest.mark.asyncio
    async def test_error_response_propagation(self):
        """Test that error responses are propagated as exceptions in futures."""
        client = create_futures_client(
            "http://invalid-host-does-not-exist:9999/v1/chat/completions",
        )

        try:
            # Send request to invalid endpoint
            query = Query(
                id="2001",
                data={
                    "prompt": "Test error",
                    "model": "gpt-3.5-turbo",
                },
            )

            future = client.issue(query)

            # Should get error
            with pytest.raises(Exception) as exc_info:
                await asyncio.wrap_future(future)

            # Error message might be empty string, just verify exception was raised
            assert exc_info.value is not None  # Exception was raised

        finally:
            client.shutdown()

    @pytest.mark.asyncio
    async def test_response_handler_error_recovery(self, futures_http_client):
        """Test that response handler recovers from errors."""
        # Send first query
        query1 = Query(
            id="3001",
            data={
                "prompt": "First query",
                "model": "gpt-3.5-turbo",
            },
        )
        future1 = futures_http_client.issue(query1)

        # Create context to inject invalid data
        context = zmq.asyncio.Context()
        response_push = context.socket(zmq.PUSH)
        # Access response address via pool transport internal state
        response_addr = futures_http_client.worker_manager.pool_transport._response_addr
        response_push.connect(response_addr)

        try:
            # Send invalid data that will cause error in handler
            await response_push.send(b"invalid msgspec data")

            # Send second query - handler should have recovered
            query2 = Query(
                id="3002",
                data={
                    "prompt": "Second query after error",
                    "model": "gpt-3.5-turbo",
                },
            )
            future2 = futures_http_client.issue(query2)

            # Wait for both futures
            result1 = await asyncio.wrap_future(future1)
            result2 = await asyncio.wrap_future(future2)

            # Both should complete successfully
            assert result1.response_output == "First query"
            assert result2.response_output == "Second query after error"

        finally:
            response_push.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    async def test_streaming_error_propagation(self):
        """Test error propagation in streaming responses."""
        # Use invalid endpoint to trigger errors
        client = create_futures_client(
            "http://invalid-endpoint-12345:9999/v1/chat/completions",
        )

        try:
            query = Query(
                id="test-error",
                data={
                    "prompt": "This will fail",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = client.issue(query)

            # Complete response should fail
            with pytest.raises(Exception):  # noqa: B017 Worker wraps errors in generic Exception
                await asyncio.wrap_future(future)

        finally:
            client.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, futures_http_client):
        """Test concurrent streaming with various payload types.

        Tests different payload lengths, special characters, unicode, and edge cases
        to verify the client handles concurrent streaming requests correctly.
        """
        # Test cases covering lengths, special chars, unicode, and edge cases
        test_cases = [
            # Length variations
            ("empty", ""),
            ("single-word", "Word"),
            ("short", "Hi there"),
            ("medium", "This is a medium length response for testing"),
            (
                "long",
                "This is a much longer response that should stream multiple chunks",
            ),
            (
                "very-long",
                ("This is a very long response " * 100).rstrip(),
            ),
            # Special characters and unicode
            ("unicode", "Hello 你好 🚀 世界"),
            ("special-chars", '@#$%^&*()_+-={}[]|\\:";<>?,./'),
            ("emoji", "Emoji fest: 😀😃😄😁😆😅😂🤣"),
        ]

        futures = []
        for name, prompt in test_cases:
            query = Query(
                id=f"stream-{name}",
                data={
                    "prompt": prompt,
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            futures.append((name, prompt, futures_http_client.issue(query)))

        # Verify all complete correctly
        for name, prompt, future in futures:
            result = await asyncio.wrap_future(future)
            assert (
                "".join(result.response_output["output"]) == prompt
            ), f"Mismatch for test case '{name}'"
