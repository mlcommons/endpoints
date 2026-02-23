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
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient

from .conftest import create_futures_client


def _create_client(
    url: str, zmq_context: ManagedZMQContext | None = None, **kwargs
) -> HTTPEndpointClient:
    config = HTTPClientConfig(
        endpoint_urls=[url],
        num_workers=kwargs.pop("num_workers", 1),
        max_connections=kwargs.pop("max_connections", 10),
        warmup_connections=kwargs.pop("warmup_connections", 0),
        **kwargs,
    )
    return HTTPEndpointClient(config, zmq_context=zmq_context)


def _make_query(id: str, prompt: str = "hello", stream: bool = False) -> Query:
    return Query(
        id=id,
        data={"prompt": prompt, "model": "test", "stream": stream},
    )


@pytest.fixture
def http_client(mock_http_echo_server):
    with ManagedZMQContext.scoped() as zmq_ctx:
        client = _create_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            zmq_context=zmq_ctx,
        )
        yield client
        client.shutdown()


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
            assert len(result.get_response_output_string()) == size
            print(f"\nSuccessfully processed {name} payload ({size} bytes)")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_many_workers(self, mock_http_echo_server):
        """Test with many workers."""
        actual_max_concurrency = 1000
        worker_counts = [16, 32]

        for num_workers in worker_counts:
            print(f"\nTesting with {num_workers} workers...")

            with ManagedZMQContext.scoped() as zmq_ctx:
                client = create_futures_client(
                    f"{mock_http_echo_server.url}/v1/chat/completions",
                    num_workers=num_workers,
                    max_connections=num_workers
                    * 10,  # ensure each worker has connections
                    warmup_connections=0,
                    zmq_context=zmq_ctx,
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
        """Test that shutdown cancels in-flight requests."""
        from aiohttp import web
        from aiohttp.test_utils import TestServer

        request_received = asyncio.Event()

        async def hang_forever(request):
            """Handler that never responds."""
            request_received.set()
            await asyncio.sleep(999)
            return web.Response(text="never reached")

        app = web.Application()
        app.router.add_post("/v1/chat/completions", hang_forever)
        server = TestServer(app)
        await server.start_server()

        try:
            with ManagedZMQContext.scoped() as zmq_ctx:
                client = create_futures_client(
                    f"http://localhost:{server.port}/v1/chat/completions",
                    num_workers=1,
                    zmq_context=zmq_ctx,
                )

                # Issue requests that will hang
                num_requests = 5
                futures = [
                    client.issue(
                        Query(id=f"test-{i}", data={"prompt": "x", "model": "test"})
                    )
                    for i in range(num_requests)
                ]

                # Wait for at least one request to reach server
                await request_received.wait()

                # Shutdown should cancel all futures
                client.shutdown()

                cancelled = sum(1 for f in futures if f.cancelled())
                assert (
                    cancelled == num_requests
                ), f"Expected {num_requests} cancelled, got {cancelled}"

        finally:
            await server.close()

    @pytest.mark.asyncio
    async def test_error_response_propagation(self):
        """Test that error responses are propagated as exceptions in futures."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            client = create_futures_client(
                "http://invalid-host-does-not-exist:9999/v1/chat/completions",
                zmq_context=zmq_ctx,
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
        with ManagedZMQContext.scoped() as zmq_ctx:
            # Use invalid endpoint to trigger errors
            client = create_futures_client(
                "http://invalid-endpoint-12345:9999/v1/chat/completions",
                warmup_connections=0,
                zmq_context=zmq_ctx,
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
                result.get_response_output_string() == prompt
            ), f"Mismatch for test case '{name}'"


class TestPoll:
    """Test non-blocking poll() response retrieval."""

    def test_poll_returns_none_when_empty(self, http_client):
        """poll() returns None when no responses are available."""
        assert http_client.poll() is None

    def test_poll_returns_response(self, http_client):
        """poll() returns a QueryResult after issue + wait."""
        http_client.issue(_make_query("poll-1"))

        result = None
        for _ in range(100):
            result = http_client.poll()
            if result is not None:
                break
            time.sleep(0.01)

        assert isinstance(result, QueryResult)
        assert result.id == "poll-1"

    def test_poll_is_non_blocking(self, http_client):
        """poll() returns immediately even with no data."""
        start = time.monotonic()
        for _ in range(1000):
            http_client.poll()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    def test_poll_drains_one_at_a_time(self, http_client):
        """Each poll() call returns at most one response."""
        n = 5
        for i in range(n):
            http_client.issue(_make_query(f"poll-multi-{i}"))

        time.sleep(1.0)

        results = []
        while (r := http_client.poll()) is not None:
            results.append(r)
            assert isinstance(r, QueryResult | StreamChunk)

        assert len(results) == n


class TestRecv:
    """Test blocking recv() response retrieval."""

    @pytest.mark.asyncio
    async def test_recv_blocks_until_response(self, http_client):
        """recv() blocks and returns a response."""
        http_client.issue(_make_query("recv-1", prompt="recv test"))

        result = await asyncio.wait_for(
            asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(http_client.recv(), http_client.loop)
            ),
            timeout=5.0,
        )

        assert isinstance(result, QueryResult)
        assert result.id == "recv-1"

    @pytest.mark.asyncio
    async def test_recv_all_responses_arrive(self, http_client):
        """recv() delivers all issued responses."""
        ids = {f"recv-order-{i}" for i in range(5)}
        for id in ids:
            http_client.issue(_make_query(id))

        received_ids = set()
        for _ in range(len(ids)):
            result = await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(
                        http_client.recv(), http_client.loop
                    )
                ),
                timeout=5.0,
            )
            assert isinstance(result, QueryResult)
            received_ids.add(result.id)

        assert received_ids == ids


class TestDrain:
    """Test non-blocking drain() bulk retrieval."""

    def test_drain_returns_empty_when_no_responses(self, http_client):
        """drain() returns empty list when nothing is available."""
        assert http_client.drain() == []

    def test_drain_returns_all_available(self, http_client):
        """drain() returns all buffered responses at once."""
        n = 5
        for i in range(n):
            http_client.issue(_make_query(f"drain-{i}"))

        time.sleep(1.0)

        results = http_client.drain()
        assert len(results) == n
        result_ids = {r.id for r in results if isinstance(r, QueryResult)}
        expected_ids = {f"drain-{i}" for i in range(n)}
        assert result_ids == expected_ids

    def test_drain_empties_buffer(self, http_client):
        """After drain(), subsequent drain() returns empty."""
        http_client.issue(_make_query("drain-once"))
        time.sleep(1.0)

        first = http_client.drain()
        assert len(first) >= 1

        second = http_client.drain()
        assert second == []

    def test_drain_is_non_blocking(self, http_client):
        """drain() returns immediately even with no data."""
        start = time.monotonic()
        for _ in range(1000):
            http_client.drain()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


class TestShutdown:
    """Test shutdown behavior."""

    def test_shutdown_is_idempotent(self, mock_http_echo_server):
        """Calling shutdown() multiple times does not raise."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            client = _create_client(
                f"{mock_http_echo_server.url}/v1/chat/completions",
                zmq_context=zmq_ctx,
            )
            client.shutdown()
            client.shutdown()

    def test_issue_after_shutdown_drops(self, mock_http_echo_server):
        """Requests issued after shutdown are silently dropped."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            client = _create_client(
                f"{mock_http_echo_server.url}/v1/chat/completions",
                zmq_context=zmq_ctx,
            )
            client.shutdown()
            client.issue(_make_query("post-shutdown"))
