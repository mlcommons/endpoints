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

"""Core functionality tests for the HTTP endpoint client."""

import asyncio
import concurrent.futures
import time

import pytest
import zmq
import zmq.asyncio
from inference_endpoint.core.types import Query
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)

from ...futures_client import FuturesHttpClient
from ...test_helpers import get_test_socket_path


def create_futures_client(
    url,
    tmp_path,
    prefix,
    num_workers=1,
    aiohttp_config=None,
    zmq_config_kwargs=None,
):
    """Helper to create a FuturesHttpClient with specific config."""
    http_config = HTTPClientConfig(
        endpoint_url=url,
        num_workers=num_workers,
    )

    zmq_kwargs = {
        "zmq_request_queue_prefix": get_test_socket_path(tmp_path, prefix, "_req"),
        "zmq_response_queue_addr": get_test_socket_path(tmp_path, prefix, "_resp"),
        "zmq_readiness_queue_addr": get_test_socket_path(tmp_path, prefix, "_ready"),
    }
    if zmq_config_kwargs:
        zmq_kwargs.update(zmq_config_kwargs)

    zmq_config = ZMQConfig(**zmq_kwargs)

    return FuturesHttpClient(http_config, aiohttp_config or AioHttpConfig(), zmq_config)


class TestHTTPEndpointClientConcurrency:
    """Test concurrent operations and future handling."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_massive_concurrency_non_streaming(
        self, mock_http_echo_server, tmp_path
    ):
        """Test high concurrent requests with proper connection management in non-streaming mode."""
        actual_max_concurrency = 10000

        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "custom",
            num_workers=1,
        )

        try:
            num_requests = actual_max_concurrency

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
                future = client.issue_query(query)
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
        finally:
            client.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_massive_concurrency_streaming(self, mock_http_echo_server, tmp_path):
        """Test high concurrent requests with proper connection management in streaming mode."""
        actual_max_concurrency = 10000

        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "custom",
            num_workers=1,
        )

        try:
            num_requests = actual_max_concurrency

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
                future = client.issue_query(query)
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
        finally:
            client.shutdown()

    @pytest.mark.asyncio
    async def test_massive_payloads(self, mock_http_echo_server, tmp_path):
        """Test handling very large payloads."""
        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "test_payloads",
            num_workers=4,
        )

        try:
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
                future = client.issue_query(query)
                futures.append((name, size, future))

            # Wait for all payloads
            for name, size, future in futures:
                result = await asyncio.wrap_future(future)
                assert result.id == f"payload-{name}"
                assert len(result.response_output) == size
                print(f"\nSuccessfully processed {name} payload ({size} bytes)")
        finally:
            client.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_many_workers(self, mock_http_echo_server, tmp_path):
        """Test with many workers."""
        actual_max_concurrency = 1000
        worker_counts = [16, 32]

        for num_workers in worker_counts:
            print(f"\nTesting with {num_workers} workers...")

            client = create_futures_client(
                f"{mock_http_echo_server.url}/v1/chat/completions",
                tmp_path,
                "custom",
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
                    future = client.issue_query(query)
                    futures.append(future)

                # Wait for all with timeout
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


class TestHTTPEndpointClientCoverage:
    """Tests to improve code coverage."""

    @pytest.mark.asyncio
    async def test_shutdown_basic(self, mock_http_echo_server, tmp_path):
        """Test shutdown() cleans up pending requests and exits."""
        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "test_shutdown_handler",
            num_workers=2,
        )

        # Verify components are running
        assert client._handler_future is not None
        assert not client._handler_future.done()
        assert client.worker_manager is not None
        assert len(client.worker_push_sockets) == 2

        # Add some pending futures
        future1 = concurrent.futures.Future()
        future2 = concurrent.futures.Future()
        client._pending[1] = future1
        client._pending[2] = future2

        # Shutdown should clean everything up
        client.shutdown()

        # Verify cleanup
        assert len(client._pending) == 0
        assert future1.cancelled()
        assert future2.cancelled()
        assert client._handler_future.cancelled()

    @pytest.mark.asyncio
    async def test_concurrent_shutdown(self, mock_http_echo_server, tmp_path):
        """Test shutdown while requests are in flight."""
        # Create a separate client for this test since we need to shut it down
        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "test_shutdown",
            num_workers=4,
        )

        try:
            # Send many requests
            futures = []
            for i in range(100):
                query = Query(
                    id=f"shutdown-{i}",
                    data={
                        "prompt": f"Shutdown test {i}",
                        "model": "gpt-3.5-turbo",
                    },
                )
                future = client.issue_query(query)
                futures.append(future)

            # Immediately shutdown
            client.shutdown()

            # Count completed vs cancelled
            completed = sum(1 for f in futures if f.done() and not f.cancelled())
            cancelled = sum(1 for f in futures if f.cancelled())

            print(f"\nShutdown test: {completed} completed, {cancelled} cancelled")

            # At least some should be cancelled
            assert cancelled > 0
        finally:
            # Ensure cleanup even if test fails
            if not client._shutdown_event.is_set():
                client.shutdown()

    @pytest.mark.asyncio
    async def test_error_response_propagation(self, tmp_path):
        """Test that error responses are propagated as exceptions in futures."""
        client = create_futures_client(
            "http://invalid-host-does-not-exist:9999/v1/chat/completions",
            tmp_path,
            "test_error_prop",
            num_workers=1,
            aiohttp_config=AioHttpConfig(client_timeout_total=2.0),
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

            future = client.issue_query(query)

            # Should get error
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.0)

            # Error message might be empty string, just verify exception was raised
            assert exc_info.value is not None  # Exception was raised

        finally:
            client.shutdown()

    @pytest.mark.asyncio
    async def test_response_handler_error_recovery(
        self, mock_http_echo_server, tmp_path
    ):
        """Test that response handler recovers from errors."""
        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "test_handler_err",
            num_workers=1,
        )

        try:
            # Send first query
            query1 = Query(
                id="3001",
                data={
                    "prompt": "First query",
                    "model": "gpt-3.5-turbo",
                },
            )
            future1 = client.issue_query(query1)

            # Create context to inject invalid data
            context = zmq.asyncio.Context()
            response_push = context.socket(zmq.PUSH)
            response_push.connect(client.zmq_config.zmq_response_queue_addr)

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
            future2 = client.issue_query(query2)

            # Wait for both futures
            result1 = await asyncio.wait_for(asyncio.wrap_future(future1), timeout=5.0)
            result2 = await asyncio.wait_for(asyncio.wrap_future(future2), timeout=5.0)

            # Both should complete successfully
            assert result1.response_output == "First query"
            assert result2.response_output == "Second query after error"

        finally:
            response_push.close()
            context.destroy(linger=0)
            client.shutdown()

    @pytest.mark.asyncio
    async def test_empty_prompt(self, mock_http_echo_server, tmp_path):
        """Test handling empty prompt."""
        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "test_empty",
            num_workers=2,
        )

        try:
            query = Query(
                id="9001",
                data={
                    "prompt": "",
                    "model": "gpt-3.5-turbo",
                },
            )

            future = client.issue_query(query)
            result = await asyncio.wrap_future(future)

            assert result.id == "9001"
            assert result.response_output == ""
        finally:
            client.shutdown()

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, mock_http_echo_server, tmp_path):
        """Test handling special characters and unicode."""
        client = create_futures_client(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "test_special",
            num_workers=2,
        )

        try:
            special_prompts = [
                "Hello 你好 🚀 世界",
                'Special chars: @#$%^&*()_+-={}[]|\\:";<>?,./',
                "Newlines\nand\ttabs\rand\\backslashes",
                "Emoji fest: 😀😃😄😁😆😅😂🤣",
                "\u0000\u0001\u0002 control chars",
            ]

            futures = []
            for i, prompt in enumerate(special_prompts):
                query = Query(
                    id=f"special-{i}",
                    data={
                        "prompt": prompt,
                        "model": "gpt-3.5-turbo",
                    },
                )
                future = client.issue_query(query)
                futures.append((prompt, future))

            # Verify all handled correctly
            for prompt, future in futures:
                result = await asyncio.wrap_future(future)
                assert result.response_output == prompt
        finally:
            client.shutdown()
