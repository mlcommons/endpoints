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

"""Streaming functionality tests for the HTTP endpoint client."""

import asyncio

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)

from ...futures_client import FuturesHttpClient
from ...test_helpers import get_test_socket_path


class TestHTTPEndpointClientStreaming:
    """Test streaming functionality with echo server integration."""

    @pytest.fixture
    def client_config(self, mock_http_echo_server, tmp_path):
        """Create client configuration for echo server."""
        # Use tmp_path for unique socket paths per test
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=10,
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_streaming", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_streaming", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "test_streaming", "_ready"
            ),
        )
        return http_config, aiohttp_config, zmq_config

    def create_client_with_configs(
        self,
        url,
        tmp_path,
        prefix,
        num_workers=1,
        max_concurrency=-1,
        aiohttp_config=None,
    ):
        """Helper to create client with specific config."""
        http_config = HTTPClientConfig(
            endpoint_url=url,
            num_workers=num_workers,
            max_concurrency=max_concurrency,
        )
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(tmp_path, prefix, "_req"),
            zmq_response_queue_addr=get_test_socket_path(tmp_path, prefix, "_resp"),
            zmq_readiness_queue_addr=get_test_socket_path(tmp_path, prefix, "_ready"),
        )
        return FuturesHttpClient(
            http_config, aiohttp_config or AioHttpConfig(), zmq_config
        )

    @pytest.mark.asyncio
    async def test_streaming_with_futures(self, mock_http_echo_server, tmp_path):
        """Test streaming responses with future handling from concurrency tests."""
        # Create client directly since we need custom config
        client = self.create_client_with_configs(
            f"{mock_http_echo_server.url}/v1/chat/completions",
            tmp_path,
            "test_stream_fut",
            num_workers=2,
        )

        await client.async_start()

        try:
            # Send both streaming and non-streaming requests
            futures = []

            # Non-streaming
            for i in range(5):
                query = Query(
                    id=f"non-stream-{i}",
                    data={
                        "prompt": f"Non-streaming request {i}",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                    },
                )
                future = await client.issue_query(query)
                futures.append(("non-stream", i, future))

            # Streaming
            for i in range(5):
                query = Query(
                    id=f"stream-{i}",
                    data={
                        "prompt": f"Streaming request {i}",
                        "model": "gpt-3.5-turbo",
                        "stream": True,
                    },
                )
                future = await client.issue_query(query)
                futures.append(("stream", i, future))

            # Wait for all
            for req_type, idx, future in futures:
                result = await future
                if req_type == "non-stream":
                    assert result.id == f"non-stream-{idx}"
                    assert result.response_output == f"Non-streaming request {idx}"
                else:
                    assert result.id == f"stream-{idx}"
                    assert (
                        "Streaming" in result.response_output
                        or result.response_output == f"Streaming request {idx}"
                    )

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_mixed_streaming_non_streaming(self, futures_http_client):
        """Test that mixed streaming and non-streaming requests work correctly."""
        # Send mixed requests
        futures = []

        # Non-streaming request
        query_non_stream = Query(
            id="non-stream-1",
            data={
                "prompt": "Non-streaming response",
                "model": "gpt-3.5-turbo",
                "stream": False,
            },
        )
        futures.append(
            ("non-stream", await futures_http_client.issue_query(query_non_stream))
        )

        # Streaming request
        query_stream = Query(
            id="stream-1",
            data={
                "prompt": "Streaming response test",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )
        futures.append(("stream", await futures_http_client.issue_query(query_stream)))

        # Another non-streaming
        query_non_stream2 = Query(
            id="non-stream-2",
            data={
                "prompt": "Another non-streaming",
                "model": "gpt-3.5-turbo",
                "stream": False,
            },
        )
        futures.append(
            ("non-stream", await futures_http_client.issue_query(query_non_stream2))
        )

        # Wait for all and verify
        for req_type, future in futures:
            result = await future

            if req_type == "non-stream":
                # Non-streaming should not have chunk metadata
                assert result.metadata is None or "first_chunk" not in result.metadata
                assert result.response_output in [
                    "Non-streaming response",
                    "Another non-streaming",
                ]
            else:
                # Streaming response
                assert "".join(result.response_output) == "Streaming response test"

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, futures_http_client):
        """Test multiple concurrent streaming requests."""
        # Send 10 concurrent streaming requests
        futures = []
        for i in range(10):
            query = Query(
                id=f"concurrent-stream-{i}",
                data={
                    "prompt": f"Concurrent streaming request number {i}",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            futures.append((i, await futures_http_client.issue_query(query)))

        # Wait for all to complete
        results = []
        for idx, future in futures:
            result = await future
            results.append((idx, result))

        # Verify all completed with correct content
        assert len(results) == 10

        for idx, result in results:
            assert result.id == f"concurrent-stream-{idx}"
            assert (
                "".join(result.response_output)
                == f"Concurrent streaming request number {idx}"
            )
            assert result.error is None

    @pytest.mark.asyncio
    async def test_streaming_future_only_resolves_with_final_content(
        self, futures_http_client
    ):
        """Test that futures are only resolved once with final complete response, not intermediate chunks."""
        # Track when future is resolved
        resolution_count = 0
        resolved_result = None

        async def track_resolution(future):
            nonlocal resolution_count, resolved_result
            result = await future
            resolution_count += 1
            resolved_result = result

        query = Query(
            id="test-single-resolution",
            data={
                "prompt": "Test single future resolution with multiple words",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future = await futures_http_client.issue_query(query)

        # Start tracking task
        track_task = asyncio.create_task(track_resolution(future))

        # Wait for completion
        await track_task

        # Verify future was only resolved once
        assert resolution_count == 1
        assert resolved_result is not None
        assert resolved_result.id == "test-single-resolution"
        assert resolved_result.response_output == (
            "Test",
            " single future resolution with multiple words",
        )

        # Verify future is done and can't be resolved again
        assert future.done()
        assert future.result() == resolved_result

    @pytest.mark.asyncio
    async def test_streaming_future_first_chunk_access(self, futures_http_client):
        """Test StreamingFuture.first property for early chunk access."""

        # Test 1: Access first chunk before completion
        query = Query(
            id="test-first-chunk",
            data={
                "prompt": "Test first chunk access functionality",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future = await futures_http_client.issue_query(query)

        # Future should be StreamingFuture
        assert hasattr(future, "first")

        # Get first chunk
        first_chunk = await future.first
        assert first_chunk == "Test"  # Echo server returns first word

        # First chunk should still be accessible
        first_chunk_again = await future.first
        assert first_chunk_again == "Test"

        # Get complete response
        result = await future
        assert result.response_output[0] == "Test"
        assert result.response_output[1] == " first chunk access functionality"

        # Test 2: Check if first chunk is ready after awaiting
        query2 = Query(
            id="test-first-ready",
            data={
                "prompt": "Quick response",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future2 = await futures_http_client.issue_query(query2)

        # Await the first chunk
        first_chunk2 = await future2.first
        assert first_chunk2 == "Quick"

        # Should be ready now since we already awaited it
        assert future2.first.done()
        assert future2.first.result() == "Quick"

    @pytest.mark.asyncio
    async def test_streaming_single_chunk_complete(self, futures_http_client):
        """Test handling of single-chunk responses marked as complete."""
        # Test single-word response
        query = Query(
            id="test-single-chunk",
            data={
                "prompt": "Hi",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future = await futures_http_client.issue_query(query)

        # Get first chunk
        first = await future.first
        assert first == "Hi"

        # Complete response should be the same
        result = await future
        assert result.response_output[0] == "Hi"
        assert result.response_output[1] == ""

    @pytest.mark.asyncio
    async def test_streaming_race_first_chunks(self, futures_http_client):
        """Test racing multiple streaming queries for first chunk."""
        # Create multiple streaming queries
        queries = [
            Query(
                id=f"race-{i}",
                data={
                    "prompt": f"Query number {i} with different content",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            for i in range(5)
        ]

        # Issue all queries
        futures = []
        for q in queries:
            futures.append(await futures_http_client.issue_query(q))

        # Race for first chunk
        first_chunks = [f.first for f in futures]
        done, pending = await asyncio.wait(
            first_chunks, return_when=asyncio.FIRST_COMPLETED
        )

        # Get winning chunk
        first_result = done.pop().result()
        assert first_result in ["Query"]  # All start with "Query"

        # Cancel pending first chunks
        for task in pending:
            task.cancel()

        # Race for first complete response
        done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)

        # Get first complete
        winner = done.pop().result()
        assert winner.id.startswith("race-")
        assert winner.response_output[0] == "Query"
        assert winner.response_output[1].startswith(" number")

        # Clean up remaining
        for f in pending:
            f.cancel()

    @pytest.mark.asyncio
    async def test_non_streaming_returns_regular_future(self, futures_http_client):
        """Test that non-streaming queries return regular Future without .first property."""
        # Non-streaming query
        query = Query(
            id="test-non-stream",
            data={
                "prompt": "Regular non-streaming response",
                "model": "gpt-3.5-turbo",
                "stream": False,
            },
        )

        future = await futures_http_client.issue_query(query)

        # Should not have .first property
        assert not hasattr(future, "first")

        # Should work as regular future
        result = await future
        assert result.id == "test-non-stream"
        assert result.response_output == "Regular non-streaming response"

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, tmp_path):
        """Test error handling in streaming responses."""
        # Use invalid endpoint to trigger errors
        client = self.create_client_with_configs(
            "http://invalid-endpoint-12345:9999/v1/chat/completions",
            tmp_path,
            "tse",
            num_workers=1,
            aiohttp_config=AioHttpConfig(client_timeout_total=2.0),
        )

        try:
            await client.async_start()

            query = Query(
                id="test-error",
                data={
                    "prompt": "This will fail",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = await client.issue_query(query)

            # Should be StreamingFuture
            assert hasattr(future, "first")

            # Both first chunk and complete response should fail
            with pytest.raises(Exception):  # noqa: B017 Worker wraps errors in generic Exception
                await asyncio.wait_for(future.first, timeout=2.0)
            with pytest.raises(Exception):  # noqa: B017
                await asyncio.wait_for(future, timeout=2.0)

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_streaming_concurrent_mixed_lengths(self, futures_http_client):
        """Test concurrent streaming with various response lengths."""
        # Different length prompts
        test_cases = [
            ("short", "Hi"),
            ("medium", "This is a medium length response"),
            (
                "long",
                "This is a much longer response that should stream multiple chunks",
            ),
            (
                "very-long",
                "This is a very long response that should stream multiple chunks" * 100,
            ),
            ("empty", ""),
            ("single", "Word"),
        ]

        futures = []
        for name, prompt in test_cases:
            query = Query(
                id=f"length-{name}",
                data={
                    "prompt": prompt,
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            futures.append((name, prompt, await futures_http_client.issue_query(query)))

        # Check all complete correctly
        for _, prompt, future in futures:
            if prompt:  # Non-empty
                first = await future.first
                # Should get first word
                assert first == prompt.split()[0] if prompt else ""

            result = await future
            assert "".join(result.response_output) == prompt
