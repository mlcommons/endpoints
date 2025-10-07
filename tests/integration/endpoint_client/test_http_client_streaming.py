"""Streaming functionality tests for the HTTP endpoint client."""

import asyncio

import pytest
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.futures_client import FuturesHttpClient

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

    @pytest.mark.asyncio
    async def test_streaming_with_futures(self, mock_http_echo_server, tmp_path):
        """Test streaming responses with future handling from concurrency tests."""
        # Use tmp_path for unique socket paths
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_stream_fut", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_stream_fut", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "test_stream_fut", "_ready"
            ),
        )

        # Create client directly since we need custom config
        client = FuturesHttpClient(
            http_config,
            AioHttpConfig(),
            zmq_config,
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
    async def test_streaming_response_complete_content(self, futures_http_client):
        """Test that streaming responses return complete content via futures."""
        # Track all responses received via callback
        received_responses = []

        def response_callback(response):
            # Handle both StreamChunk and QueryResult messages
            match response:
                case StreamChunk(id=qid, response_chunk=chunk, metadata=meta):
                    received_responses.append(
                        {
                            "id": qid,
                            "content": chunk,
                            "metadata": meta,
                            "type": "StreamChunk",
                        }
                    )
                case QueryResult(id=qid, response_output=output, metadata=meta):
                    received_responses.append(
                        {
                            "id": qid,
                            "content": output,
                            "metadata": meta,
                            "type": "QueryResult",
                        }
                    )

        futures_http_client.complete_callback = response_callback

        # Test 1: Single word response
        query1 = Query(
            id="test-stream-1",
            data={
                "prompt": "Hello",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future1 = await futures_http_client.issue_query(query1)
        result1 = await future1

        # Verify we got the complete response
        assert result1.id == "test-stream-1"
        assert result1.response_output == "Hello"

        # Test 2: Multi-word response
        query2 = Query(
            id="test-stream-2",
            data={
                "prompt": "This is a longer streaming test message",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future2 = await futures_http_client.issue_query(query2)
        result2 = await future2

        # Verify complete response
        assert result2.id == "test-stream-2"
        assert result2.response_output == "This is a longer streaming test message"

        # Test 3: Empty response
        query3 = Query(
            id="test-stream-3",
            data={
                "prompt": "",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future3 = await futures_http_client.issue_query(query3)
        result3 = await future3

        assert result3.id == "test-stream-3"
        assert result3.response_output == ""

        # Verify callback received messages
        # Note: The callback receives first StreamChunk and final QueryResult
        # For empty responses, only final QueryResult is sent (no separate first chunk)
        # - test-stream-1: 2 messages (first chunk + final result)
        # - test-stream-2: 2 messages (first chunk + final result)
        # - test-stream-3: 1 message (final result only, as it's empty)
        assert len(received_responses) >= 5

        # Verify we have responses for each query
        for sample_id in ["test-stream-1", "test-stream-2", "test-stream-3"]:
            query_responses = [r for r in received_responses if r["id"] == sample_id]

            # Get QueryResult messages
            query_results = [r for r in query_responses if r["type"] == "QueryResult"]

            # Should have exactly one QueryResult per query
            assert (
                len(query_results) == 1
            ), f"Should have exactly one QueryResult for {sample_id}"

            # Callback is called with first StreamChunk and final QueryResult
            # For empty responses, only QueryResult is sent (no StreamChunk)
            # First chunk can be accessed via future.first, final result via awaiting the future

            # Verify final content
            final_response = query_results[0]
            if sample_id == "test-stream-1":
                assert final_response["content"] == "Hello"
            elif sample_id == "test-stream-2":
                assert (
                    final_response["content"]
                    == "This is a longer streaming test message"
                )
            elif sample_id == "test-stream-3":
                assert final_response["content"] == ""

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
                assert result.response_output == "Streaming response test"

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
                result.response_output == f"Concurrent streaming request number {idx}"
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
        assert (
            resolved_result.response_output
            == "Test single future resolution with multiple words"
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
        assert result.response_output == "Test first chunk access functionality"

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
        assert result.response_output == "Hi"

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
        assert "Query number" in winner.response_output

        # Clean up remaining
        for f in pending:
            f.cancel()

    @pytest.mark.asyncio
    async def test_streaming_metadata_propagation(self, futures_http_client):
        """Test that StreamChunk metadata is properly handled."""

        responses_received = []

        def capture_responses(response):
            responses_received.append(response)

        futures_http_client.complete_callback = capture_responses

        query = Query(
            id="test-metadata",
            data={
                "prompt": "Test metadata propagation",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future = await futures_http_client.issue_query(query)

        # Wait for first chunk
        first = await future.first
        assert first == "Test"

        # Wait for completion
        result = await future
        assert result.response_output == "Test metadata propagation"

        # Give callback time to process
        await asyncio.sleep(0.1)

        # Should have received responses via callback (first chunk + final result)
        assert len(responses_received) >= 2

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
        http_config = HTTPClientConfig(
            endpoint_url="http://invalid-endpoint-12345:9999/v1/chat/completions",
            num_workers=1,
        )

        # Use tmp_path for unique socket paths (shortened to avoid path length limit)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(tmp_path, "tse", "_req"),
            zmq_response_queue_addr=get_test_socket_path(tmp_path, "tse", "_resp"),
        )

        # Create client directly since we need custom config
        client = FuturesHttpClient(
            http_config,
            AioHttpConfig(client_timeout_total=2.0),
            zmq_config,
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
            assert result.response_output == prompt
