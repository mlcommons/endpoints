"""Streaming functionality tests for the HTTP endpoint client."""

import asyncio

import aiohttp
import pytest
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)


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
            zmq_request_queue_prefix=f"ipc://{tmp_path}/test_streaming_req",
            zmq_response_queue_addr=f"ipc://{tmp_path}/test_streaming_resp",
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
            zmq_request_queue_prefix=f"ipc://{tmp_path}/test_stream_fut_req",
            zmq_response_queue_addr=f"ipc://{tmp_path}/test_stream_fut_resp",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.start()

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
                future = client.issue_query(query)
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
                future = client.issue_query(query)
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
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_response_complete_content(self, client_config):
        """Test that streaming responses return complete content via futures."""
        http_config, aiohttp_config, zmq_config = client_config

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

        client = HTTPEndpointClient(
            http_config, aiohttp_config, zmq_config, complete_callback=response_callback
        )

        try:
            await client.start()
            await asyncio.sleep(0.5)  # Let workers initialize

            # Test 1: Single word response
            query1 = Query(
                id="test-stream-1",
                data={
                    "prompt": "Hello",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future1 = client.issue_query(query1)
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

            future2 = client.issue_query(query2)
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

            future3 = client.issue_query(query3)
            result3 = await future3

            assert result3.id == "test-stream-3"
            assert result3.response_output == ""

            # Verify callback received both StreamChunk and QueryResult messages
            # Each streaming query should produce:
            # - StreamChunk messages for first chunks (except empty responses)
            # - QueryResult message for final response
            # At least one QueryResult per query
            assert len(received_responses) >= 3

            # Verify we have both chunk and final responses for each query
            for sample_id in ["test-stream-1", "test-stream-2", "test-stream-3"]:
                query_responses = [
                    r for r in received_responses if r["id"] == sample_id
                ]

                # Get StreamChunk and QueryResult messages separately
                stream_chunks = [
                    r for r in query_responses if r["type"] == "StreamChunk"
                ]
                query_results = [
                    r for r in query_responses if r["type"] == "QueryResult"
                ]

                # Should have exactly one QueryResult per query
                assert (
                    len(query_results) == 1
                ), f"Should have exactly one QueryResult for {sample_id}"

                # Non-empty responses should have StreamChunk
                if sample_id != "test-stream-3":  # Not empty
                    assert (
                        len(stream_chunks) >= 1
                    ), f"Should have at least one StreamChunk for {sample_id}"
                else:  # Empty response has no chunks
                    assert (
                        len(stream_chunks) == 0
                    ), "Empty response should have no StreamChunk"

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

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_mixed_streaming_non_streaming(self, client_config):
        """Test that mixed streaming and non-streaming requests work correctly."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

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
            futures.append(("non-stream", client.issue_query(query_non_stream)))

            # Streaming request
            query_stream = Query(
                id="stream-1",
                data={
                    "prompt": "Streaming response test",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            futures.append(("stream", client.issue_query(query_stream)))

            # Another non-streaming
            query_non_stream2 = Query(
                id="non-stream-2",
                data={
                    "prompt": "Another non-streaming",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )
            futures.append(("non-stream", client.issue_query(query_non_stream2)))

            # Wait for all and verify
            for req_type, future in futures:
                result = await future

                if req_type == "non-stream":
                    # Non-streaming should not have chunk metadata
                    assert (
                        result.metadata is None or "first_chunk" not in result.metadata
                    )
                    assert result.response_output in [
                        "Non-streaming response",
                        "Another non-streaming",
                    ]
                else:
                    # Streaming response
                    assert result.response_output == "Streaming response test"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, client_config):
        """Test multiple concurrent streaming requests."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

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
                futures.append((i, client.issue_query(query)))

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
                    result.response_output
                    == f"Concurrent streaming request number {idx}"
                )
                assert result.error is None

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_future_only_resolves_with_final_content(
        self, client_config
    ):
        """Test that futures are only resolved once with final complete response, not intermediate chunks."""
        http_config, aiohttp_config, zmq_config = client_config

        # Track when future is resolved
        resolution_count = 0
        resolved_result = None

        async def track_resolution(future):
            nonlocal resolution_count, resolved_result
            result = await future
            resolution_count += 1
            resolved_result = result

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

            query = Query(
                id="test-single-resolution",
                data={
                    "prompt": "Test single future resolution with multiple words",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = client.issue_query(query)

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

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_future_first_chunk_access(self, client_config):
        """Test StreamingFuture.first property for early chunk access."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

            # Test 1: Access first chunk before completion
            query = Query(
                id="test-first-chunk",
                data={
                    "prompt": "Test first chunk access functionality",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = client.issue_query(query)

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

            # Test 2: Check if first chunk is ready without waiting
            query2 = Query(
                id="test-first-ready",
                data={
                    "prompt": "Quick response",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future2 = client.issue_query(query2)

            # Initially may not be ready
            await asyncio.sleep(0.1)  # Small delay

            # Should be ready now
            assert future2.first.done()
            assert future2.first.result() == "Quick"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_single_chunk_complete(self, client_config):
        """Test handling of single-chunk responses marked as complete."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

            # Test single-word response
            query = Query(
                id="test-single-chunk",
                data={
                    "prompt": "Hi",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = client.issue_query(query)

            # Get first chunk
            first = await future.first
            assert first == "Hi"

            # Complete response should be the same
            result = await future
            assert result.response_output == "Hi"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_race_first_chunks(self, client_config):
        """Test racing multiple streaming queries for first chunk."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

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
            futures = [client.issue_query(q) for q in queries]

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
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )

            # Get first complete
            winner = done.pop().result()
            assert winner.id.startswith("race-")
            assert "Query number" in winner.response_output

            # Clean up remaining
            for f in pending:
                f.cancel()

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_metadata_propagation(self, client_config):
        """Test that StreamChunk metadata is properly handled."""
        http_config, aiohttp_config, zmq_config = client_config

        responses_received = []

        def capture_responses(response):
            responses_received.append(response)

        client = HTTPEndpointClient(
            http_config, aiohttp_config, zmq_config, complete_callback=capture_responses
        )

        try:
            await client.start()

            query = Query(
                id="test-metadata",
                data={
                    "prompt": "Test metadata propagation",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = client.issue_query(query)

            # Wait for first chunk
            first = await future.first
            assert first == "Test"

            # Wait for completion
            result = await future
            assert result.response_output == "Test metadata propagation"

            # Give callback time to process
            await asyncio.sleep(0.1)

            # Should have received responses via callback
            assert len(responses_received) >= 1  # At least final response

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_non_streaming_returns_regular_future(self, client_config):
        """Test that non-streaming queries return regular Future without .first property."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

            # Non-streaming query
            query = Query(
                id="test-non-stream",
                data={
                    "prompt": "Regular non-streaming response",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            future = client.issue_query(query)

            # Should not have .first property
            assert not hasattr(future, "first")

            # Should work as regular future
            result = await future
            assert result.id == "test-non-stream"
            assert result.response_output == "Regular non-streaming response"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, client_config, tmp_path):
        """Test error handling in streaming responses."""
        # Use invalid endpoint to trigger errors
        http_config = HTTPClientConfig(
            endpoint_url="http://invalid-endpoint-12345:9999/v1/chat/completions",
            num_workers=1,
        )

        # Use tmp_path for unique socket paths (shortened to avoid path length limit)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_path}/tse_req",
            zmq_response_queue_addr=f"ipc://{tmp_path}/tse_resp",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(client_timeout_total=2.0),
            zmq_config=zmq_config,
        )

        try:
            await client.start()

            query = Query(
                id="test-error",
                data={
                    "prompt": "This will fail",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = client.issue_query(query)

            # Should be StreamingFuture
            assert hasattr(future, "first")

            # Both first chunk and complete response should fail
            with pytest.raises((aiohttp.ClientError, asyncio.TimeoutError, Exception)):
                await asyncio.wait_for(future.first, timeout=3.0)

            with pytest.raises((aiohttp.ClientError, asyncio.TimeoutError, Exception)):
                await asyncio.wait_for(future, timeout=3.0)

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_streaming_concurrent_mixed_lengths(self, client_config):
        """Test concurrent streaming with various response lengths."""
        http_config, aiohttp_config, zmq_config = client_config

        client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)

        try:
            await client.start()
            await asyncio.sleep(0.5)

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
                    "This is a very long response that should stream multiple chunks"
                    * 100,
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
                futures.append((name, prompt, client.issue_query(query)))

            # Check all complete correctly
            for _, prompt, future in futures:
                if prompt:  # Non-empty
                    first = await future.first
                    # Should get first word
                    assert first == prompt.split()[0] if prompt else ""

                result = await future
                assert result.response_output == prompt

        finally:
            await client.shutdown()
