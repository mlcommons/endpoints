"""Integration tests for the Worker module core functionality using real HTTP requests."""

import asyncio
import pickle
import signal

import pytest
import zmq
import zmq.asyncio
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import Worker


class TestWorkerBasicFunctionality:
    """Test basic Worker functionality for request/response handling."""

    @pytest.fixture
    def zmq_config(self, tmp_path):
        """Create unique ZMQ configuration for each test."""
        # Use tmp_path for unique socket paths per test
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_path}/test_worker_req",
            zmq_response_queue_addr=f"ipc://{tmp_path}/test_worker_resp",
            zmq_high_water_mark=100,
        )

    @pytest.fixture
    def worker_config(self, mock_http_echo_server):
        """Create worker configuration with echo server URL."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
        )
        aiohttp_config = AioHttpConfig()
        return http_config, aiohttp_config

    @pytest.mark.asyncio
    async def test_worker_non_streaming_request(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling non-streaming requests with real HTTP calls."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Create ZMQ context and sockets
        context = zmq.asyncio.Context()

        try:
            # Create request push socket - connect (worker binds the PULL side)
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            # Create response pull socket - bind (worker connects with PUSH)
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Set socket options
            request_push.setsockopt(zmq.SNDHWM, zmq_config.zmq_high_water_mark)
            response_pull.setsockopt(zmq.RCVHWM, zmq_config.zmq_high_water_mark)

            # Start worker in background
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker process to initialize
            await asyncio.sleep(0.1)

            # Send test query
            query = Query(
                id="test-non-streaming",
                data={
                    "prompt": "Hello, echo server!",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Receive response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify response
            assert isinstance(response, QueryResult)
            assert response.id == "test-non-streaming"
            assert response.response_output == "Hello, echo server!"
            assert response.error is None

            # Shutdown worker
            worker._shutdown = True
            await worker_task

        finally:
            # Cleanup
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_request(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling streaming requests with real HTTP calls."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Create ZMQ context and sockets
        context = zmq.asyncio.Context()

        try:
            # Create sockets - request socket connects (worker binds), response socket binds (worker connects)
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Set socket options
            request_push.setsockopt(zmq.SNDHWM, zmq_config.zmq_high_water_mark)
            response_pull.setsockopt(zmq.RCVHWM, zmq_config.zmq_high_water_mark)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Send streaming query
            query = Query(
                id="test-streaming",
                data={
                    "prompt": "Stream this response please",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Collect streaming responses
            responses = []
            final_content = ""

            while True:
                try:
                    response_data = await response_pull.recv()
                    response = pickle.loads(response_data)
                    responses.append(response)

                    # Check if this is the final chunk
                    if response.metadata.get("final_chunk", False):
                        final_content = response.response_output
                        break

                except Exception:
                    break

            # Verify we got multiple chunks
            assert len(responses) >= 2  # At least first chunk and final chunk

            # Verify first chunk
            assert responses[0].metadata.get("first_chunk") is True
            assert responses[0].id == "test-streaming"

            # Verify final response
            assert final_content == "Stream this response please"

            # Shutdown
            worker._shutdown = True
            await worker_task

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_multiple_requests(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling multiple concurrent requests."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Send multiple queries
            num_queries = 5
            queries = []
            for i in range(num_queries):
                query = Query(
                    id=f"test-multi-{i}",
                    data={
                        "prompt": f"Request number {i}",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                    },
                )
                queries.append(query)
                await request_push.send(pickle.dumps(query))

            # Collect responses
            responses = {}
            for _ in range(num_queries):
                response_data = await asyncio.wait_for(
                    response_pull.recv(), timeout=2.0
                )
                response = pickle.loads(response_data)
                responses[response.id] = response

            # Verify all responses
            assert len(responses) == num_queries
            for i in range(num_queries):
                id = f"test-multi-{i}"
                assert id in responses
                assert responses[id].response_output == f"Request number {i}"
                assert responses[id].error is None

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_signal_handling(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker responds to signals correctly."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create minimal sockets
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Verify worker is running
            assert not worker._shutdown

            # Send signal
            worker._handle_signal(signal.SIGTERM, None)

            # Verify shutdown flag is set
            assert worker._shutdown

            # Worker should exit gracefully
            await asyncio.wait_for(worker_task, timeout=3.0)

            response_pull.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_first_last_token_metadata(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker correctly sets first_chunk and final_chunk metadata in streaming responses."""
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Send streaming query with multi-word response to ensure multiple chunks
            query = Query(
                id="test-first-last-token",
                data={
                    "prompt": "Hello world this is a test",  # Multi-word to get multiple chunks
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Collect all streaming responses
            stream_chunks = []
            final_result = None

            while True:
                try:
                    response_data = await asyncio.wait_for(
                        response_pull.recv(), timeout=0.2
                    )
                    response = pickle.loads(response_data)

                    # Check if it's a StreamChunk or final QueryResult
                    if hasattr(response, "response_chunk"):
                        # It's a StreamChunk
                        stream_chunks.append(response)
                    elif hasattr(response, "response_output"):
                        # It's the final QueryResult
                        final_result = response
                        break

                except TimeoutError:
                    break

            # Verify we got at least one StreamChunk and the final QueryResult
            assert len(stream_chunks) >= 1, "Should have at least one StreamChunk"
            assert final_result is not None, "Should have a final QueryResult"

            # Verify first chunk metadata
            first_chunk = stream_chunks[0]
            assert first_chunk.metadata.get("first_chunk") is True
            assert first_chunk.metadata.get("final_chunk") is False
            assert first_chunk.id == "test-first-last-token"
            assert first_chunk.response_chunk  # Should have content

            # Verify final result metadata
            assert final_result.metadata.get("final_chunk") is True
            assert final_result.id == "test-first-last-token"
            assert final_result.response_output == "Hello world this is a test"

            # Verify intermediate chunks (if any) don't have first_chunk set to True
            for chunk in stream_chunks[1:]:
                assert chunk.metadata.get("first_chunk") is not True
                assert chunk.id == "test-first-last-token"

            # Shutdown
            worker._shutdown = True
            await worker_task

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_empty_chunks(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling streaming responses with empty chunks."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Send streaming query with empty prompt (should still get response structure)
            query = Query(
                id="test-empty-chunks",
                data={
                    "prompt": "",  # Empty prompt
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Collect responses
            responses = []

            while True:
                try:
                    response_data = await asyncio.wait_for(
                        response_pull.recv(), timeout=2.0
                    )
                    response = pickle.loads(response_data)
                    responses.append(response)

                    if response.metadata.get("final_chunk", False):
                        break

                except TimeoutError:
                    break

            # Should get at least the final chunk even with empty content
            assert len(responses) >= 1

            # Verify final response
            final_response = responses[-1]
            assert final_response.metadata.get("final_chunk") is True
            assert final_response.id == "test-empty-chunks"
            assert final_response.response_output == ""  # Empty content

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_cleanup_with_resources(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker cleanup when resources are properly initialized."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Start worker to initialize resources
        worker_task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.5)

        # Verify resources are initialized
        assert worker._session is not None
        assert worker._zmq_context is not None
        assert worker._request_socket is not None
        assert worker._response_socket is not None

        # Trigger shutdown
        worker._shutdown = True
        await asyncio.wait_for(worker_task, timeout=2.0)

        # Verify cleanup occurred (resources should be closed/terminated)
        # Note: We can't directly verify closure state, but the test ensures
        # the cleanup code path is executed without errors

    @pytest.mark.asyncio
    async def test_worker_cleanup_with_partial_resources(
        self, worker_config, zmq_config
    ):
        """Test worker cleanup when only some resources are initialized."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Manually set some resources to None to test partial cleanup
        worker._session = None
        worker._request_socket = None

        # Call cleanup directly
        await worker._cleanup()

        # Should complete without errors even with None resources

    @pytest.mark.asyncio
    async def test_worker_different_signal_types(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling different signal types."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Test SIGTERM
        worker._handle_signal(signal.SIGTERM, None)
        assert worker._shutdown is True

        # Reset and test SIGINT
        worker._shutdown = False
        worker._handle_signal(signal.SIGINT, None)
        assert worker._shutdown is True

    @pytest.mark.asyncio
    async def test_worker_aiohttp_session_creation(self, worker_config, zmq_config):
        """Test worker aiohttp session creation with custom timeouts."""
        http_config, aiohttp_config = worker_config

        # Set custom timeout values
        aiohttp_config.client_timeout_total = 30.0
        aiohttp_config.client_timeout_connect = 10.0
        aiohttp_config.client_timeout_sock_read = 5.0

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Start worker briefly to initialize session
        worker_task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.5)

        # Verify session was created with correct timeout
        assert worker._session is not None
        assert worker._session.timeout.total == 30.0

        # Shutdown
        worker._shutdown = True
        await asyncio.wait_for(worker_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_worker_headers_passthrough(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker properly passes through custom headers."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Create query with custom headers
            query = Query(
                id="test-headers",
                data={
                    "prompt": "Test with headers",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )
            # Add headers attribute
            query.headers = {
                "X-Custom-Header": "test-value",
                "Authorization": "Bearer test-token",
            }

            await request_push.send(pickle.dumps(query))

            # Receive response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Echo server should return our prompt
            assert isinstance(response, QueryResult)
            assert response.id == "test-headers"
            assert response.response_output == "Test with headers"
            assert response.error is None

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_large_response_handling(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test worker handling very large responses."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Send query with large content (100KB of text with spaces)
            large_content = "Hello world! " * (100 * 1024 // 13)  # ~100KB
            query = Query(
                id="test-large-response",
                data={
                    "prompt": large_content,
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Receive response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify response
            assert isinstance(response, QueryResult)
            assert response.id == "test-large-response"

            # Check if there's an error first
            if response.error:
                print(f"Response error: {response.error}")
                raise AssertionError(f"Unexpected error: {response.error}")

            # Verify content
            assert response.response_output == large_content
            assert response.error is None
            assert len(response.response_output) == len(large_content)

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_chunk_before_final_single_word(
        self, mock_http_echo_server, worker_config, zmq_config
    ):
        """Test that first streaming chunk always arrives before final response, even for single-word echo."""
        http_config, aiohttp_config = worker_config

        # Create unique socket addresses (using short suffixes to avoid path length limits)
        worker_id = 0
        request_addr = f"{zmq_config.zmq_request_queue_prefix}_{worker_id}_sw"
        response_addr = f"{zmq_config.zmq_response_queue_addr}_sw"
        readiness_addr = f"{zmq_config.zmq_readiness_queue_addr}_sw"

        worker = Worker(
            worker_id=worker_id,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=request_addr,
            response_socket_addr=response_addr,
            readiness_socket_addr=readiness_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets
            request_push = context.socket(zmq.PUSH)
            request_push.connect(request_addr)

            response_pull = context.socket(zmq.PULL)
            response_pull.bind(response_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())
            await asyncio.sleep(0.1)

            # Test single-word prompt that will echo as single word
            query = Query(
                id="test-single-word-order",
                data={
                    "prompt": "Hi",  # Single word that will be echoed
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Collect all responses in order
            responses_in_order = []

            while True:
                try:
                    response_data = await asyncio.wait_for(
                        response_pull.recv(), timeout=0.5
                    )
                    response = pickle.loads(response_data)
                    responses_in_order.append(response)

                    # Check if it's the final QueryResult
                    if isinstance(response, QueryResult) and not isinstance(
                        response, StreamChunk
                    ):
                        break

                except TimeoutError:
                    break

            # Verify we got at least 2 responses (1 StreamChunk + 1 final QueryResult)
            assert (
                len(responses_in_order) >= 2
            ), f"Expected at least 2 responses, got {len(responses_in_order)}"

            # Verify first response is a StreamChunk
            first_response = responses_in_order[0]
            assert isinstance(
                first_response, StreamChunk
            ), "First response should be a StreamChunk"
            assert first_response.response_chunk == "Hi"
            assert first_response.metadata.get("first_chunk") is True
            assert first_response.id == "test-single-word-order"

            # Verify last response is the final QueryResult
            last_response = responses_in_order[-1]
            assert isinstance(
                last_response, QueryResult
            ), "Last response should be a QueryResult"
            assert not isinstance(
                last_response, StreamChunk
            ), "Last response should not be a StreamChunk"
            assert last_response.response_output == "Hi"
            assert last_response.metadata.get("final_chunk") is True
            assert last_response.id == "test-single-word-order"

            # Verify ordering: StreamChunk(s) before final QueryResult
            found_final = False
            for i, resp in enumerate(responses_in_order):
                if isinstance(resp, QueryResult) and not isinstance(resp, StreamChunk):
                    found_final = True
                    # Ensure all previous responses were StreamChunks
                    for j in range(i):
                        assert isinstance(
                            responses_in_order[j], StreamChunk
                        ), f"Response {j} before final should be a StreamChunk"

            assert found_final, "Should have found a final QueryResult"

            # Test with another edge case: empty prompt
            query2 = Query(
                id="test-empty-order",
                data={
                    "prompt": "",  # Empty prompt
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            await request_push.send(pickle.dumps(query2))

            # Collect responses for empty prompt
            responses_empty = []
            while True:
                try:
                    response_data = await asyncio.wait_for(
                        response_pull.recv(), timeout=0.5
                    )
                    response = pickle.loads(response_data)
                    responses_empty.append(response)

                    if isinstance(response, QueryResult) and not isinstance(
                        response, StreamChunk
                    ):
                        break

                except TimeoutError:
                    break

            # For empty prompt, we only expect the final response
            assert (
                len(responses_empty) == 1
            ), "Empty prompt should only return final response"
            assert isinstance(
                responses_empty[0], QueryResult
            ), "Should be final QueryResult"
            assert not isinstance(
                responses_empty[0], StreamChunk
            ), "Should not be a StreamChunk"
            assert responses_empty[0].response_output == ""
            assert responses_empty[0].id == "test-empty-order"

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()
