"""Integration tests for the HTTP endpoint client using echo server."""

import asyncio
import time

import pytest
import pytest_asyncio
from inference_endpoint.core.types import ChatCompletionQuery, QueryResult
from inference_endpoint.endpoint_client import (
    AsyncHTTPEndpointClient,
    HTTPEndpointClient,
)
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)


class TestHTTPEndpointClientIntegration:
    """Integration tests for HTTPEndpointClient with echo server."""

    @pytest.fixture
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )

    @pytest.fixture
    def aiohttp_config(self):
        """Create aiohttp configuration."""
        return AioHttpConfig(
            client_timeout_total=10.0,
            tcp_connector_limit=100,
        )

    @pytest.fixture
    def zmq_config(self):
        """Create ZMQ configuration with unique addresses."""
        timestamp = int(time.time() * 1000)
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_http_worker_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_http_responses_{timestamp}",
        )

    @pytest_asyncio.fixture
    async def http_client(self, http_config, aiohttp_config, zmq_config):
        """Create and start HTTP endpoint client."""
        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        )
        await client.start()
        yield client
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_single_request_response(self, http_client, mock_http_echo_server):
        """Test sending a single request and receiving response."""
        # Track responses
        responses = []

        async def handle_response(result: QueryResult):
            responses.append(result)

        http_client.complete_callback = handle_response

        # Send request
        query = ChatCompletionQuery(
            id="single-test-123",
            prompt="Hello, echo server!",
            model="gpt-3.5-turbo",
        )

        await http_client.send_request(query)

        # Wait for response
        await asyncio.sleep(1.0)

        # Verify response
        assert len(responses) == 1
        result = responses[0]
        assert result.query_id == "single-test-123"
        assert (
            result.response_output == "Hello, echo server!"
        )  # Echo server returns prompt

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, http_client):
        """Test multiple concurrent requests."""
        responses = {}

        async def handle_response(result: QueryResult):
            responses[result.query_id] = result

        http_client.complete_callback = handle_response

        # Send multiple requests
        num_requests = 10
        for i in range(num_requests):
            query = ChatCompletionQuery(
                id=f"concurrent-{i}",
                prompt=f"Concurrent request {i}",
                model="gpt-3.5-turbo",
            )
            await http_client.send_request(query)

        # Wait for all responses
        await asyncio.sleep(2.0)

        # Verify all responses received
        assert len(responses) == num_requests
        for i in range(num_requests):
            query_id = f"concurrent-{i}"
            assert query_id in responses
            assert responses[query_id].response_output == f"Concurrent request {i}"

    @pytest.mark.asyncio
    async def test_worker_distribution(self, http_client):
        """Test that requests are distributed across workers."""
        # This test verifies round-robin distribution
        responses = []

        async def handle_response(result: QueryResult):
            responses.append(result)

        http_client.complete_callback = handle_response

        # Send exactly num_workers * 2 requests
        num_requests = http_client.config.num_workers * 2
        for i in range(num_requests):
            query = ChatCompletionQuery(
                id=f"worker-dist-{i}",
                prompt=f"Worker distribution test {i}",
                model="gpt-3.5-turbo",
            )
            await http_client.send_request(query)

        # Wait for responses
        await asyncio.sleep(2.0)

        # Verify all responses received
        assert len(responses) == num_requests

    @pytest.mark.asyncio
    async def test_large_payload(self, http_client):
        """Test handling large payloads."""
        responses = []

        async def handle_response(result: QueryResult):
            responses.append(result)

        http_client.complete_callback = handle_response

        # Create large prompt
        large_prompt = "Large prompt: " + "x" * 5000
        query = ChatCompletionQuery(
            id="large-payload-test",
            prompt=large_prompt,
            model="gpt-3.5-turbo",
            max_tokens=1000,
        )

        await http_client.send_request(query)

        # Wait for response
        await asyncio.sleep(1.0)

        # Verify response
        assert len(responses) == 1
        assert responses[0].query_id == "large-payload-test"
        assert responses[0].response_output == large_prompt

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test concurrency limiting."""
        # Create client with concurrency limit
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:12345/v1/chat/completions",
            num_workers=2,
            max_concurrency=3,  # Limit to 3 concurrent requests
        )

        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_concurrency_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_concurrency_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Track when requests are sent
            send_times = []

            async def track_send(original_send):
                async def wrapper(query):
                    send_times.append(time.time())
                    return await original_send(query)

                return wrapper

            # Patch the send implementation
            for socket in client.worker_push_sockets:
                original = socket.send
                socket.send = await track_send(original)

            # Send 6 requests (should be limited to 3 at a time)
            tasks = []
            for i in range(6):
                query = ChatCompletionQuery(
                    id=f"concurrency-{i}",
                    prompt=f"Test {i}",
                    model="gpt-3.5-turbo",
                )
                task = asyncio.create_task(client.send_request(query))
                tasks.append(task)

            # Wait for all to complete
            await asyncio.gather(*tasks)

            # Verify concurrency was limited
            # With limit of 3, we should see two "waves" of sends
            assert len(send_times) == 6

        finally:
            await client.shutdown()


class TestAsyncHTTPEndpointClientIntegration:
    """Integration tests for AsyncHTTPEndpointClient."""

    @pytest.fixture
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
        )

    @pytest.fixture
    def zmq_config(self):
        """Create ZMQ configuration with unique addresses."""
        timestamp = int(time.time() * 1000)
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_async_worker_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_async_responses_{timestamp}",
        )

    @pytest_asyncio.fixture
    async def async_client(self, http_config, zmq_config):
        """Create and start async HTTP client."""
        client = AsyncHTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )
        await client.start()
        yield client
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_future_based_request(self, async_client):
        """Test future-based request/response pattern."""
        query = ChatCompletionQuery(
            id="future-test-123",
            prompt="Testing future pattern",
            model="gpt-3.5-turbo",
        )

        # Send request and get future
        future = await async_client.send_request(query)

        # Wait for result
        result = await future

        # Verify response
        assert result.query_id == "future-test-123"
        assert result.response_output == "Testing future pattern"

    @pytest.mark.asyncio
    async def test_multiple_futures(self, async_client):
        """Test handling multiple futures."""
        # Send multiple requests
        queries = []
        futures = []

        for i in range(5):
            query = ChatCompletionQuery(
                id=f"multi-future-{i}",
                prompt=f"Future test {i}",
                model="gpt-3.5-turbo",
            )
            queries.append(query)
            future = await async_client.send_request(query)
            futures.append(future)

        # Wait for all futures
        results = await asyncio.gather(*futures)

        # Verify all responses
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.query_id == f"multi-future-{i}"
            assert result.response_output == f"Future test {i}"

    @pytest.mark.asyncio
    async def test_future_with_callback(self, async_client, http_config):
        """Test using both future and callback patterns."""
        callback_results = []

        async def user_callback(result: QueryResult):
            callback_results.append(result)

        # Create unique ZMQ config for second client to avoid conflicts
        timestamp = int(time.time() * 1000)
        unique_zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_callback_worker_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_callback_responses_{timestamp}",
        )

        # Create client with callback
        client = AsyncHTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=unique_zmq_config,
            complete_callback=user_callback,
        )
        await client.start()

        try:
            query = ChatCompletionQuery(
                id="callback-future-test",
                prompt="Testing both patterns",
                model="gpt-3.5-turbo",
            )

            # Send request
            future = await client.send_request(query)

            # Wait for future
            future_result = await future

            # Both should have the result
            assert future_result.query_id == "callback-future-test"
            assert len(callback_results) == 1
            assert callback_results[0].query_id == "callback-future-test"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_error_propagation(self, http_config, zmq_config):
        """Test error propagation through futures."""
        # Create client with invalid endpoint
        invalid_config = HTTPClientConfig(
            endpoint_url="http://invalid-endpoint:99999/v1/chat/completions",
            num_workers=1,
        )

        client = AsyncHTTPEndpointClient(
            config=invalid_config,
            aiohttp_config=AioHttpConfig(client_timeout_total=1.0),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            query = ChatCompletionQuery(
                id="error-test",
                prompt="This should fail",
                model="gpt-3.5-turbo",
            )

            # Send request
            future = await client.send_request(query)

            # Wait for error - expecting generic Exception with connection error
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(future, timeout=3.0)

            # Verify the error message contains the invalid endpoint
            assert "invalid-endpoint" in str(exc_info.value)

        finally:
            await client.shutdown()
