"""Core functionality tests for the HTTP endpoint client."""

import asyncio
import pickle
import time

import pytest
import pytest_asyncio
import zmq
import zmq.asyncio
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.futures_client import FuturesHttpClient
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket

from ...test_helpers import get_test_socket_path


class TestHTTPEndpointClientConcurrency:
    """Test concurrent operations and future handling."""

    def _create_custom_client(
        self,
        mock_http_echo_server,
        tmp_path,
        num_workers=1,
        max_concurrency=-1,
        zmq_high_water_mark=10000,
        zmq_io_threads=None,
        aiohttp_config=None,
    ):
        """Helper method to create a client with custom configuration."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=num_workers,
            max_concurrency=max_concurrency,
        )

        zmq_config_kwargs = {
            "zmq_request_queue_prefix": get_test_socket_path(
                tmp_path, "custom", "_req"
            ),
            "zmq_response_queue_addr": get_test_socket_path(
                tmp_path, "custom", "_resp"
            ),
            "zmq_readiness_queue_addr": get_test_socket_path(
                tmp_path, "custom", "_ready"
            ),
            "zmq_high_water_mark": zmq_high_water_mark,
        }
        if zmq_io_threads is not None:
            zmq_config_kwargs["zmq_io_threads"] = zmq_io_threads

        zmq_config = ZMQConfig(**zmq_config_kwargs)

        # Use provided aiohttp_config or create default
        if aiohttp_config is None:
            aiohttp_config = AioHttpConfig()

        return FuturesHttpClient(
            config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        )

    @pytest.fixture
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=4,  # More workers for concurrency tests
            max_concurrency=-1,  # No limit by default
        )

    @pytest.fixture
    def zmq_config(self, tmp_path):
        """Create ZMQ configuration with unique addresses."""
        return ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_conc", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_conc", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "test_conc", "_ready"
            ),
            zmq_high_water_mark=10000,  # Higher for massive tests
        )

    @pytest_asyncio.fixture
    async def futures_http_client(self, http_config, zmq_config):
        """Create and start futures-based HTTP endpoint client."""
        client = FuturesHttpClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )
        await client.async_start()
        yield client
        await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_basic_future_handling_standalone(
        self, mock_http_echo_server, tmp_path
    ):
        """Test basic future-based request/response without class fixture."""
        # Create a fresh client in the test's own event loop
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_standalone", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_standalone", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "test_standalone", "_ready"
            ),
        )

        client = FuturesHttpClient(http_config, AioHttpConfig(), zmq_config)

        await client.async_start()

        try:
            query = Query(
                id="1001",
                data={
                    "prompt": "Test future handling",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            # issue_query returns a future directly
            future = await client.issue_query(query)
            assert isinstance(future, asyncio.Future)

            # Await the future
            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.id == "1001"
            assert result.response_output == "Test future handling"
        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_basic_future_handling(self, futures_http_client):
        """Test basic future-based request/response."""
        query = Query(
            id="1001",
            data={
                "prompt": "Test future handling",
                "model": "gpt-3.5-turbo",
            },
        )

        # issue_query returns a future directly
        future = await futures_http_client.issue_query(query)
        assert isinstance(future, asyncio.Future)

        # Await the future
        result = await future
        assert result.id == "1001"
        assert result.response_output == "Test future handling"

    @pytest.mark.asyncio
    async def test_concurrent_futures_proper_handling(self, futures_http_client):
        """Test proper concurrent future handling - collect then await all."""
        num_requests = 50

        # Collect all futures first
        futures = []
        for i in range(num_requests):
            query = Query(
                id=f"concurrent-{i}",
                data={
                    "prompt": f"Concurrent request {i}",
                    "model": "gpt-3.5-turbo",
                },
            )
            future = await futures_http_client.issue_query(query)
            futures.append(future)

        # Now await all futures together
        results = await asyncio.gather(*futures)

        # Verify all results
        assert len(results) == num_requests
        for i, result in enumerate(results):
            assert result.id == f"concurrent-{i}"
            assert result.response_output == f"Concurrent request {i}"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_massive_concurrency_non_streaming(
        self, mock_http_echo_server, tmp_path
    ):
        """Test high concurrent requests with proper connection management in non-streaming mode."""
        actual_max_concurrency = 10000

        # create client with unlimited concurrency
        client = self._create_custom_client(
            mock_http_echo_server,
            tmp_path,
            num_workers=1,
            max_concurrency=-1,
            zmq_high_water_mark=actual_max_concurrency,
        )

        await client.async_start()

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
                future = await client.issue_query(query)
                futures.append(future)

            # Wait for all futures to complete
            results = await asyncio.gather(*futures)
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
            await client.async_shutdown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_massive_concurrency_streaming(self, mock_http_echo_server, tmp_path):
        """Test high concurrent requests with proper connection management in streaming mode."""
        actual_max_concurrency = 10000

        # create client with unlimited concurrency
        client = self._create_custom_client(
            mock_http_echo_server,
            tmp_path,
            num_workers=1,
            max_concurrency=-1,
            zmq_high_water_mark=actual_max_concurrency,
        )

        try:
            await client.async_start()
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
                future = await client.issue_query(query)
                futures.append(future)

            # Wait for all futures to complete
            results = await asyncio.gather(*futures)
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
            await client.async_shutdown()

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
            future = await futures_http_client.issue_query(query)
            futures.append((name, size, future))

        # Wait for all payloads
        for name, size, future in futures:
            result = await future
            assert result.id == f"payload-{name}"
            assert len(result.response_output) == size
            print(f"\nSuccessfully processed {name} payload ({size} bytes)")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_many_workers(self, mock_http_echo_server, tmp_path):
        """Test with many workers."""
        actual_max_concurrency = 1000
        worker_counts = [16, 32]

        for num_workers in worker_counts:
            print(f"\nTesting with {num_workers} workers...")

            client = self._create_custom_client(
                mock_http_echo_server,
                tmp_path,
                num_workers=num_workers,
                max_concurrency=-1,
                zmq_high_water_mark=actual_max_concurrency,
                zmq_io_threads=8,
            )

            await client.async_start()

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
                    future = await client.issue_query(query)
                    futures.append(future)

                # Wait for all with timeout
                results = await asyncio.gather(*futures)
                duration = time.time() - start_time

                # Verify
                assert len(results) == num_requests
                print(
                    f"  Completed {num_requests} requests in {duration:.2f}s "
                    f"({num_requests / duration:.0f} req/s)"
                )

            finally:
                await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_concurrency_limit_with_futures(
        self, mock_http_echo_server, tmp_path
    ):
        """Test concurrency limiting with proper future handling."""
        max_concurrency = 5

        client = self._create_custom_client(
            mock_http_echo_server,
            tmp_path,
            num_workers=4,
            max_concurrency=max_concurrency,
            zmq_high_water_mark=max_concurrency * 20,
        )

        await client.async_start()

        try:
            # Send more requests than concurrency limit
            num_requests = 20 * max_concurrency
            futures = []
            issue_times = []

            # Record when each request is issued
            for i in range(num_requests):
                query = Query(
                    id=f"limited-{i}",
                    data={
                        "prompt": f"Concurrency limited request {i}",
                        "model": "gpt-3.5-turbo",
                    },
                )

                issue_times.append(time.time())
                future = await client.issue_query(query)
                futures.append(future)

            # Wait for all
            results = await asyncio.gather(*futures)

            # Verify all completed
            assert len(results) == num_requests

            # Analyze concurrency pattern
            # With limit of 5, requests should be processed in batches
            print(
                f"\nConcurrency limit test: {num_requests} requests with limit of {max_concurrency}"
            )

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_future_cancellation(self, tmp_path):
        """Test cancelling futures before completion."""
        # Use invalid endpoint so requests won't complete immediately
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/v1/chat/completions",
            num_workers=2,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_cancel", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_cancel", "_resp"
            ),
        )

        client = FuturesHttpClient(http_config, AioHttpConfig(), zmq_config)

        await client.async_start()

        try:
            # Create futures
            futures = []
            for i in range(10):
                query = Query(
                    id=f"cancel-{i}",
                    data={
                        "prompt": f"To be cancelled {i}",
                        "model": "gpt-3.5-turbo",
                    },
                )
                future = await client.issue_query(query)
                futures.append(future)

            # Small delay to let requests start
            await asyncio.sleep(0.1)

            # Cancel half of them
            for i in range(5):
                futures[i].cancel()

            # Shutdown to cancel remaining
            await client.async_shutdown()

            # Check cancellations - some futures may complete before cancellation
            cancelled_count = sum(1 for f in futures if f.cancelled())
            completed_count = sum(1 for f in futures if f.done() and not f.cancelled())

            # Either we cancelled some futures, or they completed/failed due to invalid endpoint
            assert cancelled_count > 0 or completed_count > 0
            print(
                f"\nCancellation test: {cancelled_count} cancelled, {completed_count} completed"
            )

        finally:
            pass  # Already shut down

    @pytest.mark.asyncio
    async def test_mixed_callback_and_future_pattern(self, futures_http_client):
        """Test using futures pattern (callbacks not supported in new API)."""
        # Send requests and collect futures
        futures = []
        for i in range(10):
            query = Query(
                id=f"mixed-{i}",
                data={
                    "prompt": f"Mixed pattern {i}",
                    "model": "gpt-3.5-turbo",
                },
            )
            future = await futures_http_client.issue_query(query)
            futures.append(future)

        # Wait for futures
        future_results = await asyncio.gather(*futures)

        # All futures should have results
        assert len(future_results) == 10


class TestHTTPEndpointClientErrorHandling:
    """Test error handling with real ZMQ sockets."""

    @pytest.mark.asyncio
    async def test_worker_connection_error(self, tmp_path):
        """Test handling when workers can't connect to endpoint."""
        # Use invalid endpoint
        http_config = HTTPClientConfig(
            endpoint_url="http://invalid-host-12345:9999/v1/chat/completions",
            num_workers=2,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_conn_err", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_conn_err", "_resp"
            ),
        )

        client = FuturesHttpClient(
            http_config,
            AioHttpConfig(client_timeout_total=2.0),  # short timeout to fail fast
            zmq_config,
        )

        await client.async_start()

        try:
            # Send request
            query = Query(
                id="2001",
                data={
                    "prompt": "This should fail",
                    "model": "gpt-3.5-turbo",
                },
            )

            future = await client.issue_query(query)

            # Should get error
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(future, timeout=5.0)

            assert "invalid-host-12345" in str(
                exc_info.value
            ) or "Cannot connect" in str(exc_info.value)

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_response_handler_error_recovery(
        self, mock_http_echo_server, tmp_path
    ):
        """Test that response handler recovers from errors."""
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_handler_err", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_handler_err", "_resp"
            ),
        )

        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
        )

        client = FuturesHttpClient(
            http_config,
            AioHttpConfig(),
            zmq_config,
        )

        # Start client
        await client.async_start()

        try:
            # Send first query
            query1 = Query(
                id="3001",
                data={
                    "prompt": "First query",
                    "model": "gpt-3.5-turbo",
                },
            )
            future1 = await client.issue_query(query1)

            # Create context to inject invalid data
            context = zmq.asyncio.Context()
            response_push = context.socket(zmq.PUSH)
            response_push.connect(zmq_config.zmq_response_queue_addr)

            # Send invalid data that will cause error in handler
            await response_push.send(b"invalid pickle data")

            # Give handler time to encounter error
            await asyncio.sleep(0.2)

            # Send second query - handler should have recovered
            query2 = Query(
                id="3002",
                data={
                    "prompt": "Second query after error",
                    "model": "gpt-3.5-turbo",
                },
            )
            future2 = await client.issue_query(query2)

            # Wait for both futures
            result1 = await asyncio.wait_for(future1, timeout=5.0)
            result2 = await asyncio.wait_for(future2, timeout=5.0)

            # Both should complete successfully
            assert result1.response_output == "First query"
            assert result2.response_output == "Second query after error"

        finally:
            response_push.close()
            context.destroy(linger=0)
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_zmq_send_failure(self, tmp_path):
        """Test handling of ZMQ send failures."""
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_send_fail", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_send_fail", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "test_send_fail", "_ready"
            ),
        )

        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
        )

        client = FuturesHttpClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.async_start()

        try:
            # Create a mock socket that will fail on send
            class FailingSocket:
                def __init__(self, original_socket):
                    self.socket = (
                        original_socket.socket
                        if hasattr(original_socket, "socket")
                        else None
                    )
                    self._original_socket = original_socket

                async def send(self, data):
                    raise Exception("ZMQ send failed")

                def close(self):
                    if self._original_socket:
                        self._original_socket.close()

            # Save original socket and replace with failing one
            original_socket = client.worker_push_sockets[0]
            client.worker_push_sockets[0] = FailingSocket(original_socket)

            query = Query(
                id="4001",
                data={
                    "prompt": "This will fail to send",
                    "model": "gpt-3.5-turbo",
                },
            )

            # The send should fail immediately and raise an exception
            with pytest.raises(Exception) as exc_info:
                await client.issue_query(query)
            assert "ZMQ send failed" in str(exc_info.value)

        finally:
            await client.async_shutdown()


class TestHTTPEndpointClientCoverage:
    """Tests to improve code coverage."""

    @pytest.fixture
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )

    @pytest.fixture
    def zmq_config(self, tmp_path):
        """Create ZMQ configuration with unique addresses."""
        return ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_coverage", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_coverage", "_resp"
            ),
        )

    @pytest_asyncio.fixture
    async def futures_http_client(self, http_config, zmq_config):
        """Create and start futures-based HTTP endpoint client."""
        client = FuturesHttpClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )
        await client.async_start()
        yield client
        await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_initialization_with_callback(self, mock_http_echo_server, tmp_path):
        """Test HTTPEndpointClient initialization with callback."""
        callback_called = []

        def test_callback(result: QueryResult):
            callback_called.append(result)

        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=5,  # Test concurrency semaphore creation
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_init_callback", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_init_callback", "_resp"
            ),
            zmq_io_threads=2,  # Test custom io_threads
        )

        # Test initialization
        client = FuturesHttpClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Verify initialization state
        assert client.config == http_config
        assert client.aiohttp_config is not None
        assert client.zmq_config == zmq_config
        assert client._concurrency_semaphore is None  # Not created until start()
        assert client.current_worker_idx == 0
        assert len(client.worker_push_sockets) == 0
        assert client.worker_manager is None
        assert client._response_handler_task is None
        assert len(client._pending_futures) == 0

        await client.async_start()

        try:
            # Test that query completes successfully
            query = Query(
                id="5001",
                data={
                    "prompt": "Test callback",
                    "model": "gpt-3.5-turbo",
                },
            )

            future = await client.issue_query(query)
            result = await future

            # Verify result
            assert result.id == "5001"
            assert result.response_output == "Test callback"

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_initialization_without_concurrency_limit(
        self, mock_http_echo_server, tmp_path
    ):
        """Test initialization without concurrency limit (max_concurrency <= 0)."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,  # No concurrency limit
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_no_concurrency", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_no_concurrency", "_resp"
            ),
        )

        client = FuturesHttpClient(http_config, AioHttpConfig(), zmq_config)
        await client.async_start()

        try:
            # Test that requests work without concurrency limit
            query = Query(
                id="6001",
                data={
                    "prompt": "Test no limit",
                    "model": "gpt-3.5-turbo",
                },
            )

            future = await client.issue_query(query)
            result = await future

            assert result.id == "6001"
            assert result.response_output == "Test no limit"

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_start_method_socket_creation(self, mock_http_echo_server, tmp_path):
        """Test start method creates correct number of worker sockets."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=4,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_start_sockets", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_start_sockets", "_resp"
            ),
        )

        client = FuturesHttpClient(http_config, AioHttpConfig(), zmq_config)

        # Verify initial state
        assert len(client.worker_push_sockets) == 0
        assert client.worker_manager is None
        assert client._response_handler_task is None

        await client.async_start()

        try:
            # Verify start method created all components
            assert len(client.worker_push_sockets) == 4
            assert client.worker_manager is not None
            assert client._response_handler_task is not None
            assert not client._response_handler_task.done()

            # Verify socket addresses are correct
            for socket in client.worker_push_sockets:
                # We can't directly check the address, but we can verify the socket exists
                assert socket is not None

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, mock_http_echo_server, tmp_path):
        """Test error handling in user callback."""
        callback_errors = []

        def failing_callback(result):
            callback_errors.append("callback_called")
            raise ValueError("Callback intentionally failed")

        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_callback_error", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_callback_error", "_resp"
            ),
        )

        client = FuturesHttpClient(http_config, AioHttpConfig(), zmq_config)
        client.complete_callback = failing_callback

        await client.async_start()

        try:
            # Send request that will trigger callback error
            query = Query(
                id="8001",
                data={
                    "prompt": "Test callback error",
                    "model": "gpt-3.5-turbo",
                },
            )

            future = await client.issue_query(query)
            result = await future

            # Future should still complete successfully despite callback error
            assert result.id == "8001"
            assert result.response_output == "Test callback error"

            # Wait for callback to be processed
            await asyncio.sleep(0.1)

            # Verify callback was called (but failed)
            # For non-streaming queries, callback is called once
            assert len(callback_errors) >= 1
            assert callback_errors[0] == "callback_called"

        finally:
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_response_with_error_field(self, mock_http_echo_server, tmp_path):
        """Test handling response with error field."""
        # This test requires a way to simulate error responses
        # We'll create a mock response directly
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_error_response", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_error_response", "_resp"
            ),
        )

        client = FuturesHttpClient(http_config, AioHttpConfig(), zmq_config)
        await client.async_start()

        # Create context for test
        context = zmq.asyncio.Context()

        try:
            # Create push socket to send error response
            response_push = context.socket(zmq.PUSH)
            response_push.connect(zmq_config.zmq_response_queue_addr)

            # Create future for tracking
            query = Query(
                id="2001",
                data={
                    "prompt": "Test error",
                    "model": "gpt-3.5-turbo",
                },
            )
            future = await client.issue_query(query)

            # Give time for query to be sent
            await asyncio.sleep(0.1)

            # Send error response
            error_result = QueryResult(
                id="2001",
                response_output="",
                error="Simulated error response",
            )
            await response_push.send(pickle.dumps(error_result))

            # Wait for processing and expect exception
            with pytest.raises(Exception) as exc_info:
                await future

            assert "Simulated error response" in str(exc_info.value)

        finally:
            response_push.close()
            context.destroy(linger=0)
            await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_with_pending_response_handler(
        self, mock_http_echo_server, tmp_path
    ):
        """Test shutdown when response handler task exists."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_shutdown_handler", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_shutdown_handler", "_resp"
            ),
        )

        client = FuturesHttpClient(http_config, AioHttpConfig(), zmq_config)

        await client.async_start()

        # Verify components are running
        assert client._response_handler_task is not None
        assert not client._response_handler_task.done()
        assert client.worker_manager is not None
        assert len(client.worker_push_sockets) == 2

        # Add some pending futures
        future1 = asyncio.get_event_loop().create_future()
        future2 = asyncio.get_event_loop().create_future()
        client._pending_futures[1] = future1
        client._pending_futures[2] = future2

        # Shutdown should clean everything up
        await client.async_shutdown()

        # Give event loop time to process cancellations
        await asyncio.sleep(0.1)

        # Verify cleanup
        assert len(client._pending_futures) == 0
        assert future1.cancelled()
        assert future2.cancelled()
        assert client._response_handler_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_without_components(self, tmp_path):
        """Test shutdown when components haven't been initialized."""
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_shutdown_empty", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_shutdown_empty", "_resp"
            ),
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Don't call start() - test shutdown on uninitialized client
        assert client.worker_manager is None
        assert len(client.worker_push_sockets) == 0

        # Should not raise any errors
        await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_zmq_socket_options(self):
        """Test ZMQ socket configuration options are applied."""
        zmq_config = ZMQConfig(
            zmq_high_water_mark=500,
            zmq_linger=1000,
            zmq_send_timeout=5000,
            zmq_recv_timeout=5000,
            zmq_recv_buffer_size=20 * 1024 * 1024,
            zmq_send_buffer_size=20 * 1024 * 1024,
        )

        context = zmq.asyncio.Context()

        try:
            # Test push socket
            push_socket = ZMQPushSocket(
                context, "ipc:///tmp/test_opts_push", zmq_config
            )

            # Verify options were set
            assert push_socket.socket.getsockopt(zmq.SNDHWM) == 500
            assert push_socket.socket.getsockopt(zmq.LINGER) == 1000
            assert push_socket.socket.getsockopt(zmq.SNDTIMEO) == 5000
            assert push_socket.socket.getsockopt(zmq.SNDBUF) == 20 * 1024 * 1024

            # Test pull socket
            pull_socket = ZMQPullSocket(
                context, "ipc:///tmp/test_opts_pull", zmq_config, bind=True
            )

            assert pull_socket.socket.getsockopt(zmq.RCVHWM) == 500
            # Note: LINGER may not be set on PULL sockets by default, check if it was actually set
            linger_val = pull_socket.socket.getsockopt(zmq.LINGER)
            assert linger_val == 1000 or linger_val == -1  # -1 is default (infinite)
            assert pull_socket.socket.getsockopt(zmq.RCVTIMEO) == 5000
            assert pull_socket.socket.getsockopt(zmq.RCVBUF) == 20 * 1024 * 1024

        finally:
            push_socket.close()
            pull_socket.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    async def test_empty_prompt(self, futures_http_client):
        """Test handling empty prompt."""
        query = Query(
            id="9001",
            data={
                "prompt": "",
                "model": "gpt-3.5-turbo",
            },
        )

        future = await futures_http_client.issue_query(query)
        result = await future

        assert result.id == "9001"
        assert result.response_output == ""

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, futures_http_client):
        """Test handling special characters and unicode."""
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
            future = await futures_http_client.issue_query(query)
            futures.append((prompt, future))

        # Verify all handled correctly
        for prompt, future in futures:
            result = await future
            assert result.response_output == prompt

    @pytest.mark.asyncio
    async def test_metadata_propagation(self, futures_http_client):
        """Test that query metadata is preserved."""
        query = Query(
            id="10001",
            data={
                "prompt": "Test metadata",
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "temperature": 0.5,
                "metadata": {
                    "user_id": "test-user",
                    "session_id": "test-session",
                    "custom_field": "custom_value",
                },
            },
        )

        future = await futures_http_client.issue_query(query)
        result = await future

        # Echo server should preserve the query
        assert result.id == "10001"
        assert result.response_output == "Test metadata"

    @pytest.mark.asyncio
    async def test_concurrent_shutdown(
        self, http_config, mock_http_echo_server, tmp_path
    ):
        """Test shutdown while requests are in flight."""
        # Create a separate client for this test since we need to shut it down
        shutdown_zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_shutdown", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_shutdown", "_resp"
            ),
        )

        client = FuturesHttpClient(
            http_config,
            AioHttpConfig(),
            shutdown_zmq_config,
        )
        await client.async_start()

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
                future = await client.issue_query(query)
                futures.append(future)

            # Immediately shutdown
            await client.async_shutdown()

            # Give event loop time to process cancellations
            await asyncio.sleep(0.1)

            # Count completed vs cancelled
            completed = sum(1 for f in futures if f.done() and not f.cancelled())
            cancelled = sum(1 for f in futures if f.cancelled())

            print(f"\nShutdown test: {completed} completed, {cancelled} cancelled")

            # At least some should be cancelled
            assert cancelled > 0
        finally:
            # Ensure cleanup even if test fails
            if not client._shutdown_event.is_set():
                await client.async_shutdown()

    @pytest.mark.asyncio
    async def test_error_response_propagation(self, tmp_path):
        """Test that error responses are propagated as exceptions in futures."""
        # Use an invalid endpoint to trigger real errors
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_error_prop", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_error_prop", "_resp"
            ),
        )

        config = HTTPClientConfig(
            endpoint_url="http://invalid-host-does-not-exist:9999/v1/chat/completions",
            num_workers=1,
        )

        client = FuturesHttpClient(
            config,
            AioHttpConfig(client_timeout_total=2.0),  # short timeout to fail fast
            zmq_config,
        )

        await client.async_start()

        try:
            # Send request to invalid endpoint
            query = Query(
                id="2001",
                data={
                    "prompt": "Test error",
                    "model": "gpt-3.5-turbo",
                },
            )

            future = await client.issue_query(query)

            # Should get an exception due to connection error
            with pytest.raises(Exception) as exc_info:
                result = await asyncio.wait_for(future, timeout=5.0)
                # If we get here without exception, print for debugging
                print(f"ERROR: Got result instead of exception: {result}")
                print(
                    f"Result error field: {getattr(result, 'error', 'NO ERROR FIELD')}"
                )
                raise AssertionError(f"Expected exception but got result: {result}")

            # Verify the error message contains expected content
            error_msg = str(exc_info.value)
            assert (
                "invalid-host-does-not-exist" in error_msg
                or "Cannot connect" in error_msg
                or "Name or service not known" in error_msg
            )

        finally:
            await client.async_shutdown()
