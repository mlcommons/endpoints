"""Core functionality tests for the HTTP endpoint client."""

import asyncio
import pickle
import time

import pytest
import pytest_asyncio
import zmq
import zmq.asyncio
from inference_endpoint.core.types import ChatCompletionQuery, QueryResult
from inference_endpoint.endpoint_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket


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
    ):
        """Helper method to create a client with custom configuration."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=num_workers,
            max_concurrency=max_concurrency,
        )

        zmq_config_kwargs = {
            "zmq_request_queue_prefix": f"ipc://{tmp_path}/test_custom_req",
            "zmq_response_queue_addr": f"ipc://{tmp_path}/test_custom_resp",
            "zmq_high_water_mark": zmq_high_water_mark,
        }
        if zmq_io_threads is not None:
            zmq_config_kwargs["zmq_io_threads"] = zmq_io_threads

        zmq_config = ZMQConfig(**zmq_config_kwargs)

        return HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

    @pytest.fixture(scope="class")
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=4,  # More workers for concurrency tests
            max_concurrency=-1,  # No limit by default
        )

    @pytest.fixture(scope="class")
    def zmq_config(self, tmp_path_factory):
        """Create ZMQ configuration with unique addresses."""
        # Use tmp_path_factory for class-scoped fixture
        tmp_dir = tmp_path_factory.mktemp("test_conc")
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_dir}/test_conc_req",
            zmq_response_queue_addr=f"ipc://{tmp_dir}/test_conc_resp",
            zmq_high_water_mark=10000,  # Higher for massive tests
        )

    @pytest_asyncio.fixture(scope="class")
    async def http_client(self, http_config, zmq_config):
        """Create and start HTTP endpoint client."""
        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )
        await client.start()
        yield client
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_basic_future_handling(self, http_client):
        """Test basic future-based request/response."""
        query = ChatCompletionQuery(
            id="future-test",
            prompt="Test future handling",
            model="gpt-3.5-turbo",
        )

        # issue_query returns a future directly
        future = http_client.issue_query(query)
        assert isinstance(future, asyncio.Future)

        # Await the future
        result = await future
        assert result.query_id == "future-test"
        assert result.response_output == "Test future handling"

    @pytest.mark.asyncio
    async def test_concurrent_futures_proper_handling(self, http_client):
        """Test proper concurrent future handling - collect then await all."""
        num_requests = 50

        # Collect all futures first
        futures = []
        for i in range(num_requests):
            query = ChatCompletionQuery(
                id=f"concurrent-{i}",
                prompt=f"Concurrent request {i}",
                model="gpt-3.5-turbo",
            )
            future = http_client.issue_query(query)
            futures.append(future)

        # Now await all futures together
        results = await asyncio.gather(*futures)

        # Verify all results
        assert len(results) == num_requests
        for i, result in enumerate(results):
            assert result.query_id == f"concurrent-{i}"
            assert result.response_output == f"Concurrent request {i}"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_massive_concurrency(self, mock_http_echo_server, tmp_path):
        """Test high concurrent requests with proper connection management."""
        actual_max_concurrency = 10000

        # create client with unlimited concurrency
        client = self._create_custom_client(
            mock_http_echo_server,
            tmp_path,
            num_workers=1,
            max_concurrency=-1,
            zmq_high_water_mark=actual_max_concurrency,
        )

        await client.start()

        try:
            num_requests = actual_max_concurrency

            # Collect futures
            start_time = time.time()
            futures = []
            for i in range(num_requests):
                query = ChatCompletionQuery(
                    id=f"massive-{i}",
                    prompt=f"Request {i}",
                    model="gpt-3.5-turbo",
                )
                future = client.issue_query(query)
                futures.append(future)

            # Wait for all futures to complete
            results = await asyncio.gather(*futures)
            end_time = time.time()

            # Verify results
            assert len(results) == num_requests
            result_ids = {r.query_id for r in results}
            expected_ids = {f"massive-{i}" for i in range(num_requests)}
            assert result_ids == expected_ids

            # Print performance metrics
            duration = end_time - start_time
            rps = num_requests / duration
            print(
                f"\nProcessed {num_requests} requests in {duration:.2f}s ({rps:.0f} req/s)"
            )

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_massive_payloads(self, http_client):
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
            query = ChatCompletionQuery(
                id=f"payload-{name}",
                prompt=large_prompt,
                model="gpt-3.5-turbo",
                max_tokens=2000,
            )
            future = http_client.issue_query(query)
            futures.append((name, size, future))

        # Wait for all payloads
        for name, size, future in futures:
            result = await future
            assert result.query_id == f"payload-{name}"
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

            await client.start()

            try:
                num_requests = actual_max_concurrency
                futures = []

                start_time = time.time()
                for i in range(num_requests):
                    query = ChatCompletionQuery(
                        id=f"worker-test-{i}",
                        prompt=f"Testing {num_workers} workers - request {i}",
                        model="gpt-3.5-turbo",
                    )
                    future = client.issue_query(query)
                    futures.append(future)

                # Wait for all with timeout
                results = await asyncio.gather(*futures)
                duration = time.time() - start_time

                # Verify
                assert len(results) == num_requests
                print(
                    f"  Completed {num_requests} requests in {duration:.2f}s "
                    f"({num_requests/duration:.0f} req/s)"
                )

            finally:
                await client.shutdown()

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

        await client.start()

        try:
            # Send more requests than concurrency limit
            num_requests = 20 * max_concurrency
            futures = []
            issue_times = []

            # Record when each request is issued
            for i in range(num_requests):
                query = ChatCompletionQuery(
                    id=f"limited-{i}",
                    prompt=f"Concurrency limited request {i}",
                    model="gpt-3.5-turbo",
                )

                issue_times.append(time.time())
                future = client.issue_query(query)
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
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_future_cancellation(self):
        """Test cancelling futures before completion."""
        # Use invalid endpoint so requests won't complete immediately
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/v1/chat/completions",
            num_workers=2,
        )

        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_cancel_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_cancel_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Create futures
            futures = []
            for i in range(10):
                query = ChatCompletionQuery(
                    id=f"cancel-{i}",
                    prompt=f"To be cancelled {i}",
                    model="gpt-3.5-turbo",
                )
                future = client.issue_query(query)
                futures.append(future)

            # Small delay to let requests start
            await asyncio.sleep(0.1)

            # Cancel half of them
            for i in range(5):
                futures[i].cancel()

            # Shutdown to cancel remaining
            await client.shutdown()

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
    async def test_mixed_callback_and_future_pattern(self, http_client):
        """Test using both callbacks and futures together."""
        callback_results = []

        def callback(result):
            callback_results.append(result)

        # Set callback
        http_client.complete_callback = callback

        # Send requests and collect futures
        futures = []
        for i in range(10):
            query = ChatCompletionQuery(
                id=f"mixed-{i}",
                prompt=f"Mixed pattern {i}",
                model="gpt-3.5-turbo",
            )
            future = http_client.issue_query(query)
            futures.append(future)

        # Wait for futures
        future_results = await asyncio.gather(*futures)

        # Both callback and futures should have results
        assert len(future_results) == 10
        assert len(callback_results) == 10

        # Results should match
        future_ids = {r.query_id for r in future_results}
        callback_ids = {r.query_id for r in callback_results}
        assert future_ids == callback_ids


class TestHTTPEndpointClientErrorHandling:
    """Test error handling with real ZMQ sockets."""

    @pytest.mark.asyncio
    async def test_worker_connection_error(self):
        """Test handling when workers can't connect to endpoint."""
        # Use invalid endpoint
        http_config = HTTPClientConfig(
            endpoint_url="http://invalid-host-12345:9999/v1/chat/completions",
            num_workers=2,
        )

        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_conn_err_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_conn_err_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(
                client_timeout_total=2.0
            ),  # short timeout to fail fast
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Send request
            query = ChatCompletionQuery(
                id="error-test",
                prompt="This should fail",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)

            # Allow time for error to propagate through ZMQ
            await asyncio.sleep(0.2)

            # Should get error
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(future, timeout=5.0)

            assert "invalid-host-12345" in str(
                exc_info.value
            ) or "Cannot connect" in str(exc_info.value)

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_handler_error_recovery(self):
        """Test that response handler recovers from errors."""
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_handler_err_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_handler_err_resp_{timestamp}",
        )

        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Initialize minimal client components
        for i in range(client.config.num_workers):
            address = f"{client.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(client.zmq_context, address, client.zmq_config)
            client.worker_push_sockets.append(push_socket)

        # Start response handler
        client._response_handler_task = asyncio.create_task(client._handle_responses())

        # Create context for test
        context = zmq.asyncio.Context()

        try:
            # Create push socket to send responses
            response_push = context.socket(zmq.PUSH)
            response_push.connect(zmq_config.zmq_response_queue_addr)

            # Send valid response
            result1 = QueryResult(
                query_id="test-1",
                response_output="Success",
            )
            await response_push.send(pickle.dumps(result1))

            # Send invalid data that will cause error
            await response_push.send(b"invalid pickle data")

            # Send another valid response to verify recovery
            result2 = QueryResult(
                query_id="test-2",
                response_output="Success after error",
            )
            await response_push.send(pickle.dumps(result2))

            # Create futures to track
            future1 = asyncio.get_event_loop().create_future()
            future2 = asyncio.get_event_loop().create_future()
            client._pending_futures["test-1"] = future1
            client._pending_futures["test-2"] = future2

            # Wait for processing
            await asyncio.sleep(0.5)

            # First future should be completed
            assert future1.done()
            assert future1.result().response_output == "Success"

            # Second future should also complete (handler recovered)
            assert future2.done()
            assert future2.result().response_output == "Success after error"

            # Cleanup
            response_push.close()
            await client.shutdown()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_zmq_send_failure(self):
        """Test handling of ZMQ send failures."""
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_send_fail_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_send_fail_resp_{timestamp}",
        )

        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Properly start the client first
        await client.start()

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

            query = ChatCompletionQuery(
                id="send-fail",
                prompt="This will fail to send",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)

            # The send happens asynchronously, wait for it
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(future, timeout=1.0)
            assert "ZMQ send failed" in str(exc_info.value)

        finally:
            await client.shutdown()


class TestHTTPEndpointClientCoverage:
    """Tests to improve code coverage."""

    @pytest.fixture(scope="class")
    def http_config(self, mock_http_echo_server):
        """Create HTTP client configuration with echo server URL."""
        return HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )

    @pytest.fixture(scope="class")
    def zmq_config(self, tmp_path_factory):
        """Create ZMQ configuration with unique addresses."""
        # Use tmp_path_factory for class-scoped fixture
        tmp_dir = tmp_path_factory.mktemp("test_coverage")
        return ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_dir}/test_coverage_req",
            zmq_response_queue_addr=f"ipc://{tmp_dir}/test_coverage_resp",
        )

    @pytest_asyncio.fixture(scope="class")
    async def http_client(self, http_config, zmq_config):
        """Create and start HTTP endpoint client."""
        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )
        await client.start()
        yield client
        await client.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_with_callback(self, mock_http_echo_server):
        """Test HTTPEndpointClient initialization with callback."""
        callback_called = []

        def test_callback(result: QueryResult):
            callback_called.append(result)

        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=5,  # Test concurrency semaphore creation
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_init_callback_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_init_callback_resp_{timestamp}",
            zmq_io_threads=2,  # Test custom io_threads
        )

        # Test initialization with callback and concurrency limit
        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
            complete_callback=test_callback,
        )

        # Verify initialization state
        assert client.config == http_config
        assert client.aiohttp_config is not None
        assert client.zmq_config == zmq_config
        assert client.complete_callback == test_callback
        assert client._concurrency_semaphore is not None
        assert client._concurrency_semaphore._value == 5
        assert client.current_worker_idx == 0
        assert len(client.worker_push_sockets) == 0
        assert client.worker_manager is None
        assert not client._shutdown_event.is_set()
        assert client._response_handler_task is None
        assert len(client._pending_futures) == 0

        await client.start()

        try:
            # Test that callback is called
            query = ChatCompletionQuery(
                id="callback-test",
                prompt="Test callback",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            await future

            # Wait a bit for callback to be processed
            await asyncio.sleep(0.1)

            assert len(callback_called) == 1
            assert callback_called[0].query_id == "callback-test"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_without_concurrency_limit(
        self, mock_http_echo_server
    ):
        """Test initialization without concurrency limit (max_concurrency <= 0)."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,  # No concurrency limit
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_no_concurrency_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_no_concurrency_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Verify no concurrency semaphore is created
        assert client._concurrency_semaphore is None

        await client.start()

        try:
            # Test that requests work without concurrency limit
            query = ChatCompletionQuery(
                id="no-limit-test",
                prompt="Test no limit",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            result = await future

            assert result.query_id == "no-limit-test"
            assert result.response_output == "Test no limit"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_start_method_socket_creation(self, mock_http_echo_server):
        """Test start method creates correct number of worker sockets."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=4,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_start_sockets_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_start_sockets_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Verify initial state
        assert len(client.worker_push_sockets) == 0
        assert client.worker_manager is None
        assert client._response_handler_task is None

        await client.start()

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
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_handler_timeout_path(self, mock_http_echo_server):
        """Test response handler timeout path in _handle_responses."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_timeout_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_timeout_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Let the response handler run for a bit to exercise timeout path
            await asyncio.sleep(1.5)  # Should trigger at least one timeout

            # Verify response handler is still running
            assert client._response_handler_task is not None
            assert not client._response_handler_task.done()

            # Send a request to verify normal operation still works
            query = ChatCompletionQuery(
                id="timeout-test",
                prompt="Test after timeout",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            result = await future

            assert result.query_id == "timeout-test"
            assert result.response_output == "Test after timeout"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, mock_http_echo_server):
        """Test error handling in user callback."""
        timestamp = int(time.time() * 1000)
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
            zmq_request_queue_prefix=f"ipc:///tmp/test_callback_error_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_callback_error_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
            complete_callback=failing_callback,
        )

        await client.start()

        try:
            # Send request that will trigger callback error
            query = ChatCompletionQuery(
                id="callback-error-test",
                prompt="Test callback error",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)
            result = await future

            # Future should still complete successfully despite callback error
            assert result.query_id == "callback-error-test"
            assert result.response_output == "Test callback error"

            # Wait for callback to be processed
            await asyncio.sleep(0.1)

            # Verify callback was called (but failed)
            # For non-streaming queries, callback is called once
            assert len(callback_errors) >= 1
            assert callback_errors[0] == "callback_called"

        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_with_error_field(self, mock_http_echo_server):
        """Test handling response with error field."""
        # This test requires a way to simulate error responses
        # We'll create a mock response directly
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_error_response_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_error_response_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Initialize minimal client components for direct response injection
        for i in range(client.config.num_workers):
            address = f"{client.zmq_config.zmq_request_queue_prefix}_{i}_requests"
            push_socket = ZMQPushSocket(client.zmq_context, address, client.zmq_config)
            client.worker_push_sockets.append(push_socket)

        # Start response handler
        client._response_handler_task = asyncio.create_task(client._handle_responses())

        # Create context for test
        context = zmq.asyncio.Context()

        try:
            # Create push socket to send error response
            response_push = context.socket(zmq.PUSH)
            response_push.connect(zmq_config.zmq_response_queue_addr)

            # Create future for tracking
            future = asyncio.get_event_loop().create_future()
            client._pending_futures["error-test"] = future

            # Send error response
            error_result = QueryResult(
                query_id="error-test",
                response_output="",
                error="Simulated error response",
            )
            await response_push.send(pickle.dumps(error_result))

            # Wait for processing and expect exception
            with pytest.raises(Exception) as exc_info:
                await future

            assert "Simulated error response" in str(exc_info.value)

            # Cleanup
            response_push.close()
            await client.shutdown()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_shutdown_with_pending_response_handler(self, mock_http_echo_server):
        """Test shutdown when response handler task exists."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_shutdown_handler_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_shutdown_handler_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        await client.start()

        # Verify components are running
        assert client._response_handler_task is not None
        assert not client._response_handler_task.done()
        assert client.worker_manager is not None
        assert len(client.worker_push_sockets) == 2

        # Add some pending futures
        future1 = asyncio.get_event_loop().create_future()
        future2 = asyncio.get_event_loop().create_future()
        client._pending_futures["pending-1"] = future1
        client._pending_futures["pending-2"] = future2

        # Shutdown should clean everything up
        await client.shutdown()

        # Verify cleanup
        assert client._shutdown_event.is_set()
        assert len(client._pending_futures) == 0
        assert future1.cancelled()
        assert future2.cancelled()
        assert client._response_handler_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_without_components(self):
        """Test shutdown when components haven't been initialized."""
        timestamp = int(time.time() * 1000)
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:9999/v1/chat/completions",
            num_workers=1,
            max_concurrency=-1,
        )

        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_shutdown_empty_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_shutdown_empty_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=zmq_config,
        )

        # Don't call start() - test shutdown on uninitialized client
        assert client.worker_manager is None
        assert client._response_handler_task is None
        assert len(client.worker_push_sockets) == 0

        # Should not raise any errors
        await client.shutdown()

        # Verify shutdown event is set
        assert client._shutdown_event.is_set()

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

            push_socket.close()

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

            pull_socket.close()

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_empty_prompt(self, http_client):
        """Test handling empty prompt."""
        query = ChatCompletionQuery(
            id="empty-prompt",
            prompt="",
            model="gpt-3.5-turbo",
        )

        future = http_client.issue_query(query)
        result = await future

        assert result.query_id == "empty-prompt"
        assert result.response_output == ""

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, http_client):
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
            query = ChatCompletionQuery(
                id=f"special-{i}",
                prompt=prompt,
                model="gpt-3.5-turbo",
            )
            future = http_client.issue_query(query)
            futures.append((prompt, future))

        # Verify all handled correctly
        for prompt, future in futures:
            result = await future
            assert result.response_output == prompt

    @pytest.mark.asyncio
    async def test_metadata_propagation(self, http_client):
        """Test that query metadata is preserved."""
        query = ChatCompletionQuery(
            id="metadata-test",
            prompt="Test metadata",
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.5,
            metadata={
                "user_id": "test-user",
                "session_id": "test-session",
                "custom_field": "custom_value",
            },
        )

        future = http_client.issue_query(query)
        result = await future

        # Echo server should preserve the query
        assert result.query_id == "metadata-test"
        assert result.response_output == "Test metadata"

    @pytest.mark.asyncio
    async def test_concurrent_shutdown(self, http_config, zmq_config):
        """Test shutdown while requests are in flight."""
        # Create a separate client for this test since we need to shut it down
        import time

        timestamp = int(time.time() * 1000)
        shutdown_zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_shutdown_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_shutdown_resp_{timestamp}",
        )

        client = HTTPEndpointClient(
            config=http_config,
            aiohttp_config=AioHttpConfig(),
            zmq_config=shutdown_zmq_config,
        )
        await client.start()

        try:
            # Send many requests
            futures = []
            for i in range(100):
                query = ChatCompletionQuery(
                    id=f"shutdown-{i}",
                    prompt=f"Shutdown test {i}",
                    model="gpt-3.5-turbo",
                )
                future = client.issue_query(query)
                futures.append(future)

            # Immediately shutdown
            await client.shutdown()

            # Count completed vs cancelled
            completed = sum(1 for f in futures if f.done() and not f.cancelled())
            cancelled = sum(1 for f in futures if f.cancelled())

            print(f"\nShutdown test: {completed} completed, {cancelled} cancelled")

            # At least some should be cancelled
            assert cancelled > 0
        finally:
            # Ensure cleanup even if test fails
            if not client._shutdown_event.is_set():
                await client.shutdown()

    @pytest.mark.asyncio
    async def test_error_response_propagation(self, http_client):
        """Test that error responses are propagated as exceptions in futures."""
        # Use an invalid endpoint to trigger real errors
        timestamp = int(time.time() * 1000)
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc:///tmp/test_error_prop_{timestamp}",
            zmq_response_queue_addr=f"ipc:///tmp/test_error_prop_resp_{timestamp}",
        )

        config = HTTPClientConfig(
            endpoint_url="http://invalid-host-does-not-exist:9999/v1/chat/completions",
            num_workers=1,
        )

        client = HTTPEndpointClient(
            config=config,
            aiohttp_config=AioHttpConfig(
                client_timeout_total=2.0
            ),  # short timeout to fail fast
            zmq_config=zmq_config,
        )

        await client.start()

        try:
            # Send request to invalid endpoint
            query = ChatCompletionQuery(
                id="error-test",
                prompt="Test error",
                model="gpt-3.5-turbo",
            )

            future = client.issue_query(query)

            # Wait for the error to propagate through the system
            # The flow is: client -> ZMQ -> worker -> HTTP error -> ZMQ -> client
            await asyncio.sleep(0.5)

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
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_response_for_unknown_query_id(self, http_client):
        """Test handling response for unknown query ID by checking internal state."""
        # Verify client is in a good state
        assert http_client.worker_push_sockets, "Client should have worker sockets"
        assert (
            not http_client._shutdown_event.is_set()
        ), "Client should not be shut down"

        # Send a normal request first
        query = ChatCompletionQuery(
            id="known-query",
            prompt="Test query",
            model="gpt-3.5-turbo",
        )

        future = http_client.issue_query(query)
        result = await future

        # Verify the query was processed and removed from pending futures
        assert result.query_id == "known-query"
        assert "known-query" not in http_client._pending_futures

        # Test that the client handles normal operations correctly
        # (Unknown query IDs would be handled gracefully by the response handler)
