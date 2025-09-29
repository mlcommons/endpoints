"""Integration tests for Worker error handling and edge cases."""

import asyncio
import pickle

import pytest
import zmq
import zmq.asyncio
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import Worker
from inference_endpoint.endpoint_client.zmq_utils import ZMQPushSocket


class TestWorkerErrorHandling:
    """Test Worker error handling for various failure scenarios."""

    @pytest.fixture
    def basic_config(self, tmp_path):
        """Create basic configuration for error handling tests."""
        # Use invalid port to trigger connection errors
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/v1/chat/completions",
            num_workers=1,
        )
        aiohttp_config = AioHttpConfig()
        # Use tmp_path for unique socket paths per test
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_path}/test_error_req",
            zmq_response_queue_addr=f"ipc://{tmp_path}/test_error_resp",
        )
        return http_config, aiohttp_config, zmq_config

    @pytest.mark.asyncio
    async def test_worker_error_handling(self, basic_config):
        """Test worker error handling with invalid endpoint."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Modify config to use invalid endpoint (localhost with invalid port for fast failure)
        http_config.endpoint_url = "http://localhost:99999/v1/chat/completions"
        aiohttp_config.client_timeout_total = 2.0  # Short timeout
        aiohttp_config.client_timeout_connect = 1.0  # Connect timeout

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
            await asyncio.sleep(0.5)

            # Send query
            query = Query(
                id="test-error",
                data={
                    "prompt": "This should fail",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=2.0)
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-error"
            assert response.error is not None
            assert (
                (
                    "connection" in response.error.lower()
                    and "refused" in response.error.lower()
                )
                or "cannot connect" in response.error.lower()
                or "99999" in response.error
            )

            # Verify error response metadata (may be empty for error responses)
            # Just check that we got an error response with the right query ID
            assert response.id == "test-error"
            assert response.error is not None

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_http_error_handling(self, basic_config):
        """Test worker handling HTTP errors in streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use invalid endpoint to trigger connection error
        http_config.endpoint_url = "http://localhost:99999/invalid"
        aiohttp_config.client_timeout_total = 1.0  # Short timeout

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
            await asyncio.sleep(0.5)

            # Send streaming query
            query = Query(
                id="test-streaming-error",
                data={
                    "prompt": "This should fail",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-streaming-error"
            assert response.error is not None
            # Should get url back as error
            assert "http://localhost:99999/invalid" in response.error

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_non_streaming_exception_handling(self, basic_config):
        """Test worker handles exceptions in _process_request for non-streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use ZMQPushSocket to send request
        context = zmq.asyncio.Context()
        request_socket = ZMQPushSocket(
            context, f"{zmq_config.zmq_request_queue_prefix}_0_requests", zmq_config
        )

        # Use raw socket to receive response (bind first before worker connects)
        response_pull = context.socket(zmq.PULL)
        response_pull.bind(zmq_config.zmq_response_queue_addr)

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        worker_task = None
        try:
            # Mock _handle_non_streaming_request to raise an exception immediately
            exception_raised = asyncio.Event()

            async def mock_handle_request(query):
                exception_raised.set()
                raise RuntimeError("Simulated processing error")

            worker._handle_non_streaming_request = mock_handle_request

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker to be ready
            await asyncio.sleep(0.5)

            # Send non-streaming query
            query = Query(
                id="test-exception-non-streaming",
                data={
                    "prompt": "Test exception handling",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )
            await request_socket.send(query)

            # Wait for exception to be raised
            try:
                await asyncio.wait_for(exception_raised.wait(), timeout=3.0)
            except TimeoutError:
                pytest.fail("Exception was not raised within timeout")

            # Receive error response with a reasonable timeout
            # The response should be sent almost immediately after the exception
            response_data = await asyncio.wait_for(
                response_pull.recv(),
                timeout=3.0,
            )
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-exception-non-streaming"
            assert response.error is not None
            assert "Simulated processing error" in response.error
            assert response.response_output is None

        finally:
            # Proper cleanup
            if worker_task and not worker_task.done():
                # Signal worker to shutdown
                worker._shutdown = True

                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(worker_task, timeout=2.0)
                except TimeoutError:
                    # Force cancel if graceful shutdown fails
                    worker_task.cancel()
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
                except Exception:
                    # Ignore other exceptions during shutdown
                    pass

            # Close sockets
            request_socket.close()
            response_pull.close()

            # Terminate context
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_exception_handling(self, basic_config):
        """Test worker handles exceptions in _process_request for streaming requests."""
        http_config, aiohttp_config, zmq_config = basic_config

        context = zmq.asyncio.Context()

        # Use raw socket to receive response (bind first before worker connects)
        response_pull = context.socket(zmq.PULL)
        response_pull.bind(zmq_config.zmq_response_queue_addr)

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        worker_task = None
        request_push = None
        try:
            # Mock _handle_streaming_request to raise an exception
            async def mock_handle_request(query):
                raise RuntimeError("Simulated streaming processing error")

            worker._handle_streaming_request = mock_handle_request

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker to be ready
            await asyncio.sleep(0.5)

            # Create the request socket after worker has bound its socket
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            # Send streaming query
            query = Query(
                id="test-exception-streaming",
                data={
                    "prompt": "Test streaming exception handling",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )
            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await asyncio.wait_for(
                response_pull.recv(),
                timeout=2.0,
            )
            response = pickle.loads(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-exception-streaming"
            assert response.error is not None
            assert "Simulated streaming processing error" in response.error
            assert response.response_output is None

        finally:
            # Proper cleanup
            if worker_task and not worker_task.done():
                # Signal worker to shutdown
                worker._shutdown = True

                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(worker_task, timeout=2.0)
                except TimeoutError:
                    # Force cancel if graceful shutdown fails
                    worker_task.cancel()
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
                except Exception:
                    # Ignore other exceptions during shutdown
                    pass

            # Close sockets
            if request_push:
                request_push.close()
            response_pull.close()

            # Terminate context
            context.term()

    @pytest.mark.asyncio
    async def test_worker_non_streaming_connection_error(self, basic_config):
        """Test worker handles connection errors in non-streaming responses."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Already uses invalid URL from basic_config
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
            await asyncio.sleep(0.5)

            # Send query
            query = Query(
                id="test-connection-error",
                data={
                    "prompt": "Test connection error",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Should receive error response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            assert isinstance(response, QueryResult)
            assert response.id == "test-connection-error"
            assert response.error is not None
            assert "99999" in response.error or "Cannot connect" in response.error

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_http_404_error(
        self, mock_http_echo_server, basic_config
    ):
        """Test worker handling HTTP 404 error in streaming request."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use echo server but with invalid endpoint to get 404
        http_config.endpoint_url = f"{mock_http_echo_server.url}/nonexistent"

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
            await asyncio.sleep(0.5)

            # Send streaming query
            query = Query(
                id="test-streaming-404",
                data={
                    "prompt": "This should get 404",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            await request_push.send(pickle.dumps(query))

            # Receive error response
            response_data = await response_pull.recv()
            response = pickle.loads(response_data)

            # Verify HTTP error response
            assert isinstance(response, QueryResult)
            assert response.id == "test-streaming-404"
            assert response.error is not None
            assert "HTTP 404" in response.error

            # Shutdown
            worker._shutdown = True
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_non_streaming_http_error_early_return(self, basic_config):
        """Test non-streaming request with HTTP error status."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Mock session to return various HTTP errors
        from unittest.mock import AsyncMock, MagicMock

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        # Create a proper async context manager mock
        mock_context_manager = MagicMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Create session mock with regular MagicMock so post doesn't return a coroutine
        mock_session = MagicMock()
        mock_session.post.return_value = mock_context_manager
        worker._session = mock_session

        context = zmq.asyncio.Context()

        try:
            # Create response pull socket BEFORE initializing worker components
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Now initialize worker components (they will connect, not bind)
            worker._zmq_context = zmq.asyncio.Context()
            worker._response_socket = ZMQPushSocket(
                worker._zmq_context, zmq_config.zmq_response_queue_addr, zmq_config
            )

            await asyncio.sleep(0.1)

            # Send query that will get HTTP error
            query = Query(
                id="test-http-500",
                data={
                    "prompt": "This will fail",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            await worker._handle_non_streaming_request(query)

            # Verify error response was sent
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=1.0)
            response = pickle.loads(response_data)

            assert isinstance(response, QueryResult)
            assert response.id == "test-http-500"
            assert "HTTP 500" in response.error
            assert "Internal Server Error" in response.error

        finally:
            response_pull.close()
            worker._response_socket.close()
            context.term()
            worker._zmq_context.term()

    @pytest.mark.asyncio
    async def test_worker_streaming_malformed_json(self, basic_config):
        """Test worker handling malformed JSON in streaming response."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create a mock server that returns malformed JSON
        from aiohttp import web

        async def malformed_json_handler(request):
            """Handler that returns malformed JSON in streaming format."""
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/plain"
            await response.prepare(request)

            # Send malformed JSON chunks
            await response.write(b'data: {"invalid": json}\n\n')
            await response.write(b"data: [DONE]\n\n")

            return response

        app = web.Application()
        app.router.add_post("/streaming", malformed_json_handler)

        # Start test server
        from aiohttp.test_utils import TestServer

        server = TestServer(app)

        try:
            await server.start_server()

            # Update config with test server URL
            http_config.endpoint_url = f"http://localhost:{server.port}/streaming"

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
                request_push.connect(
                    f"{zmq_config.zmq_request_queue_prefix}_0_requests"
                )

                response_pull = context.socket(zmq.PULL)
                response_pull.bind(zmq_config.zmq_response_queue_addr)

                # Start worker
                worker_task = asyncio.create_task(worker.run())
                await asyncio.sleep(0.5)

                # Send streaming query
                query = Query(
                    id="test-malformed-json",
                    data={
                        "prompt": "Test malformed JSON",
                        "model": "gpt-3.5-turbo",
                        "stream": True,
                    },
                )

                await request_push.send(pickle.dumps(query))

                # Should get error response due to malformed JSON
                response_data = await response_pull.recv()
                response = pickle.loads(response_data)

                # Verify we get an error response
                assert isinstance(response, QueryResult)
                assert response.id == "test-malformed-json"
                assert response.error is not None
                assert (
                    "unexpected character" in response.error
                    or "JSONDecodeError" in response.error
                )
                assert response.response_output is None

                # Shutdown
                worker._shutdown = True
                await asyncio.wait_for(worker_task, timeout=2.0)

            finally:
                request_push.close()
                response_pull.close()
                context.term()

        finally:
            await server.close()

    @pytest.mark.asyncio
    async def test_worker_zmq_socket_binding_error(self, basic_config):
        """Test worker handling ZMQ socket binding errors."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create worker with invalid socket address
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr="invalid://socket/address",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Worker run should handle the error gracefully
        with pytest.raises(
            zmq.ZMQError
        ):  # ZMQ will raise an exception for invalid address
            await worker.run()

    @pytest.mark.asyncio
    async def test_worker_zmq_socket_error(self, basic_config):
        """Test worker handling ZMQ socket errors."""
        http_config, aiohttp_config, zmq_config = basic_config

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
            # Initialize worker resources manually
            worker._zmq_context = context
            worker._response_socket = ZMQPushSocket(
                context, zmq_config.zmq_response_queue_addr, zmq_config
            )

            # Create query
            query = Query(
                id="test-zmq-error",
                data={
                    "prompt": "Test ZMQ error",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            # Close the response socket to simulate error
            worker._response_socket.socket.close()

            # Try to send response - should handle the error gracefully
            response = QueryResult(
                id=query.id,
                response_output="test",
                error="ZMQ socket closed",
            )

            # This should not raise an exception
            try:
                await worker._response_socket.send(pickle.dumps(response))
            except Exception:
                # Expected - socket is closed
                pass

            # Worker should handle this gracefully without crashing

        finally:
            context.term()

    @pytest.mark.asyncio
    async def test_worker_concurrent_error_handling(self, basic_config):
        """Test multiple workers handling errors concurrently."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use invalid endpoint (localhost with invalid port for fast failure)
        http_config.endpoint_url = "http://localhost:99999/api"
        http_config.num_workers = 3  # Test with multiple workers
        aiohttp_config.client_timeout_total = 2.0
        aiohttp_config.client_timeout_connect = 1.0

        workers = []
        worker_tasks = []
        context = zmq.asyncio.Context()

        try:
            # Create response pull socket
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            # Start multiple workers
            for i in range(3):
                worker = Worker(
                    worker_id=i,
                    http_config=http_config,
                    aiohttp_config=aiohttp_config,
                    zmq_config=zmq_config,
                    request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_{i}_requests",
                    response_socket_addr=zmq_config.zmq_response_queue_addr,
                    readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
                )
                workers.append(worker)
                worker_tasks.append(asyncio.create_task(worker.run()))

            await asyncio.sleep(0.5)

            # Send queries to all workers
            request_sockets = []
            for i in range(3):
                request_push = context.socket(zmq.PUSH)
                request_push.connect(
                    f"{zmq_config.zmq_request_queue_prefix}_{i}_requests"
                )
                request_sockets.append(request_push)

                # Send a query to this worker
                query = Query(
                    id=f"test-concurrent-error-{i}",
                    data={
                        "prompt": f"Worker {i} error test",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                    },
                )
                await request_push.send(pickle.dumps(query))

            # Collect error responses from all workers
            responses = {}
            for _ in range(3):
                response_data = await response_pull.recv()
                response = pickle.loads(response_data)
                responses[response.id] = response

            # Verify all workers handled errors
            assert len(responses) == 3
            for i in range(3):
                sample_id = f"test-concurrent-error-{i}"
                assert sample_id in responses
                assert responses[sample_id].error is not None
                assert (
                    (
                        "connection" in responses[sample_id].error.lower()
                        and "refused" in responses[sample_id].error.lower()
                    )
                    or "cannot connect" in responses[sample_id].error.lower()
                    or "99999" in responses[sample_id].error
                )

            # Shutdown all workers
            for worker in workers:
                worker._shutdown = True

            await asyncio.gather(*worker_tasks, return_exceptions=True)

        finally:
            # Close sockets
            for sock in request_sockets:
                sock.close()
            response_pull.close()
            context.term()

    @pytest.mark.asyncio
    async def test_non_streaming_invalid_json_early_return(self, basic_config):
        """Test non-streaming request with invalid JSON response."""
        http_config, aiohttp_config, zmq_config = basic_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            request_socket_addr=f"{zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=zmq_config.zmq_readiness_queue_addr,
        )

        # Initialize components
        worker._zmq_context = zmq.asyncio.Context()
        worker._response_socket = ZMQPushSocket(
            worker._zmq_context, zmq_config.zmq_response_queue_addr, zmq_config
        )

        # Mock session to return invalid JSON
        from unittest.mock import AsyncMock, MagicMock

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="not valid json at all")

        # Create a proper async context manager mock
        mock_context_manager = MagicMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Create session mock with regular MagicMock so post doesn't return a coroutine
        mock_session = MagicMock()
        mock_session.post.return_value = mock_context_manager
        worker._session = mock_session

        context = zmq.asyncio.Context()

        try:
            # Create response pull socket
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            await asyncio.sleep(0.1)

            # Send query that will get invalid JSON
            query = Query(
                id="test-bad-json",
                data={
                    "prompt": "Will get bad JSON",
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                },
            )

            # Call through _process_request to get proper error handling
            await worker._process_request(query)

            # Verify error response was sent
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=1.0)
            response = pickle.loads(response_data)

            assert isinstance(response, QueryResult)
            assert response.id == "test-bad-json"
            assert response.error is not None
            assert (
                "invalid literal" in response.error
                or "JSONDecodeError" in response.error
            )

        finally:
            response_pull.close()
            worker._response_socket.close()
            context.term()
            worker._zmq_context.term()
