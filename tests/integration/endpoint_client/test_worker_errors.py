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

"""Integration tests for HttpClient worker process error handling"""

import asyncio

import msgspec
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

from ...test_helpers import get_test_socket_path


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
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_error", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_error", "_resp"
            ),
        )
        return http_config, aiohttp_config, zmq_config

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_connection_error_handling(self, basic_config, stream):
        """Test worker error handling with invalid endpoint for both streaming and non-streaming."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Use invalid endpoint to trigger connection error
        http_config.endpoint_url = "http://localhost:99999/v1/chat/completions"
        aiohttp_config.client_timeout_total = 2.0  # Short timeout
        aiohttp_config.client_timeout_connect = 1.0  # Connect timeout

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

            # Send query
            query_id = (
                f"test-connection-error-{'streaming' if stream else 'non-streaming'}"
            )
            query = Query(
                id=query_id,
                data={
                    "prompt": "This should fail",
                    "model": "gpt-3.5-turbo",
                    "stream": stream,
                },
            )

            encoder = msgspec.msgpack.Encoder()
            await request_push.send(encoder.encode(query))

            # Receive error response
            response_data = await asyncio.wait_for(response_pull.recv(), timeout=3.0)
            decoder = msgspec.msgpack.Decoder(QueryResult | StreamChunk)
            response = decoder.decode(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == query_id
            assert response.error is not None
            assert (
                (
                    "connection" in response.error.lower()
                    and "refused" in response.error.lower()
                )
                or "cannot connect" in response.error.lower()
                or "99999" in response.error
            )

            # Shutdown
            worker.shutdown()
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_exception_handling(self, basic_config, stream):
        """Test worker handles exceptions in _process_request for both streaming and non-streaming."""
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
            # Mock the appropriate handler to raise an exception
            error_msg = f"Simulated {'streaming' if stream else 'non-streaming'} processing error"

            async def mock_handle_request(query):
                raise RuntimeError(error_msg)

            if stream:
                worker._handle_streaming_request = mock_handle_request
            else:
                worker._handle_non_streaming_request = mock_handle_request

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Create request socket after worker has bound its socket
            request_push = context.socket(zmq.PUSH)
            request_push.connect(f"{zmq_config.zmq_request_queue_prefix}_0_requests")

            # Send query
            query_id = f"test-exception-{'streaming' if stream else 'non-streaming'}"
            query = Query(
                id=query_id,
                data={
                    "prompt": "Test exception handling",
                    "model": "gpt-3.5-turbo",
                    "stream": stream,
                },
            )
            encoder = msgspec.msgpack.Encoder()
            await request_push.send(encoder.encode(query))

            # Receive error response
            response_data = await asyncio.wait_for(
                response_pull.recv(),
                timeout=3.0,
            )
            decoder = msgspec.msgpack.Decoder(QueryResult | StreamChunk)
            response = decoder.decode(response_data)

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == query_id
            assert response.error is not None
            assert error_msg in response.error
            assert response.response_output is None

        finally:
            # Proper cleanup
            if worker_task and not worker_task.done():
                worker.shutdown()
                try:
                    await asyncio.wait_for(worker_task, timeout=2.0)
                except TimeoutError:
                    worker_task.cancel()
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
                except Exception:
                    pass

            if request_push:
                request_push.close()
            response_pull.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_malformed_json(self, basic_config, stream):
        """Test worker handling malformed JSON in response."""
        http_config, aiohttp_config, zmq_config = basic_config

        # Create a mock server that returns malformed JSON
        from aiohttp import web

        async def malformed_json_streaming_handler(request):
            """Handler that returns malformed JSON in streaming format."""
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/plain"
            await response.prepare(request)

            # Send malformed JSON chunks
            await response.write(b'data: {"invalid": json}\n\n')
            await response.write(b"data: [DONE]\n\n")

            return response

        async def malformed_json_non_streaming_handler(request):
            """Handler that returns malformed JSON in non-streaming format."""
            return web.Response(
                body=b'{"invalid": json}',
                content_type="application/json",
            )

        app = web.Application()
        if stream:
            app.router.add_post("/malformed", malformed_json_streaming_handler)
        else:
            app.router.add_post("/malformed", malformed_json_non_streaming_handler)

        # Start test server
        from aiohttp.test_utils import TestServer

        server = TestServer(app)

        try:
            await server.start_server()

            # Update config with test server URL
            http_config.endpoint_url = f"http://localhost:{server.port}/malformed"

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

                # Send query
                query_id = (
                    f"test-malformed-json-{'streaming' if stream else 'non-streaming'}"
                )
                query = Query(
                    id=query_id,
                    data={
                        "prompt": "Test malformed JSON",
                        "model": "gpt-3.5-turbo",
                        "stream": stream,
                    },
                )

                encoder = msgspec.msgpack.Encoder()
                await request_push.send(encoder.encode(query))

                response_data = await response_pull.recv()
                decoder = msgspec.msgpack.Decoder(QueryResult | StreamChunk)
                response = decoder.decode(response_data)

                # Verify we get a response
                assert isinstance(response, QueryResult)
                assert response.id == query_id

                if stream:
                    # Streaming: malformed JSON is skipped, so we get an empty response
                    # (see _parse_sse_chunk which catches exceptions for non-content SSE messages)
                    assert response.error is None
                    assert response.response_output == {"output": ()}
                else:
                    # Non-streaming: malformed JSON causes a decode error
                    assert response.error is not None
                    assert (
                        "decode" in response.error.lower()
                        or "json" in response.error.lower()
                    )

                # Shutdown
                worker.shutdown()
                await asyncio.wait_for(worker_task, timeout=2.0)

            finally:
                request_push.close()
                response_pull.close()
                context.destroy(linger=0)

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
            SystemExit
        ):  # Worker exits with code 1 on initialization failure
            await worker.run()
