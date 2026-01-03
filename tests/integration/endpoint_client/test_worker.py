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

"""Integration tests for HttpClient worker process core functionality."""

import asyncio
import signal

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


class TestWorkerBasicFunctionality:
    """Test basic Worker functionality for request/response handling."""

    @pytest.fixture
    def zmq_config(self, tmp_path):
        """Create ZMQ configuration for worker tests."""
        return ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(tmp_path, "worker", "_req"),
            zmq_response_queue_addr=get_test_socket_path(tmp_path, "worker", "_resp"),
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
    @pytest.mark.parametrize(
        "requests",
        [
            # Mixed streaming and non-streaming requests
            # Format: (prompt, stream, expected_output)
            # For streaming: expected_output is tuple (first_chunk, full_output_dict)
            #   where full_output_dict = {"output": (first_chunk, joined_rest)}
            # For non-streaming: expected_output is just the string
            [
                ("Non-streaming first", False, "Non-streaming first"),
                (
                    "Streaming second",
                    True,
                    ("Streaming", {"output": ("Streaming", " second")}),
                ),
                ("Non-streaming third", False, "Non-streaming third"),
            ],
            # Empty prompts for both streaming and non-streaming
            [
                ("", False, ""),
                ("", True, ("", {"output": ()})),
            ],
        ],
        ids=[
            "mixed_streaming_non_streaming",
            "empty_prompts",
        ],
    )
    async def test_worker_request_handling(
        self,
        worker_config,
        zmq_config,
        requests,
    ):
        """Test worker handling various request patterns including streaming, non-streaming, and multiple requests."""
        http_config, aiohttp_config = worker_config

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
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

            encoder = msgspec.msgpack.Encoder()
            decoder = msgspec.msgpack.Decoder(QueryResult | StreamChunk)

            # Send all queries
            for i, (prompt, stream, _) in enumerate(requests):
                query = Query(
                    id=f"test-{i}",
                    data={
                        "prompt": prompt,
                        "model": "gpt-3.5-turbo",
                        "stream": stream,
                    },
                )
                await request_push.send(encoder.encode(query))

            # Collect responses
            final_responses: dict[str, QueryResult] = {}
            streaming_chunks: dict[str, list[StreamChunk]] = {}

            # Receive all responses (streaming queries produce multiple messages)
            while len(final_responses) < len(requests):
                try:
                    response_data = await response_pull.recv()
                    response = decoder.decode(response_data)

                    if isinstance(response, StreamChunk):
                        # Intermediate streaming chunk
                        if response.id not in streaming_chunks:
                            streaming_chunks[response.id] = []
                        streaming_chunks[response.id].append(response)
                    elif isinstance(response, QueryResult):
                        if response.metadata.get("final_chunk", False):
                            # Final streaming response
                            final_responses[response.id] = response
                        else:
                            # Non-streaming response
                            final_responses[response.id] = response

                except TimeoutError:
                    break

            # Verify all responses received
            assert len(final_responses) == len(
                requests
            ), f"Expected {len(requests)} responses, got {len(final_responses)}"

            # Verify each response
            for i, (_, stream, expected) in enumerate(requests):
                query_id = f"test-{i}"

                assert query_id in final_responses
                response = final_responses[query_id]
                assert response.error is None

                if stream:
                    # Streaming response - expected is (first_chunk_content, full_output_tuple)
                    expected_first_chunk, expected_output = expected
                    assert response.metadata.get("final_chunk") is True
                    assert response.response_output == expected_output

                    # Verify first chunk metadata and content
                    if query_id in streaming_chunks and streaming_chunks[query_id]:
                        first_chunk = streaming_chunks[query_id][0]
                        assert first_chunk.metadata.get("first_chunk") is True
                        assert first_chunk.response_chunk == expected_first_chunk
                else:
                    # Non-streaming response
                    assert response.response_output == expected

            # Shutdown
            worker.shutdown()
            await worker_task

        finally:
            request_push.close()
            response_pull.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sig",
        [
            signal.SIGTERM,
            signal.SIGINT,
            signal.SIGHUP,
            signal.SIGQUIT,
        ],
        ids=["SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"],
    )
    async def test_worker_signal_handling(self, worker_config, zmq_config, sig):
        """Test worker responds to various signals correctly.

        Tests graceful shutdown handling for common signals:
        - SIGTERM: Standard termination signal
        - SIGINT: Interrupt signal (Ctrl+C)
        - SIGHUP: Hangup signal (terminal closed)
        - SIGQUIT: Quit signal (Ctrl+\\)
        """
        http_config, aiohttp_config = worker_config

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets - bind before worker starts so worker can connect
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(zmq_config.zmq_response_queue_addr)

            readiness_pull = context.socket(zmq.PULL)
            readiness_pull.bind(zmq_config.zmq_readiness_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker readiness signal
            await readiness_pull.recv()

            # Verify worker is running
            assert not worker._shutdown

            # Send signal via shutdown method (simulates signal handler)
            worker.shutdown(sig, None)

            # Verify shutdown flag is set
            assert worker._shutdown

            # Worker should exit gracefully after the receive timeout
            await worker_task

        finally:
            readiness_pull.close()
            response_pull.close()
            context.destroy(linger=0)


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

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
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
            response_data = await response_pull.recv()
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
            await worker_task

        finally:
            request_push.close()
            response_pull.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_exception_handling(
        self, mock_http_echo_server, tmp_path, stream
    ):
        """Test worker handles exceptions in response handlers."""
        # Need a working server so _fire_request succeeds before handler is called
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(tmp_path, "test_exc", "_req"),
            zmq_response_queue_addr=get_test_socket_path(tmp_path, "test_exc", "_resp"),
        )

        context = zmq.asyncio.Context()

        # Use raw socket to receive response (bind first before worker connects)
        response_pull = context.socket(zmq.PULL)
        response_pull.bind(zmq_config.zmq_response_queue_addr)

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        )

        worker_task = None
        request_push = None
        try:
            # Mock the appropriate handler to raise an exception
            error_msg = f"Simulated {'streaming' if stream else 'non-streaming'} processing error"

            async def mock_handle_response(prepared):
                raise RuntimeError(error_msg)

            if stream:
                worker._handle_streaming_response = mock_handle_response
            else:
                worker._handle_non_streaming_response = mock_handle_response

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
            response_data = await response_pull.recv()
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
                    await worker_task
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
                await worker_task

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

        # Set invalid socket address prefix in zmq_config to trigger binding error
        zmq_config.zmq_request_queue_prefix = "invalid://socket/address"

        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
        )

        # Worker run should handle the error gracefully
        with pytest.raises(
            SystemExit
        ):  # Worker exits with code 1 on initialization failure
            await worker.run()
