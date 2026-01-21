# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.transport import ZmqWorkerPoolTransport
from inference_endpoint.endpoint_client.worker import Worker


class TestWorkerBasicFunctionality:
    """Test basic Worker functionality for request/response handling."""

    @pytest.fixture
    def worker_config(self, mock_http_echo_server):
        """Create worker configuration with echo server URL."""
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{mock_http_echo_server.url}/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=False,
        )
        return http_config

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
    async def test_worker_request_handling(self, worker_config, requests):
        """Test worker handling various request patterns including streaming, non-streaming, and multiple requests."""
        http_config = worker_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        pool = ZmqWorkerPoolTransport.create(loop, num_workers=1)

        worker = Worker(
            worker_id=0,
            connector=pool.worker_connector,
            http_config=http_config,
        )

        # Start worker
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait for worker readiness
            await pool.wait_for_workers_ready(timeout=0.5)

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
                pool.send(0, query)

            # Collect responses
            final_responses: dict[str, QueryResult] = {}
            streaming_chunks: dict[str, list[StreamChunk]] = {}

            # Receive all responses (streaming queries produce multiple messages)
            while len(final_responses) < len(requests):
                response = await pool.recv()

                if response is None:
                    break

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

        finally:
            worker.shutdown()
            await asyncio.gather(worker_task, return_exceptions=True)
            pool.cleanup()

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
    async def test_worker_signal_handling(self, worker_config, sig):
        """Test worker responds to various signals correctly.

        Tests graceful shutdown handling for common signals:
        - SIGTERM: Standard termination signal
        - SIGINT: Interrupt signal (Ctrl+C)
        - SIGHUP: Hangup signal (terminal closed)
        - SIGQUIT: Quit signal (Ctrl+\\)
        """
        http_config = worker_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        pool = ZmqWorkerPoolTransport.create(loop, num_workers=1)

        worker = Worker(
            worker_id=0,
            connector=pool.worker_connector,
            http_config=http_config,
        )

        # Start worker
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait for worker readiness
            await pool.wait_for_workers_ready(timeout=0.5)

            # Verify worker is running
            assert not worker._shutdown

            # Send signal via shutdown method (simulates signal handler)
            worker.shutdown(sig, None)

            # Verify shutdown flag is set
            assert worker._shutdown

            # Await worker exit
            await asyncio.gather(worker_task, return_exceptions=True)

        finally:
            if not worker_task.done():
                worker.shutdown()
                await asyncio.gather(worker_task, return_exceptions=True)
            pool.cleanup()


class TestWorkerErrorHandling:
    """Test Worker error handling for various failure scenarios."""

    @pytest.fixture
    def worker_config(self, mock_http_echo_server):
        """Create worker configuration with echo server URL."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=1,
            max_connections=10,
            warmup_connections=False,
        )
        return http_config

    @pytest.fixture
    def error_config(self):
        """Create configuration with invalid endpoint for error tests."""
        http_config = HTTPClientConfig(
            endpoint_urls=["http://localhost:59999/v1/chat/completions"],
            num_workers=1,
            max_connections=10,
            warmup_connections=False,
        )
        return http_config

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_connection_error_handling(self, error_config, stream):
        """Test worker error handling with invalid endpoint for both streaming and non-streaming."""
        http_config = error_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        pool = ZmqWorkerPoolTransport.create(loop, num_workers=1)

        worker = Worker(
            worker_id=0,
            connector=pool.worker_connector,
            http_config=http_config,
        )

        # Start worker
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait for worker readiness
            await pool.wait_for_workers_ready(timeout=0.5)

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
            pool.send(0, query)

            # Receive error response
            response = await pool.recv()

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == query_id
            assert response.error is not None
            # Check for connection error indicators
            error_lower = response.error.lower()
            assert (
                ("connect" in error_lower and "failed" in error_lower)
                or ("connection" in error_lower and "refused" in error_lower)
                or ("cannot connect" in error_lower)
            ), f"Unexpected error message: {response.error}"

        finally:
            worker.shutdown()
            await asyncio.gather(worker_task, return_exceptions=True)
            pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_exception_handling(self, worker_config, stream):
        """Test worker handles exceptions in body handler for both streaming and non-streaming."""
        http_config = worker_config

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        pool = ZmqWorkerPoolTransport.create(loop, num_workers=1)

        worker = Worker(
            worker_id=0,
            connector=pool.worker_connector,
            http_config=http_config,
        )

        # Mock the appropriate body handler to raise an exception
        error_msg = (
            f"Simulated {'streaming' if stream else 'non-streaming'} processing error"
        )

        async def mock_handle_body(prepared):
            raise RuntimeError(error_msg)

        if stream:
            worker._handle_streaming_body = mock_handle_body
        else:
            worker._handle_non_streaming_body = mock_handle_body

        # Start worker
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait for worker readiness
            await pool.wait_for_workers_ready(timeout=0.5)

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
            pool.send(0, query)

            # Receive error response
            response = await pool.recv()

            # Verify error response
            assert isinstance(response, QueryResult)
            assert response.id == query_id
            assert response.error is not None
            assert error_msg in response.error
            assert response.response_output is None

        finally:
            worker.shutdown()
            await asyncio.gather(worker_task, return_exceptions=True)
            pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stream", [False, True], ids=["non_streaming", "streaming"]
    )
    async def test_worker_malformed_json(self, stream):
        """Test worker handling malformed JSON in response."""
        from aiohttp import web
        from aiohttp.test_utils import TestServer

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
        server = TestServer(app)
        await server.start_server()

        # Create pool transport with the running event loop
        loop = asyncio.get_running_loop()
        pool = ZmqWorkerPoolTransport.create(loop, num_workers=1)

        # Create config with test server URL
        http_config = HTTPClientConfig(
            endpoint_urls=[f"http://localhost:{server.port}/malformed"],
            num_workers=1,
            max_connections=10,
            warmup_connections=False,
        )

        worker = Worker(
            worker_id=0,
            connector=pool.worker_connector,
            http_config=http_config,
        )

        # Start worker
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait for worker readiness
            await pool.wait_for_workers_ready(timeout=0.5)

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
            pool.send(0, query)

            # Receive response
            response = await pool.recv()

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

        finally:
            worker.shutdown()
            await asyncio.gather(worker_task, return_exceptions=True)
            pool.cleanup()
            await server.close()
