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

"""Streaming functionality tests for the HTTP endpoint client."""

import asyncio

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)

from ...futures_client import FuturesHttpClient
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

    def create_client_with_configs(
        self,
        url,
        tmp_path,
        prefix,
        num_workers=1,
        aiohttp_config=None,
    ):
        """Helper to create client with specific config."""
        http_config = HTTPClientConfig(
            endpoint_url=url,
            num_workers=num_workers,
        )
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(tmp_path, prefix, "_req"),
            zmq_response_queue_addr=get_test_socket_path(tmp_path, prefix, "_resp"),
            zmq_readiness_queue_addr=get_test_socket_path(tmp_path, prefix, "_ready"),
        )
        return FuturesHttpClient(
            http_config, aiohttp_config or AioHttpConfig(), zmq_config
        )

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
            futures.append((i, futures_http_client.issue_query(query)))

        # Wait for all to complete
        results = []
        for idx, future in futures:
            result = await asyncio.wrap_future(future)
            results.append((idx, result))

        # Verify all completed with correct content
        assert len(results) == 10

        for idx, result in results:
            assert result.id == f"concurrent-stream-{idx}"
            assert (
                "".join(result.response_output["output"])
                == f"Concurrent streaming request number {idx}"
            )
            assert result.error is None

    @pytest.mark.asyncio
    async def test_streaming_complete_response(self, futures_http_client):
        """Test streaming response returns complete result."""
        query = Query(
            id="test-streaming",
            data={
                "prompt": "Test streaming response",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future = futures_http_client.issue_query(query)
        result = await asyncio.wrap_future(future)

        assert result.id == "test-streaming"
        assert (
            "".join(result.response_output["output"]) == "Test streaming response"
        )

        # Check first chunk is properly separated for streaming (word-by-word from echo server)
        assert isinstance(result.response_output["output"], tuple)
        assert len(result.response_output["output"]) == 2
        assert result.response_output["output"][0] == "Test"
        assert result.response_output["output"][1] == " streaming response"

    @pytest.mark.asyncio
    async def test_streaming_single_word(self, futures_http_client):
        """Test streaming with single-word response."""
        query = Query(
            id="test-single-word",
            data={
                "prompt": "Hi",
                "model": "gpt-3.5-turbo",
                "stream": True,
            },
        )

        future = futures_http_client.issue_query(query)
        result = await asyncio.wrap_future(future)

        assert result.id == "test-single-word"
        assert "".join(result.response_output["output"]) == "Hi"

        # Single word: only one chunk (no second element when there's only 1 chunk)
        assert isinstance(result.response_output["output"], tuple)
        assert len(result.response_output["output"]) == 1
        assert result.response_output["output"][0] == "Hi"

    @pytest.mark.asyncio
    async def test_streaming_error_propagation(self, tmp_path):
        """Test error propagation in streaming responses."""
        # Use invalid endpoint to trigger errors
        client = self.create_client_with_configs(
            "http://invalid-endpoint-12345:9999/v1/chat/completions",
            tmp_path,
            "tse",
            num_workers=1,
            aiohttp_config=AioHttpConfig(client_timeout_total=2.0),
        )

        try:
            query = Query(
                id="test-error",
                data={
                    "prompt": "This will fail",
                    "model": "gpt-3.5-turbo",
                    "stream": True,
                },
            )

            future = client.issue_query(query)

            # Complete response should fail
            with pytest.raises(Exception):  # noqa: B017 Worker wraps errors in generic Exception
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.0)

        finally:
            client.shutdown()

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
            futures.append((name, prompt, futures_http_client.issue_query(query)))

        # Check all complete correctly
        for _, prompt, future in futures:
            result = await asyncio.wrap_future(future)
            assert "".join(result.response_output["output"]) == prompt
