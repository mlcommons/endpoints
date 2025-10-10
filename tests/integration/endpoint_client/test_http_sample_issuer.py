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

"""Integration tests for HttpClientSampleIssuer."""

import asyncio
from unittest.mock import patch

import pytest
from inference_endpoint.core.types import QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator.sample import Sample

from ...test_helpers import get_test_socket_path


@pytest.fixture(scope="function")
def issuer_http_client(tmp_path, mock_http_echo_server):
    """Create HTTPEndpointClient for testing HttpClientSampleIssuer."""
    http_config = HTTPClientConfig(
        endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
        num_workers=1,
    )
    aiohttp_config = AioHttpConfig()
    zmq_config = ZMQConfig(
        zmq_request_queue_prefix=get_test_socket_path(
            tmp_path, "http_sample_issuer", "_req"
        ),
        zmq_response_queue_addr=get_test_socket_path(
            tmp_path, "http_sample_issuer", "_resp"
        ),
        zmq_readiness_queue_addr=get_test_socket_path(
            tmp_path, "http_sample_issuer", "_ready"
        ),
    )

    client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
    client.start()

    yield client

    client.shutdown()


class TestHttpClientSampleIssuer:
    """Test HttpClientSampleIssuer functionality."""

    @pytest.mark.asyncio
    async def test_issuer_initialization(self, issuer_http_client):
        """Test HttpClientSampleIssuer initialization."""
        issuer = HttpClientSampleIssuer(issuer_http_client)

        assert issuer.http_client is issuer_http_client
        assert issuer.response_task is None
        assert issuer._shutdown is False
        assert issuer.n_inflight == 0
        assert not issuer._client_idle_event.is_set()

    @pytest.mark.asyncio
    async def test_issue_single_query(self, issuer_http_client):
        """Test issuing a single query and receiving response."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.start()

        try:
            # Create a sample (adapter will convert prompt to OpenAI messages format)
            sample = Sample(
                data={
                    "prompt": "Hello",
                    "model": "test-model",
                    "stream": False,
                }
            )

            # Mock the SampleEventHandler to capture events
            with patch(
                "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
            ) as mock_handler:
                # Issue the sample
                issuer.issue(sample)

                # Wait for completion
                completed = issuer.wait_for_all_complete(timeout=5.0)
                assert completed, "Query did not complete within timeout"

                # Verify the handler was called
                mock_handler.query_result_complete.assert_called_once()
                call_args = mock_handler.query_result_complete.call_args[0][0]
                assert isinstance(call_args, QueryResult)
                assert call_args.id == sample.uuid
                assert call_args.error is None

        finally:
            issuer.shutdown()

    @pytest.mark.asyncio
    async def test_issue_multiple_queries(self, issuer_http_client):
        """Test issuing multiple queries concurrently."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.start()

        try:
            samples = [
                Sample(
                    data={
                        "prompt": f"Query {i}",
                        "model": "test-model",
                        "stream": False,
                    }
                )
                for i in range(5)
            ]

            with patch(
                "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
            ) as mock_handler:
                # Issue all samples
                for sample in samples:
                    issuer.issue(sample)

                # Wait for all to complete
                completed = issuer.wait_for_all_complete(timeout=5.0)
                assert completed, "Not all queries completed within timeout"

                # Verify all were processed
                assert mock_handler.query_result_complete.call_count == 5

        finally:
            issuer.shutdown()

    @pytest.mark.asyncio
    async def test_handle_streaming_response(self, issuer_http_client):
        """Test handling streaming responses."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.start()

        try:
            sample = Sample(
                data={
                    "prompt": "Stream test",
                    "model": "test-model",
                    "stream": True,
                }
            )

            with patch(
                "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
            ) as mock_handler:
                issuer.issue(sample)

                # Wait for completion
                completed = issuer.wait_for_all_complete(timeout=5.0)
                assert completed

                # Should have received stream chunks and final result
                assert mock_handler.stream_chunk_complete.call_count > 0
                assert mock_handler.query_result_complete.call_count == 1

        finally:
            issuer.shutdown()

    @pytest.mark.asyncio
    async def test_handle_error_response(self, tmp_path):
        """Test handling error responses from HTTP endpoint."""
        # Use invalid endpoint to trigger error
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/invalid",
            num_workers=1,
        )
        # Create unique ZMQ config to avoid socket conflicts
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "error_test", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "error_test", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "error_test", "_ready"
            ),
        )

        error_client = HTTPEndpointClient(
            http_config,
            AioHttpConfig(client_timeout_total=1.0),
            zmq_config,
        )

        error_client.start()

        try:
            issuer = HttpClientSampleIssuer(error_client)
            issuer.start()

            sample = Sample(
                data={
                    "prompt": "Error test",
                    "model": "test-model",
                    "stream": False,
                }
            )

            with patch(
                "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
            ) as mock_handler:
                issuer.issue(sample)

                completed = issuer.wait_for_all_complete(timeout=5.0)
                assert completed

                # Should have received error response
                mock_handler.query_result_complete.assert_called_once()
                call_args = mock_handler.query_result_complete.call_args[0][0]
                assert isinstance(call_args, QueryResult)
                assert call_args.error is not None

            issuer.shutdown()

        finally:
            error_client.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_all_complete_timeout(self, issuer_http_client):
        """Test wait_for_all_complete with timeout."""
        issuer = HttpClientSampleIssuer(issuer_http_client)

        # Don't start response handler - queries won't complete
        sample = Sample(
            data={
                "prompt": "Timeout test",
                "model": "test-model",
                "stream": False,
            }
        )
        issuer.issue(sample)

        # Should timeout
        completed = issuer.wait_for_all_complete(timeout=0.1)
        assert not completed

    @pytest.mark.asyncio
    async def test_inflight_counter(self, issuer_http_client):
        """Test that inflight counter is properly maintained."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.start()

        try:
            assert issuer.n_inflight == 0

            sample = Sample(
                data={
                    "prompt": "Test",
                    "model": "test-model",
                    "stream": False,
                }
            )

            with patch(
                "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
            ):
                issuer.issue(sample)
                assert issuer.n_inflight == 1

                issuer.wait_for_all_complete(timeout=5.0)
                assert issuer.n_inflight == 0

        finally:
            issuer.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self, issuer_http_client):
        """Test that shutdown properly cleans up resources."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.start()

        assert issuer.response_task is not None

        issuer.shutdown()

        # Should set shutdown flag and cancel task
        assert issuer._shutdown is True

        # Give task a moment to be cancelled
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_handle_single_response_stream_chunk(self, issuer_http_client):
        """Test _handle_single_response with StreamChunk."""
        issuer = HttpClientSampleIssuer(issuer_http_client)

        chunk = StreamChunk(
            id="test-1",
            response_chunk="Hello",
            metadata={"first_chunk": True},
            is_complete=False,
        )

        with patch(
            "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
        ) as mock_handler:
            issuer._handle_single_response(chunk)
            mock_handler.stream_chunk_complete.assert_called_once_with(chunk)

    @pytest.mark.asyncio
    async def test_handle_single_response_query_result_success(
        self, issuer_http_client
    ):
        """Test _handle_single_response with successful QueryResult."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.n_inflight = 1

        result = QueryResult(id="test-1", response_output="Success", error=None)

        with patch(
            "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
        ) as mock_handler:
            issuer._handle_single_response(result)

            mock_handler.query_result_complete.assert_called_once_with(result)
            assert issuer.n_inflight == 0
            assert issuer._client_idle_event.is_set()

    @pytest.mark.asyncio
    async def test_handle_single_response_query_result_error(self, issuer_http_client):
        """Test _handle_single_response with error QueryResult."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.n_inflight = 1

        result = QueryResult(
            id="test-1", response_output=None, error="Connection failed"
        )

        with patch(
            "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
        ) as mock_handler:
            issuer._handle_single_response(result)

            mock_handler.query_result_complete.assert_called_once_with(result)
            assert issuer.n_inflight == 0
            assert issuer._client_idle_event.is_set()

    @pytest.mark.asyncio
    async def test_handle_single_response_invalid_type(self, issuer_http_client):
        """Test _handle_single_response with invalid response type."""
        issuer = HttpClientSampleIssuer(issuer_http_client)

        invalid_response = "not a valid response"

        with pytest.raises(ValueError, match="Unexpected response type"):
            issuer._handle_single_response(invalid_response)

    @pytest.mark.asyncio
    async def test_process_sample_data_not_implemented(self, issuer_http_client):
        """Test that process_sample_data raises NotImplementedError."""
        issuer = HttpClientSampleIssuer(issuer_http_client)

        with pytest.raises(NotImplementedError):
            issuer.process_sample_data("uuid", {"data": "test"})

    @pytest.mark.asyncio
    async def test_concurrent_issues_and_completions(self, issuer_http_client):
        """Test that issuer handles concurrent issues and completions correctly."""
        issuer = HttpClientSampleIssuer(issuer_http_client)
        issuer.start()

        try:
            with patch(
                "inference_endpoint.endpoint_client.http_sample_issuer.SampleEventHandler"
            ):
                # Issue multiple samples rapidly
                for i in range(10):
                    sample = Sample(
                        data={
                            "prompt": f"Test {i}",
                            "model": "test-model",
                            "stream": False,
                        }
                    )
                    issuer.issue(sample)

                # All should complete
                completed = issuer.wait_for_all_complete(timeout=10.0)
                assert completed
                assert issuer.n_inflight == 0

        finally:
            issuer.shutdown()
