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
        """Create unique ZMQ configuration for each test."""
        # Use tmp_path for unique socket paths per test
        return ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_worker", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_worker", "_resp"
            ),
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

            request_push.setsockopt(zmq.SNDHWM, zmq_config.zmq_high_water_mark)
            response_pull.setsockopt(zmq.RCVHWM, zmq_config.zmq_high_water_mark)

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
                    response_data = await asyncio.wait_for(
                        response_pull.recv(), timeout=3.0
                    )
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
            await asyncio.wait_for(worker_task, timeout=2.0)

        finally:
            request_push.close()
            response_pull.close()
            context.destroy(linger=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sig",
        [signal.SIGTERM, signal.SIGINT],
        ids=["SIGTERM", "SIGINT"],
    )
    async def test_worker_signal_handling(self, worker_config, tmp_path, sig):
        """Test worker responds to signals correctly."""
        http_config, aiohttp_config = worker_config

        # Use a short timeout for this test so worker exits quickly
        test_zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, f"test_worker_sig_{sig.name}", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, f"test_worker_sig_{sig.name}", "_resp"
            ),
            zmq_recv_timeout=100,  # Short timeout (100ms) for fast shutdown
        )

        # Create worker
        worker = Worker(
            worker_id=0,
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=test_zmq_config,
            request_socket_addr=f"{test_zmq_config.zmq_request_queue_prefix}_0_requests",
            response_socket_addr=test_zmq_config.zmq_response_queue_addr,
            readiness_socket_addr=test_zmq_config.zmq_readiness_queue_addr,
        )

        context = zmq.asyncio.Context()

        try:
            # Create sockets - bind before worker starts so worker can connect
            response_pull = context.socket(zmq.PULL)
            response_pull.bind(test_zmq_config.zmq_response_queue_addr)

            readiness_pull = context.socket(zmq.PULL)
            readiness_pull.bind(test_zmq_config.zmq_readiness_queue_addr)

            # Start worker
            worker_task = asyncio.create_task(worker.run())

            # Wait for worker readiness signal
            await asyncio.wait_for(readiness_pull.recv(), timeout=2.0)

            # Verify worker is running
            assert not worker._shutdown

            # Send signal via shutdown method (simulates signal handler)
            worker.shutdown(sig, None)

            # Verify shutdown flag is set
            assert worker._shutdown

            # Worker should exit gracefully after the receive timeout
            await asyncio.wait_for(worker_task, timeout=1.0)

        finally:
            readiness_pull.close()
            response_pull.close()
            context.destroy(linger=0)
