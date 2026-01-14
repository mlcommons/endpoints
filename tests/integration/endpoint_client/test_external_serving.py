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

import asyncio
import logging
import time

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
)

from tests.futures_client import FuturesHttpClient


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("num_requests", [100])
async def test_external_serving(vllm_docker_server, streaming, num_requests):
    """Test high concurrent requests with proper connection management in streaming mode."""

    def _create_custom_client(
        vllm_docker_server,
        num_workers=1,
        aiohttp_config=None,
    ):
        """Helper method to create a client with custom configuration."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{vllm_docker_server['url']}/v1/chat/completions",
            num_workers=num_workers,
        )

        # Use provided aiohttp_config or create default
        if aiohttp_config is None:
            aiohttp_config = AioHttpConfig()

        return FuturesHttpClient(
            config=http_config,
            aiohttp_config=aiohttp_config,
        )

    # Configure aiohttp with connection limits to prevent "too many open files"
    # Limit concurrent connections to avoid exceeding system file descriptor limits
    aiohttp_config = AioHttpConfig(
        tcp_connector_limit=50,  # Total connection pool limit
        tcp_connector_limit_per_host=50,  # Limit per host (vLLM server)
        tcp_connector_force_close=False,  # Enable connection pooling/reuse
        client_timeout_total=300.0,
        client_timeout_connect=30.0,
        client_timeout_sock_read=60.0,
    )

    # create client with unlimited concurrency
    client = _create_custom_client(
        vllm_docker_server,
        num_workers=1,
        aiohttp_config=aiohttp_config,
    )

    try:
        # Collect futures
        start_time = time.time()
        futures = []
        for i in range(num_requests):
            query = Query(
                id=f"massive-streaming-{i}",
                data={
                    "prompt": f"Streaming request {i}",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "stream": streaming,
                },
            )
            future = client.issue(query)
            futures.append(future)

        # Wait for all futures to complete
        results = await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
        end_time = time.time()

        # Verify results
        assert len(results) == num_requests
        result_ids = {r.id for r in results}
        expected_ids = {f"massive-streaming-{i}" for i in range(num_requests)}
        assert result_ids == expected_ids

        # Print performance metrics
        duration = end_time - start_time
        rps = num_requests / duration
        logging.info(
            f"\nStreaming mode performance: {num_requests} requests in {duration:.2f}s = {rps:.0f} RPS"
        )
    finally:
        client.shutdown()
