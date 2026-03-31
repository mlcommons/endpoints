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
from inference_endpoint.endpoint_client.config import HTTPClientConfig

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
    ):
        """Helper method to create a client with custom configuration."""
        http_config = HTTPClientConfig(
            endpoint_urls=[f"{vllm_docker_server['url']}/v1/chat/completions"],
            num_workers=num_workers,
            max_connections=50,
            warmup_connections=0,
        )

        # TODO(vir):
        # verify if this still works, custom http doesnt really support timeouts
        return FuturesHttpClient(config=http_config)

    # create client
    client = _create_custom_client(
        vllm_docker_server,
        num_workers=1,
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
