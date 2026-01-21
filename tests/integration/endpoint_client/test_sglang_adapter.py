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

"""Integration tests for SGLang adapter with real GPT-OSS server.

This test assumes a server running GPT-OSS is available at localhost:30000.
"""

import asyncio

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
)

from tests.futures_client import FuturesHttpClient

# Configuration for external GPT-OSS server
SGLANG_SERVER_HOST = "localhost"
SGLANG_SERVER_PORT = 30000
SGLANG_ENDPOINT = f"http://{SGLANG_SERVER_HOST}:{SGLANG_SERVER_PORT}/generate"


@pytest.fixture
def sglang_futures_client():
    """Create a FuturesHttpClient configured for SGLang endpoint.

    This fixture creates a client that connects to a GPT-OSS server
    running at localhost:30000.
    """
    http_config = HTTPClientConfig(
        endpoint_urls=[SGLANG_ENDPOINT],
        num_workers=4,
        api_type="sglang",
    )
    aiohttp_config = AioHttpConfig()

    client = FuturesHttpClient(http_config, aiohttp_config)
    yield client
    client.shutdown()


class TestSGLangAdapterIntegration:
    """Integration tests for SGLang adapter with real GPT-OSS server."""

    @pytest.mark.skip(
        reason="Running this test requires a running GPT-OSS server at localhost:30000."
    )
    @pytest.mark.asyncio
    @pytest.mark.run_explicitly
    @pytest.mark.integration
    async def test_sglang_non_streaming_request(self, sglang_futures_client):
        """Test non-streaming request through SGLang adapter.

        This test sends a single non-streaming request and verifies
        the response is properly decoded.
        """
        # Tokens are gotten from tokenizing "What is the capital of France?"
        # using the GPT-OSS-120b tokenizer from HuggingFace.
        input_tokens = [4827, 382, 290, 9029, 328, 10128, 30]
        query = Query(
            id="sglang-test-1",
            data={
                "input_tokens": input_tokens,
                "stream": False,
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        future = sglang_futures_client.issue(query)
        result = await asyncio.wrap_future(future)

        # Verify result structure
        assert result.id == "sglang-test-1"
        assert "response_output" in dir(result)
        assert result.response_output is not None
        assert len(result.response_output) > 0

        # Verify metadata
        assert result.metadata is not None
        assert "token_ids" in result.metadata
        assert "n_tokens" in result.metadata
        assert isinstance(result.metadata["token_ids"], list)
        assert isinstance(result.metadata["n_tokens"], int)

    @pytest.mark.skip(
        reason="Running this test requires a running GPT-OSS server at localhost:30000."
    )
    @pytest.mark.asyncio
    @pytest.mark.run_explicitly
    @pytest.mark.integration
    async def test_sglang_streaming_request(self, sglang_futures_client):
        """Test streaming request through SGLang adapter.

        This test sends a streaming request and verifies the response
        includes streaming chunks.
        """
        # Tokens are gotten from tokenizing "Tell me a short story about a robot."
        # using the GPT-OSS-120b tokenizer from HuggingFace.
        input_tokens = [60751, 668, 261, 4022, 4869, 1078, 261, 20808, 13]
        query = Query(
            id="sglang-test-stream-1",
            data={
                "input_tokens": input_tokens,
                "max_new_tokens": 100,
                "temperature": 0.8,
                "stream": True,
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )

        future = sglang_futures_client.issue(query)
        result = await asyncio.wrap_future(future)

        # Verify result structure
        assert result.id == "sglang-test-stream-1"
        assert "response_output" in dir(result)
        assert result.response_output is not None

        assert result.metadata is not None
        assert "token_ids" in result.metadata
        assert "n_tokens" in result.metadata
        assert isinstance(result.metadata["token_ids"], list)
        assert isinstance(result.metadata["n_tokens"], int)

        # Check that something was generated, but no more than max_new_tokens
        assert 0 < result.metadata["n_tokens"] and result.metadata["n_tokens"] <= 100

        # The token IDs in the result should be at most n_tokens because of retractions
        if result.metadata["retraction_occurred"]:
            assert len(result.metadata["token_ids"]) <= result.metadata["n_tokens"]
        else:
            # STOP token is not included in the response, but counts towards generated
            assert len(result.metadata["token_ids"]) + 1 == result.metadata["n_tokens"]
