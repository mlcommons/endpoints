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

"""
Tests demonstrating the usage of HTTP echo mock fixtures.

These tests show how to use the mock fixtures for testing HTTP clients
with real HTTP server that echoes requests back.
"""

import logging

import aiohttp
import pytest
from inference_endpoint.core.types import Query, TextModelOutput
from inference_endpoint.openai.openai_adapter import OpenAIAdapter
from inference_endpoint.openai.openai_types_gen import CreateChatCompletionResponse


class TestHttpEchoMockFixtures:
    """Test suite demonstrating HTTP mock fixture usage."""

    @pytest.mark.asyncio
    async def test_http_echo_server_post_request(self, mock_http_echo_server):
        """Test POST request to real HTTP server."""

        async with aiohttp.ClientSession() as session:
            payload = {
                "query": "What is machine learning?",
                "parameters": {"temperature": 0.7, "max_tokens": 150},
            }

            async with session.post(
                f"{mock_http_echo_server.url}/echo", json=payload
            ) as response:
                assert response.status == 200

                response_data = await response.json()

                # Verify echo response structure
                assert response_data["echo"] is True
                assert response_data["request"]["method"] == "POST"
                assert response_data["request"]["endpoint"] == "/echo"
                assert response_data["request"]["json_payload"] == payload

    @pytest.mark.asyncio
    async def test_mock_http_echo_server_chat_completions(self, mock_http_echo_server):
        """Test basic echo functionality of the real HTTP server."""

        # Make a real HTTP OpenAI chat completions request to the server
        async with aiohttp.ClientSession() as session:
            prompt_text = "Test prompt for mock server"
            payload = OpenAIAdapter.to_endpoint_request(
                Query(
                    id="test-chat-completions",
                    data={"prompt": prompt_text, "model": "gpt-3.5-turbo"},
                )
            ).model_dump(mode="json")

            logging.debug("payload", payload)
            async with session.post(
                f"{mock_http_echo_server.url}/v1/chat/completions", json=payload
            ) as response:
                assert response.status == 200

                json_payload = await response.json()
                query_result = OpenAIAdapter.from_endpoint_response(
                    CreateChatCompletionResponse(**json_payload)
                )

                assert query_result.response_output == TextModelOutput(
                    output=prompt_text
                )

    @pytest.mark.asyncio
    async def test_real_http_server_post_request_with_max_osl(
        self, mock_http_echo_server
    ):
        """Test POST request to real HTTP server."""
        old_max_osl = mock_http_echo_server.get_max_osl()
        mock_http_echo_server.set_max_osl(100)
        async with aiohttp.ClientSession() as session:
            prompt_text = "What is machine learning?"
            payload = OpenAIAdapter.to_endpoint_request(
                Query(
                    id="test-chat-completions",
                    data={"prompt": prompt_text, "model": "gpt-3.5-turbo"},
                )
            ).model_dump(mode="json")

            async with session.post(
                f"{mock_http_echo_server.url}/v1/chat/completions", json=payload
            ) as response:
                assert response.status == 200

                response_data = await response.json()
                response = OpenAIAdapter.from_endpoint_response(
                    CreateChatCompletionResponse.model_validate(response_data)
                )

                assert len(str(response.response_output)) == 100

        mock_http_echo_server.set_max_osl(5)
        async with aiohttp.ClientSession() as session:
            prompt_text = "What is machine learning?"
            payload = OpenAIAdapter.to_endpoint_request(
                Query(
                    id="test-chat-completions",
                    data={"prompt": prompt_text, "model": "gpt-3.5-turbo"},
                )
            ).model_dump(mode="json")

            async with session.post(
                f"{mock_http_echo_server.url}/v1/chat/completions", json=payload
            ) as response:
                assert response.status == 200
                response_data = await response.json()
                # Verify echo response structure
                response = OpenAIAdapter.from_endpoint_response(
                    CreateChatCompletionResponse.model_validate(response_data)
                )

                assert len(str(response.response_output)) == 5

        mock_http_echo_server.set_max_osl(old_max_osl)
