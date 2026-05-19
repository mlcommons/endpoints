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

import json
import logging

import aiohttp
import pytest
from aiohttp import web
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


@pytest.mark.unit
class TestHttpEchoServerFactory:
    """Tests for the mock_http_echo_server_factory fixture."""

    @pytest.mark.asyncio
    async def test_factory_default_is_echo(self, mock_http_echo_server_factory):
        """Factory with no handler behaves identically to mock_http_echo_server."""
        server = mock_http_echo_server_factory()
        async with aiohttp.ClientSession() as session:
            payload = {"query": "hello"}
            async with session.post(f"{server.url}/echo", json=payload) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["echo"] is True
                assert data["request"]["json_payload"] == payload

    @pytest.mark.asyncio
    async def test_factory_sync_handler_overrides_response(
        self, mock_http_echo_server_factory
    ):
        """A plain (sync) lambda completely replaces the built-in handler."""
        server = mock_http_echo_server_factory(
            lambda req: web.json_response({"custom": True}, status=201)
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server.url}/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
            ) as resp:
                assert resp.status == 201
                data = await resp.json()
                assert data == {"custom": True}

    @pytest.mark.asyncio
    async def test_factory_async_handler_overrides_response(
        self, mock_http_echo_server_factory
    ):
        """An async handler is awaited and its response is returned."""

        async def async_handler(request: web.Request) -> web.Response:
            body = await request.json()
            return web.json_response({"received_model": body.get("model")})

        server = mock_http_echo_server_factory(async_handler)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server.url}/v1/chat/completions",
                json={"model": "my-model", "messages": []},
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data == {"received_model": "my-model"}

    @pytest.mark.asyncio
    async def test_factory_handler_receives_request_body(
        self, mock_http_echo_server_factory
    ):
        """Handler can inspect the raw request body for assertions."""
        captured: list[bytes] = []

        async def capturing_handler(request: web.Request) -> web.Response:
            captured.append(await request.read())
            return web.json_response({"ok": True})

        server = mock_http_echo_server_factory(capturing_handler)
        payload = {"hello": "world"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server.url}/v1/chat/completions", json=payload
            ) as resp:
                assert resp.status == 200

        assert len(captured) == 1
        assert json.loads(captured[0]) == payload

    @pytest.mark.asyncio
    async def test_factory_handler_applies_to_echo_route_too(
        self, mock_http_echo_server_factory
    ):
        """Custom handler intercepts /echo requests as well."""
        server = mock_http_echo_server_factory(
            lambda req: web.json_response({"intercepted": True})
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server.url}/echo", json={}) as resp:
                assert resp.status == 200
                assert (await resp.json()) == {"intercepted": True}

    @pytest.mark.asyncio
    async def test_factory_creates_independent_servers(
        self, mock_http_echo_server_factory
    ):
        """Two servers from the same factory operate independently."""
        echo_server = mock_http_echo_server_factory()
        custom_server = mock_http_echo_server_factory(
            lambda req: web.json_response({"server": "custom"})
        )
        assert echo_server.url != custom_server.url

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{custom_server.url}/echo", json={}) as resp:
                data = await resp.json()
                assert data == {"server": "custom"}

            async with session.post(f"{echo_server.url}/echo", json={"x": 1}) as resp:
                data = await resp.json()
                assert data["echo"] is True
