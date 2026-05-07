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

from inference_endpoint.core.types import Query, QueryResult, TextModelOutput
from inference_endpoint.openai.openai_adapter import OpenAIAdapter
from inference_endpoint.openai.openai_types_gen import (
    ChatCompletionResponseMessage,
    Choice,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    Logprobs,
    Object7,
    ReasoningEffort,
    Role6,
    ServiceTier,
)


class TestOpenAIAPITypes:
    def test_create_chat_completion_request(self):
        query = CreateChatCompletionRequest(
            service_tier=ServiceTier.auto,
            reasoning_effort=ReasoningEffort.medium,
            model="test-model",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test prompt"},
            ],
        )
        assert query.model_dump(mode="json")["messages"] == [
            {
                "role": "developer",
                "content": "You are a helpful assistant.",
                "name": None,
            },
            {
                "role": "user",
                "content": "Test prompt",
                "name": None,
            },
        ]

    def test_create_chat_completion_request_from_query(self):
        query = OpenAIAdapter.to_endpoint_request(
            Query(
                id="test-123",
                data={"model": "test-model", "prompt": "Test prompt"},
            )
        )

        messages = query.model_dump(mode="json")["messages"]
        assert len(messages) == 1, f"Expected 1 messages, got {len(messages)}"
        # TODO : cleanup this once we have a way to handle the assistant message
        for message in messages:
            assert message["role"] in [
                "assistant",
                "user",
            ], f"Expected role to be assistant or user, got {message['role']}"
            assert (
                message["content"] is not None
            ), f"Expected content to be not None, got {message['content']}"
            assert (
                message["name"] is None
            ), f"Expected name to be None, got {message['name']}"
            if message["role"] == "user":
                assert message["content"] == "Test prompt"
            # TODO : cleanup this once we have a way to handle the assistant message
            #     assert message["content"] == "You are a helpful assistant."

    def test_create_chat_completion_response_from_query_result(self):
        message_content = "You are a helpful assistant."
        message = ChatCompletionResponseMessage(
            role=Role6.assistant,
            refusal="",
            content=message_content,
        )
        choices = [
            Choice(
                role=Role6.assistant,
                finish_reason="stop",
                index=0,
                message=message,
                logprobs=Logprobs(
                    content=[],
                    refusal=[],
                ),
            )
        ]
        created_time = 1715328000
        model_name = "test-model"
        response = CreateChatCompletionResponse(
            id="test-id",
            object=Object7.chat_completion,
            created=created_time,
            model=model_name,
            choices=choices,
            service_tier=ServiceTier.auto,
        )

        assert response.model_dump(mode="json") == {
            "id": "test-id",
            "object": "chat.completion",
            "created": created_time,
            "model": model_name,
            "system_fingerprint": None,
            "service_tier": ServiceTier.auto.value,
            "usage": None,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": message_content,
                        "refusal": "",
                        "tool_calls": None,
                        "annotations": None,
                        "role": Role6.assistant.value,
                        "function_call": None,
                        "audio": None,
                    },
                    "logprobs": {"content": [], "refusal": []},
                }
            ],
        }

    def test_create_chat_completion_response(self):
        message_content = "You are a helpful assistant."
        response = OpenAIAdapter.to_endpoint_response(
            QueryResult(
                id="test-123", response_output=TextModelOutput(output=message_content)
            )
        ).model_dump(mode="json")
        assert response["choices"][0]["message"]["content"] == message_content
        assert response["id"] == "test-123"
        assert response["object"] == "chat.completion"
        assert response["system_fingerprint"] is None
        assert response["service_tier"] == ServiceTier.auto.value
        assert response["usage"] is None
