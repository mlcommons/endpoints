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

import time

import msgspec
import orjson
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from .openai_types_gen import (
    ChatCompletionResponseMessage,
    Choice,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    FinishReason,
    Logprobs,
    ModelIdsShared,
    Object7,
    ReasoningEffort,
    Role,
    Role5,
    Role6,
    ServiceTier,
)


# msgspec structs for typed SSE message parsing (OpenAI streaming format)
class SSEDelta(msgspec.Struct):
    """SSE delta object containing content."""

    content: str = ""


class SSEChoice(msgspec.Struct):
    """SSE choice object containing delta."""

    delta: SSEDelta = msgspec.field(default_factory=SSEDelta)
    finish_reason: str | None = None


class SSEMessage(msgspec.Struct):
    """SSE message structure for OpenAI streaming responses."""

    choices: list[SSEChoice] = msgspec.field(default_factory=list)


class OpenAIAdapter(HttpRequestAdapter):
    """Adapter for OpenAI API."""

    @staticmethod
    def encode_query(query: Query) -> bytes:
        """Encode a Query to bytes for HTTP transmission."""
        request = OpenAIAdapter.to_endpoint_request(query)
        return OpenAIAdapter.encode_request(request)

    @staticmethod
    def decode_response(response_bytes: bytes, query_id: str) -> QueryResult:
        """Decode HTTP response bytes to QueryResult."""
        openai_response = OpenAIAdapter.decode_endpoint_response(response_bytes)
        return OpenAIAdapter.from_endpoint_response(openai_response, result_id=query_id)

    @staticmethod
    def decode_sse_message(json_bytes: bytes) -> str:
        """Decode SSE message and extract content string."""
        msg = msgspec.json.decode(json_bytes, type=SSEMessage)
        return msg.choices[0].delta.content

    # ========================================================================
    # Internal APIs
    # ========================================================================

    @staticmethod
    def to_endpoint_request(query: Query) -> CreateChatCompletionRequest:
        """Convert a Query to an OpenAI request."""
        if "prompt" not in query.data:
            raise ValueError("prompt not found in json_value")

        request = CreateChatCompletionRequest(
            model=ModelIdsShared(query.data.get("model", "no-model-name")),
            # service_tier=ServiceTier.auto,
            reasoning_effort=ReasoningEffort.medium,
            messages=[
                {
                    "role": Role.assistant.value,
                    "content": "You are a helpful assistant.",
                },
                {"role": Role5.user.value, "content": query.data["prompt"]},
            ],
            stream=query.data.get("stream", False),
            max_completion_tokens=query.data.get("max_completion_tokens", 100),
            temperature=query.data.get("temperature", 0.7),
        )
        return request

    @staticmethod
    def from_endpoint_response(
        response: CreateChatCompletionResponse,
        result_id: str | None = None,
    ) -> QueryResult:
        """Convert an OpenAI response to a QueryResult."""
        if not response.choices:
            raise ValueError("Response must contain at least one choice")

        if result_id is None:
            result_id = response.id

        return QueryResult(
            id=result_id,
            response_output=response.choices[0].message.content,
        )

    @staticmethod
    def to_endpoint_response(result: QueryResult) -> CreateChatCompletionResponse:
        """Convert a QueryResult to an OpenAI response."""
        return CreateChatCompletionResponse(
            id=result.id,
            choices=[
                Choice(
                    finish_reason=FinishReason.stop,
                    index=0,
                    message=ChatCompletionResponseMessage(
                        content=result.response_output, role=Role6.assistant, refusal=""
                    ),
                    logprobs=Logprobs(content=[], refusal=[]),
                )
            ],
            created=int(time.time()),
            model="model",
            object=Object7.chat_completion,
            service_tier=ServiceTier.auto,
        )

    @staticmethod
    def encode_request(request: CreateChatCompletionRequest) -> bytes:
        """Encode request to JSON bytes using orjson."""
        return orjson.dumps(request.model_dump(mode="json"))

    @staticmethod
    def decode_endpoint_response(response_bytes: bytes) -> CreateChatCompletionResponse:
        """Decode response from JSON bytes using orjson."""
        response_dict = orjson.loads(response_bytes)

        # Set default values for optional fields if missing
        response_dict["choices"][0]["message"]["refusal"] = "None"
        response_dict["choices"][0]["logprobs"] = {"content": [], "refusal": []}
        return CreateChatCompletionResponse(**response_dict, ignore_extra=True)
