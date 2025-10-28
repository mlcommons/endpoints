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

import re
import time

import msgspec
from inference_endpoint.core.types import Query, QueryResult

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


class OpenAIAdapter:
    """Adapter for OpenAI API."""

    # Pre-compiled regex for extracting SSE data fields with JSON content
    # Matches "data: {json content}" and captures the JSON part
    SSE_DATA_PATTERN = re.compile(rb"data:\s*(\{[^\n]+\})", re.MULTILINE)

    @staticmethod
    def to_openai_request(query: Query) -> CreateChatCompletionRequest:
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
    def from_openai_request(request: CreateChatCompletionRequest) -> Query:
        """Convert an OpenAI request to a Query."""
        if not request.messages or len(request.messages) == 0:
            raise ValueError("Request must contain at least one message")
        return Query(
            data={
                "prompt": request.messages[0].root.content,
                "model": request.model,
                "stream": request.stream,
            },
        )

    @staticmethod
    def from_openai_response(
        response: CreateChatCompletionResponse,
        result_id: str | None = None,
    ) -> QueryResult:
        """Convert an OpenAI response to a QueryResult.
        Args:
            response: The OpenAI response to convert.
            result_id: If provided, use this as the ID for the QueryResult. Otherwise,
                       uses the response ID from the OpenAI response. This is useful
                       since QueryResult is a frozen dataclass, and `id` cannot be changed
                       after creation. (Default: None)
        Returns:
            A QueryResult object.
        """
        if not response.choices:
            raise ValueError("Response must contain at least one choice")

        if result_id is None:
            result_id = response.id

        return QueryResult(
            id=result_id,
            response_output=response.choices[0].message.content,
        )

    @staticmethod
    def from_json_response(query_id, response: dict) -> QueryResult:
        """Convert an OpenAI response data to a QueryResult.
        Note that this function fixes the fields to be compatible with
        OpenAI pydantic definitions. This includes updating the refusal and
        logprobs fields to be compatible with the OpenAI pydantic definitions.
        Args:
            query_id: The ID of the query.
            response: The OpenAI response data to convert.
        Returns:
            A QueryResult object.
        """
        response["choices"][0]["message"]["refusal"] = "None"
        response["choices"][0]["logprobs"] = {"content": [], "refusal": []}
        return OpenAIAdapter.from_openai_response(
            CreateChatCompletionResponse(**response, ignore_extra=True),
            result_id=query_id,
        )

    @staticmethod
    def to_openai_response(result: QueryResult) -> CreateChatCompletionResponse:
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
