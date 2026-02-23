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

import time

import msgspec
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from ..config.schema import ModelParams, StreamingMode
from ..dataset_manager.transforms import AddStaticColumns, ColumnFilter, Transform
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
    Role3,
    Role5,
    Role6,
    ServiceTier,
)
from .types import SSEMessage


class OpenAIAdapter(HttpRequestAdapter):
    """Adapter for OpenAI API."""

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        metadata = {
            "model": model_params.name,
            "stream": (model_params.streaming == StreamingMode.ON),
            "max_completion_tokens": model_params.max_new_tokens,
            "temperature": model_params.temperature,
            "top_p": model_params.top_p,
            "top_k": model_params.top_k,
            "repetition_penalty": model_params.repetition_penalty,
        }

        return [
            ColumnFilter(
                required_columns=["prompt"],
                optional_columns=["system"],
            ),
            AddStaticColumns(metadata),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Encode a Query to bytes for HTTP transmission."""
        request = cls.to_endpoint_request(query)
        return cls.encode_request(request)

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """Decode HTTP response bytes to QueryResult."""
        openai_response = cls.decode_endpoint_response(response_bytes)
        return cls.from_endpoint_response(openai_response, result_id=query_id)

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> str:
        """Decode SSE message and extract content string."""
        msg = msgspec.json.decode(json_bytes, type=SSEMessage)
        return msg.choices[0].delta

    # ========================================================================
    # Internal APIs
    # ========================================================================

    @classmethod
    def to_endpoint_request(cls, query: Query) -> CreateChatCompletionRequest:
        """Convert a Query to an OpenAI request."""
        if "prompt" not in query.data:
            raise ValueError("prompt not found in json_value")

        messages = [{"role": Role5.user.value, "content": query.data["prompt"]}]
        if "system" in query.data:
            messages.insert(
                0, {"role": Role3.system.value, "content": query.data["system"]}
            )

        request = CreateChatCompletionRequest(
            model=ModelIdsShared(query.data.get("model", "no-model-name")),
            reasoning_effort=ReasoningEffort.medium,
            messages=messages,
            stream=query.data.get("stream", False),
            max_completion_tokens=query.data.get("max_completion_tokens", 100),
            temperature=query.data.get("temperature", 0.7),
        )
        return request

    @classmethod
    def from_endpoint_response(
        cls,
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

    @classmethod
    def to_endpoint_response(cls, result: QueryResult) -> CreateChatCompletionResponse:
        """Convert a QueryResult to an OpenAI response."""
        return CreateChatCompletionResponse(
            id=result.id,
            choices=[
                Choice(
                    finish_reason=FinishReason.stop,
                    index=0,
                    message=ChatCompletionResponseMessage(
                        content=result.get_response_output_string(),
                        role=Role6.assistant,
                        refusal="",
                    ),
                    logprobs=Logprobs(content=[], refusal=[]),
                )
            ],
            created=int(time.time()),
            model="model",
            object=Object7.chat_completion,
            service_tier=ServiceTier.auto,
        )

    @classmethod
    def encode_request(cls, request: CreateChatCompletionRequest) -> bytes:
        """Encode request to JSON bytes using msgspec."""
        return msgspec.json.encode(request.model_dump(mode="json"))

    @classmethod
    def decode_endpoint_response(
        cls, response_bytes: bytes
    ) -> CreateChatCompletionResponse:
        """Decode response from JSON bytes using msgspec."""
        response_dict = msgspec.json.decode(response_bytes)

        # Set default values for optional fields if missing
        response_dict["choices"][0]["message"]["refusal"] = "None"
        response_dict["choices"][0]["logprobs"] = {"content": [], "refusal": []}
        if (
            "content" not in response_dict["choices"][0]["message"]
            or response_dict["choices"][0]["message"]["content"] is None
        ):
            response_dict["choices"][0]["message"]["content"] = "None"
        return CreateChatCompletionResponse(**response_dict, ignore_extra=True)
