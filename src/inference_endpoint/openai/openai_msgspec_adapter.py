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
Msgspec-based OpenAI adapter for fast serialization/deserialization.
"""

import time

import msgspec
from inference_endpoint.config.schema import ModelParams, StreamingMode
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnFilter,
    Transform,
)

# Import base class and shared SSE types
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from .types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatMessage,
    SSEMessage,
)

# ============================================================================
# msgspec-based OpenAI Adapter
# ============================================================================


class OpenAIMsgspecAdapter(HttpRequestAdapter):
    """OpenAI adapter using msgspec for serialization/deserialization."""

    # Reusable encoders/decoders
    _request_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        ChatCompletionResponse
    )
    _sse_decoder: msgspec.json.Decoder = msgspec.json.Decoder(SSEMessage)

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

        # These fields are used in .to_endpoint_request() but don't exist in ModelParams,
        # so they currently cannot be configured unless they are specified in the dataset file
        # or added with a custom transform in the pipeline.
        # See: https://platform.openai.com/docs/api-reference/chat/create for more details on
        # what the fields mean.
        allowed = [
            "name",  # NOT the model name, but rather a proper noun like 'Bob' for the LLM to keep track of entities
            "n",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "logit_bias",
            "user",
            "chat_template",
        ]
        return [
            ColumnFilter(
                required_columns=["prompt"],
                optional_columns=["system"]
                + allowed,  # Allow for custom passthrough for OpenAI params
            ),
            AddStaticColumns(metadata),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Encode a Query directly to bytes for HTTP transmission."""
        request = cls.to_endpoint_request(query)
        return cls.encode_request(request)

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """Decode HTTP response bytes directly to QueryResult."""
        openai_response = cls.decode_endpoint_response(response_bytes)
        return cls.from_endpoint_response(openai_response, result_id=query_id)

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> str:
        """Decode SSE message and extract content string."""
        msg = cls._sse_decoder.decode(json_bytes)
        return msg.choices[0].delta

    # ========================================================================
    # Internal APIs
    # ========================================================================

    @classmethod
    def to_endpoint_request(cls, query: Query) -> ChatCompletionRequest:
        """
        Convert a Query to an OpenAI request struct.

        Args:
            query: Input query with prompt and parameters

        Returns:
            msgspec.Struct ChatCompletionRequest
        """
        if "prompt" not in query.data:
            raise ValueError("prompt not found in query.data")

        messages = [
            ChatMessage(
                role="user",
                content=query.data["prompt"],
                name=query.data.get("name"),
            ),
        ]
        if "system" in query.data:
            messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=query.data["system"],
                ),
            )

        return ChatCompletionRequest(
            model=query.data.get("model", "no-model-name"),
            messages=messages,
            stream=query.data.get("stream"),
            max_completion_tokens=query.data.get("max_completion_tokens"),
            temperature=query.data.get("temperature"),
            top_p=query.data.get("top_p"),
            top_k=query.data.get("top_k"),
            repetition_penalty=query.data.get("repetition_penalty"),
            n=query.data.get("n"),
            presence_penalty=query.data.get("presence_penalty"),
            frequency_penalty=query.data.get("frequency_penalty"),
            stop=query.data.get("stop"),
            logit_bias=query.data.get("logit_bias"),
            user=query.data.get("user"),
            chat_template=query.data.get("chat_template", None),
        )

    @classmethod
    def encode_request(cls, request: ChatCompletionRequest) -> bytes:
        """Encode request to JSON bytes using msgspec."""
        return cls._request_encoder.encode(request)

    @classmethod
    def decode_endpoint_response(cls, response_bytes: bytes) -> ChatCompletionResponse:
        """Decode response from JSON bytes using msgspec."""
        return cls._response_decoder.decode(response_bytes)

    @classmethod
    def from_endpoint_response(
        cls, response: ChatCompletionResponse, result_id: str | None = None
    ) -> QueryResult:
        """Convert an OpenAI response struct to a QueryResult."""
        if not response.choices:
            raise ValueError("Response must contain at least one choice")

        return QueryResult(
            id=result_id or response.id,
            response_output=response.choices[0].message.content,
        )

    @classmethod
    def to_endpoint_response(cls, result: QueryResult) -> ChatCompletionResponse:
        """
        Convert a QueryResult to an OpenAI response struct.

        Args:
            result: QueryResult to convert

        Returns:
            ChatCompletionResponse struct
        """
        return ChatCompletionResponse(
            id=result.id,
            created=int(time.time()),
            model="model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(
                        role="assistant",
                        content=result.response_output,
                    ),
                    finish_reason="stop",
                )
            ],
        )
