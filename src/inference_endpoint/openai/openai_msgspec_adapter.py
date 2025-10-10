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

"""
Fast msgspec-based OpenAI adapter for high-performance serialization/deserialization.

This adapter uses msgspec.Struct for zero-copy deserialization and efficient encoding,
providing significant performance improvements over Pydantic-based approaches.
"""

import re
import time

import msgspec
from inference_endpoint.core.types import Query, QueryResult

# Import shared SSE types from openai_adapter
from .openai_adapter import SSEMessage

# ============================================================================
# msgspec Structs for OpenAI API Types
# ============================================================================


class ChatMessage(msgspec.Struct, kw_only=True):
    """Chat message in OpenAI format."""

    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(msgspec.Struct, kw_only=True, omit_defaults=True):
    """OpenAI chat completion request (msgspec version)."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_completion_tokens: int = 100
    stream: bool = False
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class ChatCompletionResponseMessage(msgspec.Struct, kw_only=True, omit_defaults=True):
    """Response message from OpenAI."""

    role: str
    content: str | None = None
    refusal: str | None = None


class ChatCompletionChoice(msgspec.Struct, kw_only=True, omit_defaults=True):
    """A single choice in the completion response."""

    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str | None = None


class CompletionUsage(msgspec.Struct, kw_only=True, omit_defaults=True):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(msgspec.Struct, kw_only=True, omit_defaults=True):
    """OpenAI chat completion response (msgspec version)."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None


# ============================================================================
# msgspec-based OpenAI Adapter
# ============================================================================


class OpenAIMsgspecAdapter:
    """
    OpenAI adapter using msgspec for serialization/deserialization.
    """

    # Pre-compiled regex for extracting SSE data fields with JSON content
    SSE_DATA_PATTERN = re.compile(rb"data:\s*(\{[^\n]+\})", re.MULTILINE)

    # Reusable encoders/decoders for maximum performance
    _request_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        ChatCompletionResponse
    )
    _sse_decoder: msgspec.json.Decoder = msgspec.json.Decoder(SSEMessage)

    @classmethod
    def to_openai_request(cls, query: Query) -> ChatCompletionRequest:
        """
        Convert a Query to an OpenAI request struct.

        Args:
            query: Input query with prompt and parameters

        Returns:
            msgspec.Struct ChatCompletionRequest
        """
        if "prompt" not in query.data:
            raise ValueError("prompt not found in query.data")

        return ChatCompletionRequest(
            model=query.data.get("model", "no-model-name"),
            messages=[
                ChatMessage(role="user", content=query.data["prompt"]),
            ],
            stream=query.data.get("stream", False),
            max_completion_tokens=query.data.get("max_completion_tokens", 100),
            temperature=query.data.get("temperature", 0.7),
            top_p=query.data.get("top_p", 1.0),
            n=query.data.get("n", 1),
            presence_penalty=query.data.get("presence_penalty", 0.0),
            frequency_penalty=query.data.get("frequency_penalty", 0.0),
        )

    @classmethod
    def encode_request(cls, request: ChatCompletionRequest) -> bytes:
        """
        Encode request to JSON bytes using msgspec.

        Args:
            request: ChatCompletionRequest struct

        Returns:
            JSON bytes
        """
        return cls._request_encoder.encode(request)

    @classmethod
    def encode_response(cls, response: ChatCompletionResponse) -> bytes:
        """
        Encode response to JSON bytes using msgspec.

        Args:
            response: ChatCompletionResponse struct

        Returns:
            JSON bytes
        """
        return cls._response_encoder.encode(response)

    @classmethod
    def decode_response(cls, response_bytes: bytes) -> ChatCompletionResponse:
        """
        Decode response from JSON bytes using msgspec.

        Args:
            response_bytes: Raw JSON bytes from HTTP response

        Returns:
            ChatCompletionResponse struct
        """
        return cls._response_decoder.decode(response_bytes)

    @classmethod
    def from_openai_response(
        cls, response: ChatCompletionResponse, result_id: str | None = None
    ) -> QueryResult:
        """
        Convert an OpenAI response struct to a QueryResult.

        Args:
            response: ChatCompletionResponse struct
            result_id: Optional ID to use for the result (overrides response.id)

        Returns:
            QueryResult with extracted content
        """
        if not response.choices:
            raise ValueError("Response must contain at least one choice")

        return QueryResult(
            id=result_id or response.id,
            response_output=response.choices[0].message.content,
        )

    @classmethod
    def to_openai_response(cls, result: QueryResult) -> ChatCompletionResponse:
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

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> SSEMessage:
        """
        Decode SSE message from JSON bytes.

        Args:
            json_bytes: Raw JSON bytes from SSE stream

        Returns:
            SSEMessage struct
        """
        return cls._sse_decoder.decode(json_bytes)

    @classmethod
    def from_openai_request(cls, request: ChatCompletionRequest) -> Query:
        """
        Convert an OpenAI request to a Query.

        Args:
            request: ChatCompletionRequest struct

        Returns:
            Query with extracted prompt and parameters
        """
        if not request.messages or len(request.messages) == 0:
            raise ValueError("Request must contain at least one message")

        return Query(
            data={
                "prompt": request.messages[0].content,
                "model": request.model,
                "stream": request.stream,
            }
        )

    @classmethod
    def from_json_response(cls, query_id: str, response: dict) -> QueryResult:
        """
        Convert an OpenAI response dict to a QueryResult.

        This is a convenience method that decodes a dict response and converts it.

        Args:
            query_id: The ID of the query
            response: The OpenAI response dict to convert

        Returns:
            QueryResult object
        """
        # Decode the dict to a ChatCompletionResponse struct
        response_bytes = msgspec.json.encode(response)
        openai_response = cls.decode_response(response_bytes)
        return cls.from_openai_response(openai_response, result_id=query_id)
