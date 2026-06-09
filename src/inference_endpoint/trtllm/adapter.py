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

"""TRT-LLM adapter for /v1/chat/completions with prompt_token_ids."""

import msgspec

from ..config.schema import ModelParams, StreamingMode
from ..core.types import Query, QueryResult
from ..dataset_manager.transforms import (
    AddStaticColumns,
    ColumnFilter,
    ColumnRemap,
    Transform,
)
from ..endpoint_client.adapter_protocol import HttpRequestAdapter
from ..openai.types import (
    ChatCompletionResponse,
    SSEDelta,
    SSEMessage,
)
from .types import TRTLLMChatRequest


class TRTLLMAdapter(HttpRequestAdapter):
    """Adapter for TRT-LLM endpoints using /v1/chat/completions with prompt_token_ids.

    TRT-LLM implements an OpenAI-compatible API where the request includes
    prompt_token_ids for pre-tokenized input. Responses (both SSE and
    non-streaming) follow the standard OpenAI format, so we reuse OpenAI's
    response types for decoding.
    """

    _request_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        ChatCompletionResponse
    )
    _sse_decoder: msgspec.json.Decoder = msgspec.json.Decoder(SSEMessage)

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        metadata: dict = {
            "stream": (model_params.streaming == StreamingMode.ON),
            "max_tokens": model_params.max_new_tokens,
        }

        for attr in ["temperature", "top_p", "top_k"]:
            if (v := getattr(model_params, attr, None)) is not None:
                metadata[attr] = v

        if model_params.name:
            metadata["model"] = model_params.name

        return [
            # Normalize common token column names to prompt_token_ids
            ColumnRemap(
                {("tok_input", "input_ids", "input_tokens"): "prompt_token_ids"},
            ),
            ColumnFilter(
                required_columns=["prompt_token_ids"],
                optional_columns=["model"],
            ),
            AddStaticColumns(metadata),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Encode a Query to bytes for HTTP transmission."""
        prompt_token_ids = query.data["prompt_token_ids"]

        request = TRTLLMChatRequest(
            messages=[],
            prompt_token_ids=prompt_token_ids,
            model=query.data.get("model", "no-model-name"),
            max_tokens=query.data.get("max_tokens", 1024),
            temperature=query.data.get("temperature"),
            top_p=query.data.get("top_p"),
            top_k=query.data.get("top_k"),
            min_tokens=query.data.get("min_tokens"),
            skip_special_tokens=query.data.get("skip_special_tokens"),
            stop_token_ids=query.data.get("stop_token_ids"),
            stream=query.data.get("stream", False),
        )
        return cls._request_encoder.encode(request)

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """Decode HTTP response bytes to QueryResult using OpenAI response types."""
        resp = cls._response_decoder.decode(response_bytes)

        metadata: dict = {}
        if resp.usage:
            metadata["n_tokens"] = resp.usage.completion_tokens
        if resp.choices and resp.choices[0].finish_reason:
            metadata["finish_reason"] = resp.choices[0].finish_reason

        return QueryResult(
            id=query_id,
            response_output=resp.choices[0].message.content if resp.choices else None,
            metadata=metadata,
        )

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> SSEDelta:
        """Decode SSE message using OpenAI SSE types (identical format)."""
        msg = cls._sse_decoder.decode(json_bytes)
        return msg.choices[0].delta
