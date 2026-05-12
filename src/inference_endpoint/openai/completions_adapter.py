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

"""OpenAI /v1/completions adapter for pre-tokenized prompts (bypasses server chat template)."""

import msgspec
from inference_endpoint.config.schema import ModelParams, StreamingMode
from inference_endpoint.core.types import Query, QueryResult, TextModelOutput
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnFilter,
    Harmonize,
    Transform,
)
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from .types import (
    SSEDelta,
    TextCompletionRequest,
    TextCompletionResponse,
    TextCompletionSSEMessage,
)


class OpenAITextCompletionsAdapter(HttpRequestAdapter):
    """Adapter for OpenAI /v1/completions endpoint with pre-tokenized input.

    Applies Harmonize() in dataset_transforms() to convert text prompts to
    token ID lists, then sends them as `prompt: [1, 2, ...]` to /v1/completions.
    This bypasses the server's chat template, which is required for gpt-oss-120b
    where the Harmony format must be applied client-side.
    """

    _request_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        TextCompletionResponse
    )
    _sse_decoder: msgspec.json.Decoder = msgspec.json.Decoder(TextCompletionSSEMessage)

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        metadata = {
            "model": model_params.name,
            "stream": (model_params.streaming == StreamingMode.ON),
            "max_tokens": model_params.max_new_tokens,
            "temperature": model_params.temperature,
            "top_p": model_params.top_p,
            "top_k": model_params.top_k,
            "repetition_penalty": model_params.repetition_penalty,
        }
        return [
            Harmonize(),
            ColumnFilter(
                required_columns=["input_tokens"],
                optional_columns=["n", "presence_penalty", "frequency_penalty", "stop"]
                + list(metadata.keys()),
            ),
            AddStaticColumns(metadata),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        if "input_tokens" not in query.data:
            raise KeyError(
                f"input_tokens not found in query.data: {list(query.data.keys())}"
            )
        return cls._request_encoder.encode(
            TextCompletionRequest(
                model=query.data.get("model", "no-model-name"),
                prompt=query.data["input_tokens"],
                stream=query.data.get("stream"),
                max_tokens=query.data.get("max_tokens"),
                temperature=query.data.get("temperature"),
                top_p=query.data.get("top_p"),
                top_k=query.data.get("top_k"),
                repetition_penalty=query.data.get("repetition_penalty"),
                n=query.data.get("n"),
                presence_penalty=query.data.get("presence_penalty"),
                frequency_penalty=query.data.get("frequency_penalty"),
                stop=query.data.get("stop"),
            )
        )

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        resp = cls._response_decoder.decode(response_bytes)
        if not resp.choices:
            raise ValueError("Response must contain at least one choice")
        return QueryResult(
            id=query_id,
            response_output=TextModelOutput(output=resp.choices[0].text),
        )

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> SSEDelta:
        msg = cls._sse_decoder.decode(json_bytes)
        if not msg.choices:
            return SSEDelta()
        return SSEDelta(content=msg.choices[0].text)
