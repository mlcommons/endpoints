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

from typing import Any

import msgspec

from ..core.types import Query, QueryResult
from ..endpoint_client.adapter_protocol import HttpRequestAdapter
from ..openai.harmony import Harmonizer


class SamplingParams(msgspec.Struct, kw_only=True, omit_defaults=True):
    max_new_tokens: int = 32768
    """int: Maximum number of tokens to generate per request (1-32768)"""

    temperature: float = 1.0
    """float: Sampling temperature (0.0 = deterministic, higher = more random). Typically 0.001-2."""

    top_k: int = -1
    """int: Top-k sampling (number of highest probability tokens to consider). -1 = disable"""

    top_p: float = 1.0
    """float: Top-p/nucleus sampling (cumulative probability threshold). 0.0-1.0, typically 1.0 for no filterin"""


class SGLangGenerateRequest(msgspec.Struct, kw_only=True, omit_defaults=True):
    input_ids: list[int]
    sampling_params: SamplingParams


class MetaInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    id: str
    finish_reason: dict[str, Any]
    prompt_tokens: int
    weight_version: str
    total_retractions: int
    completion_tokens: int
    cached_tokens: int
    e2e_latency: float


class SGLangGenerateResponse(msgspec.Struct, kw_only=True, omit_defaults=True):
    text: str
    output_ids: list[int]
    meta_info: MetaInfo


class SGLangSSEDelta(msgspec.Struct):
    text: str = ""
    token_delta: int = 0
    total_completion_tokens: int = 0
    has_retractions: bool = False


class SGLangGenerateAdapter(HttpRequestAdapter):
    """Adapter for the native SGLang /generate endpoint for text generation."""

    _harmonizer: Harmonizer | None = None
    _request_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        SGLangGenerateResponse
    )

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Encode a Query to bytes for HTTP transmission.

        There are 2 ways to do this:
        1. If "input_tokens" is present in query.data, we assume that the input is
           already harmonized and tokenized, so we can use it directly. The sampling
           parameters should be in query.data as well, either directly, or in a nested
           'sampling_parameters' key.
        2. If "prompt", "text_input", or "question" is present in query.data, we assume
           that this is the user prompt as plaintext, which needs to be harmonized and then
           tokenized. Likewise, here, the sampling parameters should be in query.data, or
           in a nested 'sampling_parameters' key.
        """

        # Get sampling parameters
        superset = query.data
        if "sampling_parameters" in query.data:
            superset = query.data["sampling_parameters"]

        # Default sampling parameters taken from the default values in the official MLPerf
        # Inference GPT-OSS implementation
        sampling_params = SamplingParams(
            max_new_tokens=superset.get("max_new_tokens", 10240),
            temperature=superset.get("temperature", 1.0),
            top_k=superset.get("top_k", -1),
            top_p=superset.get("top_p", 1.0),
        )

        # Get the input tokens
        if "input_tokens" in query.data:
            input_tokens = query.data["input_tokens"]
        else:
            input_text = None
            for key in ("prompt", "text_input", "question"):
                if key in query.data:
                    input_text = query.data[key]
                    break
            if input_text is None:
                raise ValueError("No input text found in query.data")

            harmony_parameters = query.metadata.get("harmony_parameters", {})
            if cls._harmonizer is None:
                # The tokenizer and harmony encoder are cached, so as long as
                # they are pre-loaded before the benchmark starts, there will be
                # no loading done on the hot-path of the benchmark session.
                cls._harmonizer = Harmonizer(**harmony_parameters)

            input_tokens = cls._harmonizer(input_text, tokenize=True)

        return cls._request_encoder.encode(
            SGLangGenerateRequest(
                input_ids=input_tokens,
                sampling_params=sampling_params,
            )
        )

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        resp = cls._response_decoder.decode(response_bytes)
        return QueryResult(
            id=query_id,
            response_output=resp.text,
            metadata={
                "token_ids": resp.output_ids,
                "n_tokens": resp.meta_info.completion_tokens,
            },
        )

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> SGLangSSEDelta:
        # SGLang uses the same response format for SSE chunks as with non-streaming so
        # we can use the same decoder, but the meaning of the fields is slightly different
        resp = cls._response_decoder.decode(json_bytes)

        # Taken from GPT-OSS MLPerf Inference reference implementation
        total_text = resp.text

        # The main difference is that although the text field contains the accumulated text,
        # the output_tokens is the delta from the previous chunk
        token_delta = resp.output_ids

        has_retractions = resp.meta_info.total_retractions > 0
        return SGLangSSEDelta(
            text=total_text,
            token_delta=token_delta,
            total_completion_tokens=resp.meta_info.completion_tokens,
            has_retractions=has_retractions,
        )
