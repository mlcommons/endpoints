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

import msgspec

from ..config.schema import ModelParams, StreamingMode
from ..core.types import Query, QueryResult
from ..dataset_manager.transforms import (
    AddStaticColumns,
    ColumnFilter,
    Harmonize,
    Transform,
)
from ..endpoint_client.adapter_protocol import HttpRequestAdapter
from .types import (
    SamplingParams,
    SGLangGenerateRequest,
    SGLangGenerateResponse,
    SGLangSSEDelta,
)


class SGLangGenerateAdapter(HttpRequestAdapter):
    """Adapter for the native SGLang /generate endpoint for text generation."""

    _request_encoder: msgspec.json.Encoder = msgspec.json.Encoder()
    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        SGLangGenerateResponse
    )

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        # Set model param defaults
        metadata = {
            "stream": (model_params.streaming == StreamingMode.ON),
            "max_new_tokens": 32768,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        }

        for attr in ["max_new_tokens", "temperature", "top_p", "top_k"]:
            if (v := getattr(model_params, attr)) is not None:
                metadata[attr] = v

        return [
            # SGLang /generate expects a pre-harmonized input with the key `input_tokens`
            Harmonize(),
            # Only keep the `input_tokens` column
            ColumnFilter(
                required_columns=["input_tokens"],
            ),
            # Add metadata columns since we don't want to do a dict update every iteration
            AddStaticColumns(metadata),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Encode a Query to bytes for HTTP transmission."""

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
        if "input_tokens" not in query.data:
            raise KeyError(f"input_tokens not found in query.data {query.data.keys()}")
        input_tokens = query.data["input_tokens"]

        return cls._request_encoder.encode(
            SGLangGenerateRequest(
                input_ids=input_tokens,
                sampling_params=sampling_params,
                stream=query.data.get("stream", False),
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
