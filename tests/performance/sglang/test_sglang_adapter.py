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
Performance benchmarks for SGLangGenerateAdapter using pytest-benchmark.

Measures ns/op for encode_query, decode_response, decode_sse_message
with varying payload sizes (0, 100, 1k, 8k, 32k). Run with:

    pytest tests/performance/sglang/test_sglang_adapter.py --benchmark-only --benchmark-columns=mean,stddev,ops
"""

import json

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.sglang.adapter import SGLangGenerateAdapter

TOKEN_SIZES = {
    "empty": [],
    "100": list(range(100)),
    "1k": list(range(1_000)),
    "8k": list(range(8_000)),
    "32k": list(range(32_000)),
}


def make_query(input_tokens: list[int]) -> Query:
    """Create a Query for benchmarks."""
    return Query(
        id="test-id",
        data={
            "input_tokens": input_tokens,
            "max_new_tokens": 100,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "stream": False,
        },
        headers={},
    )


def make_response_bytes(text: str, n_tokens: int) -> bytes:
    """Create SGLang /generate response JSON bytes."""
    return json.dumps(
        {
            "text": text,
            "output_ids": list(range(n_tokens)),
            "meta_info": {
                "id": "test-id",
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 10,
                "weight_version": "v1",
                "total_retractions": 0,
                "completion_tokens": n_tokens,
                "cached_tokens": 0,
                "e2e_latency": 0.1,
            },
        }
    ).encode()


# Map token sizes to equivalent text/token counts for response benchmarks
RESPONSE_SIZES = {
    "empty": ("", 0),
    "100": ("x" * 100, 100),
    "1k": ("x" * 1_000, 1_000),
    "8k": ("x" * 8_000, 8_000),
    "32k": ("x" * 32_000, 32_000),
}


@pytest.mark.performance
@pytest.mark.parametrize(
    "size_name,input_tokens", TOKEN_SIZES.items(), ids=TOKEN_SIZES.keys()
)
def test_encode_query(benchmark, size_name, input_tokens):
    """Benchmark encode_query (Query -> HTTP bytes)."""
    query = make_query(input_tokens)
    benchmark.group = "sglang_adapter_encode_query"
    benchmark(SGLangGenerateAdapter.encode_query, query)


@pytest.mark.performance
@pytest.mark.parametrize(
    "size_name,text_and_tokens", RESPONSE_SIZES.items(), ids=RESPONSE_SIZES.keys()
)
def test_decode_response(benchmark, size_name, text_and_tokens):
    """Benchmark decode_response (HTTP bytes -> QueryResult)."""
    text, n_tokens = text_and_tokens
    response_bytes = make_response_bytes(text, n_tokens)
    benchmark.group = "sglang_adapter_decode_response"
    benchmark(SGLangGenerateAdapter.decode_response, response_bytes, "test-id")


@pytest.mark.performance
@pytest.mark.parametrize(
    "size_name,text_and_tokens", RESPONSE_SIZES.items(), ids=RESPONSE_SIZES.keys()
)
def test_decode_sse(benchmark, size_name, text_and_tokens):
    """Benchmark decode_sse_message (SSE bytes -> SGLangSSEDelta)."""
    text, n_tokens = text_and_tokens
    sse_bytes = make_response_bytes(text, n_tokens)
    benchmark.group = "sglang_adapter_decode_sse"
    benchmark(SGLangGenerateAdapter.decode_sse_message, sse_bytes)
