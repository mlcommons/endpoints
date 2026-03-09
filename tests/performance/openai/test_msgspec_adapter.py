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
Performance benchmarks for OpenAIMsgspecAdapter using pytest-benchmark.

Measures ns/op for encode_query, decode_response, decode_sse_message
with varying payload sizes (0, 100, 1k, 8k, 32k). Run with:

    pytest tests/performance/openai/test_msgspec_adapter.py --benchmark-only --benchmark-columns=mean,stddev,ops
"""

import json

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.openai.openai_msgspec_adapter import OpenAIMsgspecAdapter

TEXT_SIZES = {
    "empty": "",
    "100": "x" * 100,
    "1k": "x" * 1_000,
    "8k": "x" * 8_000,
    "32k": "x" * 32_000,
}


def make_query(text: str) -> Query:
    """Create a Query for benchmarks."""
    return Query(
        id="test-id",
        data={"prompt": text, "model": "test-model", "max_completion_tokens": 100},
        headers={"Authorization": "Bearer token"},
    )


def make_response_bytes(text: str) -> bytes:
    """Create OpenAI-compatible response JSON bytes."""
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text, "refusal": None},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "system_fingerprint": "fp_test",
        }
    ).encode()


def make_sse_bytes(text: str) -> bytes:
    """Create SSE message JSON bytes."""
    return json.dumps(
        {
            "choices": [
                {"delta": {"content": text, "reasoning": ""}, "finish_reason": None}
            ]
        }
    ).encode()


@pytest.mark.performance
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_encode_query(benchmark, size_name, text):
    """Benchmark encode_query (Query -> HTTP bytes)."""
    query = make_query(text)
    benchmark.group = "msgspec_adapter_encode_query"
    benchmark(OpenAIMsgspecAdapter.encode_query, query)


@pytest.mark.performance
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_decode_response(benchmark, size_name, text):
    """Benchmark decode_response (HTTP bytes -> QueryResult)."""
    response_bytes = make_response_bytes(text)
    benchmark.group = "msgspec_adapter_decode_response"
    benchmark(OpenAIMsgspecAdapter.decode_response, response_bytes, "test-id")


@pytest.mark.performance
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_decode_sse(benchmark, size_name, text):
    """Benchmark decode_sse_message (SSE bytes -> content)."""
    sse_bytes = make_sse_bytes(text)
    benchmark.group = "msgspec_adapter_decode_sse"
    benchmark(OpenAIMsgspecAdapter.decode_sse_message, sse_bytes)
