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
Performance benchmarks for OpenAI msgspec types serialization using pytest-benchmark.

Measures ns/op for msgspec.json serialization of SSE*, ChatCompletion* types
with varying payload sizes (0, 100, 1k, 8k, 32k). Run with:

    pytest tests/performance/openai/test_types.py --benchmark-only --benchmark-columns=mean,stddev,ops

To save results for comparison:
    pytest tests/performance/openai/test_types.py --benchmark-only --benchmark-save=baseline

To compare against saved results:
    pytest tests/performance/openai/test_types.py --benchmark-only --benchmark-compare=baseline
"""

import msgspec
import pytest
from inference_endpoint.openai.types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatMessage,
    CompletionUsage,
    SSEChoice,
    SSEDelta,
    SSEMessage,
)

TEXT_SIZES = {
    "empty": "",
    "100": "x" * 100,
    "1k": "x" * 1_000,
    "8k": "x" * 8_000,
    "32k": "x" * 32_000,
}

JSON_ENCODER = msgspec.json.Encoder()
JSON_DECODER_SSE = msgspec.json.Decoder(SSEMessage)
JSON_DECODER_REQUEST = msgspec.json.Decoder(ChatCompletionRequest)
JSON_DECODER_RESPONSE = msgspec.json.Decoder(ChatCompletionResponse)


def make_sse_message(text: str) -> SSEMessage:
    """Create SSEMessage for benchmarks."""
    return SSEMessage(
        choices=[
            SSEChoice(
                delta=SSEDelta(content=text, reasoning=""),
                finish_reason=None,
            )
        ]
    )


def make_chat_request(text: str) -> ChatCompletionRequest:
    """Create ChatCompletionRequest for benchmarks."""
    return ChatCompletionRequest(
        model="gpt-4",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content=text),
        ],
        temperature=0.7,
        max_completion_tokens=1024,
    )


def make_chat_response(text: str) -> ChatCompletionResponse:
    """Create ChatCompletionResponse for benchmarks."""
    return ChatCompletionResponse(
        id="chatcmpl-test",
        created=1234567890,
        model="gpt-4",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    role="assistant",
                    content=text,
                    refusal=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
        system_fingerprint="fp_test",
    )


# SSE Message benchmarks
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_sse_encode(benchmark, size_name, text):
    """Benchmark SSEMessage encoding."""
    instance = make_sse_message(text)
    benchmark.group = "sse_message_encode"
    benchmark(JSON_ENCODER.encode, instance)


@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_sse_decode(benchmark, size_name, text):
    """Benchmark SSEMessage decoding."""
    instance = make_sse_message(text)
    encoded = JSON_ENCODER.encode(instance)
    benchmark.group = "sse_message_decode"
    benchmark(JSON_DECODER_SSE.decode, encoded)


# ChatCompletionRequest benchmarks
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_request_encode(benchmark, size_name, text):
    """Benchmark ChatCompletionRequest encoding."""
    instance = make_chat_request(text)
    benchmark.group = "chat_request_encode"
    benchmark(JSON_ENCODER.encode, instance)


@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_request_decode(benchmark, size_name, text):
    """Benchmark ChatCompletionRequest decoding."""
    instance = make_chat_request(text)
    encoded = JSON_ENCODER.encode(instance)
    benchmark.group = "chat_request_decode"
    benchmark(JSON_DECODER_REQUEST.decode, encoded)


# ChatCompletionResponse benchmarks
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_response_encode(benchmark, size_name, text):
    """Benchmark ChatCompletionResponse encoding."""
    instance = make_chat_response(text)
    benchmark.group = "chat_response_encode"
    benchmark(JSON_ENCODER.encode, instance)


@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
def test_response_decode(benchmark, size_name, text):
    """Benchmark ChatCompletionResponse decoding."""
    instance = make_chat_response(text)
    encoded = JSON_ENCODER.encode(instance)
    benchmark.group = "chat_response_decode"
    benchmark(JSON_DECODER_RESPONSE.decode, encoded)
