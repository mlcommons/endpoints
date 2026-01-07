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

"""Tests for endpoint_client accumulator implementations."""

from inference_endpoint.core.types import QueryResult
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.types import SSEDelta as OpenAISSEDelta
from inference_endpoint.sglang.accumulator import SGLangSSEAccumulator
from inference_endpoint.sglang.types import SGLangSSEDelta


class TestOpenAISSEAccumulator:
    """Test OpenAI-compatible SSE accumulator."""

    def test_streaming_accumulation(self):
        """Test full streaming lifecycle: first chunk, accumulation, final output."""
        acc = OpenAISSEAccumulator("q1", stream_all_chunks=True)

        # First chunk has metadata
        chunk1 = acc.add_chunk(OpenAISSEDelta(content="Hello"))
        assert chunk1 is not None
        assert chunk1.metadata["first_chunk"] is True
        assert chunk1.response_chunk == "Hello"

        # Subsequent chunks accumulate
        acc.first_chunk_sent = True
        chunk2 = acc.add_chunk(OpenAISSEDelta(content=" world"))
        assert chunk2 is not None
        assert chunk2.response_chunk == " world"

        # Empty delta ignored
        assert acc.add_chunk(OpenAISSEDelta()) is None

        # Wrong delta type ignored
        assert acc.add_chunk(SGLangSSEDelta(text="wrong")) is None

        # Final output
        result = acc.get_final_output()
        assert isinstance(result, QueryResult)
        assert result.id == "q1"
        assert result.metadata["final_chunk"] is True
        assert result.response_output["output"] == ("Hello", " world")

    def test_stream_all_chunks_disabled(self):
        """Only first chunk emitted when stream_all_chunks=False."""
        acc = OpenAISSEAccumulator("q1", stream_all_chunks=False)

        chunk1 = acc.add_chunk(OpenAISSEDelta(content="Hello"))
        acc.first_chunk_sent = True
        chunk2 = acc.add_chunk(OpenAISSEDelta(content=" world"))

        assert chunk1 is not None
        assert chunk2 is None  # Suppressed


class TestSGLangSSEAccumulator:
    """Test SGLang SSE accumulator."""

    def test_streaming_accumulation(self):
        """Test full streaming lifecycle: cumulative diff, dedup, final output."""
        acc = SGLangSSEAccumulator("q1", stream_all_chunks=True)

        # First chunk: text goes from "" to "Hello"
        chunk1 = acc.add_chunk(SGLangSSEDelta(text="Hello", total_completion_tokens=1))
        assert chunk1 is not None
        assert chunk1.response_chunk == "Hello"
        assert chunk1.metadata["first_chunk"] is True

        # Second chunk: cumulative text diff
        acc.first_chunk_sent = True
        chunk2 = acc.add_chunk(
            SGLangSSEDelta(text="Hello world", total_completion_tokens=2)
        )
        assert chunk2 is not None
        assert chunk2.response_chunk == " world"  # Diff only

        # Duplicate token count ignored
        assert (
            acc.add_chunk(SGLangSSEDelta(text="Hello world", total_completion_tokens=2))
            is None
        )

        # Wrong delta type ignored
        assert acc.add_chunk(OpenAISSEDelta(content="wrong")) is None

        # Final output is complete text
        result = acc.get_final_output()
        assert result.id == "q1"
        assert result.response_output == "Hello world"
