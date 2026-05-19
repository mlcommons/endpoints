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

"""Unit tests for OpenAISSEAccumulator tool-call handling."""

import pytest
from inference_endpoint.core.types import StreamChunk, TextModelOutput
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.types import SSEChoice, SSEDelta


def _make_tc_partial(idx: int, tc_id: str, name: str, args: str) -> SSEChoice:
    """Create an SSEChoice with a single partial tool_call delta."""
    return SSEChoice(
        delta=SSEDelta(
            tool_calls=[
                {
                    "index": idx,
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ]
        )
    )


def _make_content_choice(content: str, first_chunk: bool = False) -> SSEChoice:
    return SSEChoice(delta=SSEDelta(content=content))


def _make_reasoning_choice(reasoning: str) -> SSEChoice:
    return SSEChoice(delta=SSEDelta(reasoning_content=reasoning))


def _make_finish_choice() -> SSEChoice:
    return SSEChoice(finish_reason="tool_calls")


@pytest.mark.unit
class TestAccumulatorPureToolCalls:
    """Pure tool-call stream (no content/reasoning text chunks)."""

    def test_tool_calls_in_text_output(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_tc_partial(0, "c1", "search", '{"q":'))
        acc.add_chunk(
            SSEChoice(
                delta=SSEDelta(
                    tool_calls=[{"index": 0, "function": {"arguments": '"test"}'}}]
                )
            )
        )
        acc.add_chunk(_make_finish_choice())

        result = acc.get_final_output()
        assert isinstance(result.response_output, TextModelOutput)
        assert result.response_output.tool_calls is not None
        assert len(result.response_output.tool_calls) == 1
        assert result.response_output.tool_calls[0]["function"]["name"] == "search"
        assert (
            result.response_output.tool_calls[0]["function"]["arguments"]
            == '{"q":"test"}'
        )

    def test_metadata_tool_calls_preserved(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_tc_partial(0, "c1", "f", "{}"))
        acc.add_chunk(_make_finish_choice())

        result = acc.get_final_output()
        assert "tool_calls" in result.metadata
        assert result.metadata["tool_calls"][0]["function"]["name"] == "f"

    def test_first_tool_call_delta_emits_sentinel_stream_chunk(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        sentinel = acc.add_chunk(_make_tc_partial(0, "c1", "f", "{}"))

        assert isinstance(sentinel, StreamChunk)
        assert sentinel.id == "qid"
        assert sentinel.response_chunk == ""
        assert sentinel.metadata.get("first_chunk") is True

    def test_subsequent_tool_call_deltas_return_none(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        # First delta: sentinel emitted
        acc.add_chunk(_make_tc_partial(0, "c1", "f", "{"))
        # Second delta: no sentinel
        second = acc.add_chunk(
            SSEChoice(
                delta=SSEDelta(
                    tool_calls=[{"index": 0, "function": {"arguments": "}"}}]
                )
            )
        )
        assert second is None

    def test_first_chunk_sent_after_sentinel(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_tc_partial(0, "c1", "f", "{}"))
        assert acc.first_chunk_sent is True


@pytest.mark.unit
class TestAccumulatorMixedReasoningAndToolCalls:
    """Mixed stream: reasoning followed by tool_calls."""

    def test_reasoning_chunk_is_first_chunk_not_tool_call(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        reasoning_chunk = acc.add_chunk(_make_reasoning_choice("Let me think"))
        # Reasoning chunk should be the first chunk
        assert isinstance(reasoning_chunk, StreamChunk)
        assert reasoning_chunk.metadata.get("first_chunk") is True

        # Now a tool_call delta should NOT emit another sentinel (first_chunk_sent=True)
        tc_chunk = acc.add_chunk(_make_tc_partial(0, "c1", "f", "{}"))
        assert tc_chunk is None

    def test_tool_calls_in_output_after_reasoning(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_reasoning_choice("Thinking..."))
        acc.add_chunk(_make_tc_partial(0, "c1", "search", '{"q":"x"}'))
        acc.add_chunk(_make_finish_choice())

        result = acc.get_final_output()
        assert isinstance(result.response_output, TextModelOutput)
        assert result.response_output.reasoning is not None
        assert result.response_output.tool_calls is not None
        assert result.response_output.tool_calls[0]["function"]["name"] == "search"

    def test_content_then_tool_calls(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_content_choice("Hello"))
        acc.add_chunk(_make_tc_partial(0, "c1", "f", "{}"))
        acc.add_chunk(_make_finish_choice())

        result = acc.get_final_output()
        assert (
            result.response_output.output == ("Hello",)
            or result.response_output.output == "Hello"
        )
        assert result.response_output.tool_calls is not None

    def test_no_tool_calls_returns_none_field(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_content_choice("Hello world"))
        acc.add_chunk(_make_finish_choice())

        result = acc.get_final_output()
        assert result.response_output.tool_calls is None
        assert "tool_calls" not in result.metadata
