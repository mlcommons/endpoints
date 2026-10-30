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

import msgspec
import pytest
from inference_endpoint.core.types import QueryResult, StreamChunk, TextModelOutput
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.completions_adapter import OpenAITextCompletionsAdapter
from inference_endpoint.openai.openai_msgspec_adapter import OpenAIMsgspecAdapter
from inference_endpoint.openai.types import SSEChoice, SSEDelta, SSEMessage


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
        tool_calls = result.response_output.as_message_parts()[2]
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"
        assert tool_calls[0]["function"]["arguments"] == '{"q":"test"}'
        assert result.response_output.as_message_parts_after_first_chunk()[2] == (
            {"type": "function", "function": {"arguments": '"test"}'}},
        )

    def test_metadata_tool_calls_preserved(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_tc_partial(0, "c1", "f", "{}"))
        acc.add_chunk(_make_finish_choice())

        result = acc.get_final_output()
        assert "tool_calls" in result.metadata
        assert result.metadata["tool_calls"][0]["function"]["name"] == "f"
        assert result.response_output.as_message_parts_after_first_chunk()[2] is None

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
        tool_calls = result.response_output.as_message_parts()[2]
        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "search"
        assert result.response_output.as_message_parts_after_first_chunk()[2] == (
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"x"}'},
            },
        )

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
        assert result.response_output.as_message_parts()[2] is not None
        assert result.response_output.as_message_parts_after_first_chunk()[2] == (
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            },
        )

    def test_no_tool_calls_returns_none_field(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_content_choice("Hello world"))
        acc.add_chunk(_make_finish_choice())

        result = acc.get_final_output()
        assert result.response_output.tool_calls is None
        assert result.response_output.as_message_parts_after_first_chunk()[2] is None
        assert "tool_calls" not in result.metadata


@pytest.mark.unit
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("stream_all_chunks", [False, True])
@pytest.mark.parametrize("with_tool_calls", [False, True])
def test_mixed_reasoning_and_content_are_both_preserved(
    reasoning_field, stream_all_chunks, with_tool_calls
):
    acc = OpenAISSEAccumulator("qid", stream_all_chunks=stream_all_chunks)
    tool_calls = (
        [
            {
                "index": 0,
                "id": "call1",
                "type": "function",
                "function": {"name": "answer", "arguments": "{}"},
            }
        ]
        if with_tool_calls
        else None
    )
    first = acc.add_chunk(
        SSEChoice(
            delta=SSEDelta(
                content="42", tool_calls=tool_calls, **{reasoning_field: "Think: "}
            )
        )
    )
    second = acc.add_chunk(
        SSEChoice(delta=SSEDelta(content="!", **{reasoning_field: "done"}))
    )

    assert first.response_chunk == "Think: 42"
    assert first.metadata["first_chunk"] is True
    if stream_all_chunks:
        assert second.response_chunk == "done!"
        assert second.metadata["first_chunk"] is False
    else:
        assert second is None
    output = acc.get_final_output().response_output
    assert output.output == ("42", "!")
    assert "".join(output.reasoning) == "Think: done"
    assert output.tool_calls == ((tuple(tool_calls),) if tool_calls else None)


def _usage_frame(**counts: object) -> SSEMessage:
    """A usage-only trailer as the adapters decode it."""
    return SSEMessage(usage=counts)


def _counts(metadata: dict) -> dict:
    return {
        k: metadata[k] for k in ("prompt_tokens", "completion_tokens") if k in metadata
    }


@pytest.mark.unit
class TestAccumulatorUsage:
    def test_usage_frame_sets_metadata_without_stream_chunk(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=True)
        assert acc.add_chunk(_usage_frame(prompt_tokens=7, completion_tokens=0)) is None
        assert acc.first_chunk_sent is False
        assert acc.add_chunk(_make_content_choice("Hi")).metadata["first_chunk"] is True
        assert acc.add_chunk(_usage_frame(prompt_tokens=7, completion_tokens=3)) is None
        result = acc.get_final_output()
        assert result.response_output.output == ("Hi",)
        assert _counts(result.metadata) == {"prompt_tokens": 7, "completion_tokens": 3}

    def test_usage_only_stream_reports_zero(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_usage_frame(prompt_tokens=7, completion_tokens=0))
        result = acc.get_final_output()
        assert result.response_output.output == ()
        assert result.metadata == {
            "first_chunk": True,
            "final_chunk": True,
            "prompt_tokens": 7,
            "completion_tokens": 0,
        }

    def test_message_applies_choice_and_usage(self):  # LiteLLM shape
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        chunk = acc.add_chunk(
            SSEMessage(
                choices=(
                    SSEChoice(delta=SSEDelta(content="Hi"), finish_reason="stop"),
                ),
                usage={"prompt_tokens": 7, "completion_tokens": 1},
            )
        )
        assert (chunk.response_chunk, chunk.metadata["first_chunk"]) == ("Hi", True)
        md = acc.get_final_output().metadata
        assert (md["finish_reason"], md["prompt_tokens"], md["completion_tokens"]) == (
            "stop",
            7,
            1,
        )

    def test_usage_on_every_frame_last_wins(self):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=True)
        for n, text in enumerate("abc", start=1):
            acc.add_chunk(
                SSEMessage(
                    choices=(_make_content_choice(text),),
                    usage={"prompt_tokens": 5, "completion_tokens": n},
                )
            )
        result = acc.get_final_output()
        assert result.response_output.output == ("a", "bc")
        assert result.metadata["completion_tokens"] == 3

    def test_null_usage_fields_do_not_erase_earlier_counts(
        self,
    ):  # proxy ends with "usage": {}
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(
            SSEMessage(
                choices=(_make_content_choice("Hi"),),
                usage={"prompt_tokens": 7, "completion_tokens": 3},
            )
        )
        assert acc.add_chunk(_usage_frame()) is None
        assert _counts(acc.get_final_output().metadata) == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
        }
        acc.add_chunk(
            _usage_frame(prompt_tokens=7, completion_tokens=5)
        )  # valid ints still override
        assert acc.get_final_output().metadata["completion_tokens"] == 5

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            (
                _usage_frame(prompt_tokens=None, completion_tokens=2),
                {"completion_tokens": 2},
            ),
            (None, {}),
        ],
        ids=["null-prompt-tokens", "no-usage"],
    )
    def test_missing_counts_omit_keys(self, chunk, expected):
        acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
        acc.add_chunk(_make_content_choice("Hi"))
        assert acc.add_chunk(chunk) is None
        assert _counts(acc.get_final_output().metadata) == expected


def _chat_choice(text: str, **fields: object) -> dict:
    return {"delta": {"content": text}, **fields}


def _completions_choice(text: str, **fields: object) -> dict:
    return {"text": text, **fields}


_WIRE_ADAPTERS = pytest.mark.parametrize(
    "adapter, choice",
    [
        (OpenAIMsgspecAdapter, _chat_choice),
        (OpenAITextCompletionsAdapter, _completions_choice),
    ],
    ids=["chat", "completions"],
)


def _accumulate_wire(adapter, frames: list[dict]) -> QueryResult:
    buffer = (
        b"".join(b"data: " + msgspec.json.encode(f) + b"\n\n" for f in frames)
        + b"data: [DONE]\n\n"
    )
    acc = OpenAISSEAccumulator("qid", stream_all_chunks=True)
    for decoded in adapter.parse_sse_chunk(buffer, len(buffer)):
        acc.add_chunk(decoded)
    return acc.get_final_output()


@pytest.mark.unit
@_WIRE_ADAPTERS
@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": 12.0}, [], "x", {"completion_tokens": -1}],
    ids=["float-count", "list", "string", "negative-count"],
)
def test_malformed_usage_keeps_frame_content(adapter, choice, usage):
    result = _accumulate_wire(
        adapter, [{"choices": [choice("Hi", finish_reason="stop")], "usage": usage}]
    )
    assert result.response_output.output == ("Hi",)
    assert result.metadata["finish_reason"] == "stop"
    assert _counts(result.metadata) == {}


@pytest.mark.unit
@_WIRE_ADAPTERS
def test_usage_on_every_frame_last_wellformed_wins(adapter, choice):
    result = _accumulate_wire(
        adapter,
        [
            {
                "choices": [choice("a")],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
            {
                "choices": [choice("b")],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            {
                "choices": [choice("c", finish_reason="stop")],
                "usage": {"completion_tokens": "bad"},
            },
        ],
    )
    assert result.response_output.output == ("a", "bc")
    assert _counts(result.metadata) == {"prompt_tokens": 5, "completion_tokens": 2}


@pytest.mark.unit
@_WIRE_ADAPTERS
def test_null_usage_frames_record_nothing(adapter, choice):
    result = _accumulate_wire(
        adapter,
        [
            {"choices": [choice("a")], "usage": None},
            {"choices": [choice("b", finish_reason="stop")], "usage": None},
            {"choices": [], "usage": None},
        ],
    )
    assert result.response_output.output == ("a", "b")
    assert _counts(result.metadata) == {}


@pytest.mark.unit
@_WIRE_ADAPTERS
def test_malformed_usage_on_the_wire_warns_once(
    adapter, choice, usage_warning_unlatched
):
    _accumulate_wire(
        adapter,
        [
            {"choices": [choice("a")], "usage": "x"},
            {"choices": [choice("b")], "usage": {"completion_tokens": -1}},
            {"choices": [], "usage": [1]},
        ],
    )
    (record,) = usage_warning_unlatched.records
    assert "'x' (type 'str')" in record.getMessage()


@pytest.mark.unit
@_WIRE_ADAPTERS
def test_well_formed_usage_on_the_wire_never_warns(
    adapter, choice, usage_warning_unlatched
):
    _accumulate_wire(
        adapter,
        [
            {"choices": [choice("a")], "usage": None},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
        ],
    )
    assert usage_warning_unlatched.records == []


@pytest.mark.unit
@_WIRE_ADAPTERS
def test_usage_bearing_frame_with_two_choices_drops_counts(adapter, choice):
    result = _accumulate_wire(
        adapter,
        [
            {"choices": [choice("a")]},
            {
                "choices": [choice("b"), choice("x", index=1)],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4},
            },
        ],
    )
    # Each frame contributes only its first choice to the output.
    assert result.response_output.output == ("a", "b")
    assert _counts(result.metadata) == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, frames",
    [
        (
            OpenAIMsgspecAdapter,
            [
                b'{"choices":[{"delta":{"role":"assistant"}}],"usage":null}',
                b'{"choices":[{"delta":{"content":"Hi"}}],"usage":null}',
                b'{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":null}',
            ],
        ),
        (
            OpenAITextCompletionsAdapter,
            [
                b'{"choices":[{"index":0,"text":"Hi"}]}',
                b'{"choices":[{"index":0,"text":"","finish_reason":"stop"}]}',
            ],
        ),
    ],
    ids=["chat", "completions"],
)
def test_wire_usage_trailer_reaches_final_metadata(adapter, frames):
    trailer = b'{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}'
    buffer = (
        b"".join(b"data: " + f + b"\n\n" for f in [*frames, trailer])
        + b"data: [DONE]\n\n"
    )
    acc = OpenAISSEAccumulator("qid", stream_all_chunks=False)
    chunks = [
        c
        for d in adapter.parse_sse_chunk(buffer, len(buffer))
        if (c := acc.add_chunk(d))
    ]
    assert [c.response_chunk for c in chunks] == ["Hi"]
    md = acc.get_final_output().metadata
    assert (md["finish_reason"], md["prompt_tokens"], md["completion_tokens"]) == (
        "stop",
        7,
        2,
    )
