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

"""Response contracts across field combinations and stream boundaries."""

from itertools import product
from typing import Any

import msgspec
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from inference_endpoint.core.types import QueryResult
from inference_endpoint.metrics.steady_state_diagnostics import text_after_first_chunk
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.openai_msgspec_adapter import OpenAIMsgspecAdapter

# Each frame describes semantic contributions before OpenAI wire encoding:
# content, reasoning, and (tool index, argument fragment) pairs.
Frame = tuple[str, str, tuple[tuple[int, str], ...]]
SHAPES = tuple(product((False, True), repeat=3))
SHAPE_NAMES = [
    "+".join(
        name
        for name, present in zip(("content", "reasoning", "tools"), shape, strict=True)
        if present
    )
    or "empty"
    for shape in SHAPES
]


def _frame(shape: tuple[bool, bool, bool], suffix: str) -> Frame:
    content, reasoning, tools = shape
    return (
        f"answer-{suffix} " if content else "",
        f"think-{suffix} " if reasoning else "",
        ((0, f"arg-{suffix} "),) if tools else (),
    )


def _expected_parts(frames: list[Frame], start: int = 0):
    content = "".join(frame[0] for frame in frames[start:])
    reasoning = "".join(frame[1] for frame in frames[start:]) or None
    tool_indices = sorted({index for frame in frames[start:] for index, _ in frame[2]})
    tools = []
    for index in tool_indices:
        arguments = "".join(
            fragment
            for frame in frames[start:]
            for tool_index, fragment in frame[2]
            if tool_index == index
        )
        tool: dict[str, Any] = {
            "type": "function",
            "function": {"arguments": arguments},
        }
        # Name and ID arrive only in the first fragment of each tool call.
        if not any(index == i for frame in frames[:start] for i, _ in frame[2]):
            tool["id"] = f"call-{index}"
            tool["function"]["name"] = f"tool-{index}"
        tools.append(tool)
    return content, reasoning, tuple(tools) or None


def _accumulate(frames: list[Frame], reasoning_field: str, stream_all_chunks: bool):
    acc = OpenAISSEAccumulator("qid", stream_all_chunks=stream_all_chunks)
    emitted = []
    seen_tools = set()

    def add_wire(choices):
        choice = OpenAIMsgspecAdapter.decode_sse_message(
            msgspec.json.encode({"choices": choices})
        )
        return acc.add_chunk(choice)

    assert add_wire([{"delta": {"role": "assistant"}}]) is None
    for content, reasoning, fragments in frames:
        tools = []
        for index, arguments in fragments:
            tool: dict[str, Any] = {
                "index": index,
                "function": {"arguments": arguments},
            }
            if index not in seen_tools:
                tool.update(id=f"call-{index}", type="function")
                tool["function"]["name"] = f"tool-{index}"
                seen_tools.add(index)
            tools.append(tool)
        chunk = add_wire(
            [
                {
                    "delta": {
                        "content": content,
                        reasoning_field: reasoning,
                        "tool_calls": tools,
                    }
                }
            ]
        )
        if chunk is not None:
            emitted.append(chunk)
        # Neither empty deltas nor usage-only trailers establish a boundary.
        assert add_wire([{"delta": {}}]) is None
        assert (
            add_wire(
                [
                    {
                        "delta": {
                            "content": None,
                            reasoning_field: None,
                            "tool_calls": None,
                        }
                    }
                ]
            )
            is None
        )
    assert add_wire([{"delta": {}, "finish_reason": "stop"}]) is None
    assert add_wire([]) is None
    return acc.get_final_output(), emitted


def _assert_contract(
    frames: list[Frame], reasoning_field: str, stream_all_chunks: bool
):
    result, emitted = _accumulate(frames, reasoning_field, stream_all_chunks)
    first = next((i for i, frame in enumerate(frames) if any(frame)), None)
    tail_start = first + 1 if first is not None else len(frames)
    full = _expected_parts(frames)
    tail = _expected_parts(frames, tail_start)
    output = result.response_output
    assert output.as_message_parts() == full
    assert output.as_message_parts_after_first_chunk() == tail
    expected_text = (tail[1] or "") + tail[0]
    if tail[2]:
        expected_text += msgspec.json.encode(list(tail[2])).decode()
    assert output.text_after_first_chunk() == expected_text
    assert result.metadata["first_chunk"] is (first is None)
    assert result.metadata["finish_reason"] == "stop"
    assert result.metadata.get("tool_calls") == (list(full[2]) if full[2] else None)

    expected_chunks = []
    for i, (content, reasoning, _) in enumerate(frames):
        if i == first or (stream_all_chunks and (content or reasoning)):
            expected_chunks.append(reasoning + content)
    assert [chunk.response_chunk for chunk in emitted] == expected_chunks
    assert [chunk.metadata["first_chunk"] for chunk in emitted] == [
        i == 0 for i in range(len(emitted))
    ]
    for codec in (msgspec.json, msgspec.msgpack):
        decoded = codec.decode(codec.encode(result), type=QueryResult)
        assert decoded.response_output.as_message_parts() == full
        assert decoded.response_output.text_after_first_chunk() == expected_text
        assert decoded.metadata == result.metadata
        assert decoded.response_output.as_message_parts_after_first_chunk() == tail
    # Offline diagnostics intentionally omit tools, but share the text boundary.
    wire_output = msgspec.json.decode(msgspec.json.encode(output))
    assert text_after_first_chunk(wire_output) == (tail[1] or "") + tail[0]
    return result


@pytest.mark.unit
@pytest.mark.parametrize("first_shape", SHAPES, ids=SHAPE_NAMES)
@pytest.mark.parametrize("next_shape", SHAPES, ids=SHAPE_NAMES)
@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize("stream_all_chunks", [False, True])
def test_response_field_combinations(
    first_shape, next_shape, reasoning_field, stream_all_chunks
):
    _assert_contract(
        [_frame(first_shape, "first"), _frame(next_shape, "next")],
        reasoning_field,
        stream_all_chunks,
    )


@pytest.mark.unit
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_NAMES)
@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_single_delta_has_no_post_first_payload(shape, reasoning_field):
    result = _assert_contract([_frame(shape, "only")], reasoning_field, True)
    assert result.response_output.as_message_parts_after_first_chunk() == (
        "",
        None,
        None,
    )


@pytest.mark.unit
@pytest.mark.parametrize("stream_all_chunks", [False, True])
def test_interleaved_tool_fragments_and_unicode(stream_all_chunks):
    frames = [
        ("", "", ()),
        ("答🙂", "考é", ((1, '{"b":'), (0, '{"a":'))),
        ("", "", ((0, '"one"}'),)),
        (" done", " more", ((1, '"two"}'), (2, "{}"))),
    ]
    _assert_contract(frames, "reasoning_content", stream_all_chunks)


# Small arbitrary strings include empty, whitespace, and multibyte characters;
# the generated sequences also exercise reasoning that starts after content/tools.
_text = st.text(alphabet="abc é🙂\n", max_size=8)
_frames = st.lists(
    st.tuples(
        _text,
        _text,
        st.lists(
            st.tuples(st.integers(0, 2), _text), max_size=2, unique_by=lambda x: x[0]
        ).map(tuple),
    ),
    max_size=8,
)


@pytest.mark.unit
@settings(max_examples=100, deadline=None)
@given(
    frames=_frames, reasoning_field=st.sampled_from(["reasoning", "reasoning_content"])
)
def test_generated_streams_preserve_payload_and_boundary(frames, reasoning_field):
    first_only = _assert_contract(frames, reasoning_field, False)
    all_chunks = _assert_contract(frames, reasoning_field, True)
    assert first_only.response_output == all_chunks.response_output
    assert first_only.metadata == all_chunks.metadata


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(content=_text, reasoning=_text, split=st.integers(0, 8))
def test_rechunking_after_fixed_first_chunk_preserves_output(content, reasoning, split):
    first = ("first content", "first reasoning", ())
    whole, _ = _accumulate([first, (content, reasoning, ())], "reasoning", True)
    split_output, _ = _accumulate(
        [
            first,
            (content[:split], reasoning[:split], ()),
            ("", "", ()),
            (content[split:], reasoning[split:], ()),
        ],
        "reasoning",
        True,
    )
    assert (
        whole.response_output.as_message_parts()
        == split_output.response_output.as_message_parts()
    )
    assert (
        whole.response_output.as_message_parts_after_first_chunk()
        == split_output.response_output.as_message_parts_after_first_chunk()
    )
