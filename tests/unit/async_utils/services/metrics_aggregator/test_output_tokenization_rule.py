# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single OSL population/rendering rule shared by every OSL consumer.

The performance window and the full-run statistic must count tokens the same
way, or the two numbers are not comparable (mlcommons/endpoints#500). These
tests pin the rule itself, independent of either consumer.
"""

from __future__ import annotations

import msgspec.json
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    TextInput,
    extract_tokenization_input,
    extract_tpot_tokenization_input,
)
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.metrics import steady_state_diagnostics

pytestmark = pytest.mark.unit


def test_tool_call_output_renders_via_chat_template() -> None:
    """A tool-call output must select MessageInput, never flattened text.

    Flattening loses the tool_calls structure, which is exactly how a
    text-only counter under-counts an agentic turn.
    """
    output = TextModelOutput(
        output="",
        tool_calls=(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            },
        ),
    )

    result = extract_tokenization_input(output)

    assert isinstance(result, MessageInput)
    assert result.tool_calls is not None


def test_reasoning_output_renders_via_chat_template() -> None:
    """Reasoning traces also require the chat-template path."""
    output = TextModelOutput(output="answer", reasoning="thinking about it")

    result = extract_tokenization_input(output)

    assert isinstance(result, MessageInput)
    assert result.reasoning == "thinking about it"


def test_plain_text_output_uses_text_path() -> None:
    """A plain output with no structure stays on the cheap text path."""
    output = TextModelOutput(output="just text")

    result = extract_tokenization_input(output)

    assert isinstance(result, TextInput)
    assert result.text == "just text"


def test_empty_output_is_excluded() -> None:
    """An empty completion is not a zero-token sample.

    A failed request still logs a COMPLETE event with output == ""; counting
    it as 0 would drag min/avg down on both the window and the full run.
    """
    assert extract_tokenization_input(TextModelOutput(output="")) is None


# --------------------------------------------------------------------------- #
# The post-hoc detector and the live trigger must agree.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "output",
    [
        pytest.param(TextModelOutput(output=("hello ", "world ", "again")), id="plain"),
        pytest.param(
            TextModelOutput(output=("out1 ", "out2"), reasoning=("think1 ", "think2 ")),
            id="reasoning",
        ),
        pytest.param(
            TextModelOutput(
                output=("a ", "b"),
                tool_calls=(
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                ),
            ),
            id="tool-calls",
        ),
        pytest.param(TextModelOutput(output="not streamed"), id="non-streaming"),
        pytest.param(TextModelOutput(output=("only-chunk",)), id="single-chunk"),
    ],
)
def test_the_detector_and_the_live_trigger_select_the_same_input(output):
    """The steady-state detector reconstructs TPOT from the event log and its
    number lands in result_summary.json beside the live trigger's. They agree
    only because both call this one rule -- which is the claim the detector's
    whole tokenization refactor rests on, and the thing that silently breaks if
    either side grows its own copy.

    Compares against the wire form the detector actually parses, so a change to
    the array layout breaks this too.
    """
    live = extract_tpot_tokenization_input(output)

    on_the_wire = msgspec.json.decode(msgspec.json.encode(output))
    post_hoc = steady_state_diagnostics.tpot_tokenization_input(on_the_wire)

    assert post_hoc == live
