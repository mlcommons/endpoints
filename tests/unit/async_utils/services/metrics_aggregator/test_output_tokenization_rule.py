# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single OSL population/rendering rule shared by every OSL consumer.

The performance window and the full-run statistic must count tokens the same
way, or the two numbers are not comparable (mlcommons/endpoints#500). These
tests pin the rule itself, independent of either consumer.
"""

from __future__ import annotations

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    TextInput,
    extract_tokenization_input,
    has_countable_output,
)
from inference_endpoint.core.types import TextModelOutput

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


_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "bash", "arguments": "{}"},
}


@pytest.mark.parametrize(
    "output",
    [
        pytest.param(TextModelOutput(output=""), id="empty-str"),
        pytest.param(TextModelOutput(output="a"), id="str"),
        pytest.param(TextModelOutput(output=("", "")), id="empty-chunks"),
        pytest.param(TextModelOutput(output=("", "x")), id="late-chunk"),
        pytest.param(
            TextModelOutput(output="", reasoning=("",)), id="empty-reasoning-chunk"
        ),
        pytest.param(
            TextModelOutput(output="", tool_calls=(_TOOL_CALL,)), id="tool-calls"
        ),
        pytest.param(
            TextModelOutput(output="", reasoning="thinking"), id="reasoning-only"
        ),
    ],
)
def test_has_countable_output_matches_the_osl_rule(output: TextModelOutput) -> None:
    assert has_countable_output(output) == (
        extract_tokenization_input(output) is not None
    )
