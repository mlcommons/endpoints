# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal token-counting inputs.

Each variant names the tokenization operation its payload requires.  Keeping
this distinction explicit prevents structured messages and complete prompts
from accidentally falling through the plain-text tokenizer path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

from inference_endpoint.core.types import TextModelOutput


@dataclass(frozen=True, slots=True)
class TokenIdsInput:
    """Already-tokenized input; counting is simply ``len(token_ids)``."""

    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TextInput:
    """Unstructured text counted by the tokenizer's text backend."""

    text: str


@dataclass(frozen=True, slots=True)
class MessageInput:
    """One structured assistant output rendered by the chat template."""

    content: str
    reasoning: str | None
    tool_calls: tuple[dict[str, Any], ...] | None


@dataclass(frozen=True, slots=True)
class PromptInput:
    """A complete structured chat prompt rendered by the chat template."""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] | None
    chat_template_kwargs: dict[str, Any] | None
    chat_template: str | None
    tool_choice: str | dict[str, Any] | None = None


TokenizationInput: TypeAlias = (  # noqa: UP040 - mypy version lacks PEP 695.
    TokenIdsInput | TextInput | MessageInput | PromptInput
)


def finite_number(value: object) -> float | None:
    """``value`` as a float, or None if it cannot be rendered as a number.

    Shared by the steady-state boundary and the report renderer so one rule
    decides what is renderable. bools are rejected because ``True`` would reach
    a report as 1.0; NaN and infinity because JSON admits the bare ``NaN`` and
    ``Infinity`` tokens and they would render as "nan"/"inf tok/s".
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_tpot_tokenization_input(
    output: TextModelOutput,
) -> MessageInput | TextInput | None:
    """The TPOT rule: how one model output is rendered for its token denominator.

    The post-first-chunk counterpart to :func:`extract_tokenization_input`, and
    the same single-source-of-truth argument applies: TPOT is measured over the
    output *after* the first chunk, and a second implementation of that rule
    drifts from this one. The steady-state detector reconstructs TPOT from the
    event log post-hoc and must land on the same number as the live trigger.

    Returns ``None`` when there is nothing after the first chunk to count --
    a non-streaming output, or a single-chunk one.
    """
    if output.reasoning or output.tool_calls:
        return MessageInput(*output.as_message_parts_after_first_chunk())
    text = output.text_after_first_chunk()
    if text:
        return TextInput(text)
    return None


def extract_tokenization_input(
    output: TextModelOutput,
) -> MessageInput | TextInput | None:
    """The OSL rule: how one model output is rendered for token counting.

    Single source of truth for both OSL populations — the performance-window
    series (``OslTrigger``) and the full-run statistic. Keeping one rule is
    what makes the two numbers comparable; a second, text-only implementation
    silently under-counts reasoning and tool calls (mlcommons/endpoints#500).

    Returns ``None`` for an output that must not be counted at all: an empty
    completion is excluded rather than counted as a zero-token sample, because
    a failed request still logs a COMPLETE event with ``output == ""`` and
    would otherwise drag ``min``/``avg`` down.
    """
    if output.reasoning or output.tool_calls:
        return MessageInput(*output.as_message_parts())
    text = str(output)
    if text:
        return TextInput(text)
    return None
