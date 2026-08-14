# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal token-counting inputs.

Each variant names the tokenization operation its payload requires.  Keeping
this distinction explicit prevents structured messages and complete prompts
from accidentally falling through the plain-text tokenizer path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias


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
