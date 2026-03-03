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
msgspec types for OpenAI API serialization/deserialization.
"""

from typing import Any

import msgspec

# ============================================================================
# Multimodal content (OpenAI vision format)
# ============================================================================

# prompt/system content: str for text, list[dict] for multimodal
# e.g. [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]
ChatMessageContent = str | list[dict[str, Any]]

# ============================================================================
# SSE (Server-Sent Events) Types for OpenAI streaming format
# ============================================================================


class SSEDelta(msgspec.Struct):
    """SSE delta object containing content."""

    content: str = ""
    reasoning: str = ""


class SSEChoice(msgspec.Struct):
    """SSE choice object containing delta."""

    delta: SSEDelta = msgspec.field(default_factory=SSEDelta)
    finish_reason: str | None = None


class SSEMessage(msgspec.Struct):
    """SSE message structure for OpenAI streaming responses."""

    choices: list[SSEChoice] = msgspec.field(default_factory=list)


# ============================================================================
# OpenAI Chat Completion Types (msgspec-based)
# ============================================================================


class ChatMessage(msgspec.Struct, kw_only=True, omit_defaults=True):  # type: ignore[call-arg]
    """Chat message in OpenAI format.

    content: str for text-only messages; list[dict] for multimodal (vision).
    """

    role: str
    content: ChatMessageContent
    name: str | None = None


class ChatCompletionRequest(msgspec.Struct, kw_only=True, omit_defaults=True):  # type: ignore[call-arg]
    """OpenAI chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_completion_tokens: int | None = None
    stream: bool | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    chat_template: str | None = None


class ChatCompletionResponseMessage(msgspec.Struct, kw_only=True, omit_defaults=True):  # type: ignore[call-arg]
    """Response message from OpenAI."""

    role: str
    content: str | None
    refusal: str | None


class ChatCompletionChoice(msgspec.Struct, kw_only=True, omit_defaults=True):  # type: ignore[call-arg]
    """A single choice in the completion response."""

    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str | None


class CompletionUsage(msgspec.Struct, kw_only=True, omit_defaults=True):  # type: ignore[call-arg]
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(msgspec.Struct, kw_only=True, omit_defaults=True):  # type: ignore[call-arg]
    """OpenAI chat completion response (msgspec version)."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None
    system_fingerprint: str | None
