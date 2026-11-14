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
msgspec types for SGLang API serialization/deserialization.
"""

from typing import Any

import msgspec

# ============================================================================
# SGLang Request/Response Types
# ============================================================================


class SamplingParams(
    msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, gc=False
):  # type: ignore[call-arg]
    max_new_tokens: int = 32768
    """int: Maximum number of tokens to generate per request (1-32768)"""

    temperature: float = 1.0
    """float: Sampling temperature (0.0 = deterministic, higher = more random). Typically 0.001-2."""

    top_k: int = -1
    """int: Top-k sampling (number of highest probability tokens to consider). -1 = disable"""

    top_p: float = 1.0
    """float: Top-p/nucleus sampling (cumulative probability threshold). 0.0-1.0, typically 1.0 for no filtering"""


class SGLangGenerateRequest(
    msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, gc=False
):  # type: ignore[call-arg]
    input_ids: list[int]
    sampling_params: SamplingParams
    stream: bool


class MetaInfo(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, gc=False):  # type: ignore[call-arg]
    # Only completion_tokens is required (it drives the stream): SGLang sends a
    # null finish_reason until the last frame, e2e_latency only on it, and names
    # the retraction counter num_retractions (older servers: total_retractions).
    id: str | None = None
    finish_reason: dict[str, Any] | None = None
    # Any JSON value decodes without raising, so a malformed count never fails
    # the frame; server_usage_count validates it (the accumulator on a stream,
    # once per changed value; the adapter on a non-streaming response). gc=False
    # stays safe: a JSON-decoded builtin cannot reference this struct.
    prompt_tokens: object = None
    weight_version: str | None = None
    total_retractions: int | None = None
    num_retractions: int | None = None
    completion_tokens: int
    cached_tokens: int | None = None
    e2e_latency: float | None = None


class SGLangGenerateResponse(
    msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, gc=False
):  # type: ignore[call-arg]
    text: str
    output_ids: list[int] = msgspec.field(default_factory=list)
    meta_info: MetaInfo


class SGLangSSEDelta(
    msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True, gc=False
):  # type: ignore[call-arg]
    text: str = ""
    token_delta: list[int] = msgspec.field(default_factory=list)
    total_completion_tokens: int = 0
    has_retractions: bool = False
    # The raw meta_info value: every frame repeats it, so the accumulator
    # validates it only when it changes. gc=False stays safe: a JSON-decoded
    # builtin cannot reference this struct.
    prompt_tokens: object = None
