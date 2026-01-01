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

"""
msgspec types for SGLang API serialization/deserialization.
"""

from typing import Any

import msgspec

# ============================================================================
# SGLang Request/Response Types
# ============================================================================


class SamplingParams(msgspec.Struct, kw_only=True, omit_defaults=True):
    max_new_tokens: int = 32768
    """int: Maximum number of tokens to generate per request (1-32768)"""

    temperature: float = 1.0
    """float: Sampling temperature (0.0 = deterministic, higher = more random). Typically 0.001-2."""

    top_k: int = -1
    """int: Top-k sampling (number of highest probability tokens to consider). -1 = disable"""

    top_p: float = 1.0
    """float: Top-p/nucleus sampling (cumulative probability threshold). 0.0-1.0, typically 1.0 for no filterin"""


class SGLangGenerateRequest(msgspec.Struct, kw_only=True, omit_defaults=True):
    input_ids: list[int]
    sampling_params: SamplingParams
    stream: bool = True


class MetaInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    id: str
    finish_reason: dict[str, Any]
    prompt_tokens: int
    weight_version: str
    total_retractions: int
    completion_tokens: int
    cached_tokens: int
    e2e_latency: float


class SGLangGenerateResponse(msgspec.Struct, kw_only=True, omit_defaults=True):
    text: str
    output_ids: list[int]
    meta_info: MetaInfo


class SGLangSSEDelta(msgspec.Struct):
    text: str = ""
    token_delta: int = 0
    total_completion_tokens: int = 0
    has_retractions: bool = False
