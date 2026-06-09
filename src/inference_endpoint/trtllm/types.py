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
msgspec types for TRT-LLM API request serialization.

Response types are not needed — TRT-LLM uses OpenAI-compatible response format,
so we reuse types from inference_endpoint.openai.types.
"""

import msgspec

# ============================================================================
# TRT-LLM Request Types
# ============================================================================


class TRTLLMChatRequest(msgspec.Struct, kw_only=True, omit_defaults=True):  # type: ignore[call-arg]
    """TRT-LLM chat completion request with prompt_token_ids support."""

    messages: list
    prompt_token_ids: list[int]
    model: str = "no-model-name"
    max_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_tokens: int | None = None
    skip_special_tokens: bool | None = None
    stop_token_ids: list[int] | None = None
    stream: bool = False
