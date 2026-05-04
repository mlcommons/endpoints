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


"""Preset transforms for the AIME25 dataset."""

from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    Transform,
    UserPromptFormatter,
)


def gptoss() -> list[Transform]:
    return [
        UserPromptFormatter(
            user_prompt_format="{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        ),
        # Enable DeepSeek thinking mode so the model uses chain-of-thought reasoning.
        # vLLM's reasoning_parser strips <think>...</think> tokens into reasoning_content;
        # the final boxed answer ends up in content where boxed_math_extractor finds it.
        AddStaticColumns({"chat_template_kwargs": {"thinking": True}}),
    ]


def gptoss_budget() -> list[Transform]:
    return [
        UserPromptFormatter(
            user_prompt_format="{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        ),
        # Same as gptoss but caps thinking at 8192 tokens via budget_tokens so the model
        # is forced to emit a final answer rather than consuming all max_new_tokens in
        # the reasoning phase (observed issue: 85% of responses had empty answer text).
        AddStaticColumns(
            {"chat_template_kwargs": {"thinking": True, "budget_tokens": 8192}}
        ),
    ]


def gptoss_budget_20k() -> list[Transform]:
    return [
        UserPromptFormatter(
            user_prompt_format="{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        ),
        AddStaticColumns(
            {"chat_template_kwargs": {"thinking": True, "budget_tokens": 20000}}
        ),
    ]


def gptoss_budget_20k_pre() -> list[Transform]:
    return [
        UserPromptFormatter(
            user_prompt_format="Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{question}",
        ),
        AddStaticColumns(
            {"chat_template_kwargs": {"thinking": True, "budget_tokens": 20000}}
        ),
    ]
