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


"""Preset transforms for the LiveCodeBench dataset."""

from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    DropColumns,
    Harmonize,
    Transform,
    UserPromptFormatter,
)


def gptoss_sglang(
    stream: bool = True,
    max_new_tokens: int = 32768,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> list[Transform]:
    return [
        # Step 1: Format the prompt from question and starter_code
        UserPromptFormatter(
            user_prompt_format=(
                "You are a python coding expert that solves problems step-by-step.\n"
                "You must provide the reasoning to arriving at your solution and the code to solve the problem.\n"
                "Do not try simulating the code execution. The code must be enclosed within ```python delimiters.\n\n\n"
                "{question}\n"
                "### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
                "```python\n"
                "{starter_code}\n"
                "```\n"
            ),
            output_column="user_prompt",
        ),
        # Step 2: Harmonize the prompt for SGLang/GPT-OSS
        Harmonize(
            prompt_column="user_prompt",
        ),
        # Step 3: Drop columns we don't need for inference
        DropColumns(
            columns=[
                "question",
                "starter_code",
                "user_prompt",
            ],
            errors="ignore",
        ),
        # Step 4: Add metadata columns since we don't want to do a dict update every iteration
        AddStaticColumns(
            {
                "stream": stream,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        ),
    ]
