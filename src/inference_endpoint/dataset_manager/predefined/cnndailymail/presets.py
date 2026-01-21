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


"""Preset transforms for the CNN/DailyMail dataset."""

from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    DropColumns,
    Transform,
    UserPromptFormatter,
)


def llama3(
    stream: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
) -> list[Transform]:
    return [
        # Step 1: Format the prompt from "article"
        UserPromptFormatter(
            user_prompt_format=f"Summarize the following news article in {max_new_tokens} tokens. Please output the summary only, without any other text.\n\nArticle:\n{{article}}\n\nSummary:",
            output_column="prompt",
        ),
        # Step 2: Drop columns we don't need for inference
        DropColumns(
            columns=[
                "article",
                "highlights",
            ],
            errors="ignore",
        ),
        # Step 3: Add metadata columns since we don't want to do a dict update every iteration
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
