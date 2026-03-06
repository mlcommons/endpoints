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

from typing import Any

from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    Harmonize,
    Transform,
    UserPromptFormatter,
)


def llama3_8b(
    stream: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
) -> list[Transform]:
    # Define custom chat template for Llama 3.1-8b (to sync with tokenized prompts from legacy implementation)
    template = (
        "{{- bos_token }}"
        "{%- for message in messages %}"
        "{{ message['content'] | trim }}"
        "{%- endfor %}"
    )
    chat_template: dict[str, Any] = {"chat_template": template}
    return [
        # Step 1: Format the prompt from "article"
        UserPromptFormatter(
            user_prompt_format=f"Summarize the following news article in {max_new_tokens} tokens. Please output the summary only, without any other text.\n\nArticle:\n{{article}}\n\nSummary:",
            output_column="prompt",
        ),
        AddStaticColumns(chat_template),
    ]


def llama3_8b_sglang(
    stream: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> list[Transform]:
    return [
        # Step 1: Format the prompt from "article"
        UserPromptFormatter(
            user_prompt_format=f"Summarize the following news article in {max_new_tokens} tokens. Please output the summary only, without any other text.\n\nArticle:\n{{article}}\n\nSummary:",
            output_column="prompt",
        ),
        # Step 2: Tokenize the raw prompt via Harmonize in plain mode.
        Harmonize(
            tokenizer_name=tokenizer_name,
            prompt_column="prompt",
            tokenized_column="input_tokens",
            harmonized_column=None,
            mode="plain",
        ),
    ]
