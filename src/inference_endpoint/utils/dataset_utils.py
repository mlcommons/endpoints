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

from transformers import PreTrainedTokenizerBase


def tokenizer_stats(
    tokenizer: PreTrainedTokenizerBase,
    start_index: int = 0,
    end_index: int = -1,
    max_length: int = 1024,
):
    """
    Prints the stats of the tokenizer for a given range of tokens.
    Args:
        start_index: The index to start the stats from.
        end_index: The index to end the stats at.
        max_length: The maximum length of the tokens.
    """
    if end_index == -1:
        end_index = tokenizer.vocab_size
    token_to_text = {}  # dictionary from token ids to text
    token_leng_counts = {}  # histogram of token lengths
    for i in range(start_index, end_index):
        token_to_text[i] = tokenizer.decode([i])
        token_leng_counts[len(token_to_text[i])] = 1 + token_leng_counts.get(
            len(token_to_text[i]), 0
        )
    print(f"Stats for tokens {start_index} to {end_index}:")
    print(f"Max token length: {max(len(text) for text in token_to_text.values())}")
    print(
        f"Average token length: {sum(len(text) for text in token_to_text.values()) / len(token_to_text.values())}"
    )
    print("Token length counts:")
    for i in sorted(token_leng_counts.keys()):
        print(f"Length {i}: {token_leng_counts[i]}")

    text_lengths = []
    prompts = []
    for i in range(start_index, end_index):
        tokens = [(i + j) % tokenizer.vocab_size for j in range(max_length)]
        text = tokenizer.decode(tokens, add_special_tokens=False)
        prompts.append(text)
        text_lengths.append(len(text))

    with open(
        f"text_lengths_filtered_{start_index}_{end_index}_{max_length}.txt", "w"
    ) as file:
        for i in range(len(text_lengths)):
            file.write(f"{i} , {prompts[i]} , {text_lengths[i]}\n")
    print(
        f"Stats for tokens {start_index} to {end_index} saved to text_lengths_filtered_{start_index}_{end_index}_{max_length}.txt"
    )
    return
