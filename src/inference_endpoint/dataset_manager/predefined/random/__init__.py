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

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer

from ...dataset import Dataset


class RandomDataset(Dataset, dataset_id="random"):
    """Random dataset"""

    COLUMN_NAMES = [
        "prompt",
        "input_tokens",
        "input_seq_length",
    ]

    @classmethod
    def generate(
        cls,
        datasets_dir,  # type: ignore
        force,  # type: ignore
        *,
        num_sequences: int = 1024,
        input_seq_length: int = 1024,
        range_ratio: float = 1.0,
        random_seed: int = 42,
        save_tokenized_data: bool = False,
        tokenizer: str | PreTrainedTokenizer,
    ) -> pd.DataFrame:
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        rng = np.random.default_rng(random_seed)
        data = []
        # Generate the input sequence lengths given the range ratio
        input_seq_lengths = rng.integers(
            int(input_seq_length * range_ratio),
            input_seq_length + 1,
            num_sequences,
        )
        # Generate the input starts randomly from the vocab size
        input_starts_array = rng.integers(0, tokenizer.vocab_size, num_sequences)

        # Generate the input sequences
        for i in range(num_sequences):
            # Generate the input sequence by adding the input starts to the input sequence lengths and modding by the vocab size
            seq_len = int(input_seq_lengths[i])
            start_val = int(input_starts_array[i])
            input_sequence = [
                (start_val + j) % tokenizer.vocab_size for j in range(seq_len)
            ]
            # Decode the input sequence to get the text prompt
            prompt = tokenizer.decode(input_sequence, add_special_tokens=False)
            # If we are saving the tokenized data, append the input sequence to the input tokens
            # This can be useful for debugging or for other purposes
            if save_tokenized_data:
                # Encode the prompt to get the input tokens back
                input_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            else:
                input_tokens = None
            data.append(
                {
                    "prompt": prompt,
                    "input_tokens": input_tokens,
                    "input_seq_length": seq_len,
                }
            )

        return pd.DataFrame(data)
