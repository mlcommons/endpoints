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

    def __init__(
        self,
        *,
        num_sequences: int = 1024,
        input_seq_length: int = 1024,
        range_ratio: float = 1.0,
        random_seed: int = 42,
        save_tokenized_data: bool = False,
        tokenizer: str | PreTrainedTokenizer,
    ):
        self.input_seq_length = input_seq_length
        self.num_sequences = num_sequences
        self.range_ratio = range_ratio
        self.random_seed = random_seed
        self.transforms = []
        self.repeats = 1
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.save_tokenized_data = save_tokenized_data
        self.rng = np.random.default_rng(random_seed)
        self.dataframe = self._generate_random_sequence()

    def _generate_random_sequence(self) -> pd.DataFrame:
        data = []
        tokenizer = self.tokenizer
        # Generate the input sequence lengths given the range ratio
        input_seq_length = self.rng.integers(
            int(self.input_seq_length * self.range_ratio),
            self.input_seq_length + 1,
            self.num_sequences,
        )
        # Generate the input starts randomly from the vocab size
        input_starts = self.rng.integers(0, tokenizer.vocab_size, self.num_sequences)

        # Generate the input sequences
        for i in range(self.num_sequences):
            # Generate the input sequence by adding the input starts to the input sequence lengths and modding by the vocab size
            input_sequence = [
                (input_starts[i] + j) % tokenizer.vocab_size
                for j in range(input_seq_length[i])
            ]
            # Decode the input sequence to get the text prompt
            prompt = tokenizer.decode(input_sequence, add_special_tokens=False)
            # If we are saving the tokenized data, append the input sequence to the input tokens
            # This can be useful for debugging or for other purposes
            if self.save_tokenized_data:
                # Encode the prompt to get the input tokens back
                input_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            else:
                input_tokens = None
            data.append(
                {
                    "prompt": prompt,
                    "input_tokens": input_tokens,
                    "input_seq_length": input_seq_length[i],
                }
            )

        self.dataframe = pd.DataFrame(data)
        return self.dataframe

    @classmethod
    def generate(cls, *args, **kwargs):
        return RandomDataset(*args, **kwargs)
