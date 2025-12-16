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

import json
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from typing import Any

import numpy as np
import pandas
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase


class DataLoader(ABC):
    """Abstract base class for loading and managing benchmark datasets.

    DataLoaders handle:
    - Loading datasets from various formats (pickle, HuggingFace, CSV, etc.)
    - Memory management for large datasets
    - Random-access sample retrieval by index
    - Optional memory-constrained caching/unloading

    The DataLoader is responsible for raw data loading only. Parsing and
    transformation (e.g., converting to request format) is handled separately
    by parser functions.

    Attributes:
        max_memory_usage_bytes: Optional memory limit. If None, no artificial limit.
    """

    def __init__(self, max_memory_usage_bytes: int | None = None):
        """Initialize the dataloader with optional memory constraints.

        Args:
            max_memory_usage_bytes: Maximum memory to use for dataset caching.
                                   If None, loads entire dataset into memory.
        """
        self.max_memory_usage_bytes = max_memory_usage_bytes

    def load(self, force: bool = False):  # noqa: B027
        """Load the dataset into memory for eager loading.

        Optional method for implementations that support eager loading.
        This enables converting/loading the entire dataset upfront for
        workloads that benefit from pre-processing.

        Not all implementations need this - implementations that stream
        from disk or use lazy loading can skip this method.

        Args:
            force: If True, reloads even if already loaded (for refreshing data).

        Raises:
            IOError: If dataset cannot be loaded.
        """
        pass

    @abstractmethod
    def load_sample(self, index: int) -> Any:
        """Load a single sample from the dataset by index.

        This method must support random access and may be called multiple times
        for the same index. Implementations should cache samples in memory when
        possible for performance.

        Args:
            index: Sample index (0 to num_samples()-1).

        Returns:
            Sample data in format specific to the dataset type.
            Typically a dict, dataclass, or custom object.

        Raises:
            IndexError: If index is out of range.
            IOError: If data cannot be loaded from disk.
        """
        raise NotImplementedError

    def mark_unneeded(self, index: int):  # noqa: B027
        """Mark a sample as no longer needed for eviction.

        Implementations with memory constraints can use this as a hint to
        unload samples from memory. The benchmark system calls this after
        a sample has been issued and is unlikely to be needed again soon.

        Optional implementation - not needed if entire dataset fits in memory.

        Args:
            index: Sample index that can be evicted.
        """
        pass

    @abstractmethod
    def num_samples(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            Total sample count (positive integer).
        """
        raise NotImplementedError


class PickleReader(DataLoader):
    """DataLoader implementation for pickle (.pkl) format datasets.

    Loads Python pickle files containing lists or arrays of samples.
    Supports optional parsing functions to transform raw data into
    the format needed by the benchmark system.

    The default parser extracts the 'text_input' attribute from each row,
    which is compatible with common benchmark dataset formats.

    Usage:
        >>> reader = PickleReader("dataset.pkl", parser=lambda x: x.text_input)
        >>> reader.load()
        >>> sample = reader.load_sample(0)

    Attributes:
        file_path: Path to the pickle file.
        parser: Function to transform raw rows into usable format.
        data: Loaded dataset (empty until load() is called).
        loaded: Whether the dataset has been loaded into memory.
    """

    def __init__(
        self,
        file_path,
        parser: Callable[[Any], Any] = None,
        max_memory_usage_bytes: int | None = None,
    ):
        """Initialize PickleReader for a pickle dataset file.

        Note: This does not load the data immediately. Call load() to load.

        Args:
            file_path: Path to the pickle file containing the dataset.
            parser: Optional function to extract/transform data from each row.
                   If None, uses default parser that extracts 'text_input' attribute.
            max_memory_usage_bytes: Optional memory limit (currently unused).
        """
        super().__init__(max_memory_usage_bytes)
        self.file_path = file_path
        self.data = []
        # self.text_inputs = []
        self.loaded = False
        self.parser = parser
        self.logger = getLogger(__name__)
        if parser is None:
            # TODO : remove this default implementation
            def extract_text_input(row):
                return {"prompt": row["text_input"]} | (self.metadata or {})

            self.parser = extract_text_input

    def load(self, force: bool = False):
        """Load the dataset from the pickle file into memory.

        This method reads the entire pickle file and stores it in memory.
        Subsequent load_sample() calls will be served from memory.

        Args:
            force: If True, reloads even if already loaded (for refreshing data).

        Raises:
            FileNotFoundError: If pickle file doesn't exist.
            pickle.UnpicklingError: If file is corrupted or not a valid pickle.
        """
        if self.loaded and not force:
            return
        with open(self.file_path, "rb") as file:
            self.data = pickle.load(file)
            # self.data = self.data[:2]
            self.logger.debug(
                f"Loading data from {self.file_path} with columns: {self.data.columns}"
            )

            self.text_inputs = []
            # this preloads the data in source
            # convert the dataframe to list of dictionaries
            for data in self.data.to_dict(orient="records"):
                # idx is not passed to the parser since it should _not_ be used in the parser
                # note that while we are appending, which is slower, the data may not be sequential and may
                # have gaps
                self.text_inputs.append(self.parser(data))
            self.text_inputs = np.ascontiguousarray(self.text_inputs)
        self.loaded = True

    def num_samples(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def write_samples(self, file_path: str, indices: list[int]):
        """
        Utility function - writes the samples to a pickle file.
        """
        if not self.loaded:
            self.load()
        samples = self.data.iloc[indices]
        with open(file_path, "wb") as file:
            # for index in indices:
            pickle.dump(samples, file)

    def write_unique_samples(self, file_path: str):
        """ "
        Utility function - writes the unique samples to a pickle file.
        """
        dataset_sources = {"math500", "aime1983", "livecodebench", "gpqa", "mmlu_pro"}
        samples = pandas.DataFrame(columns=self.data.columns)
        for dataset_source in dataset_sources:
            filtered = self.data[self.data["dataset"] == dataset_source]
            samples = pandas.concat([samples, filtered.iloc[[0]]], ignore_index=True)

        with open(file_path, "wb") as file:
            # for sample in samples:
            pickle.dump(samples, file)

    def load_sample(self, index: int) -> Any:
        """
        Loads a sample from the data.
        """
        assert self.loaded, "Data is not loaded. Call load() to load the data."
        x = self.text_inputs[index]
        self.logger.debug(f"Loaded sample from pickle file at {index} with result: {x}")
        return x

    def get_column_names(self):
        return self.data.columns


class HFDataLoader(DataLoader):
    def __init__(
        self,
        dataset_name,
        *,
        parser: Callable[[Any], Any] = None,
        split: str = "train",
        format: str | None = None,
        max_memory_usage_bytes: int | None = None,
    ):
        super().__init__(max_memory_usage_bytes)
        self.logger = getLogger(__name__)
        self.dataset_name = dataset_name
        self.data = []
        self.text_inputs = []
        self.parser = parser
        self.split = split
        self.format = format
        if parser is None:

            def extract_row(row):
                return row  # by default, return the training data which is a dictionary

            self.parser = extract_row

    def load(self):
        if self.format is None:
            self.data = load_dataset(self.dataset_name)
        else:
            # huggingface uses a different method to load local arrow datasets
            self.data = load_from_disk(self.dataset_name)
        self.text_inputs = []  # reset the text inputs
        for d in self.data[self.split]:
            self.text_inputs.append(self.parser(d))

    def load_sample(self, index: int) -> Any:
        return self.text_inputs[index]

    def num_samples(self):
        """
        Returns the total number of samples in the specified dataset split.

        Returns:
            int: Number of samples in the current dataset split.
        """
        return len(self.data[self.split])


class DeepSeekR1ChatCompletionDataLoader(PickleReader):
    def __init__(self, file_path, parser: Callable[[Any], Any] = None):
        """
        Initialize a DeepSeekR1ChatCompletionDataLoader for loading chat completion datasets from a pickle file.

        Args:
            file_path (str): Path to the pickle file containing chat completion data.
            parser (Callable[[Any], Any], optional): Callable to parse individual data samples. If not provided, defaults to the parent class's parsing mechanism.
        """
        super().__init__(file_path, parser=parser)


class JsonlReader(DataLoader):
    def __init__(
        self,
        file_path,
        parser: Callable[[Any], Any] = None,
        metadata: dict | None = None,
    ):
        if parser is None:
            # TODO: Implement a parser interface where yaml files specify the fields to pars
            def default_parser(x):
                # Use cnn/daily mail dataset as an example for now.
                return {"prompt": x["article"]} | metadata

            parser = default_parser
        super().__init__()
        self.file_path = file_path
        self.data = []
        self.parser = parser

    def load(self):
        with open(self.file_path) as file:
            for line in file:
                if line := line.strip():
                    self.data.append(self.parser(json.loads(line)))

    def load_sample(self, index: int) -> Any:
        return self.data[index]

    def num_samples(self):
        return len(self.data)


class RandomDataLoader(DataLoader):
    """
    DataLoader implementation for generating random data.
    This is useful for testing and benchmarking purposes. It generates random data based on the tokenizer and the input sequence length.
    The data is generated by selecting a random length from the range [input_seq_length * range_ratio, input_seq_length] and a random start index from the vocab size.
    Then it generates a random sequence of input tokens by adding the start index to the input sequence length wrapping around the vocab size.
    The input tokens are then decoded to get the text prompt. Note that the length of the tokenized prompt may be different from the input tokens due to the decoding-encoding which may coalesce some sequences to newer tokens.
    The prompt is then appended to the data list.
    If save_tokenized_data is True, the input tokens are also appended to the input tokens list. This can be useful for debugging or for other purposes.
    """

    def __init__(
        self,
        *,
        max_memory_usage_bytes: int | None = None,
        num_sequences: int = 1024,
        input_seq_length: int = 1024,
        range_ratio: float = 1.0,
        random_seed: int = 42,
        tokenizer: PreTrainedTokenizerBase,
        save_tokenized_data: bool = False,
        metadata: dict | None = None,
    ):
        """
        Initialize a RandomDataLoader for generating random data.

        Args:
            max_memory_usage_bytes: Optional memory limit (currently unused).
            num_sequences: Number of sequences to generate.
            input_seq_length: Maximum length of the input sequence.
            range_ratio: Ratio for the range of input sequence lengths to the maximum input sequence length.
            random_seed: Random seed for reproducibility.
            tokenizer: Tokenizer to use for encoding and decoding (required).
            save_tokenized_data: Whether to save the tokenized data (useful for debugging or other purposes).
            metadata: Metadata dictionary to add to the data (optional).
        """
        super().__init__(max_memory_usage_bytes=max_memory_usage_bytes)
        self.data = []
        self.input_seq_length = input_seq_length
        self.num_sequences = num_sequences
        self.range_ratio = range_ratio
        assert (
            0 < self.range_ratio <= 1
        ), "Range ratio must be greater than 0 and less than or equal to 1"
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.tokenizer = tokenizer
        self.input_tokens = []
        self.save_tokenized_data = save_tokenized_data
        self.metadata = metadata
        # Now generate the tokens
        self._generate_random_sequence()

    def _generate_random_sequence(self) -> None:
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
            # Append the generated prompt to the data
            self.data.append(prompt)
            # If we are saving the tokenized data, append the input sequence to the input tokens
            # This can be useful for debugging or for other purposes
            if self.save_tokenized_data:
                # Encode the prompt to get the input tokens back
                input_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                self.input_tokens.append(input_tokens)

    def load_sample(self, index: int) -> Any:
        assert index < self.num_samples(), "Index is out of range."
        return {"prompt": self.data[index]} | (self.metadata or {})

    def num_samples(self) -> int:
        return len(self.data)
