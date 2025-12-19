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

import csv
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas
from transformers import PreTrainedTokenizerBase

from .dataset import Dataset, DatasetFormat


class DataLoader(ABC):
    """Abstract base class for loading and managing benchmark datasets.
    It is expected that datasets are stored in a tabular format.

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

    IMPLEMENTATIONS: ClassVar[dict[str, list[type["DataLoader"]]]] = {}

    # Only used by subclasses
    FORMAT: ClassVar[DatasetFormat | None] = None

    def __init_subclass__(cls, format: DatasetFormat | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if format is not None:
            cls.FORMAT = format
        else:
            raise ValueError("Must specify 'format' when subclassing DataLoader")

        if format in DataLoader.IMPLEMENTATIONS:
            DataLoader.IMPLEMENTATIONS[format].append(cls)
        else:
            DataLoader.IMPLEMENTATIONS[format] = [cls]

    @classmethod
    def get_loader_for_format(
        cls, format: DatasetFormat, option: int = 0
    ) -> type["DataLoader"]:
        """Get the loader for a given format. By default, the first registered loader
        for the format is returned. If option is specified, the option-th loader is returned instead.

        Args:
            format: The format of the dataloader to get.
            option: The option-th loader to get.

        Returns:
            The loader for the given format.

        Raises:
            ValueError: If the format is not registered, or if option is out of range.
        """
        if format not in DataLoader.IMPLEMENTATIONS:
            raise ValueError(f"No loader registered for format: {format}")
        if option >= len(DataLoader.IMPLEMENTATIONS[format]):
            raise ValueError(f"Option {option} is out of range for format {format}")
        return DataLoader.IMPLEMENTATIONS[format][option]

    def __init__(
        self,
        dataset: Dataset,
        datasets_dir: Path = Path("datasets"),
        process_row: Callable[[dict[str, Any]], Any] | None = None,
        max_memory_usage_bytes: int | None = None,
    ):
        """Initialize the dataloader with optional memory constraints.

        Args:
            dataset: The dataset to load.
            datasets_dir: Directory to load the dataset from. (Default: "$CWD/datasets")
            process_row: Optional function applied to each row of the dataset right before
                returning it. This can be used to filter columns, add metadata, etc.
            max_memory_usage_bytes: Maximum memory to use for dataset caching.
                                    If None, loads entire dataset into memory.
        """
        self.dataset = dataset
        self.datasets_dir = datasets_dir
        self.process_row = process_row
        self.max_memory_usage_bytes = max_memory_usage_bytes

        self.data = None  # Used by subclasses to store the loaded dataset
        self.loaded = False
        self.logger = getLogger(__name__)

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


class RowToPrompt:
    """Utility callable class that converts a row of a dataset (in the form of a dictionary)
    to a prompt string given a format string. Meant to be used as the 'process_row' argument
    in the DataLoader constructor.

    The keys used in format tags must match the column names of the dataset it is used with.
    """

    def __init__(self, prompt_format: str):
        self.prompt_format = prompt_format

    def __call__(self, row: dict[str, Any]) -> str:
        return self.prompt_format.format(**row)


class CSVLoader(DataLoader, format=DatasetFormat.CSV):
    """Dataloader implementation for CSV format datasets.

    Loads CSV files using csv.DictReader, where each row is represented
    as a dictionary with column names as keys.

    Attributes:
        data: List of dictionaries, where each dictionary represents a row.
        loaded: Whether the dataset has been loaded into memory.
    """

    def load(self, force: bool = False):
        """Load the dataset from the CSV file into memory.

        This method reads the entire CSV file using csv.DictReader and stores
        it in memory as a list of dictionaries. Each dictionary represents a row
        with column names as keys.

        Args:
            force: If True, reloads even if already loaded (for refreshing data).

        Raises:
            FileNotFoundError: If CSV file doesn't exist.
            csv.Error: If file is corrupted or not a valid CSV.
        """
        if self.loaded and not force:
            return

        file_path = self.datasets_dir / self.dataset.filename
        self.logger.debug(f"Loading CSV data from {file_path}")

        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if self.process_row is None:
                self.data = list(reader)
            else:
                self.data = [self.process_row(row) for row in reader]

        self.logger.debug(f"Loaded {len(self.data)} samples from {file_path}")
        self.loaded = True

    def load_sample(self, index: int) -> dict[str, Any]:
        """Load a single sample from the dataset by index.

        Args:
            index: Sample index (0 to num_samples()-1).

        Returns:
            A dictionary with column names as keys (as returned by csv.DictReader).

        Raises:
            AssertionError: If data is not loaded.
            IndexError: If index is out of range.
        """
        assert self.loaded, "Data is not loaded. Call load() to load the data."
        return self.data[index]

    def num_samples(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            Total sample count (positive integer).
        """
        return len(self.data)


class PandasDataFrameLoader(DataLoader, format=DatasetFormat.PANDAS_DF):
    """Dataloader implementation for Pandas DataFrame format datasets.

    Loads files that were saved via pandas.DataFrame.to_pickle().
    """

    def load(self, force: bool = False):
        if self.loaded and not force:
            return
        self.data = pandas.read_pickle(self.datasets_dir / self.dataset.filename)
        self.logger.debug(
            f"Loaded {len(self.data)} samples from {self.datasets_dir / self.dataset.filename}"
        )
        self.loaded = True

    def load_sample(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index].to_dict()
        if self.process_row:
            return self.process_row(row)
        return row

    def num_samples(self) -> int:
        return len(self.data)


class DeepSeekR1ChatCompletionDataLoader(PandasDataFrameLoader):
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
