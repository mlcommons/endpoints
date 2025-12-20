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
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

import pandas

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


class ParquetLoader(DataLoader, format=DatasetFormat.PARQUET):
    """Dataloader implementation for Parquet format datasets.

    Loads files that were saved via pandas.DataFrame.to_parquet().
    """

    def load(self, force: bool = False):
        if self.loaded and not force:
            return
        self.data = pandas.read_parquet(self.datasets_dir / self.dataset.filename)
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
