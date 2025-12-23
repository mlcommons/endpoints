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

import inspect
import os
from abc import ABC
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

import datasets as hf_datasets
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

logger = getLogger(__name__)


class DatasetFormat(Enum):
    """Enum defining possible supported formats for accuracy datasets to be saved. The value of the enum
    defines the file extension to be saved as.
    """

    CSV = ".csv"
    """Comma-separated values file with a column header."""

    PARQUET = ".parquet"
    """Apache Parquet file."""

    PANDAS_DF = ".pandas_pkl"
    """Pandas DataFrame."""

    NUMPY_ARRAY = ".npy"
    """NumPy array. Assumed to be a 2D array using the datatype np.dtypes.StringDType supported in Numpy 2.x+.
    The first row is assumed to denote the column names."""

    PICKLE = ".pkl"
    """Python list of rows. Each row is a dictionary with the keys being the column names. This format is
    equivalent to PY_DICT where each row is dict(zip(py_dict["column_names"], py_dict["rows"][i]))."""

    JSON = ".json"
    """JSON file in the same structure as the PY_DICT format, but saved as a JSON file instead of Pickle."""

    JSONL = ".jsonl"
    """JSON Lines file. Each line is a JSON object where the keys are the column names. It is assumed that
    every row has the same keys."""

    HF = "huggingface"
    """HuggingFace dataset."""

    RANDOM = "random"
    """Random dataset. This is a dataset that is generated randomly."""


class Dataset(ABC):
    """Base class for datasets. Each dataset must define a static set of columns that defines the schema of the
    dataset. It is assumed that after preprocessing, the dataset will be stored in a tabular format with the
    specified columns.

    The format of the dataset that is saved to disk is fixed and determined by the 'format' class parameter. If
    other formats are needed, new subclasses should be created with their own unique names. This is to prevent
    ambiguity and discrepancies when specifying a dataset name in a benchmark config file.
    """

    IMPLEMENTATIONS: ClassVar[dict[str, type["Dataset"]]] = {}

    # Only used by subclasses
    FORMAT: ClassVar[DatasetFormat | None] = None

    def __init_subclass__(
        cls,
        format: DatasetFormat | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        is_abstract_class = inspect.isabstract(cls)
        if is_abstract_class:
            # Abstract classes are not registered as implementations
            # nor do they require columns/formats.
            return

        if format is not None:
            cls.FORMAT = format
        else:
            raise ValueError("Must specify 'format' when subclassing Dataset")

        Dataset.IMPLEMENTATIONS[cls.FORMAT.value] = cls

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataframe: pd.DataFrame | None = None

    def get_dataframe(self) -> pd.DataFrame:
        """Get the dataset as a pandas dataframe."""
        return self.dataframe

    def get_num_samples(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataframe)

    @classmethod
    def get_loader(
        cls, file_path: os.PathLike, format: DatasetFormat | None = None
    ) -> "Dataset":
        """Get the loader for the dataset."""

        if format is not None:
            return Dataset.IMPLEMENTATIONS[format.value]
        else:
            ext = Path(file_path).suffix
        if Dataset.IMPLEMENTATIONS.get(ext):
            return Dataset.IMPLEMENTATIONS[ext]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


class ParquetDataset(Dataset, format=DatasetFormat.PARQUET):
    def __init__(
        self,
        file_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parquet_path = Path(file_path)
        self.dataframe = pd.read_parquet(self.parquet_path)


class HuggingFaceDataset(Dataset, format=DatasetFormat.HF):
    def __init__(
        self,
        file_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.file_path = Path(file_path)
        self.hf_format = kwargs.get("hf_format", "arrow")
        self.split = kwargs.get("split", "train")

        if self.hf_format is None:
            self.data = load_dataset(self.file_path)
        else:
            # huggingface uses a different method to load local arrow datasets
            self.data = load_from_disk(self.file_path)
        self.dataframe = self.data[self.split].to_pandas()


class CSVDataset(Dataset, format=DatasetFormat.CSV):
    def __init__(
        self,
        csv_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.csv_path = Path(csv_path)
        self.dataframe = pd.read_csv(self.csv_path)


class PickleListDataset(Dataset, format=DatasetFormat.PICKLE):
    def __init__(
        self,
        file_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pickle_path = Path(file_path)
        self.dataframe = pd.read_pickle(self.pickle_path)


class JsonlDataset(Dataset, format=DatasetFormat.JSONL):
    def __init__(
        self,
        jsonl_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.jsonl_path = Path(jsonl_path)
        self.dataframe = pd.read_json(self.jsonl_path, lines=True)


class JsonDataset(Dataset, format=DatasetFormat.JSON):
    def __init__(
        self,
        json_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.json_path = Path(json_path)
        self.dataframe = pd.read_json(self.json_path)


class RandomDataset(Dataset, format=DatasetFormat.RANDOM):
    def __init__(
        self,
        *,
        num_sequences: int = 1024,
        input_seq_length: int = 1024,
        range_ratio: float = 1.0,
        random_seed: int = 42,
        save_tokenized_data: bool = False,
        tokenizer: str,
    ):
        super().__init__()
        self.input_seq_length = input_seq_length
        self.num_sequences = num_sequences
        self.range_ratio = range_ratio
        self.random_seed = random_seed
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
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


def load_from_huggingface(
    dataset_path: str,
    dataset_name: str | None = None,
    split: str = "train",
    cache_dir: Path | None = None,
    load_options: dict[str, Any] | None = None,
    cache_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a dataset from HuggingFace"""
    if cache_dir is not None and cache_dir.exists():
        try:
            ds = hf_datasets.load_from_disk(str(cache_dir), **load_options)
            return ds[split].to_pandas()
        except Exception as e:
            logger.warning(f"Error loading dataset from cache: {e}")

    ds = hf_datasets.load_dataset(dataset_path, name=dataset_name)
    if cache_dir is not None:
        try:
            ds.save_to_disk(str(cache_dir), **cache_options)
        except Exception as e:
            logger.warning(f"Error caching dataset: {e}")
    return ds[split].to_pandas()
