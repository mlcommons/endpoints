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

from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

import datasets as hf_datasets

logger = getLogger(__name__)


class DatasetFormat(Enum):
    """Enum defining possible supported formats for accuracy datasets to be saved. The value of the enum
    defines the file extension to be saved as.
    """

    CSV = "csv"
    """Comma-separated values file with a column header."""

    PARQUET = "parquet"
    """Apache Parquet file."""

    PANDAS_DF = "pandas_pkl"
    """Pandas DataFrame."""

    NUMPY_ARRAY = "npy"
    """NumPy array. Assumed to be a 2D array using the datatype np.dtypes.StringDType supported in Numpy 2.x+.
    The first row is assumed to denote the column names."""

    PY_DICT = "py_pkl"
    """Python dictionary. Must have the keys 'column_names' (a list of strings) and 'rows' (a list of lists of
    strings, each having the same length as 'column_names')."""

    PY_LIST = "py_list"
    """Python list of rows. Each row is a dictionary with the keys being the column names. This format is
    equivalent to PY_DICT where each row is dict(zip(py_dict["column_names"], py_dict["rows"][i]))."""

    JSON = "json"
    """JSON file in the same structure as the PY_DICT format, but saved as a JSON file instead of Pickle."""

    JSONL = "jsonl"
    """JSON Lines file. Each line is a JSON object where the keys are the column names. It is assumed that
    every row has the same keys."""


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
    COLUMNS: ClassVar[list[str] | None] = None
    FORMAT: ClassVar[DatasetFormat | None] = None
    GROUND_TRUTH_COLUMN: ClassVar[str] = "ground_truth"

    def __init_subclass__(
        cls,
        columns: list[str] | None = None,
        format: DatasetFormat | None = None,
        ground_truth_column: str = "ground_truth",
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if columns is not None:
            cls.COLUMNS = columns
        else:
            raise ValueError("Must specify 'columns' when subclassing Dataset")

        if format is not None:
            cls.FORMAT = format
        else:
            raise ValueError("Must specify 'format' when subclassing Dataset")

        if ground_truth_column is not None:
            cls.GROUND_TRUTH_COLUMN = ground_truth_column
        else:
            raise ValueError(
                "Must specify 'ground_truth_column' when subclassing Dataset"
            )

        Dataset.IMPLEMENTATIONS[cls.__name__] = cls

    def __init__(
        self,
        *args,
        variant: str = "full",
        has_labels: bool = True,
        **kwargs,
    ):
        """Initializes a dataset.

        Args:
            variant: The 'variant' of the dataset being generated. Dataset variants are specific to the dataset
                being derived from. For instance, training, validation, and test splits can be considered as
                variants. If a dataset has popular used subsets, such as GPQA Diamond, these are also considered
                as variants. By convention, we treat the default, full dataset as the default variant with the name
                'full'.
        """
        super().__init__(*args, **kwargs)
        self.variant = variant

    @property
    def filename(self) -> str:
        ext = self.__class__.FORMAT.value
        name = self.__class__.__name__
        return f"{name}.{self.variant}.{ext}"

    @abstractmethod
    def generate(self, datasets_dir: Path):
        """Generates the dataset and saves it to a file in the specified format. The file
        will be saved with the name in the format:

            <datasets_dir>/<dataset_name>.<variant>.<format extension>

        where <variant> is the specific variant of the dataset that is being generated
        (e.g. subsets, splits, yearly versions, etc.)
        and <format extension> is the extension of the format specified by the FORMAT class
        variable.

        If <variant> is not specified or not applicable, the default value is 'full'. The
        variant should be specied as an instance variable: .variant.

        Args:
            datasets_dir: Directory to save the dataset to.
        """
        raise NotImplementedError


def load_from_huggingface(
    dataset_path: str,
    dataset_name: str | None = None,
    split: str = "train",
    cache_dir: Path | None = None,
    load_options: dict[str, Any] | None = None,
    cache_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a dataset from HuggingFace"""
    if load_options is None:
        load_options = {}
    if cache_options is None:
        cache_options = {}

    if cache_dir is not None and cache_dir.exists():
        try:
            ds = hf_datasets.load_from_disk(str(cache_dir), **load_options)
            return ds[split].to_pandas()
        except Exception as e:
            logger.warning(f"Error loading dataset from cache: {e}")

    ds = hf_datasets.load_dataset(dataset_path, name=dataset_name)
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache_dir), **cache_options)
        except Exception as e:
            logger.warning(f"Error caching dataset: {e}")
    return ds[split].to_pandas()
