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
from pathlib import Path
from typing import Any, ClassVar


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

    JSON = "json"
    """JSON file in the same structure as the PY_DICT format, but saved as a JSON file instead of Pickle."""

    JSONL = "jsonl"
    """JSON Lines file. Each line is a JSON object where the keys are the column names. It is assumed that
    every row has the same keys."""


class AccuracyDataset(ABC):
    """Base class for accuracy datasets. Each accuracy dataset must define a static
    set of columns that defines the schema of the dataset. Each dataset will be
    preprocessed and stored in tabular format.

    The format of the dataset that is saved to disk is fixed and determined by the 'format'
    parameter. If other formats are needed, new subclasses should be created with their own
    unique names. This is to prevent ambiguity when specifying the accuracy dataset in benchmark
    config files.
    """

    IMPLEMENTATIONS: ClassVar[dict[str, type["AccuracyDataset"]]] = {}

    # Only used by subclasses.
    COLUMNS: ClassVar[list[str] | None] = None
    FORMAT: ClassVar[DatasetFormat | None] = None

    def __init_subclass__(
        cls,
        columns: list[str] | None = None,
        format: DatasetFormat = DatasetFormat.CSV,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if columns is not None:
            cls.COLUMNS = columns
        else:
            raise ValueError("Must specify 'columns' when subclassing AccuracyDataset")

        if format is not None:
            cls.FORMAT = format
        else:
            raise ValueError("Must specify 'format' when subclassing AccuracyDataset")

        AccuracyDataset.IMPLEMENTATIONS[cls.__name__] = cls

    def __init__(self, *args, variant: str = "full", **kwargs):
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

    @abstractmethod
    def load(self, datasets_dir: Path) -> Any:
        """Loads the dataset from a file in the specified format. See the generate method
        for the expected format of the file name.

        Args:
            datasets_dir: Directory to load the dataset from.

        Returns:
            The dataset in the format specified by the FORMAT class variable.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self, index: int, loaded_dataset: Any) -> str:
        """Get the ground truth for a sample at the given index. This must be implemented
        for use in Evaluators, as each dataset can be loaded as different types of objects.

        Args:
            index: The index of the sample.
            loaded_dataset: The loaded dataset returned by self.load()

        Returns:
            The ground truth for the sample.
        """
        raise NotImplementedError
