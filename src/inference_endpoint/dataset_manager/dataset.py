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

import inspect
import os
from abc import ABC
from enum import Enum
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from datasets import load_dataset, load_from_disk

from .transforms import Transform, apply_transforms

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


class DatafileLoader(ABC):
    """Base class for dataset loaders. It is assumed that after preprocessing, the dataset will be stored in a tabular format as a pandas dataframe.

    The format of the dataset that is saved to disk is fixed and determined by the 'format' class parameter. If
    other formats are needed, new subclasses should be created with their own unique names. This is to prevent
    ambiguity and discrepancies when specifying a dataset name in a benchmark config file.
    """

    IMPLEMENTATIONS: ClassVar[dict[str, type["DatafileLoader"]]] = {}

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

        DatafileLoader.IMPLEMENTATIONS[cls.FORMAT.value] = cls

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataframe: pd.DataFrame | None = None

    def read(self) -> None:
        """Read the dataset from the file."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_dataframe(self) -> pd.DataFrame:
        """Get the dataset as a pandas dataframe."""
        return self.dataframe

    def get_num_samples(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataframe)

    @classmethod
    def get_loader(
        cls, file_path: os.PathLike, format: DatasetFormat | None = None
    ) -> "DatafileLoader":
        """Get the loader for the dataset."""

        if format is not None:
            return DatafileLoader.IMPLEMENTATIONS[format.value]
        else:
            ext = Path(file_path).suffix
        if DatafileLoader.IMPLEMENTATIONS.get(ext):
            return DatafileLoader.IMPLEMENTATIONS[ext]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


class ParquetLoader(DatafileLoader, format=DatasetFormat.PARQUET):
    def __init__(
        self,
        file_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parquet_path = Path(file_path)

    def read(self) -> None:
        # Note we need the dtype_backend="pyarrow" to avoid issues with numpy arrays in the dataframe
        self.dataframe = pd.read_parquet(self.parquet_path, dtype_backend="pyarrow")


class HuggingFaceLoader(DatafileLoader, format=DatasetFormat.HF):
    def __init__(
        self,
        file_path: Path | str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.dataset_name = kwargs.get("dataset_name", None)
        if not self.file_path and not self.dataset_name:
            raise ValueError("Either dataset_path or dataset_name must be provided")
        self.split = kwargs.get("split", "train")

    def read(self) -> None:
        if self.file_path:
            ds = load_from_disk(self.file_path)
            self.dataframe = ds[self.split].to_pandas()
        else:
            ds = load_dataset(
                path=self.file_path, name=self.dataset_name, split=self.split
            )
            self.dataframe = ds.to_pandas()


class CSVLoader(DatafileLoader, format=DatasetFormat.CSV):
    def __init__(
        self,
        csv_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.csv_path = Path(csv_path)

    def read(self) -> None:
        self.dataframe = pd.read_csv(self.csv_path)


class PickleListLoader(DatafileLoader, format=DatasetFormat.PICKLE):
    def __init__(
        self,
        file_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pickle_path = Path(file_path)

    def read(self) -> None:
        self.dataframe = pd.read_pickle(self.pickle_path)


class JsonlLoader(DatafileLoader, format=DatasetFormat.JSONL):
    def __init__(
        self,
        jsonl_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.jsonl_path = Path(jsonl_path)

    def read(self) -> None:
        self.dataframe = pd.read_json(self.jsonl_path, lines=True)


class JsonLoader(DatafileLoader, format=DatasetFormat.JSON):
    def __init__(
        self,
        json_path: Path | str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.json_path = Path(json_path)

    def read(self) -> None:
        self.dataframe = pd.read_json(self.json_path)


def load_from_huggingface(
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    cache_dir: Path | None = None,
    load_options: dict[str, Any] | None = None,
    cache_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a dataset from HuggingFace.

    Args:
        dataset_path: The path to the dataset on HuggingFace. See HuggingFace docs for more details.
        dataset_name: The name of the dataset from the path to load. See HuggingFace docs for more details.
        split: The split of the dataset. Defaults to "train".
        cache_dir: Optional explicit cache directory to load dataset from. This is useful if your dataset is
            saved to an external storage location not in your local HuggingFace cache.
        load_options: Optional additional options to pass to the load_dataset function. See HuggingFace docs for more details.
        cache_options: Optional additional options to pass to the save_to_disk function. See HuggingFace docs for more details.

    Returns:
        A pandas dataframe containing the dataset.
    """
    load_options = load_options or {}
    cache_options = cache_options or {}

    if cache_dir is not None and cache_dir.exists():
        try:
            ds = load_from_disk(str(cache_dir), **cache_options)
            return ds[split].to_pandas()
        except Exception as e:
            logger.warning(f"Error loading dataset from cache: {e}")
    ds = load_dataset(dataset_path, dataset_name, **load_options)

    if cache_dir is not None:
        try:
            ds.save_to_disk(str(cache_dir), **cache_options)
        except Exception as e:
            logger.warning(f"Error caching dataset: {e}")
    return ds[split].to_pandas()


class Dataset:
    """Class for loading and managing benchmark datasets.

    DataLoaders handle:
    - Loading datasets from various formats (pickle, HuggingFace, CSV, etc.)
    - Memory management for large datasets
    - Random-access sample retrieval by index
    - Optional memory-constrained caching/unloading

    The DataLoader is responsible for raw data loading only. Parsing and
    transformation (e.g., converting to request format) is handled separately
    by parser functions.
    """

    COLUMN_NAMES: ClassVar[list[str] | None] = None
    """The column names of the dataset. If proovided by a subclass, upon creation of an instance,
    an error will be raised if all elements of the list are not present in the columns of the dataframe."""

    PREDEFINED: ClassVar[dict[str, type["Dataset"]]] = {}
    """A dictionary of predefined datasets, as subclasses of Dataset."""

    def __init_subclass__(
        cls,
        dataset_id: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            if dataset_id is None:
                dataset_id = cls.__name__
            cls.DATASET_ID = dataset_id
            Dataset.PREDEFINED[dataset_id] = cls

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        transforms: list[Transform] | None = None,
        repeats: int = 1,
    ):
        if self.__class__.COLUMN_NAMES is not None:
            common = set(self.__class__.COLUMN_NAMES) & set(dataframe.columns)
            if len(common) != len(self.__class__.COLUMN_NAMES):
                missing = set(self.__class__.COLUMN_NAMES) - common
                raise ValueError(
                    f"Required columns {missing} are not present in the dataframe"
                )

        self.dataframe = dataframe
        self.logger = getLogger(__name__)
        self.transforms = transforms
        self.repeats = repeats

    @classmethod
    def load_from_file(
        cls,
        file_path: PathLike,
        transforms: list[Transform] | None = None,
        format: DatasetFormat | None = None,
        dataset_id: str | None = None,
    ) -> "Dataset":
        assert format is None or isinstance(
            format, DatasetFormat
        ), "Format must be a DatasetFormat"
        # TODO add arguments to the loader class
        LoaderClass = DatafileLoader.get_loader(file_path, format=format)
        loader = LoaderClass(file_path)
        loader.read()

        ds_class = cls
        if dataset_id is not None:
            ds_class = Dataset.PREDEFINED[dataset_id]
        return ds_class(
            loader.get_dataframe(),
            transforms=transforms,
        )

    def load(self):
        """Load the dataset into memory for pre-processing. After transforms are applied,
        the dataset is converted to a contiguous numpy array.

        Args:
            force: If True, reloads even if already loaded (for refreshing data).
        """
        df = self.dataframe
        if self.transforms is not None:
            df = apply_transforms(df, self.transforms)
        # Convert numpy arrays to lists because msgspec does not support numpy arrays
        for col in df.columns:
            if isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].map(np.ndarray.tolist)
        # Repeat the dataframe if the number of repeats is greater than 1
        if self.repeats > 1:
            df = pd.concat([df] * self.repeats, ignore_index=True)
        self.data = df.to_dict(orient="records")

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
        return self.data[index]

    def num_samples(self) -> int:
        return len(self.data)

    @classmethod
    def get_dataloader(
        cls,
        datasets_dir: Path = Path("datasets"),
        num_repeats: int = 1,
        transforms: list[Transform] | None = None,
        force_regenerate: bool = False,
    ) -> "Dataset":
        if not hasattr(cls, "generate"):
            raise ValueError(
                f"Dataset {cls.__name__} does not have a generate method and cannot be auto-loaded"
            )

        if not callable(cls.generate):
            raise ValueError(
                f"Dataset {cls.__name__} has a generate method that is not callable and cannot be auto-loaded"
            )

        df = cls.generate(datasets_dir=datasets_dir, force=force_regenerate)
        return cls(df, transforms=transforms, repeats=num_repeats)


class EmptyDataset(Dataset):
    """Empty dataset to be used as performance dataset when running only accuracy tests."""

    def __init__(self):
        super().__init__(None)

    def load_sample(self, index: int):
        return None

    def num_samples(self):
        return 0
