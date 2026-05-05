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
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk

from ..config.schema import APIType, ModelParams
from .transforms import Transform, apply_transforms, get_transforms_for_api_type

if TYPE_CHECKING:
    from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

logger = getLogger(__name__)


class DatasetFormat(Enum):
    """Enum defining possible supported formats for accuracy datasets to be saved. The value of the enum
    defines the file extension to be saved as.
    """

    CSV = ".csv"
    """Comma-separated values file with a column header."""

    PARQUET = ".parquet"
    """Apache Parquet file."""

    JSON = ".json"
    """JSON file containing a list of records (dictionaries), where keys are column names."""

    JSONL = ".jsonl"
    """JSON Lines file. Each line is a JSON object where the keys are the column names. It is assumed that
    every row has the same keys."""

    HF = "huggingface"
    """HuggingFace dataset."""


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
    ) -> None:
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
        assert self.dataframe is not None
        return len(self.dataframe)

    @classmethod
    def get_loader(
        cls, file_path: os.PathLike, format: DatasetFormat | None = None
    ) -> type["DatafileLoader"]:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
        super().__init__(*args, **kwargs)
        self.csv_path = Path(csv_path)

    def read(self) -> None:
        self.dataframe = pd.read_csv(self.csv_path)


class JsonlLoader(DatafileLoader, format=DatasetFormat.JSONL):
    def __init__(
        self,
        jsonl_path: Path | str,
        *args,
        **kwargs,
    ) -> None:
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
    ) -> None:
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
    - Loading datasets from various formats (JSONL, HuggingFace, CSV, etc.)
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

    DATASET_ID: ClassVar[str]
    """The unique identifier for the dataset. Automatically set by __init_subclass__."""

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
    ) -> None:
        if self.__class__.COLUMN_NAMES is not None:
            if dataframe is None:
                raise ValueError(
                    f"dataframe cannot be None when COLUMN_NAMES is specified for {self.__class__.__name__}"
                )
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
        self.data: list[dict[str, Any]] | None = None

    @classmethod
    def load_from_file(
        cls,
        file_path: PathLike,
        transforms: list[Transform] | None = None,
        format: DatasetFormat | None = None,
        dataset_id: str | None = None,
        num_repeats: int = 1,
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
            repeats=num_repeats,
        )

    def load(
        self,
        adapter: "HttpRequestAdapter | None" = None,
        api_type: APIType | None = None,
        model_params: ModelParams | None = None,
        force: bool = False,
    ):
        """Load the dataset into memory for pre-processing. After transforms are applied,
        the dataset is converted to a contiguous numpy array.

        Args:
            adapter: If set, will apply the adapter's required transforms to the dataset after any
                user-defined transforms. (Default: None)
            api_type: If adapter is not specified, will use the API type to get the transforms for
                the adapter. (Default: None)
            force: If True, reloads even if already loaded (for refreshing data). (Default: False)
        """
        if not force and hasattr(self, "data") and self.data is not None:
            return

        df = self.dataframe
        if df is None:
            raise ValueError(
                f"Cannot load dataset {self.__class__.__name__}: dataframe is None"
            )

        transforms = []
        if self.transforms is not None:
            transforms.extend(self.transforms)

        # If adapter is specified, use it to get transforms, otherwise fallback to use APIType to
        # get transforms.
        if adapter is not None and model_params is not None:
            transforms.extend(adapter.dataset_transforms(model_params))
        elif api_type is not None and model_params is not None:
            transforms.extend(get_transforms_for_api_type(api_type, model_params))

        if transforms:
            df = apply_transforms(df, transforms)

        # Convert numpy arrays to lists because msgspec does not support numpy arrays
        for col in df.columns:
            if isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].map(np.ndarray.tolist)
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
        assert self.data is not None, "Dataset not loaded. Call load() first."
        return self.data[index]

    def num_samples(self) -> int:
        assert self.data is not None, "Dataset not loaded. Call load() first."
        return len(self.data)

    @classmethod
    def get_dataloader(
        cls,
        datasets_dir: Path = Path("dataset_cache"),
        num_repeats: int = 1,
        transforms: list[Transform] | None = None,
        force_regenerate: bool = False,
        **kwargs,
    ) -> "Dataset":
        if not hasattr(cls, "generate"):
            raise ValueError(
                f"Dataset {cls.__name__} does not have a generate method and cannot be auto-loaded"
            )

        if not callable(cls.generate):
            raise ValueError(
                f"Dataset {cls.__name__} has a generate method that is not callable and cannot be auto-loaded"
            )

        df = cls.generate(datasets_dir=datasets_dir, force=force_regenerate, **kwargs)
        return cls(df, transforms=transforms, repeats=num_repeats)


class SaltedDataset(Dataset):
    """Wraps a loaded Dataset, prepending a unique random salt to each prompt on load_sample().

    Each call to load_sample() generates a fresh salt, so reused samples (when
    n_requests > dataset size) each receive a distinct salt.
    """

    def __init__(self, inner: Dataset) -> None:
        # Skip Dataset.__init__ — all state is delegated to inner
        self._inner = inner
        self.data = inner.data
        self.dataframe = None
        self.transforms = None
        self.repeats = inner.repeats
        self.logger = getLogger(__name__)

    def load(
        self,
        adapter: "HttpRequestAdapter | None" = None,
        api_type: APIType | None = None,
        model_params: ModelParams | None = None,
        force: bool = False,
    ) -> None:
        pass  # Inner dataset already loaded

    def load_sample(self, index: int) -> Any:
        data = self._inner.load_sample(index)
        if not (isinstance(data, dict) and "prompt" in data):
            return data
        prompt = data["prompt"]
        salt = os.urandom(8).hex()
        if isinstance(prompt, str):
            return {**data, "prompt": f"[{salt}] {prompt}"}
        if (
            isinstance(prompt, list)
            and prompt
            and isinstance(prompt[0], dict)
            and prompt[0].get("type") == "text"
        ):
            salted_parts = [
                {**prompt[0], "text": f"[{salt}] {prompt[0]['text']}"},
                *prompt[1:],
            ]
            return {**data, "prompt": salted_parts}
        return data  # unsupported prompt type — skip salting

    def num_samples(self) -> int:
        return self._inner.num_samples()


class EmptyDataset(Dataset):
    """Empty dataset to be used as performance dataset when running only accuracy tests."""

    def __init__(self) -> None:
        super().__init__(None)

    def load_sample(self, index: int) -> None:
        return None

    def num_samples(self) -> int:
        return 0
