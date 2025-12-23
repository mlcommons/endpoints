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

from logging import getLogger
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd

from inference_endpoint.dataset_manager.dataset import Dataset, DatasetFormat


class RowProcessor:
    """Class for processing rows of a dataframe."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, row: dict[str, Any]) -> Any:
        """Process a row of a dataframe."""
        return row


class DataLoader:
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

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        row_processor: RowProcessor | None = None,
    ):
        self.dataframe = dataframe
        self.logger = getLogger(__name__)
        self.row_processor = row_processor or RowProcessor()

    @classmethod
    def load_from_file(
        cls,
        file_path: PathLike,
        row_processor: RowProcessor | None = None,
        format: DatasetFormat | None = None,
    ) -> "DataLoader":
        assert format is None or isinstance(
            format, DatasetFormat
        ), "Format must be a DatasetFormat"
        # TODO add arguments to the loader class
        LoaderClass = Dataset.get_loader(file_path, format=format)
        return DataLoader(
            LoaderClass(file_path).get_dataframe(),
            row_processor=row_processor or RowProcessor(),
        )

    def load(self):
        """Load the dataset into memory for pre-processing.

        Args:
            force: If True, reloads even if already loaded (for refreshing data).
        """
        self.data = []
        for row in self.dataframe.to_dict(orient="records"):
            self.data.append(self.row_processor(row))
        self.data = np.ascontiguousarray(self.data)

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
