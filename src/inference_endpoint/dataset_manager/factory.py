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

"""Dataset loader factory for creating appropriate loaders based on format.

TODO: Very simple factory for now. Will be expanded to support multiple formats and datasets.
"""

import logging

from inference_endpoint.dataset_manager.dataset import DatasetFormat

from .dataset import Dataset
from .transforms import AddStaticColumns, ColumnNameRemap

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating dataset loaders based on format.

    Supports:
    - pkl: Pickle format (PickleReader)
    - parquet: Parquet format (ParquetReader)
    - jsonl: JSON Lines format (JsonlReader)
    - hf: HuggingFace datasets (HFDataLoader)
    """

    @staticmethod
    def create_loader(
        config: Dataset,
        metadata: dict | None = None,
        **kwargs,
    ) -> Dataset:
        """Create appropriate dataset loader based on format.

        Args:
            config: Dataset configuration
            metadata: Dictionary of metadata for the loader
        """
        dataset_path = config.path
        format = config.format
        remap = config.parser
        name = config.name
        if name in Dataset.PREDEFINED:
            return Dataset.PREDEFINED[name].get_dataloader(**kwargs)
        if format is not None:
            format = DatasetFormat(format)

        if remap is None:
            remap = {"prompt": "text_input"}

        transforms = [ColumnNameRemap(remap)]
        if metadata is not None:
            transforms.append(AddStaticColumns(metadata))
        return Dataset.load_from_file(
            dataset_path,
            transforms=transforms,
            format=format,
        )
