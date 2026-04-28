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
from pathlib import Path

from inference_endpoint.config.schema import Dataset as DatasetConfig
from inference_endpoint.dataset_manager.dataset import Dataset, DatasetFormat

from .multi_turn_dataset import MultiTurnDataset
from .transforms import ColumnRemap, MakeAdapterCompatible, Transform

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating dataset loaders based on format.

    Supports:
    - csv: CSV format (CSVLoader)
    - json: JSON format (JsonLoader)
    - parquet: Parquet format (ParquetLoader)
    - jsonl: JSON Lines format (JsonlLoader)
    - huggingface: HuggingFace datasets (HuggingFaceLoader)
    """

    @staticmethod
    def create_loader(config: DatasetConfig, num_repeats: int = 1, **kwargs) -> Dataset:
        """Create appropriate dataset loader based on format.

        Args:
            config: Dataset configuration
            num_repeats: Number of times to repeat the dataset
            **kwargs: Additional keyword arguments to use for predefined datasets. Passed to
                Dataset.get_dataloader()
        """
        dataset_path = config.path
        file_format = config.format
        remap = config.parser
        name = config.name
        preset = None

        if "::" in name:
            parts = name.split("::")
            if len(parts) != 2:
                raise ValueError(f"Invalid dataset name: {name}")
            name = parts[0]
            preset = parts[1]

        if name in Dataset.PREDEFINED:
            ds_cls = Dataset.PREDEFINED[name]
            preset_transforms = None

            # If preset is provided, search for the preset in the dataset class
            if preset is not None:
                if not hasattr(ds_cls, "PRESETS"):
                    raise ValueError(
                        f"Dataset {name} does not have preset model transforms"
                    )

                if not hasattr(ds_cls.PRESETS, preset):
                    raise ValueError(
                        f"Dataset {name} does not have a preset model transform for {preset}"
                    )

                preset_transforms = getattr(ds_cls.PRESETS, preset)()
            return ds_cls.get_dataloader(
                transforms=preset_transforms,
                num_repeats=num_repeats,
                **kwargs,
            )

        if name not in Dataset.PREDEFINED and dataset_path is None:
            raise ValueError(
                f"Dataset {name} is not predefined and no dataset path provided - predefined datasets are: {list(Dataset.PREDEFINED.keys())}"
            )

        format_enum: DatasetFormat | None = None
        if file_format is not None:
            format_enum = DatasetFormat(file_format)

        dataset_id = None
        if config.multi_turn is not None:
            dataset_id = MultiTurnDataset.DATASET_ID

        transforms: list[Transform] = []
        if remap is not None:
            # Parser convention is {target: source} (e.g. {prompt: article}).
            # ColumnRemap expects {source: target} — flip it.
            flipped = {src: dst for dst, src in remap.items()}
            transforms.append(ColumnRemap(flipped))  # type: ignore[arg-type]
        if dataset_id != MultiTurnDataset.DATASET_ID:
            transforms.append(MakeAdapterCompatible())

        assert dataset_path is not None
        return Dataset.load_from_file(
            Path(dataset_path),
            transforms=transforms,
            format=format_enum,
            dataset_id=dataset_id,
            num_repeats=num_repeats,
        )
