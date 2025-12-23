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

"""Dataset loader factory for creating appropriate loaders based on format.

TODO: Very simple factory for now. Will be expanded to support multiple formats and datasets.
"""

import logging
from pathlib import Path
from typing import Any

from inference_endpoint.dataset_manager.dataset import DatasetFormat

from .dataloader import (
    DataLoader,
    RowProcessor,
)

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
        dataset_path: Path | str,
        format: str | None = None,
        key_maps: list[dict[str, str]] | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> DataLoader:
        """Create appropriate dataset loader based on format.

        Args:
            dataset_path: Path to dataset file or directory
            key_maps: Dictionary of key mappings for the parser
            metadata: Dictionary of metadata for the loader
            **kwargs: Additional arguments for specific loaders

        Returns:
            DataLoader instance

        Raises:
            ValueError: If format is unsupported
        """

        class KeyMapRowProcessor(RowProcessor):
            def __init__(self):
                if key_maps is None:
                    self.key_maps = [{"prompt": "text_input"}]
                else:
                    self.key_maps = key_maps
                super().__init__()

            def __call__(self, row: dict[str, Any]) -> Any:
                return {k: row[v] for k, v in self.key_maps[0].items()} | (
                    metadata or {}
                )

        if format is not None:
            format = DatasetFormat(format)
        return DataLoader.load_from_file(
            dataset_path, row_processor=KeyMapRowProcessor(), format=format
        )
