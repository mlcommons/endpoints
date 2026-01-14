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

from pathlib import Path

import pandas as pd
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnNameRemap,
)

from ...dataset import Dataset


class OpenOrca(
    Dataset,
    dataset_id="open-orca-accuracy",
):
    """OpenOrca GPT4 tokenized dataset for accuracy evaluation."""

    @classmethod
    def get_dataloader(
        cls,
        dataset_path: str | Path,
        metadata: dict | None = None,
        remap: dict | None = None,
    ):
        """Load the OpenOrca dataset from a file.

        Args:
            dataset_path: Path to the dataset file (typically .pkl format)
            metadata: Optional metadata to add to the dataset
            remap: Column name remapping dict. If None, defaults to {"prompt": "text_input"}

        Returns:
            OpenOrca dataset instance
        """
        # Load the dataset from file
        df = pd.read_pickle(dataset_path)

        # Apply same parser/remap logic as factory.py
        if remap is None:
            remap = {"prompt": "text_input"}

        # Create transforms
        transforms = [ColumnNameRemap(remap)]

        if metadata is not None:
            transforms.append(AddStaticColumns(metadata))

        return cls(df, transforms=transforms)


__all__ = ["OpenOrca"]
