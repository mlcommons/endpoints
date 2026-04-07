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

"""
Dataset Manager for the MLPerf Inference Endpoint Benchmarking System.

This module handles dataset loading, preprocessing, and management.
"""

from .dataset import Dataset, EmptyDataset
from .factory import DataLoaderFactory
from .predefined.aime25 import AIME25
from .predefined.cnndailymail import CNNDailyMail
from .predefined.gpqa import GPQA
from .predefined.livecodebench import LiveCodeBench
from .predefined.open_orca import OpenOrca
from .predefined.random import RandomDataset
from .predefined.shopify_product_catalogue import ShopifyProductCatalogue

# Import workload-specific datasets so they register in Dataset.PREDEFINED
from inference_endpoint.wan22.dataset import Wan22Dataset  # noqa: E402

from .transforms import (
    AddStaticColumns,
    ColumnFilter,
    ColumnRemap,
    FusedRowProcessor,
    Harmonize,
    MakeAdapterCompatible,
    UserPromptFormatter,
    apply_transforms,
)

__all__ = [
    "Dataset",
    "EmptyDataset",
    "DataLoaderFactory",
    "ColumnFilter",
    "ColumnRemap",
    "AddStaticColumns",
    "UserPromptFormatter",
    "FusedRowProcessor",
    "Harmonize",
    "MakeAdapterCompatible",
    "apply_transforms",
    "AIME25",
    "GPQA",
    "OpenOrca",
    "LiveCodeBench",
    "CNNDailyMail",
    "RandomDataset",
    "ShopifyProductCatalogue",
    "Wan22Dataset",
]
