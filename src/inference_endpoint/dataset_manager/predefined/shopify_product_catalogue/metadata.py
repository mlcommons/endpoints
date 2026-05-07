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

"""Pydantic models for Shopify product catalogue (shared with presets to avoid circular imports)."""

from pydantic import BaseModel, ConfigDict


class ProductMetadata(BaseModel):
    """JSON format for expected VLM responses (matches MLCommons Q3VL schema).

    Reference: https://github.com/mlcommons/inference/blob/master/multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/schema.py
    """

    model_config = ConfigDict(extra="forbid")

    category: str
    """Complete category path, e.g. 'Clothing & Accessories > Clothing > Shirts > Polo Shirts'."""

    brand: str
    """Brand of the product, e.g. 'giorgio armani'."""

    is_secondhand: bool
    """True if second-hand, False otherwise."""
