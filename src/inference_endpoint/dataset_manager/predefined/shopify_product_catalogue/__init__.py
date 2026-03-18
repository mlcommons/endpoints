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

"""Shopify product catalogue dataset for multimodal product taxonomy classification."""

import base64
import json
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from ...dataset import Dataset, load_from_huggingface
from . import presets

logger = getLogger(__name__)

EXT_TO_FORMAT = {"": "JPEG", ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


def _process_sample_to_row(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a single HF dataset sample to a row dict for parquet storage.

    Handles product_image as either PIL Image (MLCommons-style) or dict with
    bytes/path. Reference: https://github.com/mlcommons/inference/blob/master/
    multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/task.py#L577
    """
    image = sample.get("product_image")
    if image is None:
        raise ValueError("product_image is missing from dataset row")

    if hasattr(image, "save") and hasattr(image, "format"):
        image_file = BytesIO()
        image_format = image.format or "JPEG"
        image.save(image_file, format=image_format)
        image_bytes = image_file.getvalue()
    elif isinstance(image, dict) and "bytes" in image:
        image_bytes = image["bytes"]
        ext = Path(image.get("path", "")).suffix.lower()
        image_format = EXT_TO_FORMAT.get(ext, "JPEG")
    else:
        raise ValueError(
            "product_image must be PIL Image or dict with 'bytes' and 'path'"
        )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    categories = sample.get("potential_product_categories", [])
    if hasattr(categories, "tolist"):
        categories = categories.tolist()

    return {
        "product_title": sample.get("product_title", ""),
        "product_description": sample.get("product_description", ""),
        "product_image_base64": image_base64,
        "product_image_format": image_format,
        "potential_product_categories": json.dumps(categories),
        "ground_truth_category": sample.get("ground_truth_category", ""),
        "ground_truth_brand": sample.get("ground_truth_brand", ""),
        "ground_truth_is_secondhand": json.dumps(
            sample.get("ground_truth_is_secondhand", False)
        ),
    }


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


class ShopifyProductCatalogue(
    Dataset,
    dataset_id="shopify_product_catalogue",
):
    """Shopify product catalogue: multimodal benchmark for product taxonomy classification.

    Reference: https://huggingface.co/datasets/Shopify/product-catalogue

    Each sample includes product image, title, description, and candidate categories.
    Compatible with OpenAI multimodal adapter (prompt/system with vision content).
    """

    COLUMN_NAMES = [
        "product_title",
        "product_description",
        "product_image_base64",
        "product_image_format",
        "potential_product_categories",
        "ground_truth_category",
        "ground_truth_brand",
        "ground_truth_is_secondhand",
    ]

    PRESETS = presets

    REPO_ID = "Shopify/product-catalogue"

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        split: list[str] | None = None,
        force: bool = False,
        token: str | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate the Shopify product catalogue dataset.

        Loads from HuggingFace and converts images to base64 for parquet storage.
        Use token for gated/private datasets.

        Args:
            datasets_dir: Directory to save transformed dataset.
            split: Splits to load (e.g. ["train", "test"]). Defaults to ["train", "test"].
            force: Regenerate even if file exists.
            token: HuggingFace token for gated datasets.
            revision: Dataset revision/branch. Defaults to "main".

        Returns:
            DataFrame with product_title, product_description, product_image_base64,
            product_image_format, potential_product_categories.
        """
        if split is None:
            split = ["train", "test"]
        split_key = "+".join(split)
        filename = f"{cls.DATASET_ID}_{split_key}.parquet"
        dst_path = datasets_dir / cls.DATASET_ID / split_key / filename
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True)

        if dst_path.exists() and not force:
            logger.info(f"Dataset already exists at {dst_path}. Loading from file.")
            return pd.read_parquet(dst_path)

        load_options: dict[str, Any] = {}
        if token is not None:
            load_options["token"] = token
        if revision is not None:
            load_options["revision"] = revision

        all_rows: list[dict[str, Any]] = []
        for s in split:
            ds = load_from_huggingface(
                dataset_path=cls.REPO_ID,
                split=s,
                load_options=load_options,
            )
            logger.info(
                f"Loaded {len(ds)} samples from Shopify product catalogue ({s})"
            )

            for i in tqdm(
                range(len(ds)),
                desc=f"Converting images ({s})",
                unit="rows",
            ):
                sample = ds[i]
                all_rows.append(_process_sample_to_row(sample))

        df = pd.DataFrame(all_rows)

        df.to_parquet(dst_path)
        logger.info(f"Saved {len(df)} samples to {dst_path}")
        return df


__all__ = ["ProductMetadata", "ShopifyProductCatalogue"]
