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
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ...dataset import Dataset, load_from_huggingface
from . import presets

logger = getLogger(__name__)


class Shopify(
    Dataset,
    dataset_id="shopify",
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
            df = load_from_huggingface(
                dataset_path=cls.REPO_ID,
                split=s,
                load_options=load_options,
            )
            logger.info(
                f"Loaded {len(df)} samples from Shopify product catalogue ({s})"
            )

            # Convert product_image (dict with bytes/path) to base64 for parquet storage.

            ext_to_format = {"": "JPEG", ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
            for row in tqdm(
                df.to_dict("records"),
                total=len(df),
                desc=f"Converting images ({s})",
                unit="rows",
            ):
                image = row.get("product_image")
                if image is None:
                    raise ValueError("product_image is missing from dataset row")
                image_base64 = base64.b64encode(image["bytes"]).decode("utf-8")
                ext = Path(image.get("path", "")).suffix.lower()
                image_format = ext_to_format.get(ext, "JPEG")

                categories = row.get("potential_product_categories", [])
                if hasattr(categories, "tolist"):
                    categories = categories.tolist()

                all_rows.append(
                    {
                        "product_title": row.get("product_title", ""),
                        "product_description": row.get("product_description", ""),
                        "product_image_base64": image_base64,
                        "product_image_format": image_format,
                        "potential_product_categories": json.dumps(categories),
                    }
                )

        df = pd.DataFrame(all_rows)

        df.to_parquet(dst_path)
        logger.info(f"Saved {len(df)} samples to {dst_path}")
        return df


__all__ = ["Shopify"]
