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
import random
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd

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
        seed: int = 0,
        max_samples: int | None = None,
        force: bool = False,
        token: str | None = None,
        revision: str = "main",
    ) -> pd.DataFrame:
        """Generate the Shopify product catalogue dataset.

        Loads from HuggingFace and converts images to base64 for parquet storage.
        Use token for gated/private datasets.

        Args:
            datasets_dir: Root datasets directory.
            split: Splits to load (e.g. ["train", "test"]). Defaults to ["train", "test"].
            seed: Random seed for sampling.
            max_samples: Max samples to keep. If None, use full split.
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
            try:
                df = load_from_huggingface(
                    dataset_path=cls.REPO_ID,
                    split=s,
                    cache_dir=datasets_dir / "hf_cache" / cls.DATASET_ID,
                    load_options=load_options,
                )
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                logger.error("Note: This dataset may require HuggingFace authentication.")
                logger.error("Run: huggingface-cli login")
                raise

            logger.info(f"Loaded {len(df)} samples from Shopify product catalogue ({s})")

            # Convert product_image (PIL) to base64 for parquet storage.
            for _, row in df.iterrows():
                image = row.get("product_image")
                if image is None:
                    raise ValueError("product_image is missing from dataset row")
                img = image if hasattr(image, "save") else image
                buf = BytesIO()
                image_format = getattr(img, "format", None) or "PNG"
                img.save(buf, format=image_format)
                image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                categories = row.get("potential_product_categories", [])
                if isinstance(categories, str):
                    try:
                        categories = json.loads(categories)
                    except json.JSONDecodeError:
                        categories = [categories]

                all_rows.append({
                    "product_title": row.get("product_title", ""),
                    "product_description": row.get("product_description", ""),
                    "product_image_base64": image_base64,
                    "product_image_format": image_format,
                    "potential_product_categories": json.dumps(categories)
                    if isinstance(categories, (list, dict))
                    else str(categories),
                })

        df = pd.DataFrame(all_rows)

        if max_samples is not None and len(df) > max_samples:
            rng = random.Random(seed)
            df = df.sample(n=max_samples, random_state=rng).reset_index(drop=True)
            logger.info(f"Sampled {max_samples} samples")

        df.to_parquet(dst_path)
        logger.info(f"Saved {len(df)} samples to {dst_path}")
        return df


__all__ = ["Shopify"]
