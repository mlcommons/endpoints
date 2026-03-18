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

"""Unit tests for Shopify product catalogue dataset initialization and transforms."""

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.predefined.shopify_product_catalogue import (
    ShopifyProductCatalogue,
)
from inference_endpoint.dataset_manager.predefined.shopify_product_catalogue.presets import (
    ShopifyMultimodalFormatter,
    q3vl,
)


def _make_mock_hf_row(
    *,
    product_title: str = "Test Product",
    product_description: str = "A test product",
    product_image_bytes: bytes = b"\xff\xd8\xff",
    product_image_path: str = "image.jpg",
    potential_product_categories: list[str] | None = None,
    ground_truth_category: str = "Clothing > Shirts",
    ground_truth_brand: str = "Test Brand",
    ground_truth_is_secondhand: bool = False,
) -> dict:
    """Create a mock row matching HuggingFace product-catalogue format."""
    if potential_product_categories is None:
        potential_product_categories = ["Clothing > Shirts", "Clothing > Tops"]
    return {
        "product_title": product_title,
        "product_description": product_description,
        "product_image": {"bytes": product_image_bytes, "path": product_image_path},
        "potential_product_categories": potential_product_categories,
        "ground_truth_category": ground_truth_category,
        "ground_truth_brand": ground_truth_brand,
        "ground_truth_is_secondhand": ground_truth_is_secondhand,
    }


class TestShopifyProductCatalogueGenerate:
    """Tests for ShopifyProductCatalogue.generate()."""

    @pytest.fixture
    def mock_hf_dataset(self) -> list[dict]:
        """Synthetic HuggingFace-style dataset for mocking load_from_huggingface.

        Returns a list of sample dicts (indexable like HF Dataset via ds[i]).
        """
        return [
            _make_mock_hf_row(
                product_title="Shirt A",
                product_description="Blue cotton shirt",
                ground_truth_category="Clothing > Shirts > Polo",
            ),
            _make_mock_hf_row(
                product_title="Shirt B",
                product_description="Red silk shirt",
                product_image_path="image.png",
                ground_truth_category="Clothing > Shirts > Dress",
            ),
        ]

    def test_generate_produces_expected_columns(
        self, tmp_path: Path, mock_hf_dataset: list[dict]
    ) -> None:
        """Generate produces DataFrame with expected column names."""
        with patch(
            "inference_endpoint.dataset_manager.predefined.shopify_product_catalogue.load_from_huggingface",
            return_value=mock_hf_dataset,
        ):
            df = ShopifyProductCatalogue.generate(
                datasets_dir=tmp_path,
                split=["train"],
                force=True,
            )
        assert list(df.columns) == ShopifyProductCatalogue.COLUMN_NAMES
        assert len(df) == 2

    def test_generate_converts_image_to_base64(
        self, tmp_path: Path, mock_hf_dataset: list[dict]
    ) -> None:
        """Images are base64-encoded in output."""
        with patch(
            "inference_endpoint.dataset_manager.predefined.shopify_product_catalogue.load_from_huggingface",
            return_value=mock_hf_dataset,
        ):
            df = ShopifyProductCatalogue.generate(
                datasets_dir=tmp_path,
                split=["train"],
                force=True,
            )
        expected_b64 = base64.b64encode(b"\xff\xd8\xff").decode("utf-8")
        assert df["product_image_base64"].iloc[0] == expected_b64
        assert df["product_image_format"].iloc[0] == "JPEG"
        assert df["product_image_format"].iloc[1] == "PNG"

    def test_generate_json_serializes_categories_and_secondhand(
        self, tmp_path: Path, mock_hf_dataset: list[dict]
    ) -> None:
        """potential_product_categories and ground_truth_is_secondhand are JSON strings."""
        with patch(
            "inference_endpoint.dataset_manager.predefined.shopify_product_catalogue.load_from_huggingface",
            return_value=mock_hf_dataset,
        ):
            df = ShopifyProductCatalogue.generate(
                datasets_dir=tmp_path,
                split=["train"],
                force=True,
            )
        cats = json.loads(df["potential_product_categories"].iloc[0])
        assert cats == ["Clothing > Shirts", "Clothing > Tops"]
        secondhand = json.loads(df["ground_truth_is_secondhand"].iloc[0])
        assert secondhand is False

    def test_generate_uses_cache_when_exists_and_not_force(
        self, tmp_path: Path, mock_hf_dataset: list[dict]
    ) -> None:
        """When file exists and force=False, loads from cache without calling HF."""
        dst_dir = tmp_path / "shopify_product_catalogue" / "train"
        dst_dir.mkdir(parents=True)
        cached_df = pd.DataFrame(
            {
                "product_title": ["Cached"],
                "product_description": [""],
                "product_image_base64": ["YQ=="],
                "product_image_format": ["JPEG"],
                "potential_product_categories": ["[]"],
                "ground_truth_category": [""],
                "ground_truth_brand": [""],
                "ground_truth_is_secondhand": ["false"],
            }
        )
        cached_df.to_parquet(dst_dir / "shopify_product_catalogue_train.parquet")

        with patch(
            "inference_endpoint.dataset_manager.predefined.shopify_product_catalogue.load_from_huggingface",
            return_value=mock_hf_dataset,
        ) as mock_load:
            df = ShopifyProductCatalogue.generate(
                datasets_dir=tmp_path,
                split=["train"],
                force=False,
            )
        mock_load.assert_not_called()
        assert df["product_title"].iloc[0] == "Cached"

    def test_generate_passes_token_and_revision(
        self, tmp_path: Path, mock_hf_dataset: list[dict]
    ) -> None:
        """Token and revision are passed to load_from_huggingface."""
        with patch(
            "inference_endpoint.dataset_manager.predefined.shopify_product_catalogue.load_from_huggingface",
            return_value=mock_hf_dataset,
        ) as mock_load:
            ShopifyProductCatalogue.generate(
                datasets_dir=tmp_path,
                split=["train"],
                force=True,
                token="hf_xxx",
                revision="v1",
            )
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["load_options"]["token"] == "hf_xxx"
        assert call_kwargs["load_options"]["revision"] == "v1"

    def test_generate_raises_when_product_image_missing(self, tmp_path: Path) -> None:
        """Raises ValueError when product_image is missing from a row."""
        bad_dataset = [
            {
                "product_title": "No Image",
                "product_description": "",
                "product_image": None,
                "potential_product_categories": [],
                "ground_truth_category": "",
                "ground_truth_brand": "",
                "ground_truth_is_secondhand": False,
            }
        ]
        with patch(
            "inference_endpoint.dataset_manager.predefined.shopify_product_catalogue.load_from_huggingface",
            return_value=bad_dataset,
        ):
            with pytest.raises(ValueError, match="product_image is missing"):
                ShopifyProductCatalogue.generate(
                    datasets_dir=tmp_path,
                    split=["train"],
                    force=True,
                )


class TestShopifyMultimodalFormatter:
    """Tests for ShopifyMultimodalFormatter transform."""

    def _make_product_row(
        self,
        *,
        product_title: str = "Test Shirt",
        product_description: str = "Blue cotton",
        product_image_base64: str = "YQ==",
        product_image_format: str = "JPEG",
        potential_product_categories: str = '["Clothing > Shirts"]',
    ) -> dict:
        return {
            "product_title": product_title,
            "product_description": product_description,
            "product_image_base64": product_image_base64,
            "product_image_format": product_image_format,
            "potential_product_categories": potential_product_categories,
        }

    def test_transform_adds_system_and_prompt(self) -> None:
        """Transform adds system and prompt columns."""
        df = pd.DataFrame([self._make_product_row()])
        formatter = ShopifyMultimodalFormatter()
        result = formatter(df)
        assert "system" in result.columns
        assert "prompt" in result.columns

    def test_transform_system_contains_schema_and_instructions(self) -> None:
        """System message includes JSON schema and instructions."""
        df = pd.DataFrame([self._make_product_row()])
        formatter = ShopifyMultimodalFormatter()
        result = formatter(df)
        system = result["system"].iloc[0]
        assert "category" in system
        assert "brand" in system
        assert "is_secondhand" in system
        assert "JSON" in system

    def test_transform_prompt_contains_title_description_categories(self) -> None:
        """Prompt includes product title, description, and categories."""
        df = pd.DataFrame(
            [
                self._make_product_row(
                    product_title="Red Polo",
                    product_description="Classic fit",
                    potential_product_categories='["Clothing > Shirts > Polo"]',
                )
            ]
        )
        formatter = ShopifyMultimodalFormatter()
        result = formatter(df)
        prompt = result["prompt"].iloc[0]
        assert isinstance(prompt, list)
        text_part = next(p for p in prompt if p["type"] == "text")
        assert "Red Polo" in text_part["text"]
        assert "Classic fit" in text_part["text"]
        assert "Clothing > Shirts > Polo" in text_part["text"]

    def test_transform_includes_image_url_when_base64_present(self) -> None:
        """Prompt includes image_url part when product_image_base64 is non-empty."""
        df = pd.DataFrame([self._make_product_row(product_image_base64="YQ==")])
        formatter = ShopifyMultimodalFormatter()
        result = formatter(df)
        prompt = result["prompt"].iloc[0]
        image_part = next((p for p in prompt if p["type"] == "image_url"), None)
        assert image_part is not None
        assert "data:image/JPEG;base64,YQ==" in image_part["image_url"]["url"]

    def test_transform_omits_image_when_base64_empty(self) -> None:
        """Prompt has no image_url part when product_image_base64 is empty."""
        df = pd.DataFrame([self._make_product_row(product_image_base64="")])
        formatter = ShopifyMultimodalFormatter()
        result = formatter(df)
        prompt = result["prompt"].iloc[0]
        image_parts = [p for p in prompt if p["type"] == "image_url"]
        assert len(image_parts) == 0


class TestShopifyQ3vlPreset:
    """Tests for q3vl preset and get_dataloader integration."""

    def test_q3vl_preset_returns_list_of_transforms(self) -> None:
        """q3vl() returns a list of Transform instances."""
        transforms = q3vl()
        assert isinstance(transforms, list)
        assert len(transforms) >= 1
        assert all(callable(t) for t in transforms)

    def test_get_dataloader_with_q3vl_preset(self, tmp_path: Path) -> None:
        """get_dataloader with q3vl preset produces loadable dataset."""
        mock_df = pd.DataFrame(
            [
                {
                    "product_title": "T1",
                    "product_description": "D1",
                    "product_image_base64": "YQ==",
                    "product_image_format": "JPEG",
                    "potential_product_categories": "[]",
                    "ground_truth_category": "A > B",
                    "ground_truth_brand": "Brand",
                    "ground_truth_is_secondhand": "false",
                }
            ]
        )
        with patch.object(
            ShopifyProductCatalogue,
            "generate",
            return_value=mock_df,
        ):
            loader = ShopifyProductCatalogue.get_dataloader(
                datasets_dir=tmp_path,
                transforms=q3vl(),
                num_repeats=1,
                force_regenerate=True,
            )
        loader.load()
        assert loader.num_samples() == 1
        sample = loader.load_sample(0)
        assert "system" in sample
        assert "prompt" in sample
        assert isinstance(sample["prompt"], list)
