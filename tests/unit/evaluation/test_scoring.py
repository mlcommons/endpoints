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

"""Unit tests for evaluation scoring module."""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from inference_endpoint.evaluation.scoring import (
    _PRED_CATEGORY_PAD,
    ProductMetadata,
    Scorer,
    ShopifyCategoryF1Scorer,
    _calculate_hierarchical_f1,
    _create_pred_pad_category,
    _get_hierarchical_components,
    _parse_response_to_category,
)
from pydantic import ValidationError


class TestGetHierarchicalComponents:
    """Tests for _get_hierarchical_components."""

    def test_exact_match(self):
        pred = "Clothing > Shirts > Polo"
        true = "Clothing > Shirts > Polo"
        inter, p_len, t_len = _get_hierarchical_components(pred, true)
        assert inter == 3
        assert p_len == 3
        assert t_len == 3

    def test_partial_match_prefix(self):
        pred = "Clothing > Shirts"
        true = "Clothing > Shirts > Polo"
        inter, p_len, t_len = _get_hierarchical_components(pred, true)
        assert inter == 2
        assert p_len == 2
        assert t_len == 3

    def test_mismatch_at_first_level(self):
        pred = "Electronics > Phones"
        true = "Clothing > Shirts > Polo"
        inter, p_len, t_len = _get_hierarchical_components(pred, true)
        assert inter == 0
        assert p_len == 2
        assert t_len == 3


class TestCalculateHierarchicalF1:
    """Tests for _calculate_hierarchical_f1."""

    def test_perfect_score(self):
        data = [
            ("A > B > C", "A > B > C"),
            ("X > Y", "X > Y"),
        ]
        assert _calculate_hierarchical_f1(data) == 1.0

    def test_zero_score(self):
        data = [
            ("A > B", "X > Y"),
            ("P > Q", "M > N"),
        ]
        assert _calculate_hierarchical_f1(data) == 0.0

    def test_partial_score(self):
        # 2/3 match on first, 1/2 on second
        # Sample 1: inter=2, pred=3, true=3
        # Sample 2: inter=1, pred=2, true=2
        # total_inter=3, total_pred=5, total_true=5
        # hp=3/5, hr=3/5, f1=3/5
        data = [
            ("A > B > X", "A > B > C"),  # inter=2
            ("A > X", "A > B"),  # inter=1
        ]
        f1 = _calculate_hierarchical_f1(data)
        assert abs(f1 - 0.6) < 1e-6

    def test_empty_data(self):
        assert _calculate_hierarchical_f1([]) == 0.0


class TestCreatePredPadCategory:
    """Tests for _create_pred_pad_category."""

    def test_matching_depth(self):
        gt = "A > B > C"
        result = _create_pred_pad_category(gt, " > ")
        assert (
            result
            == f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"
        )

    def test_single_level(self):
        gt = "Electronics"
        result = _create_pred_pad_category(gt, " > ")
        assert result == _PRED_CATEGORY_PAD


class TestParseResponseToCategory:
    """Tests for _parse_response_to_category."""

    def test_valid_json(self):
        out = json.dumps(
            {"category": "Clothing > Shirts", "brand": "Nike", "is_secondhand": False}
        )
        assert (
            _parse_response_to_category(out, "Clothing > Shirts") == "Clothing > Shirts"
        )

    def test_json_with_extra_whitespace(self):
        out = '  { "category" : " A > B ", "brand": "x", "is_secondhand": false }  '
        assert _parse_response_to_category(out, "A > B") == "A > B"

    def test_raw_json_only(self):
        """Reference implementation expects raw JSON; no markdown stripping."""
        out = '{"category": "X > Y", "brand": "x", "is_secondhand": false}'
        assert _parse_response_to_category(out, "X > Y") == "X > Y"

    def test_invalid_json_uses_pred_pad(self):
        gt = "A > B > C"
        result = _parse_response_to_category("not json", gt)
        expected = f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"
        assert result == expected

    def test_missing_category_uses_pred_pad(self):
        out = '{"brand": "Nike", "is_secondhand": false}'
        result = _parse_response_to_category(out, "Electronics > Phones")
        expected = f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"
        assert result == expected

    def test_invalid_schema_uses_pred_pad(self):
        out = '{"category": "x", "brand": "y"}'  # missing required is_secondhand
        result = _parse_response_to_category(out, "A > B")
        assert result == f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"

    def test_markdown_wrapped_uses_pred_pad(self):
        """Reference passes raw string; markdown-wrapped JSON fails validation."""
        out = (
            '```json\n{"category": "X > Y", "brand": "x", "is_secondhand": false}\n```'
        )
        result = _parse_response_to_category(out, "A > B")
        assert result == f"{_PRED_CATEGORY_PAD} > {_PRED_CATEGORY_PAD}"


class TestProductMetadata:
    """Tests for ProductMetadata model."""

    def test_valid_parse(self):
        data = '{"category": "A > B", "brand": "Nike", "is_secondhand": false}'
        parsed = ProductMetadata.model_validate_json(data)
        assert parsed.category == "A > B"
        assert parsed.brand == "Nike"
        assert parsed.is_secondhand is False

    def test_missing_field_raises(self):
        data = '{"category": "A", "brand": "x"}'
        with pytest.raises(ValidationError):
            ProductMetadata.model_validate_json(data)


class TestShopifyCategoryF1ScorerRegistration:
    """Test that ShopifyCategoryF1Scorer is registered."""

    def test_scorer_registered(self):
        assert "shopify_category_f1" in Scorer.available_scorers()
        scorer_cls = Scorer.get("shopify_category_f1")
        assert scorer_cls is ShopifyCategoryF1Scorer


class TestShopifyCategoryF1Scorer:
    """Integration tests for ShopifyCategoryF1Scorer."""

    @pytest.fixture
    def mock_dataset(self):
        df = pd.DataFrame(
            {
                "ground_truth_category": [
                    "Clothing > Shirts > Polo",
                    "Electronics > Phones",
                ],
            }
        )
        dataset = MagicMock()
        dataset.dataframe = df
        dataset.num_samples.return_value = 2
        return dataset

    @pytest.fixture
    def report_dir(self, tmp_path):
        return tmp_path

    def test_score_requires_sample_index_map_and_events(self, mock_dataset, report_dir):
        """Scorer raises when sample_idx_map.json missing (checked in __init__)."""
        with pytest.raises(FileNotFoundError, match="Sample index map"):
            ShopifyCategoryF1Scorer(
                dataset_name="test",
                dataset=mock_dataset,
                report_dir=report_dir,
            )
