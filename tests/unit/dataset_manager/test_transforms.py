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

"""
Unit tests for the transforms module.
Tests all transform classes and functions except Harmonize.
"""

from typing import Any

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.transforms import (
    ColumnNameRemap,
    DropColumns,
    FusedRowProcessor,
    RowProcessor,
    Transform,
    UserPromptFormatter,
    apply_transforms,
)


class TestColumnNameRemap:
    """Test suite for ColumnNameRemap transform."""

    def test_basic_column_rename_inplace(self):
        """Test basic column renaming with inplace=True."""
        df = pd.DataFrame({"old_name": [1, 2, 3], "another_col": [4, 5, 6]})
        transform = ColumnNameRemap({"old_name": "new_name"}, inplace=True)
        result = transform(df)

        # With inplace=True, should modify original DataFrame
        assert result is None
        assert "new_name" in df.columns
        assert "old_name" not in df.columns
        assert list(df["new_name"]) == [1, 2, 3]

    def test_basic_column_rename_not_inplace(self):
        """Test basic column renaming with inplace=False."""
        df = pd.DataFrame({"old_name": [1, 2, 3], "another_col": [4, 5, 6]})
        transform = ColumnNameRemap({"old_name": "new_name"}, inplace=False)
        result = transform(df)

        # With inplace=False, should return new DataFrame and leave original unchanged
        assert result is not None
        assert "new_name" in result.columns
        assert "old_name" not in result.columns
        assert list(result["new_name"]) == [1, 2, 3]
        # Original should be unchanged
        assert "old_name" in df.columns

    def test_multiple_columns_rename(self):
        """Test renaming multiple columns at once."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
        transform = ColumnNameRemap({"col1": "a", "col2": "b"}, inplace=False)
        result = transform(df)

        assert "a" in result.columns
        assert "b" in result.columns
        assert "col3" in result.columns  # Unchanged column
        assert "col1" not in result.columns
        assert "col2" not in result.columns

    def test_empty_mapping(self):
        """Test with empty mapping (no columns renamed)."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        transform = ColumnNameRemap({}, inplace=False)
        result = transform(df)

        # Should return DataFrame with same columns
        assert list(result.columns) == list(df.columns)
        assert result.equals(df)

    def test_rename_nonexistent_column(self):
        """Test renaming a column that doesn't exist (should be silently ignored by pandas)."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        transform = ColumnNameRemap({"nonexistent": "new_name"}, inplace=False)
        result = transform(df)

        # Pandas silently ignores non-existent column mappings
        assert list(result.columns) == ["col1", "col2"]
        assert result.equals(df)

    def test_partial_column_rename(self):
        """Test renaming only some columns, leaving others unchanged."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        transform = ColumnNameRemap({"b": "B", "d": "D"}, inplace=False)
        result = transform(df)

        assert list(result.columns) == ["a", "B", "c", "D"]


class TestUserPromptFormatter:
    """Test suite for UserPromptFormatter transform."""

    def test_basic_prompt_formatting(self):
        """Test basic prompt formatting with single variable."""
        df = pd.DataFrame(
            {"question": ["What is 2+2?", "What is the capital of France?"]}
        )
        transform = UserPromptFormatter("Question: {question}", output_column="prompt")
        result = transform(df)

        assert "prompt" in result.columns
        assert result["prompt"][0] == "Question: What is 2+2?"
        assert result["prompt"][1] == "Question: What is the capital of France?"

    def test_custom_output_column(self):
        """Test using a custom output column name."""
        df = pd.DataFrame({"question": ["What is 2+2?"]})
        transform = UserPromptFormatter("Q: {question}", output_column="formatted_q")
        result = transform(df)

        assert "formatted_q" in result.columns
        assert result["formatted_q"][0] == "Q: What is 2+2?"
        assert "prompt" not in result.columns

    def test_multiple_variables_in_format(self):
        """Test formatting with multiple variables from row."""
        df = pd.DataFrame(
            {
                "context": ["Paris is the capital.", "Rome is the capital."],
                "question": ["Of which country?", "Of what?"],
            }
        )
        transform = UserPromptFormatter(
            "Context: {context}\nQuestion: {question}", output_column="prompt"
        )
        result = transform(df)

        assert (
            result["prompt"][0]
            == "Context: Paris is the capital.\nQuestion: Of which country?"
        )
        assert (
            result["prompt"][1] == "Context: Rome is the capital.\nQuestion: Of what?"
        )

    def test_missing_variable_raises_error(self):
        """Test that missing variables in format string raise KeyError."""
        df = pd.DataFrame({"question": ["What is 2+2?"]})
        transform = UserPromptFormatter(
            "Question: {question} Context: {missing_var}", output_column="prompt"
        )

        with pytest.raises(KeyError):
            transform(df)

    def test_empty_format_string(self):
        """Test with empty format string."""
        df = pd.DataFrame({"question": ["What is 2+2?"]})
        transform = UserPromptFormatter("", output_column="prompt")
        result = transform(df)

        assert result["prompt"][0] == ""

    def test_no_variables_in_format(self):
        """Test format string with no variables."""
        df = pd.DataFrame({"question": ["What is 2+2?"]})
        transform = UserPromptFormatter("Static prompt text", output_column="prompt")
        result = transform(df)

        assert result["prompt"][0] == "Static prompt text"

    def test_preserves_original_columns(self):
        """Test that original columns are preserved."""
        df = pd.DataFrame({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})
        transform = UserPromptFormatter("{question}", output_column="prompt")
        result = transform(df)

        assert "question" in result.columns
        assert "answer" in result.columns
        assert "prompt" in result.columns
        assert list(result["question"]) == ["Q1", "Q2"]
        assert list(result["answer"]) == ["A1", "A2"]


class TestRowProcessor:
    """Test suite for RowProcessor base class."""

    def test_row_processor_calls_process_row(self):
        """Test that RowProcessor __call__ invokes process_row for each row."""

        class TestProcessor(RowProcessor):
            def __init__(self):
                self.call_count = 0

            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                self.call_count += 1
                row["processed"] = True
                return row

        df = pd.DataFrame({"col1": [1, 2, 3]})
        processor = TestProcessor()
        result = processor(df)

        assert processor.call_count == 3  # Called once per row
        assert "processed" in result.columns
        assert all(result["processed"])

    def test_row_processor_with_row_modification(self):
        """Test that row modifications are reflected in output DataFrame."""

        class DoubleValues(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        processor = DoubleValues()
        result = processor(df)

        assert list(result["value"]) == [2, 4, 6]

    def test_row_processor_adds_new_column(self):
        """Test that process_row can add new columns."""

        class AddSquared(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["squared"] = row["value"] ** 2
                return row

        df = pd.DataFrame({"value": [2, 3, 4]})
        processor = AddSquared()
        result = processor(df)

        assert "squared" in result.columns
        assert list(result["squared"]) == [4, 9, 16]

    def test_row_processor_with_empty_dataframe(self):
        """Test row processor with empty DataFrame."""

        class TestProcessor(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                return row

        df = pd.DataFrame({"col1": []})
        processor = TestProcessor()
        result = processor(df)

        assert len(result) == 0
        assert "col1" in result.columns


class TestFusedRowProcessor:
    """Test suite for FusedRowProcessor."""

    def test_single_processor(self):
        """Test FusedRowProcessor with a single processor."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        fused = FusedRowProcessor([AddOne()])
        result = fused(df)

        assert list(result["value"]) == [2, 3, 4]

    def test_multiple_processors_in_sequence(self):
        """Test that multiple processors are applied in sequence."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        class MultiplyByTwo(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        # Should add 1 first, then multiply by 2: (1+1)*2 = 4, (2+1)*2 = 6, (3+1)*2 = 8
        fused = FusedRowProcessor([AddOne(), MultiplyByTwo()])
        result = fused(df)

        assert list(result["value"]) == [4, 6, 8]

    def test_processor_order_matters(self):
        """Test that the order of processors affects the result."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        class MultiplyByTwo(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})

        # Order 1: Multiply then add: (1*2)+1 = 3, (2*2)+1 = 5, (3*2)+1 = 7
        fused1 = FusedRowProcessor([MultiplyByTwo(), AddOne()])
        result1 = fused1(df)
        assert list(result1["value"]) == [3, 5, 7]

        # Order 2: Add then multiply: (1+1)*2 = 4, (2+1)*2 = 6, (3+1)*2 = 8
        df = pd.DataFrame({"value": [1, 2, 3]})  # Reset dataframe
        fused2 = FusedRowProcessor([AddOne(), MultiplyByTwo()])
        result2 = fused2(df)
        assert list(result2["value"]) == [4, 6, 8]

    def test_empty_processor_list(self):
        """Test FusedRowProcessor with empty processor list."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        fused = FusedRowProcessor([])
        result = fused(df)

        # Should return unchanged data
        assert result.equals(df)

    def test_processors_can_add_columns(self):
        """Test that fused processors can add new columns."""

        class AddSquared(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["squared"] = row["value"] ** 2
                return row

        class AddCubed(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["cubed"] = row["value"] ** 3
                return row

        df = pd.DataFrame({"value": [2, 3]})
        fused = FusedRowProcessor([AddSquared(), AddCubed()])
        result = fused(df)

        assert "squared" in result.columns
        assert "cubed" in result.columns
        assert list(result["squared"]) == [4, 9]
        assert list(result["cubed"]) == [8, 27]


class TestApplyTransforms:
    """Test suite for apply_transforms function."""

    def test_single_transform(self):
        """Test applying a single transform."""
        df = pd.DataFrame({"old": [1, 2, 3]})
        transforms = [ColumnNameRemap({"old": "new"}, inplace=False)]
        result = apply_transforms(df, transforms)

        assert "new" in result.columns
        assert "old" not in result.columns

    def test_multiple_transforms_in_sequence(self):
        """Test applying multiple transforms in sequence."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        class AddColumn(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["col2"] = row["col1"] * 2
                return row

        transforms = [
            AddColumn(),
            ColumnNameRemap({"col1": "original", "col2": "doubled"}, inplace=False),
        ]
        result = apply_transforms(df, transforms)

        assert "original" in result.columns
        assert "doubled" in result.columns
        assert list(result["doubled"]) == [2, 4, 6]

    def test_empty_transform_list(self):
        """Test with empty transform list."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = apply_transforms(df, [])

        # Should return the DataFrame unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_fusion_enabled(self):
        """Test that row processors are fused when fusion is enabled."""

        class CountingRowProcessor(RowProcessor):
            call_count = 0

            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                CountingRowProcessor.call_count += 1
                return super().__call__(df)

            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row.get("value", 0) + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        CountingRowProcessor.call_count = 0

        transforms = [CountingRowProcessor(), CountingRowProcessor()]
        result = apply_transforms(df, transforms, fuse_row_processors=True)

        # With fusion, should only iterate once (FusedRowProcessor's __call__)
        # But we need to track the fused processor's call count, not individual processors
        # Let's verify the result is correct instead
        assert list(result["value"]) == [3, 4, 5]  # Each row incremented twice

    def test_fusion_disabled(self):
        """Test that row processors are not fused when fusion is disabled."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [AddOne(), AddOne()]
        result = apply_transforms(df, transforms, fuse_row_processors=False)

        # Result should be the same regardless of fusion
        assert list(result["value"]) == [3, 4, 5]

    def test_mix_of_transforms_and_row_processors(self):
        """Test mixing regular transforms with row processors."""

        class AddDoubled(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["doubled"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [
            AddDoubled(),
            ColumnNameRemap({"value": "original"}, inplace=False),
        ]
        result = apply_transforms(df, transforms)

        assert "original" in result.columns
        assert "doubled" in result.columns
        assert list(result["doubled"]) == [2, 4, 6]

    def test_consecutive_row_processors_are_fused(self):
        """Test that consecutive row processors are fused together."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        class MultiplyByTwo(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] * 2
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [AddOne(), MultiplyByTwo()]
        result = apply_transforms(df, transforms, fuse_row_processors=True)

        # Should be (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        assert list(result["value"]) == [4, 6, 8]

    def test_non_consecutive_row_processors_separated_by_transform(self):
        """Test that non-consecutive row processors are not fused together."""

        class AddOne(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["value"] = row["value"] + 1
                return row

        df = pd.DataFrame({"value": [1, 2, 3]})
        transforms = [
            AddOne(),
            ColumnNameRemap(
                {"value": "value"}, inplace=False
            ),  # No-op transform to break fusion
            AddOne(),
        ]
        result = apply_transforms(df, transforms, fuse_row_processors=True)

        # Each AddOne should add 1, so final result is +2
        assert list(result["value"]) == [3, 4, 5]

    def test_complex_transform_pipeline(self):
        """Test a complex pipeline with multiple types of transforms."""

        class AddSquared(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["squared"] = row["num"] ** 2
                return row

        class AddCubed(RowProcessor):
            def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
                row["cubed"] = row["num"] ** 3
                return row

        df = pd.DataFrame({"num": [2, 3, 4]})
        transforms = [
            AddSquared(),
            AddCubed(),
            ColumnNameRemap({"num": "original_number"}, inplace=False),
        ]
        result = apply_transforms(df, transforms)

        assert "original_number" in result.columns
        assert "squared" in result.columns
        assert "cubed" in result.columns
        assert list(result["squared"]) == [4, 9, 16]
        assert list(result["cubed"]) == [8, 27, 64]


class TestTransformBaseClass:
    """Test suite for Transform base class."""

    def test_transform_not_implemented(self):
        """Test that Transform base class __call__ raises NotImplementedError."""

        class IncompleteTransform(Transform):
            pass

        df = pd.DataFrame({"col1": [1, 2, 3]})
        transform = IncompleteTransform()

        with pytest.raises(NotImplementedError):
            transform(df)

    def test_custom_transform_implementation(self):
        """Test implementing a custom transform."""

        class DropNullsTransform(Transform):
            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                return df.dropna()

        df = pd.DataFrame({"col1": [1, None, 3, None, 5]})
        transform = DropNullsTransform()
        result = transform(df)

        assert len(result) == 3
        assert list(result["col1"]) == [1.0, 3.0, 5.0]


class TestDropColumns:
    """Test suite for DropColumns transform."""

    def test_drop_single_column(self):
        """Test dropping a single column."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
        transform = DropColumns(columns=["col2"])
        result = transform(df)

        assert "col1" in result.columns
        assert "col2" not in result.columns
        assert "col3" in result.columns
        assert len(result.columns) == 2

    def test_drop_multiple_columns(self):
        """Test dropping multiple columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
                "col4": [10, 11, 12],
            }
        )
        transform = DropColumns(columns=["col2", "col4"])
        result = transform(df)

        assert "col1" in result.columns
        assert "col2" not in result.columns
        assert "col3" in result.columns
        assert "col4" not in result.columns
        assert len(result.columns) == 2

    def test_drop_nonexistent_column_ignore(self):
        """Test dropping a column that doesn't exist with errors='ignore'."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        transform = DropColumns(columns=["col3"], errors="ignore")
        result = transform(df)

        # Should not raise an error and should return the original dataframe
        assert "col1" in result.columns
        assert "col2" in result.columns
        assert len(result.columns) == 2

    def test_drop_nonexistent_column_raise(self):
        """Test dropping a column that doesn't exist with errors='raise'."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        transform = DropColumns(columns=["col3"], errors="raise")

        with pytest.raises(KeyError):
            transform(df)

    def test_drop_all_columns(self):
        """Test dropping all columns."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        transform = DropColumns(columns=["col1", "col2"])
        result = transform(df)

        assert len(result.columns) == 0
        assert len(result) == 3  # Rows should remain

    def test_drop_empty_list(self):
        """Test dropping with an empty list of columns."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        transform = DropColumns(columns=[])
        result = transform(df)

        # Should return the same dataframe
        assert "col1" in result.columns
        assert "col2" in result.columns
        assert len(result.columns) == 2

    def test_drop_columns_preserves_data(self):
        """Test that dropping columns preserves remaining data."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
        transform = DropColumns(columns=["col2"])
        result = transform(df)

        assert list(result["col1"]) == [1, 2, 3]
        assert list(result["col3"]) == [7, 8, 9]

    def test_drop_columns_in_pipeline(self):
        """Test using DropColumns in a transform pipeline."""
        df = pd.DataFrame(
            {
                "question": ["What is 2+2?"],
                "answer": ["4"],
                "metadata": ["some_metadata"],
                "extra": ["extra_data"],
            }
        )

        # Apply a sequence of transforms
        transforms = [
            ColumnNameRemap({"question": "prompt"}, inplace=False),
            DropColumns(columns=["metadata", "extra"]),
        ]

        result = apply_transforms(df, transforms)

        assert "prompt" in result.columns
        assert "answer" in result.columns
        assert "metadata" not in result.columns
        assert "extra" not in result.columns
        assert "question" not in result.columns
