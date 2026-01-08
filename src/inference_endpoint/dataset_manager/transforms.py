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


from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..openai.harmony import Harmonizer


class Transform(ABC):
    """Base class for transforms. Transforms are single parameter functions that are applied to either each row of
    a dataframe, or to the entire dataframe.

    These can be chained together in a pipeline to perform more complex transformations.
    """

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transform to a pandas DataFrame.

        Args:
            df: Input DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        raise NotImplementedError("Subclasses must implement this method.")


class RowProcessor(Transform):
    """Base class for processing rows of a dataframe.

    This is a special Transform subclass that loops through each row in a dataframe
    and applies the process_row method to each row.
    """

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process each row of a dataframe.

        Args:
            df: Input DataFrame to process

        Returns:
            DataFrame with processed rows
        """
        return df.apply(self.process_row, axis=1, result_type="expand")

    @abstractmethod
    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of a dataframe.

        Args:
            row: A dictionary representing a single row from the dataframe

        Returns:
            Processed row as a dictionary
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ColumnNameRemap(Transform):
    """Transform that renames columns in a DataFrame.

    This transform takes a mapping dictionary where keys are the current column names
    and values are the new column names.
    """

    def __init__(self, column_mapping: dict[str, str], inplace: bool = False):
        """Initialize the ColumnNameRemap transform.

        Args:
            column_mapping: Dictionary mapping old column names to new column names
            inplace: If True, the DataFrame is modified in place. Otherwise, a new DataFrame is returned.
        """
        self.column_mapping = column_mapping
        self.inplace = inplace

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns in the DataFrame according to the mapping.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with renamed columns
        """
        retval = df.rename(columns=self.column_mapping, inplace=self.inplace)
        if self.inplace:
            return df
        else:
            return retval


class UserPromptFormatter(RowProcessor):
    """Transform that formats user prompts from DataFrame rows.

    This transform takes a format string and applies it to each row using the row's
    values as keyword arguments. The result is stored in a new column.
    """

    def __init__(self, user_prompt_format: str, output_column: str = "prompt"):
        """Initialize the UserPromptFormatter transform.

        Args:
            user_prompt_format: Format string to apply to each row (using .format(**row))
            output_column: Name of the column to store the formatted prompt (default: "prompt")
        """
        self.user_prompt_format = user_prompt_format
        self.output_column = output_column

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Format the prompt for a single row.

        Args:
            row: Dictionary representing a single row from the dataframe

        Returns:
            Row dictionary with the formatted prompt added
        """
        # Format the prompt using the row values as kwargs
        formatted_prompt = self.user_prompt_format.format(**row)
        # Add the formatted prompt to the row
        row[self.output_column] = formatted_prompt
        return row


class AddStaticColumns(RowProcessor):
    """Transform that adds columns with constant values to a DataFrame."""

    def __init__(self, data: dict[str, Any]):
        """Initialize the AddStaticColumns transform."""
        self.data = data

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Add the static columns to the row."""
        # row is technically a Series object, so we can't directly use the | operator
        for key, value in self.data.items():
            row[key] = value
        return row


class Harmonize(RowProcessor):
    """Transform to convert a user prompt to an OpenAI Harmony-compatible format."""

    def __init__(
        self,
        tokenizer_name: str = "openai/gpt-oss-120b",
        encoding_name: str = "HARMONY_GPT_OSS",
        reasoning_effort: str = "high",
        conversation_start_date: str | None = None,
        prompt_column: str = "prompt",
        tokenized_column: str = "input_tokens",
        harmonized_column: str | None = "harmonized_prompt",
    ):
        """Initialize the Harmonize transform.

        Args:
            tokenizer_name: The name of the tokenizer to use for the dataset.
            encoding_name: The name of the HarmonyEncoding enum member to use.
            reasoning_effort: The reasoning effort to use for the dataset.
            conversation_start_date: The start date of the conversation.
            prompt_column: The name of the column containing the user prompt.
            tokenized_column: The name of the column containing the tokenized prompt.
            harmonized_column: The name of the column containing the harmonized prompt. If None,
                the harmonized prompt will not be stored as text.
        """
        self.prompt_column = prompt_column
        self.tokenized_column = tokenized_column
        self.harmonized_column = harmonized_column
        self.harmonizer = Harmonizer(
            tokenizer_name=tokenizer_name,
            encoding_name=encoding_name,
            reasoning_effort=reasoning_effort,
            conversation_start_date=conversation_start_date,
        )

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Harmonize the user prompt for a single row.

        Args:
            row: Dictionary representing a single row from the dataframe

        Returns:
            Row dictionary with the harmonized prompt added
        """
        row[self.tokenized_column] = self.harmonizer(row[self.prompt_column])
        if self.harmonized_column is not None:
            row[self.harmonized_column] = self.harmonizer.to_text(
                row[self.tokenized_column]
            )
        return row


class DropColumns(Transform):
    """Transform that drops specified columns from a DataFrame.

    This transform removes columns from a DataFrame by name.
    """

    def __init__(self, columns: list[str], errors: str = "ignore"):
        """Initialize the DropColumns transform.

        Args:
            columns: List of column names to drop from the DataFrame
            errors: How to handle errors. Options: 'raise' or 'ignore' (default: 'ignore')
        """
        self.columns = columns
        self.errors = errors

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with specified columns removed
        """
        return df.drop(columns=self.columns, errors=self.errors)


class FusedRowProcessor(RowProcessor):
    """Row processor that fuses consecutive row processors into a single row processor."""

    def __init__(self, row_processors: list[RowProcessor]):
        """Initialize the FusedRowProcessor."""
        self.row_processors = row_processors

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        for processor in self.row_processors:
            row = processor.process_row(row)
        return row


def _create_fused_transform(row_processors: list[RowProcessor]) -> Transform:
    """Create a fused transform from a list of row processors.

    Args:
        row_processors: Non-empty list of row processors to fuse

    Returns:
        A single Transform (either the original processor if only one, or a FusedRowProcessor
        if multiple)
    """
    if len(row_processors) == 1:
        return row_processors[0]
    else:
        return FusedRowProcessor(row_processors)


def apply_transforms(
    df: pd.DataFrame,
    transforms: list[Transform],
    fuse_row_processors: bool = True,
) -> pd.DataFrame:
    """Apply a list of transforms to a dataframe.

    Args:
        df: Input DataFrame to transform
        transforms: List of transforms to apply
        fuse_row_processors: If True, consecutive row processors will be fused into a single row
            processor to prevent unnecessary iterations over the dataframe. (Default: True)

    Returns:
        Transformed DataFrame
    """
    if fuse_row_processors:
        new_transforms = []
        fused_transforms = []

        for transform in transforms:
            if isinstance(transform, RowProcessor):
                fused_transforms.append(transform)
            else:
                # Flush any accumulated row processors before adding non-row-processor transform
                if fused_transforms:
                    new_transforms.append(_create_fused_transform(fused_transforms))
                    fused_transforms = []
                new_transforms.append(transform)

        # Flush any remaining row processors at the end
        if fused_transforms:
            new_transforms.append(_create_fused_transform(fused_transforms))

        transforms = new_transforms

    for transform in transforms:
        df = transform(df)
    return df
