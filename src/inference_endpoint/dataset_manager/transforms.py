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

from ..config.schema import APIType, ModelParams
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


class AddStaticColumns(Transform):
    """Transform that adds columns with constant values to a DataFrame."""

    def __init__(self, data: dict[str, Any]):
        """Initialize the AddStaticColumns transform."""
        self.data = data

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the static columns to the row."""
        for key, value in self.data.items():
            df[key] = value
        return df


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

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transform, skipping if the target column already exists."""
        if self.tokenized_column in df.columns:
            return df
        return super().__call__(df)

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


class ColumnFilter(Transform):
    """Transform that filters columns from a DataFrame as an allow-list. Only the specified columns
    will be kept in the DataFrame.
    """

    def __init__(
        self,
        required_columns: list[str],
        optional_columns: list[str] | None = None,
    ):
        """Initialize the ColumnFilter transform.

        Args:
            required_columns: List of column names to keep in the DataFrame
            optional_columns: List of column names to keep in the DataFrame if present
        """
        self.required_columns = required_columns
        self.optional_columns = optional_columns

        # Check that required and optional columns are mutually exclusive
        if optional_columns is not None and (
            set(required_columns) & set(optional_columns)
        ):
            raise ValueError("Required and optional columns must be mutually exclusive")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter columns from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with filtered columns
        """
        columns_to_keep = self.required_columns
        if self.optional_columns is not None:
            found_cols = set(df.columns) & set(self.optional_columns)
            columns_to_keep += list(found_cols)

        # Filter the columns
        df = df[columns_to_keep]
        return df


class ColumnRemap(Transform):
    """Remaps columns in a DataFrame. This Transform is has an added feature on top of the
    normal dataframe.rename() method in that rather than remapping an old column name to a new
    column name, a list of candidate column names can be provided.
    This transform will iterate through the candidate column names and use the first one found
    as the column to rename. As an example:

    ColumnRemap(
        remap={
            "abc": "def",
            ("123", "456", "789"): "numbers",
        },
        strict=False,
    )

    when applied to a dataframe with the columns ["789", "456", "abc"] will result in a new
    dataframe with the columns ["789", "numbers", "def"], since "456" is the first column in the
    remap key found in the original column list.

    If `strict` is True, an error will be raised in the above example, since both "456" and "789"
    exist in the original column list.
    """

    def __init__(
        self,
        remap: dict[str | tuple[str, ...], str],
        strict: bool = True,
    ):
        self.remap = remap
        self.strict = strict

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remap the columns in the DataFrame.

        Args:
            df: Input DataFrame
        """
        new_cols = {}
        for src, dst in self.remap.items():
            if isinstance(src, str):
                new_cols[src] = dst
            elif isinstance(src, tuple):
                old_cols = set(df.columns)
                found = None
                for candidate in src:
                    if candidate in old_cols:
                        if found is None:
                            new_cols[candidate] = dst
                            found = candidate
                        elif self.strict:
                            raise ValueError(
                                f"Multiple columns found for fuzzy remap: {found} and {candidate}"
                            )
        df = df.rename(columns=new_cols, errors="ignore")
        return df


class MakeAdapterCompatible(ColumnRemap):
    """Special transform for arbitrary load_from_file() datasets which may have arbitrary
    structure.

    When using an arbitrary Dataset.load_from_file() dataframe, it is expected that the user
    prompt will be stored in a column and is ready to be used for inference.

    This transform will search for through a set of common column names and rename the column
    to 'prompt', which is the expected column name for adapter transforms.

    If no column is found, an error will be raised.
    """

    def __init__(self):
        super().__init__(
            remap={
                (
                    "user_prompt",
                    "question",
                    "input",
                    "input_text",
                    "problem",
                    "query",
                ): "prompt",
                "system_prompt": "system",
            },
            strict=True,
        )


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


def get_transforms_for_api_type(
    api_type: APIType, model_params: ModelParams
) -> list[Transform]:
    """Utility function to get the transforms required for a given API type.

    Args:
        api_type: The API type to get the transforms for

    Returns:
        A list of transforms required for the given API type
    """
    from importlib import import_module

    from inference_endpoint.endpoint_client.config import ADAPTER_MAP

    adapter_path = ADAPTER_MAP.get(api_type)
    if not adapter_path:
        raise ValueError(f"Invalid or unsupported API type: {api_type}")

    module_path, class_name = adapter_path.rsplit(".", 1)
    module = import_module(module_path)
    adapter = getattr(module, class_name)
    return adapter.dataset_transforms(model_params)
