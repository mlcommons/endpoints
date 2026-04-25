#!/usr/bin/env python3
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
Prepare test dataset for Qwen benchmark.

This script creates a test dataset with the 'prompt' column required by
the inference-endpoint benchmarking tool.
"""

import pickle
import sys
from pathlib import Path


def prepare_dataset(
    input_path: str = "tests/datasets/dummy_1k.pkl",
    output_dir: str = "examples/08_Qwen2.5-0.5B_Example/data",
    output_filename: str = "test_dataset.pkl",
) -> None:
    """
    Prepare the test dataset by renaming columns to match expected format.

    Args:
        input_path: Path to the input dataset
        output_dir: Directory to save the output dataset
        output_filename: Name of the output file
    """
    print(f"Loading dataset from: {input_path}")

    # Load the original dataset
    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input dataset not found at {input_path}")
        print("Make sure you're running from the repository root directory")
        sys.exit(1)

    print(f"Loaded dataset with {len(data)} samples")
    print(f"Original columns: {data.columns.tolist()}")

    # Rename text_input to prompt
    if "text_input" in data.columns:
        data = data.rename(columns={"text_input": "prompt"})
        print("Renamed 'text_input' to 'prompt'")
    elif "prompt" not in data.columns:
        print("ERROR: Dataset must have 'text_input' or 'prompt' column")
        sys.exit(1)

    print(f"Final columns: {data.columns.tolist()}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    full_output_path = output_path / output_filename
    with open(full_output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Dataset saved to: {full_output_path}")
    print(f"   Samples: {len(data)}")
    print(f"   Columns: {data.columns.tolist()}")


if __name__ == "__main__":
    # Allow custom input path as command-line argument
    input_path = sys.argv[1] if len(sys.argv) > 1 else "tests/datasets/dummy_1k.pkl"
    prepare_dataset(input_path=input_path)
