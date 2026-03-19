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

"""This script will generate the LiveCodeBench dataset. Because LiveCodeBench
has different dependencies (i.e. datasets==3.6.0), and evaluation needs to be
containerized, we need to generate the dataset from within the container.

To minimize the number of dependencies, this script is detached from the main
Inference Endpoints codebase, and instead is used as a standalone script.

It can be invoked by running it as a module, but it is recommended to use the
LiveCodeBench-Serve workflow to generate the dataset and perform evaluation.
"""

# This file is under the evaluation submodule rather than the dataset_manager submodule
# since this requires being run in the lcb-service container.
# The .generate() method in the dataset_manager instead uses the /get_dataset route in the
# lcb-service container to retrieve the dataset.

import argparse
import base64
import json
import logging
import pickle
import random
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

from datasets import load_dataset

logger = logging.getLogger(__name__)


SCRIPT_PATH = Path(__file__)


def deserialize_private_test(private_test_cases: str) -> list[dict[str, Any]]:
    """Unpack the private test cases. In most cases, the private test cases are stored as either a
    JSON string, or a Base64-encoded zlib-compressed pickle'd string, which needs to be JSON decoded.
    Reference: https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/benchmarks/code_generation.py#L64-L74

    If any error is raised, the dataset is probably malformed, and should be a fatal error.
    The error type indicates the step in the deserialization process that failed.

    Args:
        private_test_cases: The encoded string of private test cases to unpack.

    Returns:
        The unpacked private test cases.
    """
    try:
        # Check if it is already JSON - if it is, do nothing.
        deserialized_private_test_cases = json.loads(private_test_cases)
        return deserialized_private_test_cases
    except json.JSONDecodeError as json_error:
        # Otherwise, it is a Base64-encoded zlib-compressed pickle'd string.
        # If any steps fail, the dataset is probably malformed, and should be a fatal error.

        # Decode the Base64 string
        decoded_bytes = base64.b64decode(private_test_cases.encode("utf-8"))

        # Uncompress the bytes
        uncompressed_bytes = zlib.decompress(decoded_bytes)

        # Unpickle the bytes
        unpickled_obj = pickle.loads(uncompressed_bytes)

        if not isinstance(unpickled_obj, str):
            raise ValueError(
                "Invalid private_test_cases format. Is the dataset malformed?"
            ) from json_error

        # Check if is valid JSON and decode
        return json.loads(unpickled_obj)


def generate_dataset(
    datasets_dir: Path,
    variant: str = "release_v6",
    seed: int = 0,
    max_samples: int | None = None,
    force: bool = False,
    save_test_cases: bool = True,
) -> pd.DataFrame:
    """Generates the LiveCodeBench dataset.

    Args:
        datasets_dir: Path to the directory where the dataset will be stored.
        variant: The variant of the dataset to generate.
        seed: The seed to use for the random number generator.
        max_samples: The maximum number of samples to generate.
        force: Whether to force the generation of the dataset.
        save_test_cases: Whether to save test cases as separate JSON files.
            If True, test cases are saved to datasets_dir/test_cases/<question_id>.json
    """

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory {datasets_dir} does not exist")

    dst_path = datasets_dir / f"livecodebench_{variant}.parquet"
    if dst_path.exists() and not force:
        logger.info(f"Dataset already exists at {dst_path}. Loading from file.")
        return pd.read_parquet(dst_path)

    df = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag=variant,
        trust_remote_code=True,
    )["test"].to_pandas()

    df = df.rename(columns={"question_content": "question"})

    # We do not care about 'competition start date' since we evaluate on the entire dataset anyway.
    # LiveCodeBench is a constantly evolving dataset, but will be pinned by version, so as long as
    # we are consistent with the version_tag being used, we should be good.
    df = df.drop(
        columns=[
            "question_title",
            "contest_date",
            "contest_id",
            "platform",
            "difficulty",
        ]
    )

    # If max_samples is specified, sample 'max_samples' rows from the dataset
    if max_samples is not None and max_samples < len(df):
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(len(df)), max_samples)
        df = df.iloc[sampled_indices].reset_index(drop=True)
        logger.info(f"Sampled {max_samples} questions")

    # Process and save test cases if requested
    if save_test_cases:
        # Save test cases to separate JSON files
        test_cases_dir = datasets_dir / "test_cases"
        test_cases_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            question_id = row["question_id"]
            test_case_json_path = test_cases_dir / f"{question_id}.json"

            public_cases = json.loads(row["public_test_cases"])
            private_cases = deserialize_private_test(row["private_test_cases"])
            func_name = None

            try:
                metadata = json.loads(row["metadata"])
                func_name = metadata.get("func_name", None)
            except json.JSONDecodeError:
                logger.info(f"Failed to load metadata for question {question_id}.")

            test_case_data = {
                "public_test_cases": public_cases,
                "private_test_cases": private_cases,
                "func_name": func_name,
            }

            with test_case_json_path.open("w", encoding="utf-8") as f:
                json.dump(test_case_data, f, indent=2)

        logger.info(f"Saved test cases to {test_cases_dir}")

    # Drop test case columns from the dataframe before saving to parquet
    df_to_save = df.drop(
        columns=["public_test_cases", "private_test_cases", "metadata"]
    )

    # Save to parquet file with pyarrow engine for better performance
    df_to_save.to_parquet(dst_path, engine="pyarrow")
    logger.info(f"Saved {len(df_to_save)} samples to {dst_path}")

    return df_to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the LiveCodeBench dataset")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        required=True,
        help="Path to the directory where the dataset will be stored",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="release_v6",
        help="The variant of the dataset to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for the random number generator",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="The maximum number of samples to generate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to force the generation of the dataset",
    )
    parser.add_argument(
        "--no-test-cases",
        action="store_true",
        default=False,
        help="Do not save test cases as separate JSON files",
    )
    args = parser.parse_args()

    generate_dataset(
        datasets_dir=args.datasets_dir,
        variant=args.variant,
        seed=args.seed,
        max_samples=args.max_samples,
        force=args.force,
        save_test_cases=not args.no_test_cases,
    )
