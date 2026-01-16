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

import argparse
import base64
import json
import logging
import pickle
import random
import zlib
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def generate_dataset(
    datasets_dir: Path,
    variant: str = "release_v6",
    seed: int = 0,
    max_samples: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Generates the LiveCodeBench dataset.

    Args:
        datasets_dir: Path to the directory where the dataset will be stored.
        variant: The variant of the dataset to generate.
        seed: The seed to use for the random number generator.
        max_samples: The maximum number of samples to generate.
        force: Whether to force the generation of the dataset.
    """

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory {datasets_dir} does not exist")

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

    # Unpack the private test cases. In most cases, the private test cases are stored as either a
    # JSON string, or a Base64-encoded zlib-compressed pickle'd string, which needs to be JSON decoded.
    # Reference: https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/benchmarks/code_generation.py#L64-L74
    def deserialize_private_test(private_test_cases: str) -> str:
        try:
            # Check if it is already JSON - if it is, do nothing.
            _ = json.loads(private_test_cases)
            return private_test_cases
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

            # Check if is valid JSON
            _ = json.loads(unpickled_obj)
            return unpickled_obj

    df["private_test_cases"] = df["private_test_cases"].apply(deserialize_private_test)

    # The only important metadata field for evaluation is the 'func_name' field
    def get_func_name(metadata_json: str) -> str:
        try:
            metadata = json.loads(metadata_json)
            return metadata.get("func_name", "")
        except json.JSONDecodeError:
            return ""

    df["func_name"] = df["metadata"].apply(get_func_name)
    df = df.drop(columns=["metadata"])

    # If max_samples is specified, sample 'max_samples' rows from the dataset
    if max_samples is not None and max_samples < len(df):
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(len(df)), max_samples)
        df = df.iloc[sampled_indices].reset_index(drop=True)
        logger.info(f"Sampled {max_samples} questions")

    # Save to parquet file
    dst_path = datasets_dir / f"livecodebench_{variant}.parquet"
    df.to_parquet(dst_path)
    logger.info(f"Saved {len(df)} samples to {dst_path}")

    return df


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
    args = parser.parse_args()

    generate_dataset(
        datasets_dir=args.datasets_dir,
        variant=args.variant,
        seed=args.seed,
        max_samples=args.max_samples,
        force=args.force,
    )
