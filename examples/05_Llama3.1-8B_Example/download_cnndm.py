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

import json
import os
import warnings
from argparse import ArgumentParser

from datasets import load_dataset
from tqdm import tqdm

PROMPT = "Summarize the following news article in 128 tokens. Please output the summary only, without any other text.\n\nArticle:\n{input}\n\nSummary:"


def download_cnndm(
    save_dir: str, split: str = "validation", calibration_ids_file: str = None
) -> None:
    """Download the CNN/DailyMail dataset and save it to the specified directory.

    Args:
        save_dir (str): The directory where the dataset will be saved.
        split (str): The dataset split to download (default: validation).
        calibration_ids_file (str): Path to a file containing calibration IDs (one per line).
            If provided, 'split' must be 'train' and only examples with these IDs will be prepared and saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)

    output_file_tag = split
    calibration_ids = set()
    if calibration_ids_file:
        with open(calibration_ids_file, encoding="utf-8") as id_file:
            for line in id_file:
                calibration_ids.add(line.strip())
        output_file_tag = "calibration"
    fname = f"cnn_dailymail_{output_file_tag}.json"
    output_file = os.path.join(save_dir, fname)

    # Add the custom prompt to each example and filter if calibration IDs are provided
    with open(output_file, "w", encoding="utf-8") as f:
        for _, example in tqdm(
            enumerate(dataset), total=len(dataset), desc="Processing examples"
        ):
            if calibration_ids and str(example["id"]) not in calibration_ids:
                continue
            f.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "input": PROMPT.format(input=example["article"]),
                        "highlights": example["highlights"],
                    }
                )
                + "\n"
            )
    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Download CNN/DailyMail dataset")
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the downloaded dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to download (default: validation)",
    )
    parser.add_argument(
        "--calibration-ids-file",
        type=str,
        default=None,
        help="Path to a file containing calibration IDs (one per line)."
        " If provided, 'split' must be 'train' and only examples with these IDs will be saved.",
    )

    args = parser.parse_args()
    if args.calibration_ids_file and args.split != "train":
        warnings.warn(
            "When --calibration-ids-file is provided, --split must be 'train'. Setting split to 'train'.",
            stacklevel=2,
        )
        args.split = "train"

    if args.calibration_ids_file and not os.path.isfile(args.calibration_ids_file):
        raise FileNotFoundError(
            f"Provided calibration IDs file not found: {args.calibration_ids_file}"
        )

    download_cnndm(
        save_dir=args.save_dir,
        split=args.split,
        calibration_ids_file=args.calibration_ids_file,
    )
