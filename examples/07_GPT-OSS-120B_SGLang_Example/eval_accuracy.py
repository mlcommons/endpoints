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


import argparse
from pathlib import Path

from inference_endpoint.dataset_manager.predefined.gpqa import GPQA
from inference_endpoint.evaluation.extractor import ABCDExtractor
from inference_endpoint.evaluation.scoring import PassAt1Scorer


def main(args):
    # Load the dataset
    ds = GPQA.load_from_file(args.dataset_path)
    ds.load()

    # Create the scorer
    scorer = PassAt1Scorer(
        GPQA.DATASET_ID,
        ds,
        args.report_dir,
        extractor=ABCDExtractor,
    )

    # Score the dataset
    score, n_repeats = scorer.score()
    print(f"Pass@1 Score ({n_repeats} repeats): {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy of the SGLang endpoint on the GPQA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to the dataset",
        default="datasets/gpqa/diamond/gpqa_diamond.parquet",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Path to the report directory",
        default="gpqa_sglang_report",
    )
    args = parser.parse_args()
    main(args)
