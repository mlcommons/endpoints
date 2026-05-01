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

"""Extract MMLU-Pro and LiveCodeBench subsets from the combined MLPerf accuracy parquet.

The combined parquet (mlperf_deepseek_r1_dataset_4388_fp8_eval_accuracy.parquet) contains
5 sub-datasets identified by the 'dataset' column. Pre-split files already exist for Math
and GPQA; this script extracts the remaining two:

  datasets/deepseek/mlperf_deepseek_r1_mmlu_pro_accuracy.parquet   (2410 rows)
  datasets/deepseek/mlperf_deepseek_r1_livecodebench_accuracy.parquet (349 rows)

Usage:
    uv run python examples/09_DeepSeek-V4-Pro_Example/extract_mlperf_subsets.py
"""

from pathlib import Path

import pandas as pd

SRC = Path("datasets/deepseek/mlperf_deepseek_r1_dataset_4388_fp8_eval_accuracy.parquet")
OUT_DIR = Path("datasets/deepseek")

SUBSETS = {
    "mmlu_pro": "mlperf_deepseek_r1_mmlu_pro_accuracy.parquet",
    "livecodebench": "mlperf_deepseek_r1_livecodebench_accuracy.parquet",
}


def main() -> None:
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df)} rows from {SRC}")
    print("Sub-dataset breakdown:")
    print(df.groupby(["dataset", "metric"]).size().to_string())
    print()

    for dataset_name, out_filename in SUBSETS.items():
        subset = df[df["dataset"] == dataset_name].reset_index(drop=True)
        out_path = OUT_DIR / out_filename
        subset.to_parquet(out_path, index=False)
        print(f"Wrote {len(subset)} rows → {out_path}")


if __name__ == "__main__":
    main()
