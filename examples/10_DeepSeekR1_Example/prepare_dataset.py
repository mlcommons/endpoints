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

"""Convert the MLPerf DeepSeek-R1 eval pickle into a benchmark-ready parquet.

The MLCommons dataset ships as a pandas pickle the endpoint dataset manager
cannot read (it loads JSONL/JSON/CSV/Parquet/HF). This emits a parquet with
exactly the columns the benchmark + accuracy scorer need:

  - input_tokens : the pre-tokenized MLPerf prompt (source column `tok_input`).
                   Present as `input_tokens` so the openai_completions adapter's
                   Harmonize() transform is a no-op and the server's chat
                   template is bypassed - the model sees the exact MLPerf prompt.
  - ground_truth : the expected answer (LCB rows: the LiveCodeBench question id).
  - dataset      : the subset id (math500 / aime* / gpqa / mmlu_pro / livecodebench)
                   used by DeepSeekR1Scorer to route per-subset grading.
  - question     : the human-readable question text.

Run inside an env with a working pandas/pyarrow (e.g. the trtllm container or
the accuracy subproject's `.venv`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# The MLPerf DeepSeek-R1 source .pkl is not bundled (it is the official
# MLCommons eval dataset). Point --source at your copy, or set the env var.
DEFAULT_SOURCE = os.environ.get("DEEPSEEK_R1_DATASET_PKL")

# Source columns required to build the benchmark parquet.
_REQUIRED = ("tok_input", "ground_truth", "dataset", "question")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        required=DEFAULT_SOURCE is None,
        help="Path to the MLPerf DeepSeek-R1 source .pkl (or set $DEEPSEEK_R1_DATASET_PKL)",
    )
    ap.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parent / "data" / "deepseek_r1_eval.parquet"
        ),
        help="Destination .parquet",
    )
    ap.add_argument(
        "--subset",
        type=int,
        default=0,
        help="If >0, also write a stratified subset of this many rows "
        "(proportional per `dataset` subset) for a quick representative run.",
    )
    args = ap.parse_args()

    df = pd.read_pickle(args.source)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Source pkl missing expected columns {missing}; found {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "input_tokens": df["tok_input"].map(
                lambda t: t.tolist() if hasattr(t, "tolist") else list(t)
            ),
            "ground_truth": df["ground_truth"].astype(str),
            "dataset": df["dataset"].astype(str),
            "question": df["question"].astype(str),
        }
    )

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"Wrote {len(out)} rows to {dest}")
    print("Subset counts:")
    print(out["dataset"].value_counts().to_string())

    # A 4-row performance parquet so --mode acc's mandatory perf phase is cheap.
    perf_tiny = dest.with_name("deepseek_r1_perf_tiny.parquet")
    out.head(4).to_parquet(perf_tiny, index=False)
    print(f"Wrote 4 rows to {perf_tiny}")

    if args.subset > 0:
        frac = args.subset / len(out)
        parts = [
            g.sample(n=min(max(1, round(len(g) * frac)), len(g)), random_state=42)
            for _, g in out.groupby("dataset")
        ]
        sub = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
        sub_dest = dest.with_name("deepseek_r1_eval_subset.parquet")
        sub.to_parquet(sub_dest, index=False)
        print(f"Wrote {len(sub)} stratified rows to {sub_dest}")
        print(sub["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()
