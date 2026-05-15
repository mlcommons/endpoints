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

"""Standalone CLI that runs VBench.evaluate on a staged video directory.

Invoked as a subprocess by `inference_endpoint.evaluation.scoring.VBenchScorer`
so the parent benchmark environment never has to import vbench (which pins
incompatible transformers/numpy versions). Writes VBench's own results JSON
to `--out-dir/{run_name}_eval_results.json`.
"""

import argparse
import sys

import torch
from vbench import VBench


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VBench on staged videos.")
    parser.add_argument("--videos-dir", required=True, help="Directory of mp4s.")
    parser.add_argument("--out-dir", required=True, help="VBench output directory.")
    parser.add_argument(
        "--name",
        required=True,
        help="Run name; results land at <name>_eval_results.json.",
    )
    parser.add_argument(
        "--dims",
        required=True,
        help="Comma-separated VBench dimension names.",
    )
    parser.add_argument(
        "--full-info-json",
        default=None,
        help="Optional path to vbench_full_info.json; uses VBench default if omitted.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimension_list = [d.strip() for d in args.dims.split(",") if d.strip()]
    vb = VBench(device, args.full_info_json, args.out_dir)
    vb.evaluate(
        videos_path=args.videos_dir,
        name=args.name,
        dimension_list=dimension_list,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
