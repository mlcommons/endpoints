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

Exit codes:
    0  — VBench evaluation completed; results JSON written.
    1  — VBench raised during evaluate(); structured error on stderr.
    2  — CUDA required but unavailable (pass --allow-cpu to override).
"""

import argparse
import json
import sys
import traceback
from importlib.resources import files as _pkg_files

import torch
import vbench as _vbench_pkg
from vbench import VBench


def _emit_error(exc: BaseException) -> None:
    """Print a structured JSON error line on stderr for the parent to surface."""
    payload = {
        "status": "error",
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    print(json.dumps(payload), file=sys.stderr, flush=True)


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
        help=(
            "Optional path to a VBench_full_info.json file. When omitted, "
            "falls back to the JSON bundled with the installed vbench "
            "package — required for vbench_standard mode (which the "
            "MLPerf WAN 2.2 prompt set, a subset of VBench's standard "
            "suite, runs under)."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Permit CPU fallback when CUDA is unavailable. VBench's per-dim "
            "models (CLIP/DINO/RAFT/AMT) effectively need a GPU; CPU runs "
            "take hours and may OOM the host. Off by default to avoid a "
            "silent foot-gun."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        print(
            json.dumps(
                {
                    "status": "error",
                    "type": "CudaUnavailable",
                    "message": (
                        "CUDA is not available; VBench requires a GPU. Pass "
                        "--allow-cpu to override (not recommended for "
                        "production accuracy runs)."
                    ),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimension_list = [d.strip() for d in args.dims.split(",") if d.strip()]
    # VBench's `full_info_dir` arg is actually a JSON *file* path (despite
    # the name). vbench_standard mode (used here) requires a valid file;
    # passing None crashes inside VBench's load_json().
    full_info_json = args.full_info_json or str(
        _pkg_files(_vbench_pkg).joinpath("VBench_full_info.json")
    )
    try:
        vb = VBench(device, full_info_json, args.out_dir)
        vb.evaluate(
            videos_path=args.videos_dir,
            name=args.name,
            dimension_list=dimension_list,
        )
    except Exception as e:
        _emit_error(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
