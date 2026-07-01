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

"""Render standardized result plots for benchmark run report directories.

Generates accuracy (overall/normalized vs gate, per-category, per-subset) and
performance (inline-IoU per-turn, turn completion, TTFT/latency distributions)
PNGs into REPORT_DIR/plots (or --out-dir).

Usage:
    python scripts/plot_results.py REPORT_DIR [REPORT_DIR ...]
    python scripts/plot_results.py results/edge_agentic_full_run \\
        --out-dir /tmp/edge_plots
"""

import argparse
import sys
from pathlib import Path

from inference_endpoint.metrics.results_plots import generate_plots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_dir",
        nargs="+",
        type=Path,
        help="Run report directory (contains results.json / scores.json / "
        "result_summary.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: REPORT_DIR/plots).",
    )
    parser.add_argument(
        "--ruleset",
        default="mlperf-edge-current",
        help="Registered ruleset name (for the accuracy gate overlay).",
    )
    parser.add_argument(
        "--model",
        default="qwen3.6-27b",
        help="Model name within the ruleset.",
    )
    args = parser.parse_args()

    any_written = False
    multiple = len(args.report_dir) > 1
    for report_dir in args.report_dir:
        # Plot filenames are fixed (accuracy.png, perf_turns.png, ...). With a
        # single shared --out-dir and multiple report dirs they would clobber
        # each other, so give each report its own subdirectory. When --out-dir
        # is unset, generate_plots defaults to REPORT_DIR/plots (already unique).
        out_dir = args.out_dir
        if out_dir is not None and multiple:
            out_dir = out_dir / report_dir.name
        written = generate_plots(report_dir, out_dir, args.ruleset, args.model)
        if written:
            any_written = True
            print(f"{report_dir}: wrote {len(written)} plot(s)")
            for p in written:
                print(f"  {p}")
        else:
            print(
                f"{report_dir}: no plots written (no artifacts or matplotlib missing)"
            )

    return 0 if any_written else 1


if __name__ == "__main__":
    sys.exit(main())
