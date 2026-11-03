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

"""Check a benchmark run's report directory against a ruleset.

Validates config-lock (deterministic/single-stream settings), the accuracy gate,
and run-validity rules. Exits 0 if all checks pass, 1 otherwise.

Usage:
    python scripts/check_compliance.py REPORT_DIR [REPORT_DIR ...]
    python scripts/check_compliance.py results/edge_agentic_full_run \\
        --ruleset mlperf-edge-current --model qwen3.6-27b
"""

import argparse
import sys
from pathlib import Path

from inference_endpoint.compliance import check_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_dir",
        nargs="+",
        type=Path,
        help="Run report directory (contains config.yaml + scorer output).",
    )
    parser.add_argument(
        "--ruleset",
        default="mlperf-edge-current",
        help="Registered ruleset name (default: mlperf-edge-current).",
    )
    parser.add_argument(
        "--model",
        default="qwen3.6-27b",
        help="Model name within the ruleset (default: qwen3.6-27b).",
    )
    args = parser.parse_args()

    all_passed = True
    for report_dir in args.report_dir:
        report = check_submission(report_dir, args.ruleset, args.model)
        print(report.render())
        print()
        all_passed = all_passed and report.passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
