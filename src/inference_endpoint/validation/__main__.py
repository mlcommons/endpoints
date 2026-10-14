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

"""The user-facing CLI: `python -m inference_endpoint.validation COHORT SUBMISSION`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from inference_endpoint.validation.catalog import Catalog
from inference_endpoint.validation.check import Cohort
from inference_endpoint.validation.runner import Finding
from inference_endpoint.validation.submission import Submission


def main(argv: Sequence[str] | None = None) -> int:
    """Run one cohort's checks against one submission; the process exit code."""
    args = _parser().parse_args(argv)
    try:
        cohort = Cohort.from_directory(args.cohort)
        submission = Submission.from_directory(
            args.submission,
            ruleset=Catalog.from_directory(args.ruleset) if args.ruleset else None,
        )
        findings = Finding.collect(cohort, submission)
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    failing = [finding for finding in findings if not finding.verdict.ok]
    for finding in findings if args.audit else failing:
        mark = "ok" if finding.verdict.ok else "FAIL"
        print(
            f"{mark} {finding.check_id} [{finding.severity.value}] "
            f"{finding.node_directory}: {finding.verdict.why}"
        )
    print(f"{len(findings)} finding(s), {len(failing)} failing")
    return 1 if failing else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference_endpoint.validation",
        description="Run one cohort's checks against one submission.",
    )
    parser.add_argument(
        "cohort", type=Path, help="cohort directory (its name names the cohort)"
    )
    parser.add_argument(
        "submission",
        type=Path,
        help="submission root (system_desc.json + results/<system>/<model>/r<N>/)",
    )
    parser.add_argument(
        "--ruleset",
        type=Path,
        default=None,
        help="ruleset directory (models.yaml + datasets.yaml): the `ruleset` fold",
    )
    parser.add_argument(
        "--audit", action="store_true", help="print held findings alongside failures"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
