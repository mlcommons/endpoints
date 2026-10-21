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
from inference_endpoint.validation.check import Cohort, Severity
from inference_endpoint.validation.runner import Finding
from inference_endpoint.validation.submission import Submission


def _escaped(text: str) -> str:
    """`text` with every control character made visible.

    Upload-controlled names reach the report (`refused:` messages and every
    line's `node_directory`); a newline or an escape sequence inside one must
    not forge or hide a report line.

    Args:
        text (str): The composed message about to be printed.
    Returns:
        str: `text` with non-printable characters escaped: a newline becomes
            the two characters `\\n`.
    """
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode()
        for char in text
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one cohort's checks against one submission.

    Returns:
        int: `0` = every applicable check held, or failed only at `warn` severity (a
            warning is reported and passes through). `1` = at least one failing `fail`
            finding. `2` = the cohort or the submission was refused before any check ran
            (a bad check file, an unwalkable tree, an unreadable artifact).
    """
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
        help="submission root (results/<system>/<model>/r<N>/)",
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
    args = parser.parse_args(argv)
    try:
        cohort = Cohort.from_directory(args.cohort)
        submission = Submission.from_directory(
            args.submission,
            ruleset=Catalog.from_directory(args.ruleset) if args.ruleset else None,
        )
        findings = Finding.collect(cohort, submission)
    except (OSError, ValueError) as exc:
        print(f"refused: {_escaped(str(exc))}", file=sys.stderr)
        return 2
    failing = [finding for finding in findings if not finding.verdict.ok]
    blocking = [finding for finding in failing if finding.severity is Severity.FAIL]
    for finding in findings if args.audit else failing:
        if finding.verdict.ok:
            mark = "OK"
        else:
            mark = "FAIL" if finding.severity is Severity.FAIL else "WARN"
        line = (
            f"{mark} {finding.check_id} [{finding.severity.value}] "
            f"{finding.node_directory}: {finding.row}"
        )
        if finding.verdict.why:
            line += f" -> {finding.verdict.why}"
        if not finding.verdict.ok:
            line += f"  [{finding.rule}]"
        print(_escaped(line))
        if not finding.verdict.ok and finding.fix:
            # The declared remedy rides its failing line: one fix sentence,
            # only where a row failed.
            print(_escaped(f"    fix: {finding.fix}"))
    instances = {check.id: 0 for check in cohort.checks}
    seen: set[tuple[str, Path]] = set()
    for finding in findings:
        key = (finding.check_id, finding.node_directory)
        if key not in seen:
            seen.add(key)
            instances[finding.check_id] += 1
    for check_id, count in instances.items():
        if count:
            print(f"{check_id} ran at {count} instance(s)")
        else:
            # A check that never ran is indistinguishable from one that passed
            # unless the report says so: its `when` gated every node off.
            print(
                _escaped(
                    f"NEVER-RAN {check_id}: no node was applicable "
                    "(its `when` gated every instance off)"
                ),
                file=sys.stderr,
            )
    print(
        f"{len(findings)} finding(s), {len(failing)} failing ({len(blocking)} blocking)"
    )
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
