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
from inference_endpoint.validation.runner import CheckRun, Finding
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


def _never_ran_cause(run: CheckRun) -> str:
    """Why a check ran at zero nodes, in the report's words.

    Args:
        run (CheckRun): The check's run record.
    Returns:
        str: The cause clause: the `when` gate, the `needs` chain, or both.
    """
    if run.silenced and not run.gated_off:
        return "no node was applicable (its `needs` chain silenced every instance)"
    if run.gated_off and not run.silenced:
        return "no node was applicable (its `when` gated every instance off)"
    return (
        f"no node was applicable ({run.gated_off} gated off, {run.silenced} silenced)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one cohort's checks against one submission.

    Returns:
        int: `0` = every applicable check held, or failed only at `warn` severity (a
            warning is reported and passes through). `1` = at least one failing `fail`
            finding. `2` = the cohort or the submission was refused before any check ran
            (a bad check file, an unwalkable tree, an unreadable artifact, a
            number no `float` can carry).
    """
    parser = argparse.ArgumentParser(
        prog="python -m inference_endpoint.validation",
        description="Run one cohort's checks against one submission.",
    )
    parser.add_argument(
        "cohort",
        type=Path,
        help="cohort directory (its `cohort.yaml` manifest names its ruleset)",
    )
    parser.add_argument(
        "submission",
        type=Path,
        help="submission root (results/<system>/<model>/r<N>/)",
    )
    parser.add_argument(
        "--audit", action="store_true", help="print held findings alongside failures"
    )
    args = parser.parse_args(argv)
    try:
        cohort = Cohort.from_directory(args.cohort)
        catalog: Catalog | None = None
        if cohort.ruleset_dir is not None:
            catalog = Catalog.from_directory(cohort.ruleset_dir)
        elif cohort.ruleset:
            catalog = Catalog.from_ruleset(cohort.ruleset)
        submission = Submission.from_directory(args.submission, ruleset=catalog)
        collected = Finding.collect(cohort, submission)
    except (OSError, ValueError, OverflowError) as exc:
        print(f"refused: {_escaped(str(exc))}", file=sys.stderr)
        return 2
    for note in collected.coercions:
        # How a numeric string was read: a shape note, not a finding.
        print(_escaped(f"coerced: {note}"), file=sys.stderr)
    failing = [finding for finding in collected.findings if not finding.verdict.ok]
    blocking = [finding for finding in failing if finding.severity is Severity.FAIL]
    for finding in collected.findings if args.audit else failing:
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
    for check in cohort.checks:
        run = collected.runs[check.id]
        if run.ran_at:
            print(f"{check.id} ran at {run.ran_at} instance(s)")
        else:
            # A check that never ran is indistinguishable from one that passed
            # unless the report says so, and the cause is the check's own
            # record: `when` gated it off, a `needs` chain silenced it, or both.
            print(
                _escaped(f"NEVER-RAN {check.id}: {_never_ran_cause(run)}"),
                file=sys.stderr,
            )
    print(
        f"{len(collected.findings)} finding(s), {len(failing)} failing "
        f"({len(blocking)} blocking)"
    )
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
