# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the selected cohort policy against classified submission evidence."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from .artifacts import load_artifacts
from .checks.power import prepare
from .evaluator_base import Evaluator
from .models import Policy
from .outcomes import result
from .planner import plan_checks
from .results import Report
from .types import Decision


def execute_submission(
    path: PathLike[str],
    policy: Policy,
) -> Report:
    """Parse evidence once, plan conditions/overrides, then execute each ready rule.

    Unavailable prerequisites and unsupported evaluators produce blocking errors.
    Parsing does not add compliance findings outside the selected policy.
    """
    artifacts = load_artifacts(Path(path), policy)
    prepare(artifacts)
    plan = plan_checks(policy, artifacts.index.subjects)
    findings = []
    for check in plan.checks:
        if check.decision is Decision.EXCLUDED:
            continue
        if check.decision is Decision.BLOCKED:
            missing = ", ".join(sorted(check.missing))
            findings.append(
                result(
                    check,
                    False,
                    check.reason + (f": {missing}" if missing else ""),
                    blocked=True,
                )
            )
            continue
        evaluator = Evaluator.lookup(check.rule.kind)
        if evaluator is None:
            findings.append(
                result(
                    check,
                    False,
                    f"Unsupported evaluator kind: {check.rule.kind}",
                    blocked=True,
                )
            )
            continue
        try:
            instance = evaluator()
            findings.extend(instance(check, artifacts))
        except (
            ValueError,
            TypeError,
            KeyError,
            ArithmeticError,
            AttributeError,
        ) as exc:
            # Malformed evidence must never turn an unevaluated policy into success.
            findings.append(
                result(
                    check, False, f"Evaluation could not complete: {exc}", blocked=True
                )
            )
    return Report(submission_path=artifacts.root, results=findings)
