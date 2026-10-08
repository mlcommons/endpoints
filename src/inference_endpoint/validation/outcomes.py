# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy finding construction shared by independent evaluator classes."""

from pathlib import Path

from .artifacts import PointArtifacts
from .planner import PlannedCheck
from .results import CheckResult, Severity


def result(
    check: PlannedCheck,
    passed: bool,
    message: str,
    *,
    key: str | None = None,
    point: PointArtifacts | None = None,
    blocked: bool = False,
) -> CheckResult:
    return CheckResult(
        rule=check.rule.id,
        key=key or ("blocked" if blocked else "pass" if passed else "fail"),
        title=check.rule.id,
        message=message,
        severity=Severity.INFO
        if passed
        else Severity.ERROR
        if blocked
        else Severity(check.rule.severity.value),
        path=point.path if point else Path(check.subject.id),
    )
