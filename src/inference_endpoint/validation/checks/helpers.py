# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared findings, numeric validation, and supported policy execution modes."""

from __future__ import annotations

import math
from typing import TypeGuard

from pydantic import JsonValue

from ..artifacts import PointArtifacts
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult, Severity


def values(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def finding(
    check: PlannedCheck,
    passed: bool,
    message: str,
    *,
    point: PointArtifacts | None = None,
    severity: Severity | None = None,
    blocked: bool = False,
) -> CheckResult:
    value = result(check, passed, message, point=point, blocked=blocked)
    return (
        value.model_copy(update={"severity": severity})
        if severity is not None and not passed and not blocked
        else value
    )


def warning(
    check: PlannedCheck, message: str, point: PointArtifacts | None = None
) -> CheckResult:
    return finding(check, False, message, point=point, severity=Severity.WARNING)


def unsupported(
    check: PlannedCheck, message: str = "Unsupported policy operation"
) -> list[CheckResult]:
    return [finding(check, False, message, blocked=True)]


def number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def block_value(point: PointArtifacts, name: str) -> dict[str, JsonValue] | None:
    value = getattr(point.config, name, None)
    return (
        value.model_dump(mode="json", exclude_unset=True)
        if value is not None
        else point.get_config(name)
    )
