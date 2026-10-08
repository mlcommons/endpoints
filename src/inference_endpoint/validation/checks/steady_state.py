# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven steady state checks."""

from __future__ import annotations

from collections.abc import Mapping

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..evidence.steady_state import MetricState, SteadyStateStatus
from ..operations import SteadyOperation
from ..planner import PlannedCheck
from ..results import CheckResult
from ..schemas.requirements_v1 import (
    SteadyStateRequirements,
)
from ..vocabulary import CheckKind
from .helpers import finding, number, unsupported, warning


class SteadyStateEvaluator(Evaluator, kind=CheckKind.STEADY_STATE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.requirements(SteadyStateRequirements)
        constants = artifacts.resolve(params.constants, check)
        if not isinstance(constants, Mapping):
            return unsupported(check, "Steady-state catalog unavailable")
        operation = params.operation
        output = []
        for point in artifacts.members(check):
            block = point.config.steady_state if point.config is not None else None
            if operation is SteadyOperation.REPORTING_BASIS:
                if block is None or not block.is_official:
                    output.append(
                        warning(
                            check,
                            f"Steady-state window is nonofficial; reporting basis: {params.fallback}",
                            point,
                        )
                    )
                else:
                    output.append(
                        finding(
                            check,
                            True,
                            "Official steady-state window is the reporting basis",
                            point=point,
                        )
                    )
                if block is not None and (
                    block.verdict is not None
                    and block.verdict.is_drifting
                    or block.drifting_metrics
                ):
                    output.append(
                        warning(
                            check, "Drifting metrics require a range or slope", point
                        )
                    )
                continue
            if block is None:
                output.append(
                    finding(
                        check,
                        False,
                        "Steady-state declaration unavailable",
                        point=point,
                        blocked=True,
                    )
                )
                continue
            problems = []
            if operation is SteadyOperation.VOCABULARY:
                for field, catalog in (("status", "statuses"), ("verdict", "verdicts")):
                    if (
                        getattr(block, field) is not None
                        and getattr(block, field) not in constants[catalog]
                    ):
                        problems.append(
                            f"{field}={getattr(block, field)!r} is outside the policy vocabulary"
                        )
                for metric, value in block.state.items():
                    if value not in constants["metric_states"]:
                        problems.append(f"{metric}={value!r} is outside metric states")
            else:
                window = block.window
                lo, hi = window.super_pass_start, window.super_pass_end
                spans = hi - lo + 1 if number(lo) and number(hi) else None
                minimum = constants["minimum_super_passes"]
                if spans is not None and spans <= 0:
                    problems.append("Window super-pass extent is reversed or empty")
                if block.status is SteadyStateStatus.WINDOWABLE:
                    if spans is not None and spans < minimum:
                        problems.append(
                            f"Official window spans {spans} super-passes; requires {minimum}"
                        )
                    if params.official_requires_plateau and any(
                        v is not MetricState.PLATEAU for v in block.state.values()
                    ):
                        problems.append("Official window contains a nonplateau metric")
                elif (
                    block.status is SteadyStateStatus.INSUFFICIENT_PASSES
                    and spans is not None
                    and spans >= minimum
                ):
                    problems.append(
                        f"Insufficient-pass status conflicts with {spans} super-passes"
                    )
                reported = (
                    window.n_super_passes
                    if "n_super_passes" in window.model_fields_set
                    else block.n_super_passes
                )
                if (
                    params.check_reported_super_pass_count
                    and reported is not None
                    and spans is not None
                    and reported != spans
                ):
                    problems.append(
                        f"Reported super-pass count {reported} differs from extent {spans}"
                    )
            output.append(
                finding(
                    check,
                    not problems,
                    "; ".join(problems)
                    or "Steady-state declaration agrees with policy",
                    point=point,
                )
            )
        return output
