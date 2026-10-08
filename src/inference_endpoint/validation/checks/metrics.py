# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy evaluators for metrics."""

from __future__ import annotations

import math
import operator

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..evidence.steady_state import SteadyStateStatus
from ..operations import (
    ComparisonOperator,
    DerivedOperation,
    DurationBasis,
    PolicyAction,
)
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult
from ..schemas.requirements_v1 import (
    ComparisonRequirements,
    ConstantOperand,
    DerivedMetricRequirements,
    DurationRequirements,
    FieldOperand,
    NumericValidityRequirements,
    Operand,
    SumOperand,
)
from ..vocabulary import CheckKind


class Comparison(Evaluator, kind=CheckKind.COMPARISON):
    operations = {
        ComparisonOperator.GREATER_THAN: operator.gt,
        ComparisonOperator.GREATER_THAN_OR_EQUAL: operator.ge,
        ComparisonOperator.EQUAL: operator.eq,
    }

    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(ComparisonRequirements)
        when = req.when
        if when:
            actual = artifacts.resolve(when.field, check)
            if actual is None or actual <= when.greater_than:
                return []

        def operand(value: Operand) -> int | float | None:
            if isinstance(value, ConstantOperand):
                return value.value
            if isinstance(value, FieldOperand):
                resolved = artifacts.resolve(value.field, check)
                if resolved is None:
                    resolved = value.default
                return (
                    resolved
                    if isinstance(resolved, (int, float))
                    and not isinstance(resolved, bool)
                    else None
                )
            assert isinstance(value, SumOperand)
            operands = [operand(child) for child in value.operands]
            if any(child is None for child in operands):
                return None
            return sum(child for child in operands if child is not None)

        left = operand(req.left)
        right = operand(req.right)
        op = self.operations[req.operator]
        valid = (
            left is not None
            and right is not None
            and not isinstance(left, bool)
            and not isinstance(right, bool)
            and (not isinstance(left, float) or math.isfinite(left))
            and (not isinstance(right, float) or math.isfinite(right))
            and op(left, right)
        )
        return [result(check, valid, f"{left!r} {req.operator} {right!r}")]


class NumericValidity(Evaluator, kind=CheckKind.NUMERIC_VALIDITY):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(NumericValidityRequirements)
        value = artifacts.resolve(req.source, check)
        if value is None and not req.require_present:
            return []
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if valid and req.finite:
            valid = math.isfinite(value)
        if valid and req.strictly_positive:
            valid = value > 0
        return [result(check, valid, f"{req.source}: {value!r}")]


class Duration(Evaluator, kind=CheckKind.DURATION):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        point = artifacts.index.points[check.subject.id]
        req = check.requirements(DurationRequirements)
        band = artifacts.region(point)
        if band is None and req.on_unclassifiable_concurrency == PolicyAction.SKIP:
            return []
        thresholds = artifacts.resolve(req.thresholds, check)
        minimum = thresholds.get(band)
        duration = None
        for basis in req.basis_precedence:
            if basis == DurationBasis.REPORTED_WINDOW_ISSUE_SPAN:
                value = point.get_config("steady_state.window.duration_s")
                if value is not None and (
                    not req.window_status_required
                    or point.get_config("steady_state.status")
                    == SteadyStateStatus.WINDOWABLE
                ):
                    duration = value * 1000
            elif basis == DurationBasis.WHOLE_RUN_DURATION:
                value = point.get_summary("duration_ns")
                if value is not None:
                    duration = value / 1e6
            if duration is not None:
                break
        if minimum is None:
            return [
                result(check, False, "Duration threshold unavailable", blocked=True)
            ]
        return [
            result(
                check,
                isinstance(duration, (int, float))
                and math.isfinite(duration)
                and duration >= minimum,
                f"Duration {duration!r} ms; {band} requires {minimum} ms",
            )
        ]


class DerivedMetric(Evaluator, kind=CheckKind.DERIVED_METRIC):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(DerivedMetricRequirements)
        operation = req.operation
        findings = []
        points = artifacts.members(check)
        constants = artifacts.resolve(req.constants, check) or {}
        if operation is DerivedOperation.CURVE_PEAK_RATIO:
            assert req.tolerance is not None
            available = [
                p
                for p in points
                if p.throughput is not None
                and isinstance(artifacts.resolve(req.stored, check, p), (int, float))
            ]
            peak = max(
                (p.throughput for p in available if p.throughput is not None), default=0
            )
            if peak <= 0:
                return []
            for point in available:
                stored = artifacts.resolve(req.stored, check, point)
                assert point.throughput is not None
                ratio = point.throughput / peak
                findings.append(
                    result(
                        check,
                        math.isfinite(stored)
                        and abs(stored - ratio) <= req.tolerance.absolute,
                        f"Utilization {stored}; derived {ratio}",
                        point=point,
                    )
                )
            return findings
        for point in points:
            stored = artifacts.resolve(req.stored, check, point)
            expected: float | None = None
            if operation is DerivedOperation.OUTPUT_TOKENS_PER_ELAPSED_SECOND:
                expected = point.throughput
            elif operation is DerivedOperation.INVERSE_TPOT_P90_MS:
                assert req.numerator is not None
                tpot = artifacts.resolve(
                    "result_summary.tpot.percentiles.90", check, point
                )
                if isinstance(tpot, (int, float)) and math.isfinite(tpot) and tpot > 0:
                    milliseconds = tpot / 1e6
                    if milliseconds <= 0:
                        findings.append(
                            result(
                                check,
                                False,
                                "TPOT duration is too small to represent",
                                point=point,
                            )
                        )
                        continue
                    expected = req.numerator / milliseconds
                else:
                    continue
            elif operation is DerivedOperation.THROUGHPUT_PER_PROVISIONED_KW:
                denominator = artifacts.resolve(req.denominator, check, point)
                if denominator is None or denominator <= 0:
                    continue
                expected = (
                    point.throughput / denominator
                    if point.throughput is not None
                    else None
                )
            elif operation is DerivedOperation.OUTPUT_TOKENS_PER_COMPLETED_TURN_SECOND:
                expected = (
                    point.summary.e2e_avg_interactivity
                    if point.summary is not None
                    else None
                )
                if expected is None and stored is None:
                    continue
            else:
                return [
                    result(
                        check,
                        False,
                        f"Unsupported derived operation {operation}",
                        blocked=True,
                    )
                ]
            if expected is None:
                findings.append(
                    result(check, False, "Metric inputs unavailable", point=point)
                )
                continue
            if not math.isfinite(expected):
                findings.append(
                    result(check, False, "Derived metric is not finite", point=point)
                )
                continue
            if stored is None:
                findings.append(
                    result(check, True, f"Derived metric {expected}", point=point)
                )
                continue
            floor = constants.get("relative_denominator_floor", 1e-9)
            tolerance = constants.get("relative_tolerance", 0.01)
            valid = (
                isinstance(stored, (int, float))
                and not isinstance(stored, bool)
                and math.isfinite(stored)
                and math.isfinite(expected)
                and abs(stored - expected) / max(abs(expected), floor) <= tolerance
            )
            findings.append(
                result(
                    check,
                    valid,
                    f"Stored {stored!r}; derived {expected}; tolerance {tolerance}",
                    point=point,
                )
            )
        return findings
