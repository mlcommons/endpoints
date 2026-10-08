# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy evaluators for metrics."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import Any

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..evidence.steady_state import SteadyStateStatus
from ..operations import (
    ComparisonOperator,
    DerivedOperation,
    DurationBasis,
    OperandKind,
)
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult
from ..vocabulary import CheckKind


class Comparison(Evaluator, kind=CheckKind.COMPARISON):
    operations = {
        ComparisonOperator.GREATER_THAN: operator.gt,
        ComparisonOperator.GREATER_THAN_OR_EQUAL: operator.ge,
        ComparisonOperator.EQUAL: operator.eq,
    }

    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        when = req.get("when")
        if when:
            actual = artifacts.resolve(when["field"], check)
            if actual is None or actual <= when["greater_than"]:
                return []

        def operand(value: Mapping[str, Any]) -> Any:
            return (
                value["value"]
                if OperandKind(value["kind"]) is OperandKind.CONSTANT
                else artifacts.resolve(value["field"], check)
            )

        left = operand(req["left"])
        right = operand(req["right"])
        op = self.operations.get(ComparisonOperator(req["operator"]))
        if op is None:
            return [
                result(
                    check,
                    False,
                    f"Unsupported comparison {req['operator']}",
                    blocked=True,
                )
            ]
        valid = (
            left is not None
            and right is not None
            and not isinstance(left, bool)
            and not isinstance(right, bool)
            and op(left, right)
        )
        return [result(check, valid, f"{left!r} {req['operator']} {right!r}")]


class NumericValidity(Evaluator, kind=CheckKind.NUMERIC_VALIDITY):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        value = artifacts.resolve(req["source"], check)
        if value is None and not req.get("require_present"):
            return []
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if valid and req.get("finite"):
            valid = math.isfinite(value)
        if valid and req.get("strictly_positive"):
            valid = value > 0
        return [result(check, valid, f"{req['source']}: {value!r}")]


class Duration(Evaluator, kind=CheckKind.DURATION):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        point = artifacts.index.points[check.subject.id]
        req = check.rule.requirements
        band = artifacts.region(point)
        if band is None and req.get("on_unclassifiable_concurrency") == "skip":
            return []
        thresholds = artifacts.resolve(req["thresholds"], check)
        minimum = thresholds.get(band)
        duration = None
        for basis in req["basis_precedence"]:
            if basis == DurationBasis.REPORTED_WINDOW_ISSUE_SPAN:
                value = point.get_config("steady_state.window.duration_s")
                if value is not None and (
                    not req.get("window_status_required")
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
        req = check.rule.requirements
        operation = DerivedOperation(req["operation"])
        findings = []
        points = artifacts.members(check)
        constants = artifacts.resolve(req.get("constants"), check) or {}
        if operation is DerivedOperation.CURVE_PEAK_RATIO:
            available = [
                p
                for p in points
                if p.throughput is not None
                and isinstance(artifacts.resolve(req["stored"], check, p), (int, float))
            ]
            peak = max(
                (p.throughput for p in available if p.throughput is not None), default=0
            )
            if peak <= 0:
                return []
            for point in available:
                stored = artifacts.resolve(req["stored"], check, point)
                assert point.throughput is not None
                ratio = point.throughput / peak
                findings.append(
                    result(
                        check,
                        math.isfinite(stored)
                        and abs(stored - ratio) <= req["tolerance"]["absolute"],
                        f"Utilization {stored}; derived {ratio}",
                        point=point,
                    )
                )
            return findings
        for point in points:
            stored = artifacts.resolve(req.get("stored"), check, point)
            expected: float | None = None
            if operation is DerivedOperation.OUTPUT_TOKENS_PER_ELAPSED_SECOND:
                expected = point.throughput
            elif operation is DerivedOperation.INVERSE_TPOT_P90_MS:
                tpot = artifacts.resolve(
                    "result_summary.tpot.percentiles.90", check, point
                )
                if isinstance(tpot, (int, float)) and math.isfinite(tpot) and tpot > 0:
                    expected = req["numerator"] / (tpot / 1e6)
                else:
                    continue
            elif operation is DerivedOperation.THROUGHPUT_PER_PROVISIONED_KW:
                denominator = artifacts.resolve(req["denominator"], check, point)
                if denominator is None or denominator <= 0:
                    continue
                expected = (
                    point.throughput / denominator
                    if point.throughput is not None
                    else None
                )
            elif operation is DerivedOperation.OUTPUT_TOKENS_PER_COMPLETED_TURN_SECOND:
                tokens = point.get_summary("output_tokens_per_turn_total")
                seconds = point.get_summary("e2e_turn_time_seconds_total")
                if (
                    isinstance(tokens, (int, float))
                    and isinstance(seconds, (int, float))
                    and seconds > 0
                ):
                    expected = tokens / seconds
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
