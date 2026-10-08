# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy evaluators for datasets."""

from __future__ import annotations

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..operations import CountOperator
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult
from ..types import SampleUnit
from ..vocabulary import CheckKind


class Count(Evaluator, kind=CheckKind.COUNT):
    """Compare a reported or configured count with a catalog reference count."""

    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        dataset = (
            artifacts.resolve(req["dataset_source"], check)
            if "dataset_source" in req
            else req["dataset"]
        )
        entry = (artifacts.resolve(req["catalog"], check) or {}).get(dataset, {})
        supported_units = req.get("supported_sample_units")
        if (
            supported_units is not None
            and entry.get("sample_unit", SampleUnit.SAMPLE) not in supported_units
        ):
            if req.get("on_unsupported_sample_unit") == "skip":
                return []
            return [result(check, False, "Unsupported sample unit", blocked=True)]
        expected = entry.get(req.get("threshold", "sample_count"))
        if expected is None:
            if req.get("on_unknown_or_null_threshold") == "skip":
                return []
            return [result(check, False, "Reference count unavailable", blocked=True)]
        if expected <= 0:
            return [
                result(check, False, "Reference count must be positive", blocked=True)
            ]
        operation = CountOperator(req["operator"])
        sources = req.get("sources") or [req["source"]]
        findings = []
        for source in sources:
            actual = artifacts.resolve(source, check)
            if operation is CountOperator.GREATER_THAN_OR_EQUAL:
                valid = (
                    isinstance(actual, (int, float))
                    and not isinstance(actual, bool)
                    and actual >= expected
                )
            else:
                valid = (
                    isinstance(actual, int)
                    and not isinstance(actual, bool)
                    and (
                        actual == expected
                        if operation is CountOperator.EQUAL
                        else actual > 0 and actual % expected == 0
                    )
                )
            findings.append(
                result(
                    check,
                    valid,
                    f"{source}: {actual!r}; expected {operation.value} {expected} for {dataset}",
                )
            )
        return findings


class FieldConstraints(Evaluator, kind=CheckKind.FIELD_CONSTRAINTS):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        point = artifacts.index.points[check.subject.id]
        if "constraints_by_model" in req:
            constraint = (
                artifacts.resolve(req["constraints_by_model"], check) or {}
            ).get(check.subject.model_id)
            if constraint is None:
                return [
                    result(
                        check, False, "Model field constraint unavailable", blocked=True
                    )
                ]
            constraints = {req["target"]: constraint}
        else:
            constraints = artifacts.resolve(req["constraints"], check)
        findings = []
        for field, constraint in constraints.items():
            value = (
                artifacts.resolve(field, check)
                if field.startswith("accuracy.")
                else point.get_config(field)
            )
            valid = (
                value is None
                if constraint.get("absent")
                else value == constraint.get("equals")
                and type(value) is type(constraint.get("equals"))
            )
            findings.append(
                result(check, valid, f"{field}: {value!r}; expected {dict(constraint)}")
            )
        return findings
