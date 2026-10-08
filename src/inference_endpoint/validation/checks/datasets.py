# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy evaluators for datasets."""

from __future__ import annotations

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..operations import DatasetCountOperator
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult
from ..types import SampleUnit
from ..vocabulary import CheckKind


class DatasetMinimum(Evaluator, kind=CheckKind.DATASET_MINIMUM):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        point = artifacts.index.points[check.subject.id]
        dataset = point.get_config("dataset")
        catalog = artifacts.resolve(req["catalog"], check) or {}
        entry = catalog.get(dataset, {})
        if entry.get("sample_unit", SampleUnit.SAMPLE) not in req.get(
            "supported_sample_units", [SampleUnit.SAMPLE]
        ):
            if req.get("on_unsupported_sample_unit") == "skip":
                return []
            return [result(check, False, "Unsupported sample unit", blocked=True)]
        minimum = entry.get(req["threshold"])
        if minimum is None:
            if req.get("on_unknown_or_null_threshold") == "skip":
                return []
            return [result(check, False, "Dataset threshold unavailable", blocked=True)]
        actual = artifacts.resolve(req["source"], check)
        return [
            result(
                check,
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and actual >= minimum,
                f"Completed {actual!r}; {dataset} requires {minimum}",
            )
        ]


class DatasetCount(Evaluator, kind=CheckKind.DATASET_COUNT):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        entry = (artifacts.resolve(req["catalog"], check) or {}).get(req["dataset"], {})
        expected = entry.get("sample_count")
        sources = req.get("sources") or [req["source"]]
        findings = []
        if expected is None or expected <= 0:
            return [result(check, False, "Dataset count unavailable", blocked=True)]
        for source in sources:
            actual = artifacts.resolve(source, check)
            valid = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and (
                    actual == expected
                    if req["operator"] == DatasetCountOperator.EQUAL
                    else actual > 0 and actual % expected == 0
                )
            )
            findings.append(
                result(
                    check,
                    valid,
                    f"{source}: {actual!r}; expected {req['operator']} {expected}",
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
