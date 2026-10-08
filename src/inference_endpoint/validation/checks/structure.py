# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy evaluators for structure."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from inference_endpoint.config.schema import LoadPatternType

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult, Severity
from ..vocabulary import CheckKind
from .helpers import values


class Presence(Evaluator, kind=CheckKind.PRESENCE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        targets = req.get("targets") or [req.get("target")]
        findings = []
        for target in targets:
            value = (
                artifacts.root / target
                if target in ("results", "docs", "src")
                else artifacts.resolve(target, check)
            )
            items = values(value)
            if req.get("pattern"):
                items = [
                    item
                    for item in items
                    if re.fullmatch(req["pattern"], getattr(item, "name", str(item)))
                ]
            object_type = req.get("object")

            def present(item: object, object_type: str | None = object_type) -> bool:
                if isinstance(item, Path):
                    return (
                        item.is_file()
                        if object_type == "file"
                        else item.is_dir()
                        if object_type == "directory"
                        else item.exists()
                    )
                return item is not None and item is not False

            count = sum(present(item) for item in items)
            findings.append(
                result(
                    check,
                    count >= req.get("minimum_count", 1),
                    f"{target}: found {count}, required {req.get('minimum_count', 1)}",
                )
            )
        return findings


class ArtifactSchema(Evaluator, kind=CheckKind.ARTIFACT_SCHEMA):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        point = artifacts.index.points[check.subject.id]
        req = check.rule.requirements
        schema = req.get("schema")
        if schema == "accuracy_result":
            present = point.evidence.accuracy.present
            if req.get("when_present") and not present:
                return []
            obj = point.accuracy
            findings = point.evidence.accuracy.errors
            if obj is not None and req.get("reject_empty") and not obj.root:
                return [result(check, False, "Accuracy results must not be empty")]
        else:
            attr = {
                "point_config": "config",
                "system_description": "system",
                "result_summary": "summary",
            }.get(cast(str, schema))
            if attr is None:
                return [
                    result(
                        check,
                        False,
                        f"Unsupported artifact schema {schema}",
                        blocked=True,
                    )
                ]
            obj = getattr(point, attr)
            findings = getattr(point.evidence, attr).errors
        if obj is None or any(f.severity == Severity.ERROR for f in findings):
            return [
                f.model_copy(
                    update={
                        "rule": check.rule.id,
                        "severity": Severity(check.rule.severity.value)
                        if f.severity is Severity.ERROR
                        else f.severity,
                    }
                )
                for f in findings
            ] or [result(check, False, f"{schema}: missing or invalid evidence")]
        return [result(check, True, f"{schema}: parsed successfully")]


class Membership(Evaluator, kind=CheckKind.MEMBERSHIP):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        catalog = artifacts.resolve(req["catalog"], check)
        items = values(artifacts.resolve(req["target"], check))
        findings = []
        seen = set()
        for value in items:
            if value is None and req.get("skip_missing"):
                continue
            if value is None and not req.get("required"):
                continue
            if req.get("deduplicate") and value in seen:
                continue
            seen.add(value)
            findings.append(
                result(
                    check,
                    value in catalog if catalog is not None else False,
                    f"{req['target']}: {value!r}; allowed {list(catalog or [])!r}",
                    blocked=catalog is None,
                )
            )
        return findings


class Consistency(Evaluator, kind=CheckKind.CONSISTENCY):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        if "left" in req:
            left = artifacts.resolve(req["left"], check)
            right = artifacts.resolve(req["right"], check)
            if left is None or right is None:
                finding = result(check, False, "Comparison evidence is missing")
                if req.get("on_missing") == "warning":
                    finding = finding.model_copy(update={"severity": Severity.WARNING})
                return [finding]
            return [result(check, left == right, f"{left!r} compared with {right!r}")]
        items = values(artifacts.resolve(req["target"], check))
        if req.get("exclude_fields"):
            items = [
                {k: v for k, v in item.items() if k not in req["exclude_fields"]}
                if isinstance(item, Mapping)
                else item
                for item in items
            ]
        if any(item is None for item in items):
            if req.get("on_missing") == "error":
                return [result(check, False, "Required comparison evidence is missing")]
            items = [item for item in items if item is not None]
        nonempty = (
            not req.get("require_nonempty")
            or bool(items)
            and all(bool(item) for item in items)
        )
        return [
            result(
                check,
                nonempty and (not items or all(item == items[0] for item in items)),
                f"{req['target']}: {len(items)} declarations compared",
            )
        ]


class Disclosure(Evaluator, kind=CheckKind.DISCLOSURE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        point = artifacts.index.points[check.subject.id]
        fields = artifacts.resolve(req["fields"], check)
        missing = [name for name in fields if point.get_config(name) in (None, "")]
        return [
            result(
                check,
                not missing,
                f"Missing disclosure fields: {missing}"
                if missing
                else "Required disclosures are present",
            )
        ]


class CohortIdentifier(Evaluator, kind=CheckKind.COHORT_IDENTIFIER):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        value = artifacts.resolve(check.rule.requirements["target"], check)
        return [
            result(
                check,
                isinstance(value, str)
                and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])-C[01]", value) is not None,
                f"Target cohort: {value!r}",
            )
        ]


class PathResolution(Evaluator, kind=CheckKind.PATH_RESOLUTION):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.rule.requirements
        root = Path(artifacts.resolve(req["base"], check)).resolve()
        findings = []
        for address in req["targets"]:
            value = artifacts.resolve(address, check)
            if value is None and req.get("on_missing") == "skip":
                continue
            if not isinstance(value, str):
                findings.append(result(check, False, f"{address}: missing path"))
                continue
            path = Path(value)
            try:
                resolved = (root / path).resolve()
            except (OSError, RuntimeError) as error:
                findings.append(
                    result(
                        check, False, f"{address}: cannot resolve {value!r}: {error}"
                    )
                )
                continue
            valid = (
                (req.get("allow_absolute", False) or not path.is_absolute())
                and (req.get("allow_parent_traversal", False) or ".." not in path.parts)
                and (
                    req.get("allow_escape_via_symlink", False)
                    or resolved.is_relative_to(root)
                )
                and (
                    resolved.is_dir()
                    if req.get("require_directory")
                    else resolved.exists()
                )
            )
            findings.append(
                result(check, valid, f"{address}: {value!r} resolves to {resolved}")
            )
        return findings


class ReportValue(Evaluator, kind=CheckKind.REPORT):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        value = artifacts.resolve(check.rule.requirements["target"], check)
        return [result(check, True, f"{check.rule.requirements['target']}: {value!r}")]


class Issuance(Evaluator, kind=CheckKind.ISSUANCE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        point = artifacts.index.points[check.subject.id]
        req = check.rule.requirements
        pattern = point.get_config(
            "runtime_settings.load_pattern", LoadPatternType.CONCURRENCY
        )
        valid = pattern in req["allowed"] and (
            not req.get("require_positive_concurrency")
            or point.concurrency is not None
            and point.concurrency > 0
        )
        return [
            result(
                check,
                valid,
                f"Load pattern {pattern!r}, concurrency {point.concurrency}",
            )
        ]
