# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven seeds checks."""

from __future__ import annotations

from collections.abc import Mapping

from ..artifacts import Artifacts, nested
from ..cohorts import Cohort, adoption_window
from ..evaluator_base import Evaluator
from ..operations import SeedOperation
from ..planner import PlannedCheck
from ..results import CheckResult
from ..schemas.requirements_v1 import (
    SeedBindingRequirements,
)
from ..vocabulary import CheckKind
from .helpers import finding, unsupported, warning


class SeedBindingEvaluator(Evaluator, kind=CheckKind.SEED_BINDING):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.requirements(SeedBindingRequirements)
        operation = params.operation
        registry = (
            artifacts.resolve(params.catalog, check)
            if operation is not SeedOperation.LEGACY_NAMES
            else None
        )
        if operation is not SeedOperation.LEGACY_NAMES and not isinstance(
            registry, Mapping
        ):
            return unsupported(check, "Seed catalog is unavailable")
        output = []
        for point in artifacts.members(check):
            runtime = point.get_config("runtime_settings.runtime", {})
            if operation is SeedOperation.LEGACY_NAMES:
                aliases = params.aliases
                assert aliases is not None
                used = [
                    key
                    for key in aliases
                    if f"runtime_settings.runtime.{key}" in point.evidence.config.fields
                ]
                if used and (
                    not params.warn_legacy_only_when_model_seed_missing
                    or runtime.get("model_seed") is None
                ):
                    output.append(
                        warning(
                            check, f"Legacy seed fields used: {', '.join(used)}", point
                        )
                    )
                continue
            set_id = point.get_config("seed_set")
            published = registry.get(set_id)
            if operation is SeedOperation.MEMBERSHIP:
                output.append(
                    finding(
                        check,
                        published is not None,
                        f"Seed set {set_id!r} {'is published' if published else 'is not published'}",
                        point=point,
                    )
                )
                continue
            if published is None:
                output.append(
                    finding(
                        check,
                        False,
                        f"Cannot evaluate unpublished seed set {set_id!r}",
                        point=point,
                        blocked=True,
                    )
                )
                continue
            if operation is SeedOperation.RUNTIME_VALUES:
                assert params.fields is not None
                canonical = nested(point.config, "runtime_settings.runtime")
                problems = []
                for field in params.fields:
                    expected = nested(published, field)
                    actual = (
                        nested(canonical, field)
                        if canonical is not None
                        else runtime.get(field)
                    )
                    if expected is None:
                        return unsupported(check, f"Published seed set has no {field}")
                    if actual != expected and (
                        actual is not None or params.require_all
                    ):
                        problems.append(f"{field}={actual!r}, expected {expected!r}")
                output.append(
                    finding(
                        check,
                        not problems,
                        "; ".join(problems)
                        or "Runtime seed values match the published set",
                        point=point,
                    )
                )
            else:
                cohorts = nested(published, "cohorts", ())
                if not cohorts:
                    output.append(
                        finding(
                            check,
                            False,
                            "Seed registry has no publication cohort; adoption cannot be verified",
                            point=point,
                            blocked=True,
                        )
                    )
                    continue
                publication = Cohort.parse(min(cohorts))
                target = Cohort.parse(point.get_config("target_cohort") or "")
                if publication is None or target is None:
                    output.append(
                        finding(
                            check,
                            False,
                            "Seed publication or target cohort cannot be parsed",
                            point=point,
                            blocked=True,
                        )
                    )
                    continue
                assert params.adoption_window_cohorts is not None
                allowed = adoption_window(publication, params.adoption_window_cohorts)
                output.append(
                    finding(
                        check,
                        target in allowed,
                        f"Seed adoption target {target}; allowed cohorts: {', '.join(map(str, allowed))}",
                        point=point,
                    )
                )
        return output
