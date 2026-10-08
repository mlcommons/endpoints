# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven drafters checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ..artifacts import Artifacts, nested
from ..cohorts import Cohort
from ..evaluator_base import Evaluator
from ..operations import DrafterOperation
from ..planner import PlannedCheck
from ..results import CheckResult
from ..vocabulary import CheckKind
from .helpers import finding, unsupported, validate_modes, warning


class DrafterBindingEvaluator(Evaluator, kind=CheckKind.DRAFTER_BINDING):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.rule.requirements
        invalid = validate_modes(check)
        if invalid is not None:
            return invalid
        try:
            operation = DrafterOperation(cast(str, params.get("operation")))
        except ValueError:
            return unsupported(check)
        catalog = artifacts.resolve(params["catalog"], check)
        if catalog is None:
            return unsupported(check, "Approved drafter catalog unavailable")
        output = []
        for point in artifacts.members(check):
            declared = point.get_config("speculative_decoding")
            if declared is None:
                declared = point.get_config("drafter")
            if declared is None:
                declared = {}
            model = point.context.model_id
            matched = None
            entries = (
                catalog.get(model, ()) if isinstance(catalog, Mapping) else catalog
            )
            for entry in entries:
                if isinstance(entry, Mapping):
                    if (
                        entry.get("model") == model
                        and entry.get("repository") == declared.get("repository")
                        and entry.get("revision") == declared.get("revision")
                    ):
                        matched = entry
                        break
                else:
                    return unsupported(
                        check,
                        "Drafter catalog entries must disclose repository and revision for model identity matching",
                    )
            if operation is DrafterOperation.MEMBERSHIP:
                output.append(
                    finding(
                        check,
                        matched is not None,
                        "Drafter matches approved model, repository and revision"
                        if matched
                        else "Drafter does not match an approved identity",
                        point=point,
                    )
                )
                continue
            if matched is None and params.get("on_unmatched_drafter") == "skip":
                continue
            approved = Cohort.parse(nested(matched, "approved_cohort") or "")
            target = Cohort.parse(point.get_config("target_cohort") or "")
            if approved is None or target is None:
                output.append(
                    warning(
                        check,
                        "Drafter approval age cannot be verified without valid approval and target cohorts",
                        point,
                    )
                )
                continue
            earliest = approved
            for _ in range(params["minimum_cohorts"]):
                earliest = earliest.next()
            output.append(
                finding(
                    check,
                    target >= earliest,
                    f"Drafter target {target}; earliest permitted cohort {earliest}",
                    point=point,
                )
            )
        return output
