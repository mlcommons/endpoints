# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven speculative decoding head checks."""

from __future__ import annotations

from collections.abc import Mapping

from ..artifacts import Artifacts, nested
from ..cohorts import Cohort
from ..evaluator_base import Evaluator
from ..models import thaw
from ..operations import PolicyAction, SpecDecodeHeadOperation
from ..planner import PlannedCheck
from ..results import CheckResult
from ..schemas.requirements_v1 import (
    SpecDecodeHeadRequirements,
)
from ..vocabulary import CheckKind
from .helpers import finding, unsupported, warning


class SpecDecodeHeadEvaluator(Evaluator, kind=CheckKind.SPEC_DECODE_HEAD):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.requirements(SpecDecodeHeadRequirements)
        operation = params.operation
        catalog = artifacts.resolve(params.catalog, check)
        if catalog is None:
            return unsupported(
                check, "Approved speculative decoding head catalog unavailable"
            )
        output = []
        for point in artifacts.members(check):
            declared = (
                point.config.speculative_decoding if point.config is not None else None
            )
            if declared is None and point.config is not None:
                declared = point.config.spec_decode_head
            model = point.context.model_id
            matched = None
            entries = (
                catalog.get(model, ()) if isinstance(catalog, Mapping) else catalog
            )
            for entry in entries:
                if isinstance(entry, Mapping):
                    if (
                        entry.get("model") == model
                        and declared is not None
                        and declared.matches(
                            entry.get("repository"),
                            entry.get("revision"),
                            target_checksum=entry.get("target_checksum"),
                            configuration=thaw(entry.get("configuration")),
                        )
                    ):
                        matched = entry
                        break
                else:
                    return unsupported(
                        check,
                        "Speculative decoding head catalog entries must disclose a supported identity for model matching",
                    )
            if operation is SpecDecodeHeadOperation.MEMBERSHIP:
                output.append(
                    finding(
                        check,
                        matched is not None,
                        "Speculative decoding head matches an approved identity for this model"
                        if matched
                        else "Speculative decoding head does not match an approved identity",
                        point=point,
                    )
                )
                continue
            if (
                matched is None
                and params.on_unmatched_spec_decode_head == PolicyAction.SKIP
            ):
                continue
            approved = Cohort.parse(nested(matched, "approved_cohort") or "")
            target = Cohort.parse(point.get_config("target_cohort") or "")
            if approved is None or target is None:
                output.append(
                    warning(
                        check,
                        "Speculative decoding head approval age cannot be verified without valid approval and target cohorts",
                        point,
                    )
                )
                continue
            assert params.minimum_cohorts is not None
            earliest = approved
            for _ in range(params.minimum_cohorts):
                earliest = earliest.next()
            output.append(
                finding(
                    check,
                    target >= earliest,
                    f"Speculative decoding head target {target}; earliest permitted cohort {earliest}",
                    point=point,
                )
            )
        return output
