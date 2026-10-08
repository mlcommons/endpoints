# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven warmup checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..operations import WarmupOperation
from ..planner import PlannedCheck
from ..results import CheckResult
from ..vocabulary import CheckKind
from .helpers import block_value, finding, unsupported, validate_modes


class WarmupEvaluator(Evaluator, kind=CheckKind.WARMUP):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.rule.requirements
        invalid = validate_modes(check)
        if invalid is not None:
            return invalid
        try:
            operation = WarmupOperation(cast(str, params.get("operation")))
        except ValueError:
            return unsupported(check)
        output = []
        for point in artifacts.members(check):
            if operation is WarmupOperation.SALT_DISABLED:
                value = artifacts.resolve(params["source"], check, point)
                if value is None and params.get("on_missing") == "skip":
                    continue
                output.append(
                    finding(
                        check,
                        value is False,
                        f"Warmup salt disabled: {value!r}",
                        point=point,
                    )
                )
                continue
            if params.get("verify_archive_contents"):
                return unsupported(
                    check, "Warmup archive content verification is unavailable"
                )
            warmup = block_value(point, "warmup")
            if not isinstance(warmup, Mapping):
                output.append(
                    finding(
                        check,
                        False,
                        "Warmup declaration unavailable",
                        point=point,
                        blocked=True,
                    )
                )
                continue
            disabled = all(
                warmup.get(key) == 0
                for key in ("duration_s", "requests_issued", "requests_completed")
            )
            if disabled and params.get("disabled_warmup") == "exempt":
                output.append(
                    finding(
                        check,
                        True,
                        "Disabled warmup is exempt from log retention",
                        point=point,
                    )
                )
            else:
                retained = warmup.get("logs_retained") is True
                output.append(
                    finding(
                        check,
                        retained or not params.get("require_declared_retained", True),
                        "Warmup logs retained"
                        if retained
                        else "Warmup log retention is not declared",
                        point=point,
                    )
                )
        return output
