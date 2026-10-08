# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven warmup checks."""

from __future__ import annotations

from collections.abc import Mapping

from ..artifacts import Artifacts
from ..evaluator_base import Evaluator
from ..operations import PolicyAction, WarmupOperation
from ..planner import PlannedCheck
from ..results import CheckResult
from ..schemas.requirements_v1 import (
    WarmupRequirements,
)
from ..vocabulary import CheckKind
from .helpers import block_value, finding, unsupported


class WarmupEvaluator(Evaluator, kind=CheckKind.WARMUP):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.requirements(WarmupRequirements)
        operation = params.operation
        output = []
        for point in artifacts.members(check):
            if operation is WarmupOperation.SALT_DISABLED:
                value = artifacts.resolve(params.source, check, point)
                if value is None and params.on_missing == PolicyAction.SKIP:
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
            if params.verify_archive_contents:
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
            if disabled and params.disabled_warmup == PolicyAction.EXEMPT:
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
                        retained or not params.require_declared_retained,
                        "Warmup logs retained"
                        if retained
                        else "Warmup log retention is not declared",
                        point=point,
                    )
                )
        return output
