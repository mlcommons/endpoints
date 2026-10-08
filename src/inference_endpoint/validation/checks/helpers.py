# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared findings, numeric validation, and supported policy execution modes."""

from __future__ import annotations

import math
from typing import TypeGuard

from pydantic import JsonValue

from ..artifacts import PointArtifacts
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult, Severity


def values(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def finding(
    check: PlannedCheck,
    passed: bool,
    message: str,
    *,
    point: PointArtifacts | None = None,
    severity: Severity | None = None,
    blocked: bool = False,
) -> CheckResult:
    value = result(check, passed, message, point=point, blocked=blocked)
    return (
        value.model_copy(update={"severity": severity})
        if severity is not None and not passed and not blocked
        else value
    )


def warning(
    check: PlannedCheck, message: str, point: PointArtifacts | None = None
) -> CheckResult:
    return finding(check, False, message, point=point, severity=Severity.WARNING)


def unsupported(
    check: PlannedCheck, message: str = "Unsupported policy operation"
) -> list[CheckResult]:
    return [finding(check, False, message, blocked=True)]


def number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def block_value(point: PointArtifacts, name: str) -> dict[str, JsonValue] | None:
    value = getattr(point.config, name, None)
    return (
        value.model_dump(mode="json", exclude_unset=True)
        if value is not None
        else point.get_config(name)
    )


def validate_modes(check: PlannedCheck) -> list[CheckResult] | None:
    """Reject policy modes whose semantics this parser revision cannot execute."""
    allowed = {
        "matching": {"model_and_repository_and_revision"},
        "on_unavailable": {"blocked"},
        "on_legacy_registry_without_cohorts": {"blocked"},
        "on_empty_catalog": {"error"},
        "on_missing_repository_or_revision": {"error"},
        "on_unmatched_drafter": {"skip"},
        "on_unparseable_cohort": {"warning"},
        "on_missing_approval_cohort": {"warning"},
        "on_undeclared_nodes": {"fully_engaged"},
        "declared_total_scaling": {"maximum_engaged_fraction"},
        "switch_scaling": {"engaged_node_fraction"},
        "on_spare_nodes": {"warning"},
        "on_missing_parallelism": {"warning"},
        "on_disaggregated": {"warning"},
        "on_unknown_accelerator_count": {"warning"},
        "on_missing_window_extent": {"skip_count_comparison"},
        "on_missing_or_nonofficial": {"warning"},
        "fallback": {"whole_run_total"},
        "on_drift": {"warn_range_or_slope_required"},
        "disabled_warmup": {"exempt"},
        "on_missing_weights": {"arithmetic_mean"},
        "on_missing_required_dataset": {"error"},
        "on_unrelated_dataset": {"exclude"},
        "on_no_sample_counts": {"skip"},
        "on_unknown_model": {"warning"},
        "on_unknown_or_unpublished": {"warning"},
        "on_missing_metric": {"skip", "error"},
        "fraction_conversion": {"legacy_value_and_threshold_heuristic"},
        "count_scope": {"per_required_dataset_or_suite"},
        "units": {"samples_times_repeats"},
        "score": {"score_or_first_metric"},
        "unnamed_scalar": {"sole_metric_only_with_warning"},
        "on_multiple_results_per_band": {"average_and_warn"},
        "on_missing_bands": {"gate_available_mean_and_warn"},
        "on_no_results": {"warning"},
    }
    for field, modes in allowed.items():
        value = check.rule.requirements.get(field)
        if value is not None and value not in modes:
            return unsupported(check, f"Unsupported policy mode {field}={value!r}")
    return None
