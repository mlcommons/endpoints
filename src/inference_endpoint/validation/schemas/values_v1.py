# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalog shapes for cohort policy revision 1."""

import math
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from inference_endpoint.config.schema import LoadPatternType

from ..types import (
    AccuracyKind,
    CohortId,
    CommitId,
    Cooling,
    DecodeMethod,
    Division,
    FrozenModel,
    Identifier,
    OfflineMode,
    Release,
    SampleUnit,
    Scope,
    Severity,
)


class Dataset(FrozenModel):
    sample_count: int | None = Field(strict=True, gt=0)
    sample_unit: SampleUnit = SampleUnit.SAMPLE
    is_legacy: bool = Field(default=False, strict=True)


class Checkpoint(FrozenModel):
    repository: Identifier
    revision: CommitId


class DecodeHead(Checkpoint):
    model: Identifier
    method: DecodeMethod
    bundled: bool = Field(default=False, strict=True)
    approved_cohort: CohortId | None = None


class Accuracy(FrozenModel):
    kind: AccuracyKind
    datasets: tuple[Identifier, ...] = Field(min_length=1)
    metrics: dict[Identifier, dict[str, JsonValue]] | None = None
    inline_minimum_percent: float | None = Field(default=None, ge=0, le=100)
    swebench_mean_minimum_percent: float | None = Field(default=None, ge=0, le=100)
    full_run_osl_range: tuple[int, int] | None = None

    @model_validator(mode="after")
    def check_profile(self):
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("Duplicate accuracy dataset")
        if self.kind is AccuracyKind.SINGLE_TURN:
            if not self.metrics or any(
                value is not None
                for value in (
                    self.inline_minimum_percent,
                    self.swebench_mean_minimum_percent,
                    self.full_run_osl_range,
                )
            ):
                raise ValueError(
                    "Single-turn accuracy requires only its metric thresholds"
                )
        elif self.metrics is not None or any(
            value is None
            for value in (
                self.inline_minimum_percent,
                self.swebench_mean_minimum_percent,
                self.full_run_osl_range,
            )
        ):
            raise ValueError(
                "Agentic accuracy requires inline, SWE-bench and OSL thresholds"
            )
        if self.full_run_osl_range is not None:
            lower, upper = self.full_run_osl_range
            if lower <= 0 or upper < lower:
                raise ValueError("Invalid full-run OSL range")
        return self


class Model(FrozenModel):
    accuracy: Accuracy


class Catalogs(FrozenModel):
    load_pattern: tuple[LoadPatternType, ...]
    divisions: tuple[Division, ...]
    offline_modes: tuple[OfflineMode, ...]
    region_names: tuple[Identifier, ...]
    declared_regions: tuple[Identifier, ...]
    mandatory_accuracy_bands: tuple[Identifier, ...]
    datasets: dict[Identifier, Dataset]
    models: dict[Identifier, Model]
    approved_sped_decode_heads: tuple[DecodeHead, ...]
    approved_client_revisions: tuple[CommitId, ...]
    approved_checkpoints: dict[Identifier, tuple[Checkpoint, ...]]
    submission_flags: dict[str, JsonValue]
    disclosure: dict[str, JsonValue]
    steady_state: dict[str, JsonValue]
    regions: dict[str, JsonValue]
    power: dict[str, JsonValue]
    metrics: dict[str, JsonValue]

    @model_validator(mode="after")
    def check_references(self):
        overhead = self.power.get("cooling_overhead")
        if not isinstance(overhead, dict) or set(overhead) != set(Cooling):
            raise ValueError(
                "Power cooling_overhead must define liquid, air, and mixed"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in overhead.values()
        ):
            raise ValueError(
                "Cooling overhead fractions must be nonnegative numeric values"
            )
        for name, model in self.models.items():
            if not set(model.accuracy.datasets) <= self.datasets.keys():
                raise ValueError(f"Unknown accuracy dataset for {name}")
        if not self.approved_checkpoints.keys() <= self.models.keys():
            raise ValueError("Checkpoint catalog references unknown models")
        for head in self.approved_sped_decode_heads:
            if head.model not in self.models:
                raise ValueError("Decode head references unknown model")
        return self


class GroupDocument(Release):
    scope: Scope
    severity: Severity = Severity.ERROR
    checks: dict[Identifier, dict[str, JsonValue]] = Field(min_length=1)


class Diagnostics(FrozenModel):
    scope: Literal[Scope.VALIDATOR]
    severity: Severity = Severity.WARNING
    checks: dict[Identifier, dict[str, JsonValue]]
