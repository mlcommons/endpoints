# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SteadyStateStatus(StrEnum):
    """Coverage status controlling the reported metric basis."""

    WINDOWABLE = "windowable"
    INSUFFICIENT_DURATION = "insufficient_duration"
    INSUFFICIENT_PASSES = "insufficient_passes"
    PARTIAL_DATASET = "partial_dataset"


class SteadyStateVerdict(StrEnum):
    """Canonical detector verdict with explicit input aliases."""

    STEADY_STATE = "steady_state"
    DRIFTING_UP = "drifting_up"
    DRIFTING_DOWN = "drifting_down"
    ANOMALY = "anomaly"
    NOT_FOUND = "not_found"

    @classmethod
    def _missing_(cls, value: object) -> SteadyStateVerdict | None:
        aliases = {"STEADY STATE": cls.STEADY_STATE, "not found": cls.NOT_FOUND}
        return aliases.get(value) if isinstance(value, str) else None

    @property
    def is_drifting(self) -> bool:
        """Whether the verdict requires drift reporting rather than a point estimate."""
        return self in (self.DRIFTING_UP, self.DRIFTING_DOWN)


class MetricState(StrEnum):
    """Canonical state of a gating metric."""

    PLATEAU = "plateau"
    DRIFTING_UP = "drifting_up"
    DRIFTING_DOWN = "drifting_down"

    @classmethod
    def _missing_(cls, value: object) -> MetricState | None:
        aliases = {
            "Plateau": cls.PLATEAU,
            "Drifting Up": cls.DRIFTING_UP,
            "Drifting Down": cls.DRIFTING_DOWN,
        }
        return aliases.get(value) if isinstance(value, str) else None

    @property
    def is_drifting(self) -> bool:
        """Whether this metric has an upward or downward trend."""
        return self in (self.DRIFTING_UP, self.DRIFTING_DOWN)


class SteadyStateWindow(BaseModel):
    """The detected window's extent (§4.4)."""

    model_config = ConfigDict(extra="ignore")
    n_super_passes: int | None = None
    super_pass_start: int | None = None
    super_pass_end: int | None = None
    super_pass_size: int | None = None
    n_samples: int | None = None
    duration_s: float | None = None


class SteadyState(BaseModel):
    """A point's §4.4 reporting block."""

    model_config = ConfigDict(extra="ignore")
    n_super_passes: int | None = None
    status: SteadyStateStatus | None = None
    verdict: SteadyStateVerdict | None = None
    window: SteadyStateWindow = Field(default_factory=SteadyStateWindow)
    state: dict[str, MetricState] = Field(default_factory=dict)
    anomaly: object | None = None

    @property
    def is_official(self) -> bool:
        """True when §4.4 makes the windowed metrics this point's official result."""
        return self.status == SteadyStateStatus.WINDOWABLE

    @property
    def drifting_metrics(self) -> list[str]:
        """Gating metrics that are drifting rather than flat."""
        return sorted((name for name, state in self.state.items() if state.is_drifting))
