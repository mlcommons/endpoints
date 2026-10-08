# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed vocabulary and policy identities shared by parsing and planning."""

from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints


def _canonical_identifier(value: str) -> str:
    if value != value.strip():
        raise ValueError("Identifiers must not contain surrounding whitespace")
    return value


Identifier = Annotated[
    str, StringConstraints(min_length=1), AfterValidator(_canonical_identifier)
]
CohortId = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])-C[01]$")]
CommitId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Release(FrozenModel):
    version: CohortId
    revision: int = Field(strict=True, ge=1)


class Scope(StrEnum):
    SUBMISSION = "submission"
    SYSTEM = "system"
    CURVE = "curve"
    POINT = "point"
    VALIDATOR = "validator"
    IMPLEMENTATION = "implementation"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class PolicyFile(StrEnum):
    CATALOG = "catalog.yaml"
    SUBMISSION = "submission_checks.yaml"
    SYSTEM = "system_checks.yaml"
    CURVE = "curve_checks.yaml"
    POINT = "point_checks.yaml"

    @property
    def scope(self) -> Scope:
        return {
            self.CATALOG: Scope.VALIDATOR,
            self.SUBMISSION: Scope.SUBMISSION,
            self.SYSTEM: Scope.SYSTEM,
            self.CURVE: Scope.CURVE,
            self.POINT: Scope.POINT,
        }[self]


class OfflineMode(StrEnum):
    NONE = "none"
    DEDICATED = "dedicated"
    ELECTED = "elected"


class Division(StrEnum):
    STANDARDIZED = "Standardized"
    SERVICED = "Serviced"
    RDI = "RDI"


class Declaration(StrEnum):
    WARMUP = "warmup"
    STEADY_STATE = "steady_state"
    POWER_DESCRIPTOR = "power_descriptor"


class EvidenceKey(StrEnum):
    COMPUTED_REGIONS = "computed_regions"
    WARMUP = "warmup"
    STEADY_STATE = "steady_state"
    SEED_CATALOG = "seed_catalog"
    APPROVED_SPED_DECODE_HEADS = "approved_sped_decode_heads"
    POINT_CONFIG = "point_config"
    POWER_COMPUTATION = "power_computation"
    POINT_POWER = "point_power"


class Decision(StrEnum):
    READY = "ready"
    EXCLUDED = "excluded"
    BLOCKED = "blocked"


class Match(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class AccuracyKind(StrEnum):
    SINGLE_TURN = "single_turn"
    AGENTIC = "agentic"


class SampleUnit(StrEnum):
    SAMPLE = "sample"
    TRAJECTORY = "trajectory"
    INSTANCE = "instance"


class DecodeMethod(StrEnum):
    DSPARK = "dspark"
    MTP = "mtp"


class Cooling(StrEnum):
    LIQUID = "liquid"
    AIR = "air"
    MIXED = "mixed"
