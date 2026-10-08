# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Typed point configuration and nested declarations."""

from __future__ import annotations

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from inference_endpoint.config.schema import LoadPatternType

from ..json_values import json_identity_equal
from ..operations import ChecksumAlgorithm
from ..types import ChecksumValue, CommitId, Division, OfflineMode
from .steady_state import SteadyState


class WarmupSpec(BaseModel):
    """Warmup procedure declaration required by §6.3.3."""

    model_config = ConfigDict(extra="ignore")
    duration_s: float = Field(ge=0)
    requests_issued: int = Field(ge=0)
    requests_completed: int = Field(ge=0)
    data_source: str = Field(min_length=1)
    concurrency: int = Field(ge=0)
    initialization_steps: list[str] = Field(default_factory=list)
    logs_retained: bool | None = None
    link_logs: str | None = None

    @property
    def is_disabled(self) -> bool:
        """Whether the declaration records no warmup duration or requests."""
        return (
            self.duration_s == 0
            and self.requests_issued == 0
            and (self.requests_completed == 0)
        )

    @model_validator(mode="after")
    def _check_completed_le_issued(self) -> WarmupSpec:
        if self.requests_completed > self.requests_issued:
            raise ValueError(
                f"warmup requests_completed ({self.requests_completed}) exceeds requests_issued ({self.requests_issued})"
            )
        if self.concurrency == 0 and (not self.is_disabled):
            raise ValueError(
                "warmup concurrency must be positive when duration or requests are nonzero"
            )
        return self


class NodesUsed(BaseModel):
    """One entry of §8.3's ``nodes_used``: the nodes of one type a point engages."""

    model_config = ConfigDict(extra="ignore")
    system_node_ensemble_id: int
    nodes: int


class DpShortfall(BaseModel):
    """§8.3's ``dp_shortfall``: a point run at fewer replicas than §4.5.3's formula gives."""

    model_config = ConfigDict(extra="ignore")
    dp_actual: int
    dp_formula: int
    reason: str = Field(min_length=1)


class CheckpointDeclaration(BaseModel):
    repository: str | None = None
    revision: str | None = None


class SpecDecodeHeadChecksum(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    algorithm: ChecksumAlgorithm
    revision: CommitId

    @classmethod
    def from_string(cls, value: str) -> SpecDecodeHeadChecksum:
        algorithm, separator, revision = value.partition(":")
        if not separator:
            raise ValueError("Weight checksum requires algorithm:revision")
        return cls.model_validate({"algorithm": algorithm, "revision": revision})

    def as_string(self) -> str:
        return f"{self.algorithm.value}:{self.revision}"


class SpecDecodeHeadDeclaration(CheckpointDeclaration):
    weight_checksum: SpecDecodeHeadChecksum | None = None
    target_checksum: ChecksumValue | None = None
    configuration: dict[str, JsonValue] | None = None

    @field_validator("weight_checksum", mode="before")
    @classmethod
    def parse_checksum(cls, value: object) -> SpecDecodeHeadChecksum | None:
        if value is None or isinstance(value, SpecDecodeHeadChecksum):
            return value
        if not isinstance(value, str):
            raise ValueError("Weight checksum must be an algorithm:revision string")
        return SpecDecodeHeadChecksum.from_string(value)

    @field_serializer("weight_checksum")
    def serialize_checksum(self, value: SpecDecodeHeadChecksum | None) -> str | None:
        return value.as_string() if value is not None else None

    @model_validator(mode="after")
    def consistent_identity(self) -> SpecDecodeHeadDeclaration:
        if self.target_checksum is not None or self.configuration is not None:
            if self.target_checksum is None or not self.configuration:
                raise ValueError(
                    "Configuration identity requires target_checksum and nonempty configuration"
                )
            if any(
                value is not None
                for value in (self.weight_checksum, self.repository, self.revision)
            ):
                raise ValueError(
                    "Weight and configuration identity forms cannot be combined"
                )
        if (
            self.weight_checksum is not None
            and self.revision is not None
            and self.revision != self.weight_checksum.revision
        ):
            raise ValueError("Weight checksum and revision must agree")
        return self

    def matches(
        self,
        repository: str | None = None,
        revision: str | None = None,
        *,
        target_checksum: str | None = None,
        configuration: JsonValue = None,
    ) -> bool:
        if target_checksum is not None:
            return (
                self.target_checksum == target_checksum
                and bool(self.configuration)
                and json_identity_equal(self.configuration, configuration)
            )
        if self.target_checksum is not None:
            return False
        if repository is None or revision is None:
            return False
        if self.weight_checksum is not None:
            return self.weight_checksum.revision == revision and (
                self.repository is None or self.repository == repository
            )
        return self.repository == repository and self.revision == revision


class AgenticSettings(BaseModel):
    enable_salt: StrictBool | None = None
    inject_tool_delay: StrictBool | None = None
    stop_issuing_on_first_user_complete: StrictBool | None = None
    num_trajectories_to_issue: StrictInt | None = None


class RuntimeSettings(BaseModel):
    """Runtime settings declared in a measurement point's ``point.yaml``."""

    model_config = ConfigDict(extra="ignore")
    load_pattern: LoadPatternType = LoadPatternType.CONCURRENCY
    min_duration_ms: int | None = None
    min_sample_count: int | None = None
    stream_all_chunks: bool = True

    class Runtime(BaseModel):
        """RNG seed configuration bound to a published seed set (§4.6)."""

        model_config = ConfigDict(extra="ignore")
        scheduler_rng_seed: int | None = None
        sample_index_rng_seed: int | None = None
        model_seed: int | None = None
        scheduler_random_seed: int | None = None
        dataloader_random_seed: int | None = None

        @model_validator(mode="after")
        def _lift_legacy_seed_names(self) -> RuntimeSettings.Runtime:
            """Fill canonical seed fields from aliases when canonical values are absent."""
            if (
                self.scheduler_rng_seed is None
                and self.scheduler_random_seed is not None
            ):
                self.scheduler_rng_seed = self.scheduler_random_seed
            if (
                self.sample_index_rng_seed is None
                and self.dataloader_random_seed is not None
            ):
                self.sample_index_rng_seed = self.dataloader_random_seed
            return self

    class WarmupLoadgen(BaseModel):
        """Loadgen warmup options."""

        model_config = ConfigDict(extra="ignore")
        salt: bool | None = None

    runtime: Runtime
    agentic_inference: AgenticSettings | None = None
    warmup: WarmupLoadgen | None = None


class PointConfig(BaseModel):
    """Parsed contents of a Pareto point's ``point.yaml`` (§8.3)."""

    model_config = ConfigDict(extra="ignore")
    concurrency: int
    region: str | None = None
    dataset: str = ""
    runtime_settings: RuntimeSettings
    warmup: WarmupSpec | None = None
    division: Division | None = None
    max_supported_concurrency: int | None = None
    model_name: str | None = None
    model_precision: str | None = None
    link_to_model: str | None = None
    link_to_model_transformation: str | None = None
    model_notes: str | None = None
    dataset_name: str | None = None
    dataset_type: str | None = None
    dataset_link: str | None = None
    offline: OfflineMode | None = None
    steady_state: SteadyState | None = None
    speculative_decoding: SpecDecodeHeadDeclaration | None = None
    spec_decode_head: SpecDecodeHeadDeclaration | None = Field(
        default=None, validation_alias=AliasChoices("spec_decode_head", "drafter")
    )
    checkpoint: CheckpointDeclaration | None = None
    accuracy: dict[str, dict[str, JsonValue]] | None = None
    nodes_used: list[NodesUsed] | None = None
    dp_shortfall: DpShortfall | None = None
    shared_src: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_spec_decode_head_aliases(cls, value: object) -> object:
        if (
            isinstance(value, dict)
            and "spec_decode_head" in value
            and "drafter" in value
            and not json_identity_equal(value["spec_decode_head"], value["drafter"])
        ):
            raise ValueError("spec_decode_head and its input alias must agree")
        return value

    shared_docs: str | None = None
    seed_set: str | None = None
    target_cohort: str | None = None
