# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..types import Division


class SystemAvailabilityStatus(StrEnum):
    """System availability status (§8.2)."""

    AVAILABLE = "Available"
    PREVIEW = "Preview"
    RDI = "RDI"


_ACCELERATOR_FIELDS = (
    "accelerator_model_name",
    "accelerators_per_node",
    "accelerator_memory_capacity",
    "accelerator_memory_type",
    "accelerator_interconnect",
    "accelerator_host_interconnect",
)


class AcceleratorInfo(BaseModel):
    """One accelerator configuration within a node type (§8.2.1)."""

    model_config = ConfigDict(extra="ignore")
    accelerator_model_name: str | None = None
    accelerators_per_node: int | None = None
    accelerator_memory_capacity: str | None = None
    accelerator_memory_type: str | None = None
    accelerator_interconnect: str | None = None
    accelerator_host_interconnect: str | None = None


class NodeType(BaseModel):
    """Per-node hardware and software configuration (§8.2.1)."""

    model_config = ConfigDict(extra="ignore")
    system_node_ensemble_id: int | None = None
    number_of_nodes: int | None = None
    host_processor_model_name: str | None = None
    host_processors_per_node: int | None = None
    host_processor_core_count: int | None = None
    host_processor_vcpu_count: int | None = None
    host_memory_capacity: str | None = None
    host_memory_configuration: str
    accelerator_info: list[AcceleratorInfo] = Field(default_factory=list)
    host_network_card_count: str
    host_networking: str | None = None
    host_storage_capacity: str | None = None
    host_storage_type: str | None = None
    other_hardware: str | None = None
    hw_notes: str | None = None
    cooling: str | None = None
    inference_backend: str | None = None
    driver: str
    operating_system: str | None = None
    filesystem: str
    container_link: str | None = None
    other_software_stack: str | None = None
    sw_notes: str | None = None

    @field_validator("system_node_ensemble_id", mode="before")
    @classmethod
    def _coerce_to_int(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return v
        return v

    @model_validator(mode="before")
    @classmethod
    def _lift_flat_accelerator_fields(cls, data: object) -> object:
        """Normalize flat accelerator fields into accelerator_info entries."""
        if not isinstance(data, dict) or data.get("accelerator_info"):
            return data
        flat = {
            name: data[name]
            for name in _ACCELERATOR_FIELDS
            if data.get(name) is not None
        }
        if flat:
            data = {**data, "accelerator_info": [flat]}
        return data

    @model_validator(mode="after")
    def _require_core_or_vcpu_count(self) -> NodeType:
        """A node must disclose at least one of physical core count or vCPU count."""
        if (
            self.host_processor_core_count is None
            and self.host_processor_vcpu_count is None
        ):
            raise ValueError(
                "a node type must give host_processor_core_count or host_processor_vcpu_count"
            )
        return self


class ConfigSummary(BaseModel):
    """Structured serving configuration and parallelism disclosure."""

    model_config = ConfigDict(extra="ignore")
    disaggregated: bool | None = None
    expert_parallel: int | None = None
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    data_parallel: int | None = None
    batch: int | None = None


_ConfigSummaryStr = Annotated[str, StringConstraints(min_length=4)]
_AVAILABILITY_SPELLINGS = (
    "publication_status",
    "system_availability_status",
    "availability_status",
)


class DatasetAccuracyScores(BaseModel):
    """Per-dataset accuracy scores for ``measured_accuracy_score`` (§8.2)."""

    model_config = ConfigDict(extra="ignore")
    scores: dict[str, StrictFloat]


class SystemDescription(BaseModel):
    """Parsed per-point system description (§8.2)."""

    model_config = ConfigDict(extra="ignore")
    submitter_org_names: str | None = None
    submitter_contact: str | None = None
    submission_id: str | None = None
    submission_date: str | None = None
    publish_date: str | None = None
    cooling: str | None = None
    system_name: str
    system_category: str | None = None
    publication_status: SystemAvailabilityStatus
    max_supported_concurrency: int
    system_size: str
    system_node_ensemble_count: int
    system_node_ensemble_total: int
    serving_framework: str | None = None
    shortened_system_name: str | None = None
    endpoint_url: str | None = None
    node_types: list[NodeType]
    node_config: str | None = None
    config_summary: ConfigSummary | _ConfigSummaryStr | None = None
    config_summary_notes: str | None = None
    disaggregated: bool | None = None
    expert_parallel: int | None = None
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    data_parallel: int | None = None
    batch: int | None = None
    link_config: str | None = None
    tps_utilization: float | None = None
    division: Division
    model_precision: str | None = None
    link_to_model: str | None = None
    link_to_model_transformation: str | None = None
    model_notes: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    input_token_average: float | None = None
    output_token_average: float | None = None
    dataset_type: str | None = None
    dataset_link: str | None = None
    measured_accuracy_score: str | float | dict[str, DatasetAccuracyScores] | None = (
        None
    )

    @field_validator("division", mode="before")
    @classmethod
    def _coerce_division(cls, v: object) -> object:
        if isinstance(v, str):
            mapping = {
                "standardized": "Standardized",
                "serviced": "Serviced",
                "rdi": "RDI",
            }
            normalized = mapping.get(v.strip().lower())
            if normalized is not None:
                return normalized
            raise ValueError(
                f"unknown division {v!r}; must be one of standardized, serviced, rdi"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _reconcile_availability_spellings(cls, data: object) -> object:
        """Accept availability field aliases and reject conflicting declarations."""
        if not isinstance(data, dict):
            return data
        present = {
            name: data[name]
            for name in _AVAILABILITY_SPELLINGS
            if data.get(name) not in (None, "")
        }
        if not present:
            return data
        distinct = {str(v).strip().lower() for v in present.values()}
        if len(distinct) > 1:
            pairs = ", ".join((f"{k}={v!r}" for k, v in sorted(present.items())))
            raise ValueError(f"the availability fields disagree ({pairs})")
        if data.get("publication_status") in (None, ""):
            data = {**data, "publication_status": next(iter(present.values()))}
        return data

    @field_validator("publication_status", mode="before")
    @classmethod
    def _coerce_availability(cls, v: object) -> object:
        if isinstance(v, str):
            mapping = {"available": "Available", "preview": "Preview", "rdi": "RDI"}
            normalized = mapping.get(v.strip().lower())
            if normalized is not None:
                return normalized
            raise ValueError(
                f"unknown availability {v!r}; must be one of available, preview, rdi"
            )
        return v

    @field_validator("input_token_average", "output_token_average", mode="before")
    @classmethod
    def _coerce_tokens_to_float(cls, v: object) -> object:
        if isinstance(v, str) and v:
            try:
                return float(v)
            except ValueError:
                return v
        return v

    @field_validator("measured_accuracy_score", mode="before")
    @classmethod
    def _coerce_empty_accuracy_to_none(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v
