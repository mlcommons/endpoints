"""System description model — §8.2 hardware and software metadata."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Division(str, Enum):
    """Submission division (§2)."""

    STANDARDIZED = "Standardized"
    SERVICED = "Serviced"
    RDI = "RDI"


class PublicationStatus(str, Enum):
    """Publication status category (§7)."""

    AVAILABLE = "Available"
    PREVIEW = "Preview"
    RDI = "RDI"


class SystemDescription(BaseModel):
    """Parsed contents of ``systems/<system_desc_id>.json`` (§8.2).

    Combines Endpoints-specific fields with the standard hardware metadata
    required by the general MLPerf submission rules (§5.7 / system_desc_id).

    Attributes:
        division: Submission division (Endpoints-specific).
        publication_status: Publication status category (Endpoints-specific).
        benchmark_model: Benchmark model name matching the supported model list.
        max_supported_concurrency: Declared maximum concurrency ``M`` (must be > 32).
        endpoint_url: URL or description of the endpoint under test.
        serving_framework: Inference serving framework and version.
        submitter: Organisation submitting these results.
        system_name: Human-readable name for the system under test.
        system_type: Deployment class — ``"datacenter"`` or ``"edge"``.
        system_type_detail: Free-form clarification of system type.
        number_of_nodes: Node count in the system.
        host_processors_per_node: Number of host CPU sockets per node.
        host_processor_model_name: CPU model name.
        host_processor_core_count: Physical core count per CPU (provide this
            or ``host_processor_vcpu_count``, or both).
        host_processor_vcpu_count: vCPU count per CPU (virtual/cloud systems).
        host_memory_capacity: Total host DRAM capacity (e.g. ``"768 GB"``).
        host_storage_type: Primary storage device type.
        host_storage_capacity: Total usable storage capacity.
        host_networking: Network technology used (e.g. ``"InfiniBand HDR"``).
        host_networking_topology: Network topology description.
        accelerators_per_node: Number of accelerator devices per node.
        accelerator_model_name: Accelerator model name.
        accelerator_memory_capacity: HBM or device memory capacity.
        operating_system: OS name and version string.
        host_processor_frequency: CPU frequency (optional).
        host_processor_caches: CPU cache hierarchy description (optional).
        host_processor_interconnect: CPU-to-CPU interconnect (optional).
        accelerator_frequency: Accelerator clock frequency (optional).
        accelerator_on_chip_memories: On-chip SRAM/cache description (optional).
        accelerator_host_interconnect: Host-to-accelerator bus (optional).
        accelerator_memory_configuration: HBM configuration details (optional).
        accelerator_interconnect: Accelerator-to-accelerator interconnect (optional).
        accelerator_interconnect_topology: Interconnect topology (optional).
        host_memory_configuration: DRAM configuration details (optional).
        host_network_card_count: NIC description / count (optional).
        cooling: Cooling method (optional).
        other_software_stack: Additional software stack notes (optional).
        hw_notes: Free-form hardware notes (optional).
        sw_notes: Free-form software notes (optional).
    """

    model_config = ConfigDict(extra="allow")

    # Endpoints-specific required fields
    division: Division
    publication_status: PublicationStatus
    benchmark_model: str
    max_supported_concurrency: int = Field(gt=32)  # M > 32 per §5.4
    endpoint_url: str
    serving_framework: str

    # General MLPerf submission rules — required hardware fields
    submitter: str
    system_name: str
    system_type: Literal["datacenter", "edge"]
    system_type_detail: str
    number_of_nodes: int = Field(gt=0)
    host_processors_per_node: int = Field(gt=0)
    host_processor_model_name: str
    host_processor_core_count: int | None = None
    host_processor_vcpu_count: int | None = None
    host_memory_capacity: str
    host_storage_type: str
    host_storage_capacity: str
    host_networking: str
    host_networking_topology: str
    accelerators_per_node: int = Field(ge=0)
    accelerator_model_name: str
    accelerator_memory_capacity: str
    operating_system: str

    # General MLPerf submission rules — optional hardware fields
    host_processor_frequency: str = ""
    host_processor_caches: str = ""
    host_processor_interconnect: str = ""
    accelerator_frequency: str = ""
    accelerator_on_chip_memories: str = ""
    accelerator_host_interconnect: str = ""
    accelerator_memory_configuration: str = ""
    accelerator_interconnect: str = ""
    accelerator_interconnect_topology: str = ""
    host_memory_configuration: str = ""
    host_network_card_count: str = ""
    cooling: str = ""
    other_software_stack: str = ""
    hw_notes: str = ""
    sw_notes: str = ""

    @model_validator(mode="after")
    def _require_core_or_vcpu(self) -> SystemDescription:
        """At least one of host_processor_core_count or host_processor_vcpu_count must be set."""
        if self.host_processor_core_count is None and self.host_processor_vcpu_count is None:
            raise ValueError(
                "At least one of host_processor_core_count"
                " or host_processor_vcpu_count must be provided"
            )
        return self
