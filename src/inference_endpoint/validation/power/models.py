# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Power descriptor schema and calculation results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..types import Cooling

_W_PER_KW = 1000.0

SourceType = Literal["vendor_spec", "publication", "public_statement", "mlc_default"]
_APPENDIX_D = frozenset({"D.1", "D.2", "D.3", "D.4"})


class SourcedValue(BaseModel):
    """E.1: a power figure and the public, verifiable reference behind it."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    value_w: float | None = Field(default=None, ge=0)
    value_kw: float | None = Field(default=None, ge=0)
    value_pj: float | None = Field(default=None, ge=0)
    source_type: SourceType
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _one_value_and_a_real_source(self) -> SourcedValue:
        values = [
            v for v in (self.value_w, self.value_kw, self.value_pj) if v is not None
        ]
        if len(values) != 1:
            raise ValueError("give exactly one of value_w, value_kw or value_pj")
        if self.source_type == "mlc_default":
            if self.source not in _APPENDIX_D:
                raise ValueError(
                    "an mlc_default source names its Appendix D subsection ({subsections}), not {source!r}".format(
                        subsections=", ".join(sorted(_APPENDIX_D)), source=self.source
                    )
                )
        elif not self.source.startswith(("https://", "http://")):
            raise ValueError(
                f"a {self.source_type} source must be a resolvable URL, not {self.source!r}"
            )
        return self

    @property
    def watts(self) -> float | None:
        """The value in watts, or ``None`` for an energy-per-bit figure."""
        if self.value_w is not None:
            return self.value_w
        if self.value_kw is not None:
            return self.value_kw * _W_PER_KW
        return None

    @property
    def is_default(self) -> bool:
        """True for an Appendix D value, which sets the estimated-power tag."""
        return self.source_type == "mlc_default"


class _Open(BaseModel):
    """Appendix E objects tolerate keys the schema does not name — ``notes`` and the like."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class Processor(_Open):
    """E.3.1's ``cpu`` block. Counts are what is provisioned, not what the chassis holds."""

    model: str | None = None
    count_per_node: int = Field(ge=0)
    tdp_per_unit: SourcedValue | None = None


class Accelerator(_Open):
    """E.3.1's ``accelerator`` block."""

    model: str | None = None
    count_per_node: int = Field(ge=0)
    tdp_per_unit: SourcedValue | None = None
    below_rated_tdp: dict[str, object] | None = None


class ScaleUpNetwork(_Open):
    """E.3.1's ``scale_up_network`` block, costed by one of three methods."""

    method: Literal["declared_tdp", "bandwidth_estimate", "none"]
    switch_count: int | None = Field(default=None, ge=0)
    tdp_per_switch: SourcedValue | None = None
    aggregate_bandwidth_tbps: float | None = Field(default=None, ge=0)
    energy_per_bit_pj: SourcedValue | None = None


class Components(_Open):
    """E.3.1: a node's power built from its parts, per node."""

    cpu: Processor | None = None
    accelerator: Accelerator | None = None
    combined_cpu_accelerator: SourcedValue | None = None
    scale_up_network: ScaleUpNetwork | None = None


class NodeSet(_Open):
    """E.3: one set of identical nodes, and how its power was established."""

    node_set_id: int
    system_node_ensemble_id: int
    nodes_provisioned: int = Field(gt=0)
    power_method: Literal["component_sum", "published_system", "node_scaling"]
    published_power: SourcedValue | None = None
    nodes_in_published_rack: int | None = None
    components: Components | None = None


class Nics(_Open):
    """E.4's scale-out adapters."""

    count_per_node: int = Field(ge=0)
    bandwidth_per_nic_gbps: float = Field(ge=0)
    counted: bool
    tdp_per_nic: SourcedValue | None = None
    excluded_from_published_power: str | None = None


class Switch(_Open):
    """One scale-out switch model in E.4's ``switches`` array."""

    model: str
    count: int = Field(gt=0)
    bandwidth_tbps: float = Field(ge=0)
    power_per_switch: SourcedValue | None = None


class ScaleOut(_Open):
    """E.4: the scale-out fabric. Everything but ``present`` is conditional on it."""

    present: bool
    cabling: Literal["passive", "active_optical"] | None = None
    required_bandwidth_tbps: float | None = Field(default=None, ge=0)
    nics: Nics | None = None
    switches: list[Switch] | None = None


class Computed(_Open):
    """E.5's derived arithmetic, as a submitter wrote it. The checker's values govern."""

    major_components_w: float | None = None
    overhead_fraction: float | None = None
    other_components_w: float | None = None
    published_node_power_w: float | None = None
    scale_out_switch_power_w: float | None = None
    total_system_power_w: float | None = None


@dataclass(frozen=True)
class SetPower:
    """One node set's share of provisioned power — §4.5.3's ``P_s`` and ``N_s``."""

    node_set_id: int
    system_node_ensemble_id: int
    nodes_provisioned: int
    power_w: float
    accelerators_per_node: int | None


@dataclass
class PowerComputation:
    """Computed provisioned power and calculation findings."""

    major_components_w: float = 0.0
    overhead_fraction: float = 0.0
    other_components_w: float = 0.0
    published_node_power_w: float = 0.0
    scale_out_switch_power_w: float = 0.0
    provisioned_power_kw: float | None = None
    sets: list[SetPower] = field(default_factory=list)
    declared: bool = False
    estimated: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_system_power_w(self) -> float:
        """E.5's ``total_system_power_w``: the four terms, unrounded."""
        return (
            self.major_components_w
            + self.other_components_w
            + self.published_node_power_w
            + self.scale_out_switch_power_w
        )


class SystemPower(_Open):
    """Parsed contents of a system's ``system_power.json`` (Appendix E.2)."""

    system_desc_id: str = Field(min_length=1)
    cooling: Cooling
    node_sets: list[NodeSet] = Field(min_length=1)
    scale_out: ScaleOut
    declared_provisioned_power: SourcedValue | None = None
    computed: Computed | None = None
    provisioned_power_kw: float | None = None
    mlc_estimated_power: bool | None = None
    notes: str | None = None


class PowerEvidence(TypedDict, total=False):
    computation: PowerComputation
    descriptor: SystemPower
    problems: list[str]
    requirements: Mapping[str, object]
