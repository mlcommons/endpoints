# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Provisioned power arithmetic driven by cohort policy values."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import (
    Components,
    NodeSet,
    PowerComputation,
    ScaleUpNetwork,
    SetPower,
    SourcedValue,
    SystemPower,
)


@dataclass(frozen=True)
class PowerDefault:
    """One Appendix D value: watts, and the subsection it came from."""

    watts: float
    source: str


_W_PER_KW = 1000.0


def _catalog_default(
    kind: str,
    catalog: Mapping[str, Any],
    model: str | None = None,
    cores: int | None = None,
    cabling: str | None = None,
) -> PowerDefault | None:
    defaults = catalog.get("defaults", {})
    text = (model or "").lower()
    if kind == "nic":
        watts = defaults.get("nic_watts")
        return PowerDefault(watts, "D.4") if watts is not None else None
    if kind == "cpu":
        if cores is None:
            match = re.search("(\\d+)[\\s-]*core", text)
            cores = int(match.group(1)) if match else None
        architecture = (
            "arm"
            if re.search(
                "\\b(grace|neoverse|arm|graviton|axion|ampere|altra|agi)\\b", text
            )
            else "x86"
            if re.search("\\b(xeon|epyc|intel|amd|x86|x86_64)\\b", text)
            else None
        )
        if architecture is None or cores is None:
            return None
        for row in defaults.get("cpu", ()):
            if row.get("architecture") == architecture and row.get(
                "minimum_cores", 0
            ) <= cores <= row.get("maximum_cores", float("inf")):
                return PowerDefault(row["watts"], "D.2")
        return None
    key = "accelerators" if kind == "accelerator" else "switches"
    names = defaults.get(key, {})
    squashed = re.sub("[^a-z0-9]", "", text)
    for name in sorted(names, key=len, reverse=True):
        if name in squashed:
            watts = names[name] if kind == "accelerator" else names[name].get(cabling)
            return (
                PowerDefault(watts, "D.3" if kind == "accelerator" else "D.4")
                if watts is not None
                else None
            )
    return None


class _Tally:
    """Collects one figure's worth of watts, defaults and gaps while summing."""

    def __init__(self) -> None:
        self.estimated: list[str] = []
        self.gaps: list[str] = []
        self.problems: list[str] = []

    def watts(
        self,
        label: str,
        value: SourcedValue | None,
        fallback: PowerDefault | None = None,
    ) -> float | None:
        """*value* in watts, or the Appendix D *fallback* where it is absent."""
        if value is not None:
            watts = value.watts
            if watts is None:
                self.problems.append(
                    f"{label} is an energy figure; give value_w or value_kw"
                )
                return None
            if value.is_default:
                self.estimated.append(f"{label} ({value.source} default)")
            return watts
        if fallback is not None:
            self.estimated.append(
                f"{label} (absent; auto-populated from {fallback.source})"
            )
            return fallback.watts
        self.gaps.append(f"{label} is absent and Appendix D has no default for it")
        return None


class PowerCalculator:
    def __init__(self, descriptor: SystemPower, policy: Mapping[str, Any]) -> None:
        self.descriptor = descriptor
        self.policy = policy

    @property
    def overhead_fraction(self) -> float:
        """§4.5.2's overhead fraction for the declared ``cooling``."""
        return self.policy["cooling_overhead"][self.descriptor.cooling]

    def compute(
        self, cores_by_ensemble: dict[int, int] | None = None
    ) -> PowerComputation:
        """E.5's arithmetic, with Appendix D filling what the descriptor leaves absent."""
        cores = cores_by_ensemble or {}
        out = PowerComputation(overhead_fraction=self.overhead_fraction)
        tally = _Tally()
        ids = [s.node_set_id for s in self.descriptor.node_sets]
        if len(set(ids)) != len(ids):
            out.problems.append("node_set_id values must be unique within the file")
        major: dict[int, float] = {}
        published: dict[int, float] = {}
        for node_set in self.descriptor.node_sets:
            set_major, set_published = self._add_node_set(node_set, cores, out, tally)
            major[node_set.node_set_id] = set_major
            published[node_set.node_set_id] = set_published
        nic_per_node_w = self._add_scale_out(out, tally)
        for node_set in self.descriptor.node_sets:
            if self._carries_nics(node_set):
                major[node_set.node_set_id] += (
                    node_set.nodes_provisioned * nic_per_node_w
                )
        out.major_components_w = sum(major.values())
        out.published_node_power_w = sum(published.values())
        out.other_components_w = out.overhead_fraction * out.major_components_w
        out.sets = [
            SetPower(
                node_set_id=s.node_set_id,
                system_node_ensemble_id=s.system_node_ensemble_id,
                nodes_provisioned=s.nodes_provisioned,
                power_w=(1 + out.overhead_fraction) * major[s.node_set_id]
                + published[s.node_set_id],
                accelerators_per_node=s.components.accelerator.count_per_node
                if s.components is not None and s.components.accelerator is not None
                else None,
            )
            for s in self.descriptor.node_sets
        ]
        out.problems.extend(tally.problems)
        declared = self.descriptor.declared_provisioned_power
        if declared is not None:
            out.declared = True
            watts = declared.watts
            if watts is None:
                out.problems.append(
                    "declared_provisioned_power must give value_kw or value_w"
                )
            else:
                out.provisioned_power_kw = round(watts / _W_PER_KW, 2)
                if declared.is_default:
                    out.estimated.append("declared_provisioned_power (mlc_default)")
        else:
            out.problems.extend(tally.gaps)
            out.estimated.extend(tally.estimated)
            if not out.problems:
                out.provisioned_power_kw = round(
                    out.total_system_power_w / _W_PER_KW, 2
                )
        if not out.problems and (not tally.gaps):
            out.problems.extend(self._computed_disagreements(out))
        if out.problems:
            out.provisioned_power_kw = None
        return out

    def _add_node_set(
        self,
        node_set: NodeSet,
        cores: dict[int, int],
        out: PowerComputation,
        tally: _Tally,
    ) -> tuple[float, float]:
        """One set's ``(major, published)`` watts. A set that cannot be costed adds 0."""
        label = f"node_sets[{node_set.node_set_id}]"
        y = node_set.nodes_provisioned
        if node_set.power_method == "component_sum":
            if node_set.components is None:
                out.problems.append(f"{label}: component_sum requires components")
                return (0.0, 0.0)
            per_node = self._per_node_w(
                node_set.components,
                cores.get(node_set.system_node_ensemble_id),
                label,
                tally,
            )
            return (0.0 if per_node is None else y * per_node, 0.0)
        if node_set.published_power is None:
            out.problems.append(
                f"{label}: {node_set.power_method} requires published_power"
            )
            return (0.0, 0.0)
        published = tally.watts(f"{label}.published_power", node_set.published_power)
        if published is None:
            return (0.0, 0.0)
        if node_set.power_method == "published_system":
            return (0.0, y * published)
        n = node_set.nodes_in_published_rack
        if n is None or n <= y:
            out.problems.append(
                f"{label}: node_scaling requires nodes_in_published_rack greater than nodes_provisioned ({y}), got {n}"
            )
            return (0.0, 0.0)
        return (0.0, published * (y / n))

    def _per_node_w(
        self, components: Components, cores: int | None, label: str, tally: _Tally
    ) -> float | None:
        """One node's CPU + accelerator + scale-up, in watts."""
        label = f"{label}.components"
        parts: list[float | None] = []
        cpu, acc = (components.cpu, components.accelerator)
        if components.combined_cpu_accelerator is not None:
            if cpu and cpu.tdp_per_unit or (acc and acc.tdp_per_unit):
                tally.problems.append(
                    f"{label}: combined_cpu_accelerator is mutually exclusive with cpu.tdp_per_unit and accelerator.tdp_per_unit"
                )
            parts.append(
                tally.watts(
                    f"{label}.combined_cpu_accelerator",
                    components.combined_cpu_accelerator,
                )
            )
        else:
            if cpu is None or acc is None:
                tally.problems.append(
                    f"{label}: needs cpu and accelerator, or combined_cpu_accelerator"
                )
                return None
            cpu_w = tally.watts(
                f"{label}.cpu.tdp_per_unit",
                cpu.tdp_per_unit,
                _catalog_default("cpu", self.policy, cpu.model, cores),
            )
            acc_w = tally.watts(
                f"{label}.accelerator.tdp_per_unit",
                acc.tdp_per_unit,
                _catalog_default("accelerator", self.policy, acc.model),
            )
            parts.append(None if cpu_w is None else cpu.count_per_node * cpu_w)
            parts.append(None if acc_w is None else acc.count_per_node * acc_w)
        up = components.scale_up_network
        if up is None:
            tally.problems.append(
                f"{label}: scale_up_network is required; use method none where there is no scale-up switch"
            )
            return None
        parts.append(
            PowerCalculator._scale_up_w(up, f"{label}.scale_up_network", tally)
        )
        if any(p is None for p in parts):
            return None
        return sum(p for p in parts if p is not None)

    @staticmethod
    def _scale_up_w(up: ScaleUpNetwork, label: str, tally: _Tally) -> float | None:
        if up.method == "none":
            return 0.0
        if up.method == "declared_tdp":
            if up.switch_count is None:
                tally.problems.append(f"{label}: declared_tdp requires switch_count")
                return None
            per_switch = tally.watts(f"{label}.tdp_per_switch", up.tdp_per_switch)
            return None if per_switch is None else up.switch_count * per_switch
        if up.aggregate_bandwidth_tbps is None:
            tally.problems.append(
                f"{label}: bandwidth_estimate requires aggregate_bandwidth_tbps"
            )
            return None
        energy = up.energy_per_bit_pj
        if energy is None:
            tally.gaps.append(
                f"{label}.energy_per_bit_pj is absent; D.1's references depend on the link protocol, so there is no default to apply"
            )
            return None
        if energy.value_pj is None:
            tally.problems.append(f"{label}.energy_per_bit_pj must give value_pj")
            return None
        if energy.is_default:
            tally.estimated.append(
                f"{label}.energy_per_bit_pj ({energy.source} default)"
            )
        return up.aggregate_bandwidth_tbps * 8.0 * energy.value_pj

    def _carries_nics(self, node_set: NodeSet) -> bool:
        """Whether counted scale-out NICs are added to *node_set*'s power."""
        nics = self.descriptor.scale_out.nics
        if not self.descriptor.scale_out.present or nics is None or (not nics.counted):
            return False
        return node_set.power_method == "component_sum" or bool(
            nics.excluded_from_published_power
        )

    def _check_fabric_declared(self, out: PowerComputation) -> None:
        """E.4: ``present`` follows from whether the nodes need a scale-out fabric."""
        nodes = sum(s.nodes_provisioned for s in self.descriptor.node_sets)
        if self.descriptor.scale_out.present:
            if nodes == 1:
                out.warnings.append(
                    "scale_out.present is true for a single-node submission; E.4 makes it false there, and the switch power is counted against this node"
                )
            return
        if nodes == 1:
            return
        no_scale_up = all(
            s.power_method == "component_sum"
            and s.components is not None
            and (s.components.scale_up_network is not None)
            and (s.components.scale_up_network.method == "none")
            for s in self.descriptor.node_sets
        )
        if no_scale_up:
            out.problems.append(
                f"scale_out.present is false, but the system has {nodes} nodes and no scale-up network joins them; E.4 allows false only for a single node or nodes joined by a fabric already counted in scale_up_network"
            )
        else:
            out.warnings.append(
                f"scale_out.present is false for {nodes} nodes; E.4 allows that only where the nodes are joined by a fabric already counted in scale_up_network, which the descriptor cannot show"
            )

    def _add_scale_out(self, out: PowerComputation, tally: _Tally) -> float:
        """Add the switches to *out*; return the counted NIC watts **per node**."""
        fabric = self.descriptor.scale_out
        nic_per_node_w = 0.0
        self._check_fabric_declared(out)
        if not fabric.present:
            return nic_per_node_w
        missing = [
            name
            for name in ("cabling", "required_bandwidth_tbps", "nics", "switches")
            if getattr(fabric, name) is None
        ]
        if missing:
            out.problems.append(
                "scale_out.present is true but {missing} is missing".format(
                    missing=", ".join(missing)
                )
            )
            return nic_per_node_w
        assert fabric.nics is not None and fabric.switches is not None
        assert fabric.required_bandwidth_tbps is not None
        nodes = sum(s.nodes_provisioned for s in self.descriptor.node_sets)
        nics = fabric.nics
        by_formula = any(
            s.power_method == "component_sum" for s in self.descriptor.node_sets
        )
        if nics.counted:
            if not by_formula and (not nics.excluded_from_published_power):
                out.problems.append(
                    "scale_out.nics.counted is true, but node power comes from a published specification, which §4.5.2 assumes includes the adapters. Set counted to false, or evidence the exclusion in excluded_from_published_power"
                )
            per_nic = tally.watts(
                "scale_out.nics.tdp_per_nic",
                nics.tdp_per_nic,
                _catalog_default("nic", self.policy),
            )
            if per_nic is not None:
                nic_per_node_w = nics.count_per_node * per_nic
        elif by_formula:
            out.problems.append(
                "scale_out.nics.counted is false, but node power is built with the MLC formula, which has no NIC term — §4.5.2 says the NICs MUST be included"
            )
        from_nics = nodes * nics.count_per_node * nics.bandwidth_per_nic_gbps / 1000.0
        required = max(fabric.required_bandwidth_tbps, from_nics)
        if fabric.required_bandwidth_tbps < from_nics - 1e-06:
            out.problems.append(
                f"scale_out.required_bandwidth_tbps is {fabric.required_bandwidth_tbps:g}, below the {from_nics:g} Tb/s of NIC bandwidth E.4 defines it as ({nodes} nodes × {nics.count_per_node} NICs × {nics.bandwidth_per_nic_gbps:g} Gb/s)"
            )
        elif fabric.required_bandwidth_tbps > from_nics + 1e-06:
            out.warnings.append(
                f"scale_out.required_bandwidth_tbps is {fabric.required_bandwidth_tbps:g}, above the {from_nics:g} Tb/s the NICs carry ({nodes} nodes × {nics.count_per_node} NICs × {nics.bandwidth_per_nic_gbps:g} Gb/s)"
            )
        offered = 0.0
        for index, switch in enumerate(fabric.switches):
            label = f"scale_out.switches[{index}].power_per_switch"
            watts = tally.watts(
                label,
                switch.power_per_switch,
                _catalog_default(
                    "switch", self.policy, switch.model, cabling=fabric.cabling
                ),
            )
            if watts is not None:
                out.scale_out_switch_power_w += switch.count * watts
            offered += switch.count * switch.bandwidth_tbps
        if offered < required - 1e-06:
            out.problems.append(
                f"scale-out switches offer {offered:g} Tb/s, below the required {required:g} Tb/s"
            )
        return nic_per_node_w

    def _computed_disagreements(self, out: PowerComputation) -> list[str]:
        """E.7: a submitter-supplied figure that disagrees with the recomputation."""
        problems: list[str] = []
        stated = self.descriptor.computed
        if stated is not None:
            ours = {
                "major_components_w": out.major_components_w,
                "other_components_w": out.other_components_w,
                "published_node_power_w": out.published_node_power_w,
                "scale_out_switch_power_w": out.scale_out_switch_power_w,
                "total_system_power_w": out.total_system_power_w,
            }
            for name, value in ours.items():
                theirs = getattr(stated, name)
                if (
                    theirs is not None
                    and abs(theirs - value) > self.policy["computed_tolerance_w"]
                ):
                    problems.append(
                        f"computed.{name} is {theirs:g}, recomputed {value:g}"
                    )
            if (
                stated.overhead_fraction is not None
                and abs(stated.overhead_fraction - out.overhead_fraction) > 1e-09
            ):
                problems.append(
                    f"computed.overhead_fraction is {stated.overhead_fraction:g}, but cooling {self.descriptor.cooling!r} fixes it at {out.overhead_fraction:g}"
                )
        kw = out.provisioned_power_kw
        stated_kw = self.descriptor.provisioned_power_kw
        if (
            stated_kw is not None
            and kw is not None
            and (abs(stated_kw - kw) > self.policy["computed_tolerance_kw"])
        ):
            problems.append(
                f"provisioned_power_kw is {stated_kw:g}, recomputed {kw:.2f}"
            )
        return problems
