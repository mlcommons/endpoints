# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven power checks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from ..artifacts import Artifacts, PointArtifacts, read_mapping
from ..evaluator_base import Evaluator
from ..operations import PowerOperation
from ..planner import PlannedCheck, plan_checks
from ..power.calculation import PowerCalculator
from ..power.models import PowerComputation, PowerEvidence, SystemPower
from ..results import CheckResult
from ..types import Cooling, Decision, EvidenceKey, Scope
from ..vocabulary import CheckKind
from .helpers import finding, unsupported, validate_modes, warning


def power_entry(artifacts: Artifacts, point: PointArtifacts) -> PowerEvidence:
    return artifacts.derived.power.get(str(point.path.parent.parent), {})


def nodes_for(
    point: PointArtifacts, computation: PowerComputation
) -> tuple[dict[int, int], list[str]]:
    declared = point.get_config("nodes_used")
    nodes = {}
    problems = []
    if declared is not None:
        for entry in declared:
            ensemble, count = entry.get("system_node_ensemble_id"), entry.get("nodes")
            candidates = [
                s for s in computation.sets if s.system_node_ensemble_id == ensemble
            ]
            if len(candidates) != 1 or ensemble in nodes:
                problems.append(f"Unknown or duplicate node ensemble {ensemble}")
                continue
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or not 1 <= count <= candidates[0].nodes_provisioned
            ):
                problems.append(
                    f"Invalid engaged node count {count!r} for ensemble {ensemble}"
                )
                continue
            nodes[ensemble] = count
    return {
        s.node_set_id: nodes.get(s.system_node_ensemble_id, s.nodes_provisioned)
        for s in computation.sets
    }, problems


def compute_descriptor(check: PlannedCheck, artifacts: Artifacts) -> None:
    """Compute only the selected system using its effective descriptor rule."""
    system_id = check.subject.id
    rule = check.rule
    existing = artifacts.derived.power.get(system_id, {})
    if existing.get("requirements") == rule.requirements:
        return
    point_ids = artifacts.index.systems.get(system_id, ())
    for point_id in point_ids:
        point = artifacts.index.points[point_id]
        point.derived.power_kw = None
        point.context = point.context.model_copy(
            update={
                "available": point.context.available
                - {EvidenceKey.POWER_COMPUTATION, EvidenceKey.POINT_POWER}
            }
        )
    constants = artifacts.resolve(rule.requirements["constants"], check)
    path = Path(system_id) / rule.requirements["artifact"]
    if not path.exists():
        artifacts.derived.power[system_id] = {
            "problems": [f"Power descriptor {path.name} is missing"]
        }
        return
    try:
        descriptor = SystemPower.model_validate(read_mapping(path))
        members = [artifacts.index.points[id] for id in point_ids]
        nodes: list[dict[str, Any]] = next(
            (p.get_system("node_types", []) for p in members if p.system is not None),
            [],
        )
        cores = {
            int(n["system_node_ensemble_id"]): n["host_processor_core_count"]
            for n in nodes
            if "system_node_ensemble_id" in n
            and isinstance(n.get("host_processor_core_count"), int)
        }
        computation = PowerCalculator(descriptor, constants).compute(cores)
        for node_set in computation.sets:
            if node_set.accelerators_per_node is None:
                node = next(
                    (
                        n
                        for n in nodes
                        if str(n.get("system_node_ensemble_id"))
                        == str(node_set.system_node_ensemble_id)
                    ),
                    {},
                )
                counts = [
                    a.get("accelerators_per_node")
                    for a in node.get("accelerator_info", ())
                    if isinstance(a, Mapping)
                ]
                count = (
                    sum(cast(int, count) for count in counts)
                    if counts and all(isinstance(c, int) for c in counts)
                    else node.get(
                        "accelerators_per_node", node.get("accelerator_count")
                    )
                )
                if isinstance(count, int):
                    computation.sets[computation.sets.index(node_set)] = replace(
                        node_set, accelerators_per_node=count
                    )
        if rule.requirements.get("cooling_must_match_system_description"):
            declared = [str(p.get_system("cooling", "")) for p in members]
            declared.extend(str(n.get("cooling", "")) for n in nodes)
            labels: set[Cooling] = set()
            for text in declared:
                words = set(re.findall(r"[a-z]+", text.lower()))
                if "mixed" in words:
                    labels.update((Cooling.AIR, Cooling.LIQUID))
                if words & {"liquid", "water", "immersion"}:
                    labels.add(Cooling.LIQUID)
                if "air" in words:
                    labels.add(Cooling.AIR)
            if labels and descriptor.cooling != (
                Cooling.MIXED if len(labels) > 1 else next(iter(labels))
            ):
                computation.problems.append(
                    "Power cooling disagrees with system description"
                )
                computation.provisioned_power_kw = None
        entry: PowerEvidence = {
            "computation": computation,
            "problems": computation.problems,
            "descriptor": descriptor,
            "requirements": rule.requirements,
        }
        artifacts.derived.power[system_id] = entry
        if computation.provisioned_power_kw is None:
            return
        for point in members:
            point.context = point.context.model_copy(
                update={
                    "available": point.context.available
                    | {EvidenceKey.POWER_COMPUTATION}
                }
            )
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        artifacts.derived.power[system_id] = {
            "problems": [f"Power descriptor could not be computed: {exc}"]
        }


def refresh_power_subjects(artifacts: Artifacts) -> None:
    artifacts.index.subjects = [
        artifacts.index.points[subject.id].context
        if subject.scope is Scope.POINT
        else subject.model_copy(
            update={
                "members": tuple(
                    artifacts.index.points[id].context
                    for id in artifacts.index.curves[subject.id]
                )
            }
        )
        if subject.scope is Scope.CURVE
        else subject
        for subject in artifacts.index.subjects
    ]


def prepare(artifacts: Artifacts) -> None:
    """Plan descriptor and point-power stages before publishing derived evidence."""
    artifacts.derived.power.clear()
    for point in artifacts.index.points.values():
        point.derived.power_kw = None
        point.context = point.context.model_copy(
            update={
                "available": point.context.available
                - {EvidenceKey.POWER_COMPUTATION, EvidenceKey.POINT_POWER}
            }
        )
    refresh_power_subjects(artifacts)
    for scope, operation in (
        (Scope.SYSTEM, PowerOperation.DESCRIPTOR),
        (Scope.POINT, PowerOperation.ENGAGED_NODES),
    ):
        subjects = [
            subject for subject in artifacts.index.subjects if subject.scope is scope
        ]
        if not subjects:
            continue
        stage = plan_checks(artifacts.policy, subjects)
        for check in stage.checks:
            if (
                check.decision is Decision.READY
                and check.rule.kind is CheckKind.POWER
                and check.rule.requirements.get("operation") == operation
            ):
                try:
                    PowerEvaluator()(check, artifacts)
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    ArithmeticError,
                    AttributeError,
                ):
                    # The main evaluation emits the blocked finding. A failed
                    # prerequisite stage must not publish partial derived facts.
                    failed = {EvidenceKey.POINT_POWER}
                    if operation is PowerOperation.DESCRIPTOR:
                        artifacts.derived.power.pop(check.subject.id, None)
                        failed.add(EvidenceKey.POWER_COMPUTATION)
                    for point in artifacts.members(check):
                        point.derived.power_kw = None
                        point.context = point.context.model_copy(
                            update={"available": point.context.available - failed}
                        )
        refresh_power_subjects(artifacts)


class PowerEvaluator(Evaluator, kind=CheckKind.POWER):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.rule.requirements
        invalid = validate_modes(check)
        if invalid is not None:
            return invalid
        try:
            operation = PowerOperation(cast(str, params.get("operation")))
        except ValueError:
            return unsupported(check)
        if operation is PowerOperation.DESCRIPTOR:
            flags = (
                "validate_public_sources",
                "validate_node_sets",
                "validate_scale_out",
                "validate_computed_values",
            )
            if any(params.get(field) is False for field in flags):
                return unsupported(
                    check,
                    "This power schema requires source, node, scale-out and computed-value validation",
                )
            compute_descriptor(check, artifacts)
        if operation in (PowerOperation.DESCRIPTOR, PowerOperation.APPENDIX_D_DEFAULTS):
            entry = artifacts.derived.power.get(check.subject.id, {})
            computation = entry.get("computation")
            if operation is PowerOperation.APPENDIX_D_DEFAULTS:
                return (
                    [
                        warning(
                            check,
                            f"Estimated power uses defaults: {', '.join(computation.estimated)}",
                        )
                    ]
                    if computation and computation.estimated
                    else []
                )
            problems = entry.get("problems", ["Power descriptor unavailable"])
            results = [
                finding(
                    check,
                    not problems,
                    "; ".join(problems)
                    or f"Provisioned power: {computation.provisioned_power_kw if computation else None} kW",
                )
            ]
            if computation:
                results.extend(
                    warning(check, message) for message in computation.warnings
                )
            return results
        output = []
        for point in artifacts.members(check):
            computation = power_entry(artifacts, point).get("computation")
            if computation is None:
                output.append(
                    finding(
                        check,
                        False,
                        "Power computation unavailable",
                        point=point,
                        blocked=True,
                    )
                )
                continue
            engaged, problems = nodes_for(point, computation)
            config_summary = point.get_system("config_summary")
            description = {
                **(
                    point.system.model_dump(mode="json", exclude_unset=True)
                    if point.system is not None
                    else {}
                ),
                **(config_summary if isinstance(config_summary, Mapping) else {}),
            }
            if operation is PowerOperation.ENGAGED_NODES:
                point.derived.power_kw = None
                point.context = point.context.model_copy(
                    update={
                        "available": point.context.available - {EvidenceKey.POINT_POWER}
                    }
                )
                if (
                    params.get("declared_total_scaling") != "maximum_engaged_fraction"
                    or params.get("switch_scaling") != "engaged_node_fraction"
                    or params.get("on_undeclared_nodes") != "fully_engaged"
                ):
                    return unsupported(
                        check, "Unsupported engaged power scaling policy"
                    )
                if not problems and computation.provisioned_power_kw is not None:
                    fractions = {
                        s.node_set_id: engaged[s.node_set_id] / s.nodes_provisioned
                        for s in computation.sets
                    }
                    if computation.declared:
                        kw = computation.provisioned_power_kw * max(
                            fractions.values(), default=1
                        )
                    else:
                        watts = sum(
                            s.power_w * fractions[s.node_set_id]
                            for s in computation.sets
                        )
                        watts += (
                            computation.scale_out_switch_power_w
                            * sum(engaged.values())
                            / sum(s.nodes_provisioned for s in computation.sets)
                        )
                        kw = watts / 1000
                    point.derived.power_kw = round(kw, params["round_kw_decimals"])
                    point.context = point.context.model_copy(
                        update={
                            "available": point.context.available
                            | {EvidenceKey.POINT_POWER}
                        }
                    )
                output.append(
                    finding(
                        check,
                        point.derived.power_kw is not None,
                        f"Engaged power: {point.derived.power_kw} kW",
                        point=point,
                    )
                )
                continue
            factors = params.get(
                "replica_factors",
                ("tensor_parallel", "pipeline_parallel", "expert_parallel"),
            )
            present = any(field in description for field in (*factors, "data_parallel"))
            replica = math.prod(description.get(field, 1) for field in factors)
            dp = description.get("data_parallel", 1)
            known = all(s.accelerators_per_node is not None for s in computation.sets)
            if operation is PowerOperation.NODE_DECLARATION:
                if (
                    params.get("require_accelerator_capacity")
                    and present
                    and known
                    and not description.get("disaggregated")
                ):
                    capacity = sum(
                        engaged[s.node_set_id] * cast(int, s.accelerators_per_node)
                        for s in computation.sets
                    )
                    if capacity < replica * dp:
                        problems.append(
                            f"Declared nodes hold {capacity} accelerators; parallelism uses {replica * dp}"
                        )
                    elif any(
                        engaged[s.node_set_id] > 0
                        and capacity - cast(int, s.accelerators_per_node)
                        >= replica * dp
                        for s in computation.sets
                    ):
                        output.append(
                            warning(
                                check,
                                "Declared nodes include spare accelerator capacity",
                                point,
                            )
                        )
                output.append(
                    finding(
                        check,
                        not problems,
                        "; ".join(problems) or "Engaged node declaration is valid",
                        point=point,
                    )
                )
                continue
            if not present or description.get("disaggregated") or not known:
                output.append(
                    warning(
                        check,
                        "Maximal engagement cannot be verified without shared parallelism and known accelerator counts",
                        point,
                    )
                )
                continue
            provisioned = sum(
                s.nodes_provisioned * cast(int, s.accelerators_per_node)
                for s in computation.sets
            )
            if replica < 1 or dp < 1:
                output.append(
                    finding(
                        check,
                        False,
                        "Parallelism factors must be positive",
                        point=point,
                    )
                )
                continue
            maximum = provisioned // replica
            shortfall = artifacts.resolve(params["shortfall"], check, point)
            valid = dp <= maximum
            if params.get("require_maximum_data_parallel") and dp < maximum:
                valid = (
                    isinstance(shortfall, Mapping)
                    and shortfall.get("dp_actual") == dp
                    and shortfall.get("dp_formula") == maximum
                    and bool(shortfall.get("reason") or shortfall.get("explanation"))
                )
            output.append(
                finding(
                    check,
                    valid,
                    f"Data parallelism {dp}; maximum {maximum}; declared shortfall {shortfall!r}",
                    point=point,
                )
            )
        return output
