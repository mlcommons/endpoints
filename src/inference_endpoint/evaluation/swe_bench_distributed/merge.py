# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The merge gate: refuse to emit an accuracy unless every id is accounted for.

The single most important property of a sharded accuracy run is that it never
divides the results of 190 instances by 200. The gate is all-or-nothing by
design: there is no force flag and no partial-credit path, because a partial
number is indistinguishable from a real one once it leaves this module.

The gate is also scoped to exactly one run. There is no ``merge_all``. Merging
"every run that looks finished" once re-merged hundreds of banked results
belonging to unrelated configurations into one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .queue import UnitOutcome, UnitResult, WorkQueue
from .units import UnitPlan


class MergeRefusal(RuntimeError):
    """The gate refused to produce an accuracy number.

    ``reasons`` lists every independent failure, so one merge attempt reports
    everything wrong rather than the first thing wrong.
    """

    def __init__(self, run_id: str, reasons: list[str]) -> None:
        self.run_id = run_id
        self.reasons = reasons
        super().__init__(
            f"refusing to score run {run_id!r}: "
            + "; ".join(reasons[:10])
            + (f" (+{len(reasons) - 10} more)" if len(reasons) > 10 else "")
        )


@dataclass(slots=True)
class MergeResult:
    run_id: str
    plan_digest: str
    total_instances: int
    resolved_instances: int
    unit_count: int

    @property
    def resolved_rate(self) -> float:
        return self.resolved_instances / self.total_instances

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan_digest": self.plan_digest,
            "total_instances": self.total_instances,
            "resolved_instances": self.resolved_instances,
            "resolved_rate": self.resolved_rate,
            "unit_count": self.unit_count,
        }


@dataclass(slots=True)
class InventoryReport:
    """Cross-check of three independently produced views of the same run.

    The units the plan asked for, the units the queue recorded results for, and
    the instance ids those results claim to have covered are produced by
    different code paths. Checking one against itself is how a verification pass
    can agree with a broken system: the instrument shares the blind spot. These
    must agree with each other.
    """

    missing_units: list[str] = field(default_factory=list)
    foreign_units: list[str] = field(default_factory=list)
    unreadable_units: list[str] = field(default_factory=list)
    ownerless_claims: list[str] = field(default_factory=list)
    claims_without_results: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not (
            self.missing_units
            or self.foreign_units
            or self.unreadable_units
            or self.ownerless_claims
        )


def verify_inventory(queue: WorkQueue) -> InventoryReport:
    """Compare the plan, the claim directory and the result directory."""
    report = InventoryReport()
    plan_units = set(queue.plan.unit_ids)

    result_files = {path.stem for path in queue.results_dir.glob("*.json")}
    report.foreign_units = sorted(result_files - plan_units)
    report.missing_units = sorted(plan_units - result_files)
    for unit_id in sorted(result_files & plan_units):
        if queue.result(unit_id) is None:
            report.unreadable_units.append(unit_id)

    for unit_id in sorted(queue.claimed_unit_ids()):
        if queue.owner(unit_id) is None:
            report.ownerless_claims.append(unit_id)
        if unit_id not in result_files:
            report.claims_without_results.append(unit_id)
    return report


def merge_run(queue: WorkQueue, run_id: str) -> MergeResult:
    """Score one run, or refuse.

    ``run_id`` is required and must match the queue's plan. Passing another
    run's id is an error, not a filter.
    """
    plan: UnitPlan = queue.plan
    if run_id != plan.run_id:
        raise MergeRefusal(
            run_id,
            [
                f"queue at {queue.root} holds run {plan.run_id!r}, not {run_id!r}; "
                "a merge is always scoped to exactly one run"
            ],
        )

    reasons: list[str] = []
    inventory = verify_inventory(queue)
    if inventory.foreign_units:
        reasons.append(
            "results present for units outside the plan: "
            + ", ".join(inventory.foreign_units[:5])
        )
    if inventory.unreadable_units:
        reasons.append(
            "unreadable result records: " + ", ".join(inventory.unreadable_units[:5])
        )
    if inventory.ownerless_claims:
        reasons.append(
            "claims with no readable owner: "
            + ", ".join(inventory.ownerless_claims[:5])
        )
    if inventory.missing_units:
        reasons.append(
            f"{len(inventory.missing_units)} of {len(plan.units)} units have no "
            "result: " + ", ".join(inventory.missing_units[:5])
        )

    results: dict[str, UnitResult] = queue.results()
    seen_ids: dict[str, str] = {}
    resolved: set[str] = set()

    for unit in plan.units:
        result = results.get(unit.unit_id)
        if result is None:
            continue
        if result.plan_digest != plan.digest:
            reasons.append(
                f"{unit.unit_id}: result belongs to plan {result.plan_digest[:12]}, "
                f"not {plan.digest[:12]}"
            )
            continue
        if result.abandoned:
            reasons.append(f"{unit.unit_id}: abandoned after {result.attempt} attempts")
            continue
        if result.outcome is not UnitOutcome.SUCCEEDED:
            reasons.append(f"{unit.unit_id}: outcome {result.outcome.value}")
            continue
        if result.infra_error_count > 0:
            reasons.append(
                f"{unit.unit_id}: {result.infra_error_count} instance(s) lost to "
                "infrastructure"
            )
            continue

        expected = set(unit.instance_ids)
        accounted = set(result.accounted_instance_ids)
        if len(result.accounted_instance_ids) != len(accounted):
            reasons.append(f"{unit.unit_id}: duplicate instance ids in its own result")
            continue
        # Compare ids, never counts. A shard with one duplicate and one missing
        # id has the right count and the wrong content.
        if accounted != expected:
            missing = sorted(expected - accounted)
            extra = sorted(accounted - expected)
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing[:5])}")
            if extra:
                detail.append(f"unplanned {', '.join(extra[:5])}")
            reasons.append(f"{unit.unit_id}: " + "; ".join(detail))
            continue

        for instance_id in result.accounted_instance_ids:
            previous = seen_ids.get(instance_id)
            if previous is not None:
                reasons.append(
                    f"instance {instance_id} accounted for by both {previous} and "
                    f"{unit.unit_id}"
                )
                continue
            seen_ids[instance_id] = unit.unit_id
        unplanned_resolved = set(result.resolved_instance_ids) - expected
        if unplanned_resolved:
            reasons.append(
                f"{unit.unit_id}: resolved ids outside its shard: "
                + ", ".join(sorted(unplanned_resolved)[:5])
            )
            continue
        resolved.update(result.resolved_instance_ids)

    planned_ids = set(plan.instance_ids)
    if not reasons and set(seen_ids) != planned_ids:
        unaccounted = sorted(planned_ids - set(seen_ids))
        reasons.append(
            f"{len(unaccounted)} planned instance(s) unaccounted for: "
            + ", ".join(unaccounted[:5])
        )

    if reasons:
        raise MergeRefusal(run_id, reasons)

    return MergeResult(
        run_id=run_id,
        plan_digest=plan.digest,
        total_instances=len(planned_ids),
        resolved_instances=len(resolved),
        unit_count=len(plan.units),
    )
