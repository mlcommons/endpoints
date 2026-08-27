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

    ``report`` carries the :class:`CompletenessReport` for the same run. A
    refusal is not an absence of information: the conditional rate, the lower
    bound and the ids that went missing are exactly what the operator needs in
    order to act, and withholding them alongside the headline is what makes
    people go and compute the wrong number by hand instead.
    """

    def __init__(
        self,
        run_id: str,
        reasons: list[str],
        report: CompletenessReport | None = None,
    ) -> None:
        self.run_id = run_id
        self.reasons = reasons
        self.report = report
        super().__init__(
            f"refusing to score run {run_id!r}: "
            + "; ".join(reasons[:10])
            + (f" (+{len(reasons) - 10} more)" if len(reasons) > 10 else "")
        )


@dataclass(slots=True)
class CompletenessReport:
    """What a run accounted for, and whether that permits an accuracy number.

    ``resolved_rate`` is published only when the run is *structurally complete*
    -- every planned instance id reached a terminal state exactly once -- **and**
    zero instances were lost to infrastructure. Those are two different
    questions and conflating them gets both wrong:

    * An instance the model attempted and failed is a legitimate score. An
      instance our own harness dropped is not: it never had the chance.
    * A run that is short of instances has the wrong denominator; a run that is
      complete but leaned on the infrastructure has the wrong provenance.

    Withholding the headline is the point. A run that lost 106 of 200 instances
    to infrastructure and printed ``resolved / planned`` reported 47.0% as
    though it were accuracy, and that number was then compared against a
    complete-run reference of 70.67% and read as a model regression. It was
    attrition.

    Two numbers are therefore *always* published, refusal or not:

    ``conditional_resolved_rate``
        Resolved over the instances that actually completed. Honest about what
        it measures, and not comparable to a complete-run reference.
    ``resolved_rate_lower_bound``
        Resolved over everything planned. Infrastructure losses can only ever
        *add* resolutions, so this bounds the true rate from below even on a
        badly degraded run.
    """

    run_id: str
    plan_digest: str
    total_instances: int
    accounted_instance_ids: tuple[str, ...] = ()
    resolved_instance_ids: tuple[str, ...] = ()
    incomplete_instance_ids: tuple[str, ...] = ()
    infra_lost_instances: int = 0
    infra_lost_unit_ids: tuple[str, ...] = ()
    unit_count: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def accounted_instances(self) -> int:
        return len(self.accounted_instance_ids)

    @property
    def resolved_instances(self) -> int:
        return len(self.resolved_instance_ids)

    @property
    def complete(self) -> bool:
        """Every planned instance id reached a terminal state exactly once."""
        return not self.incomplete_instance_ids

    @property
    def publishable(self) -> bool:
        return self.complete and self.infra_lost_instances == 0 and not self.reasons

    @property
    def conditional_resolved_rate(self) -> float | None:
        if not self.accounted_instances:
            return None
        return self.resolved_instances / self.accounted_instances

    @property
    def resolved_rate_lower_bound(self) -> float | None:
        if not self.total_instances:
            return None
        return self.resolved_instances / self.total_instances

    @property
    def resolved_rate(self) -> float | None:
        """The headline number, or ``None`` when it must be withheld."""
        if not self.publishable:
            return None
        return self.resolved_rate_lower_bound

    @property
    def withheld_reason(self) -> str | None:
        if self.publishable:
            return None
        why: list[str] = []
        if self.incomplete_instance_ids:
            why.append(
                f"{len(self.incomplete_instance_ids)} of {self.total_instances} "
                "instances never reached a terminal state"
            )
        if self.infra_lost_instances:
            why.append(
                f"{self.infra_lost_instances} instance(s) were lost to "
                "infrastructure, not to the model"
            )
        if self.reasons and not why:
            why.append("; ".join(self.reasons[:5]))
        conditional = self.conditional_resolved_rate
        lower = self.resolved_rate_lower_bound
        return (
            "NO PUBLISHABLE ACCURACY: "
            + "; ".join(why)
            + ". Lower bound "
            + ("n/a" if lower is None else f"{lower:.4f}")
            + " (infrastructure losses can only add resolutions); conditional "
            + ("n/a" if conditional is None else f"{conditional:.4f}")
            + f" over the {self.accounted_instances} instance(s) that completed. "
            "Neither is comparable to a complete-run reference."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan_digest": self.plan_digest,
            "complete": self.complete,
            "total_instances": self.total_instances,
            "accounted_instances": self.accounted_instances,
            "resolved_instances": self.resolved_instances,
            "unit_count": self.unit_count,
            "incomplete_instance_ids": list(self.incomplete_instance_ids),
            "infra_lost_instances": self.infra_lost_instances,
            "infra_lost_unit_ids": list(self.infra_lost_unit_ids),
            "resolved_rate": self.resolved_rate,
            "conditional_resolved_rate": self.conditional_resolved_rate,
            "resolved_rate_lower_bound": self.resolved_rate_lower_bound,
            "resolved_rate_withheld_reason": self.withheld_reason,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class MergeResult:
    run_id: str
    plan_digest: str
    total_instances: int
    resolved_instances: int
    unit_count: int
    #: The accounting this result was published from. Present on the success
    #: path too, so a caller never has to decide which of two shapes it holds.
    report: CompletenessReport | None = None

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
            **(self.report.to_dict() if self.report is not None else {}),
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


def assess_run(queue: WorkQueue, run_id: str) -> CompletenessReport:
    """Account for every planned instance id. Never raises on a bad run.

    This is the whole of the gate's arithmetic, separated from the decision to
    refuse, so that a refused run still yields the conditional rate, the lower
    bound and the ids that went missing. :func:`merge_run` is the strict
    all-or-nothing wrapper over it.
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
    infra_lost = 0
    infra_lost_units: set[str] = set()

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
            infra_lost += len(unit.instance_ids)
            infra_lost_units.add(unit.unit_id)
            continue
        if result.outcome is not UnitOutcome.SUCCEEDED:
            reasons.append(f"{unit.unit_id}: outcome {result.outcome.value}")
            continue
        if result.infra_error_count > 0:
            # Recorded, not merely refused: how many instances the harness lost
            # is the difference between an accuracy and an attrition figure.
            reasons.append(
                f"{unit.unit_id}: {result.infra_error_count} instance(s) lost to "
                "infrastructure"
            )
            infra_lost += result.infra_error_count
            infra_lost_units.add(unit.unit_id)
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
    unaccounted = sorted(planned_ids - set(seen_ids))
    if unaccounted:
        reasons.append(
            f"{len(unaccounted)} planned instance(s) unaccounted for: "
            + ", ".join(unaccounted[:5])
        )

    return CompletenessReport(
        run_id=run_id,
        plan_digest=plan.digest,
        total_instances=len(planned_ids),
        accounted_instance_ids=tuple(sorted(seen_ids)),
        resolved_instance_ids=tuple(sorted(resolved & planned_ids)),
        incomplete_instance_ids=tuple(sorted(planned_ids - set(seen_ids))),
        infra_lost_instances=infra_lost,
        infra_lost_unit_ids=tuple(sorted(infra_lost_units)),
        unit_count=len(plan.units),
        reasons=reasons,
    )


def merge_run(queue: WorkQueue, run_id: str) -> MergeResult:
    """Score one run, or refuse.

    ``run_id`` is required and must match the queue's plan. Passing another
    run's id is an error, not a filter.
    """
    report = assess_run(queue, run_id)
    if not report.publishable:
        raise MergeRefusal(
            run_id, report.reasons or [report.withheld_reason or ""], report
        )
    return MergeResult(
        run_id=run_id,
        plan_digest=report.plan_digest,
        total_instances=report.total_instances,
        resolved_instances=report.resolved_instances,
        unit_count=report.unit_count,
        report=report,
    )
