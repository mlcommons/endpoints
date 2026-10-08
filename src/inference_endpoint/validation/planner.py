# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select scoped checks and resolve prerequisites without evaluating compliance."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TypeVar

from .conditions import Context
from .models import Override, Policy, Rule
from .references import evidence_references, needs_model
from .requirements import CheckRequirements
from .types import Decision, EvidenceKey, Match

_Requirements = TypeVar("_Requirements", bound=CheckRequirements)


@dataclass(frozen=True)
class PlannedCheck:
    subject: Context
    rule: Rule
    decision: Decision
    reason: str
    missing: frozenset[EvidenceKey] = frozenset()
    override: str | None = None
    selected_members: tuple[str, ...] | None = None

    def requirements(self, contract: type[_Requirements]) -> _Requirements:
        parameters = self.rule.requirements
        if not isinstance(parameters, contract):
            raise TypeError(f"Rule {self.rule.id} requires {contract.__name__}")
        return parameters


@dataclass(frozen=True)
class CheckPlan:
    policy_digest: str
    checks: tuple[PlannedCheck, ...]

    @property
    def ready(self) -> tuple[PlannedCheck, ...]:
        return tuple(check for check in self.checks if check.decision is Decision.READY)

    @property
    def blocked(self) -> tuple[PlannedCheck, ...]:
        return tuple(
            check for check in self.checks if check.decision is Decision.BLOCKED
        )


def _select(rule: Rule, subject: Context) -> tuple[Decision, str]:
    included = rule.applies_to(subject) if rule.applies_to else Match.YES
    excluded = rule.unless(subject) if rule.unless else Match.NO
    if excluded is Match.YES:
        return Decision.EXCLUDED, "Exclusion conditions match"
    if included is Match.NO:
        return Decision.EXCLUDED, "Inclusion conditions do not match"
    if Match.UNKNOWN in (included, excluded):
        return (
            Decision.BLOCKED,
            "Classification facts required for selection are unknown",
        )
    return Decision.READY, "Applicable"


def _member_subject(subject: Context, member: Context) -> Context:
    """Inherit shared identity while preserving unknown point-specific facts."""
    fields = ("model_id", "division")
    return member.model_copy(
        update={
            field: getattr(subject, field)
            for field in fields
            if getattr(member, field) is None
        }
    )


def _select_collection(rule: Rule, subject: Context):
    if subject.members is None or not (rule.applies_to or rule.unless):
        return (*_select(rule, subject), None)
    selected = []
    unknown = False
    for member in subject.members:
        decision, _reason = _select(rule, _member_subject(subject, member))
        if decision is Decision.READY:
            selected.append(member.id)
        elif decision is Decision.BLOCKED:
            unknown = True
    if unknown:
        return (
            Decision.BLOCKED,
            "Collection selection needs member classification facts",
            tuple(selected),
        )
    if not selected:
        return Decision.EXCLUDED, "No collection members match", ()
    return Decision.READY, "Applicable to selected collection members", tuple(selected)


def _override_partitions(
    policy: Policy,
    rule: Rule,
    subject: Context,
    selected_members: tuple[str, ...] | None,
) -> list[tuple[Override | None, bool, tuple[str, ...] | None]]:
    overrides = [override for override in policy.overrides if override.rule == rule.id]
    if not overrides:
        return [(None, False, selected_members)]
    member_selection = bool(subject.members)
    contexts = (
        [
            _member_subject(subject, member)
            for member in (subject.members or ())
            if selected_members is None or member.id in selected_members
        ]
        if member_selection
        else [subject]
    )
    partitions: dict[tuple[str | None, bool], tuple[Override | None, list[str]]] = {}
    for context in contexts:
        matches = []
        unknown = False
        for override in overrides:
            match = override.applies_to(context)
            if match is Match.YES:
                matches.append(override)
            elif match is Match.UNKNOWN:
                unknown = True
        if len(matches) > 1:
            raise ValueError(f"Overlapping overrides for {rule.id} on {context.id}")
        selected_override = matches[0] if matches and not unknown else None
        key = (selected_override.id if selected_override else None, unknown)
        if key not in partitions:
            partitions[key] = (selected_override, [])
        partitions[key][1].append(context.id)
    return [
        (override, unknown, tuple(members) if member_selection else selected_members)
        for (_, unknown), (override, members) in partitions.items()
    ]


def plan_checks(policy: Policy, subjects: Iterable[Context]) -> CheckPlan:
    subjects = tuple(subjects)
    if not subjects:
        raise ValueError("Planning requires at least one classified subject")
    if len({(subject.scope, subject.id) for subject in subjects}) != len(subjects):
        raise ValueError("Duplicate scoped subject IDs")
    planned = []
    for subject in subjects:
        for original in policy.checks:
            if original.scope is not subject.scope:
                continue
            selection, selection_reason, selected_members = _select_collection(
                original, subject
            )
            partitions = (
                _override_partitions(policy, original, subject, selected_members)
                if selection is Decision.READY
                else [(None, False, selected_members)]
            )
            for override, unresolved, members in partitions:
                rule = original
                decision, reason = selection, selection_reason
                missing: frozenset[EvidenceKey] = frozenset()
                applied = None
                if unresolved:
                    decision, reason = (
                        Decision.BLOCKED,
                        "Override selection needs classification facts",
                    )
                elif override:
                    applied = override.id
                    if not override.enabled:
                        decision, reason = Decision.EXCLUDED, override.reason
                    else:
                        requirements = rule.requirements.with_updates(
                            override.requirements
                        )
                        rule = replace(
                            rule,
                            requirements=requirements,
                            references=evidence_references(requirements.wire()),
                            needs_model=needs_model(requirements.wire()),
                        )
                        reason = override.reason
                if decision is Decision.READY:
                    if rule.needs_model and subject.model_id not in policy.models:
                        decision, reason = (
                            Decision.BLOCKED,
                            "Model is missing or not enrolled",
                        )
                    else:
                        missing = rule.requires - subject.available
                        if missing:
                            decision, reason = (
                                Decision.BLOCKED,
                                "Required evidence is missing or invalid",
                            )
                planned.append(
                    PlannedCheck(
                        subject, rule, decision, reason, missing, applied, members
                    )
                )
    return CheckPlan(policy.digest, tuple(planned))
