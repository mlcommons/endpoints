# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check-runner core: one `Finding` per evaluated `assert` row across the cohort.

`Finding.collect(cohort, submission, regions)` walks each `Check.scope` kind
over the submission's `Point`/`Node` tree, gates on `when` rows, and evaluates
each `assert` row through `Verdict.evaluate`. `needs` gate-checks run at their
own scope first; `optional_needs` are non-gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inference_endpoint.validation.check import Check, Cohort, Severity
from inference_endpoint.validation.compare import Verdict
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.submission import Submission
from inference_endpoint.validation.tree import Node, RuleScopes


@dataclass(frozen=True)
class Finding:
    """One `Check`'s outcome at one applicable node (its row-Verdict plus provenance)."""

    check_id: str
    severity: Severity
    scope: RuleScopes
    node_directory: Path
    verdict: Verdict
    rule: str

    @classmethod
    def from_check_row(
        cls,
        check: Check,
        scope: RuleScopes,
        node_directory: Path,
        verdict: Verdict,
    ) -> Finding:
        """`Finding.from_check_row(check, RuleScopes.POINT, dir, Verdict(True, ""))`."""
        return cls(
            check_id=check.id,
            severity=check.severity,
            scope=scope,
            node_directory=node_directory,
            verdict=verdict,
            rule=check.rule,
        )

    @classmethod
    def collect(
        cls, cohort: Cohort, submission: Submission, regions: Regions
    ) -> tuple[Finding, ...]:
        """One `Finding` per evaluated `assert` row at each applicable node (holds too).

        Three phases: `_collect_candidates` validates `needs` names;
        `_needs_gate` selects check ids whose `needs` chain fully passes
        (each chain-token must hold its own rows *and* its own chain); and
        `_evaluate_rows` emits one Finding per row at each `when`-applicable
        node of the gate-approved checks. A check's own row-failure suppresses
        its *dependents*, never its own Findings.
        """
        candidates = _collect_candidates(cohort)
        to_run = _filter_should_run(
            candidates, _needs_gate(candidates, submission, regions)
        )
        return _evaluate_rows(to_run, submission, regions)


def _filter_should_run(
    candidates: tuple[Check, ...], should_run: frozenset[str]
) -> tuple[Check, ...]:
    """`candidates` narrowed to ids whose `needs` chain passed (see `_needs_gate`)."""
    return tuple(check for check in candidates if check.id in should_run)


def _collect_candidates(cohort: Cohort) -> tuple[Check, ...]:
    """Every check in the cohort with `needs`/`optional_needs` names validated against itself.

    An id referenced but absent from the cohort raises ValueError -- a
    load-level mistake, not a report item.
    """
    by_id = {check.id: check for check in cohort.checks}
    for check in cohort.checks:
        for token in (*check.needs, *check.optional_needs):
            if token not in by_id:
                raise ValueError(f"check {check.id!r} needs unknown check {token!r}")
    return cohort.checks


def _needs_gate(
    candidates: tuple[Check, ...], submission: Submission, regions: Regions
) -> frozenset[str]:
    """Ids whose every `needs` token holds its own rows and its own chain (recursive memo).

    A YAML check's `needs` gate is declared as ids of upstream checks; the
    dependent fires only when every named chain holds at that upstream's own
    scope (`optional_needs` are recorded but non-gating by their name):

        id: gauge_price_sane
        needs: [concurrency_in_order]
        optional_needs: [point_count_lookback]
        assert:
          - left: submission.offline_ordering
        op: eq
            right: true

    `holds_self` recurses the `needs` chain first (cycles trip a `resolving`
    set into a ValueError -- a load-level mistake). The gate ignores the
    check's *own* row status by design: a check whose own rows fail still emits
    its Findings; only its dependents stay silent.
    """
    by_id = {check.id: check for check in candidates}
    resolved: dict[str, bool] = {}
    resolving: set[str] = set()

    def holds_self(check: Check) -> bool:
        """Whether `check`'s own rows hold at its scope with its `needs` chain passing."""
        if check.id in resolved:
            return resolved[check.id]
        if check.id in resolving:
            raise ValueError(f"`needs` cycle through {check.id!r}")
        resolving.add(check.id)
        needs_hold = all(holds_self(by_id[token]) for token in check.needs)
        held = needs_hold and _rows_hold(check, submission, regions)
        resolved[check.id] = held
        resolving.discard(check.id)
        return held

    should_run = set()
    for check in candidates:
        if all(holds_self(by_id[token]) for token in check.needs):
            should_run.add(check.id)
    return frozenset(should_run)


def _evaluate_rows(
    to_run: tuple[Check, ...],
    submission: Submission,
    regions: Regions,
) -> tuple[Finding, ...]:
    """One `Finding` per evaluated `assert` row at each `when`-applicable node (holds too).

    Emits holds and failures alike -- the report side filters `not
    verdict.ok` for a failure summary; the audit trail keeps the record.
    """
    findings = []
    for check in to_run:
        for node in _scope_nodes(check.scope, submission, regions):
            if not _when_rows_hold(check, node):
                continue
            for row in check.assert_:
                findings.append(
                    Finding.from_check_row(
                        check,
                        check.scope,
                        node.dir,
                        Verdict.evaluate(row, node.scopes()),
                    )
                )
    return tuple(findings)


def _rows_hold(check: Check, submission: Submission, regions: Regions) -> bool:
    """The check's own `assert` rows all hold at its `when`-applicable scope nodes."""
    return all(
        Verdict.evaluate(row, node.scopes()).ok
        for node in _scope_nodes(check.scope, submission, regions)
        if _when_rows_hold(check, node)
        for row in check.assert_
    )


def _scope_nodes(
    scope: RuleScopes, submission: Submission, regions: Regions
) -> tuple[Node, ...]:
    """Unique nodes of `scope` kind reachable from every point's tree position.

    Point-scope reaches the points' own `PointNode`s; `curve`/`system`/
    `submission` reach the 1/2/3-hop parent of each (deduplicated because a
    curve's points share a curve node).
    """
    nodes: list[Node] = []
    if scope == RuleScopes.POINT:
        nodes = [point.node for point in submission.points]
    else:
        parent_hops = {
            RuleScopes.CURVE: 1,
            RuleScopes.SYSTEM: 2,
            RuleScopes.SUBMISSION: 3,
        }[scope]
        for point in submission.points:
            node = point.node
            for _ in range(parent_hops):
                if node.parent is None:
                    raise ValueError(
                        f"check scope {scope.value!r} walks past the submission root"
                    )
                node = node.parent
            nodes.append(node)
    return tuple({node.dir: node for node in nodes}.values())


def _when_rows_hold(check: Check, node: Node) -> bool:
    """Every `when` row holds at this node (a `when`-free check always applies)."""
    scopes = node.scopes()
    return all(Verdict.evaluate(row, scopes).ok for row in check.when)
