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

`Finding.collect(cohort, submission)` walks each `Check.scope` kind over the
submission's `Point`/`Node` tree, gates on `when` rows, and evaluates each
assert` row through `Verdict.evaluate`. A `needs` gate speaks per node: a
check runs only where every named upstream chain ran and held at the nodes
that vouch for it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from inference_endpoint.validation.check import Check, Cohort, Severity
from inference_endpoint.validation.compare import Ref, Verdict
from inference_endpoint.validation.submission import (
    REGION_DERIVATION_METHODS,
    Point,
    Submission,
    normalize_offline,
)
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
        cls, check: Check, scope, node_directory: Path, verdict: Verdict
    ) -> Finding:
        """`Finding.from_check_row(check, RuleScopes.POINT, dir, Verdict(True, ""))`."""
        return cls(
            check_id=check.id,
            severity=check.severity,
            scope=scope if isinstance(scope, RuleScopes) else RuleScopes(scope),
            node_directory=node_directory,
            verdict=verdict,
            rule=check.rule,
        )

    @classmethod
    def collect(cls, cohort: Cohort, submission: Submission) -> tuple[Finding, ...]:
        """One `Finding` per evaluated `assert` row at each `when`-applicable node (holds too).

        A load-level check first (`_validate_*`: a submission the walk cannot
        cover, an unreadable artifact, hand-mirrored facts that disagree with
        the parsed tree, or a fold the cohort names but the tree lacks are all
        errors -- silence must never be the verdict); then
        `_collect_candidates` validates `needs` names (unknown -> ValueError)
        and `_GateResolver` resolves every `needs` chain per node. Row
        evaluation emits one Finding per row at each `when`-applicable node
        whose chain holds -- holds and failures alike (the report side filters
        `not verdict.ok`). `derived.<name>` rows resolve via
        `submission.derived()` folded into each node's row-scope at evaluation.
        A check whose own rows fail still emits its Findings; only its
        dependents stay silent, and only at the nodes whose upstreams did not
        hold there.
        """
        _validate_points(submission)
        _validate_artifacts(submission)
        _validate_agreement(submission)
        _validate_fold_needs(cohort, submission)
        candidates = _collect_candidates(cohort)
        gate = _GateResolver({check.id: check for check in candidates}, submission)
        findings: list[Finding] = []
        for check in candidates:
            for node in _scope_nodes(check.scope, submission):
                if not _when_rows_hold(check, node, submission):
                    continue
                if not gate.to_run_at(check, node):
                    continue
                findings.extend(_row_findings(check, node, submission))
        return tuple(findings)


def _validate_points(submission: Submission) -> None:
    """Fail loud where the hand-carried `Point` set and the walked tree disagree.

    The two are views of one upload: every `r<N>` run directory needs exactly
    one carried `Point`, bound to a point node of *this* tree. An unlisted run
    would otherwise escape every check (while still vouching for dependents),
    and a shared or foreign node would corrupt counts, regions and derivations.
    """
    if not submission.points:
        raise ValueError("submission has no measurement points to check")
    for point in submission.points:
        if not isinstance(point.node, Node) or point.node.kind != RuleScopes.POINT:
            raise ValueError(
                f"point at concurrency {point.concurrency} needs its point node"
            )
    root = _tree_root(submission)
    tree = {
        id(node): node for node in _all_nodes(root) if node.kind == RuleScopes.POINT
    }
    seen: dict[int, Point] = {}
    for point in submission.points:
        node = point.node
        assert node is not None  # checked above; narrows for the type checker
        if node.root() is not root:
            raise ValueError(
                f"point at concurrency {point.concurrency} is bound outside "
                "this submission's tree"
            )
        token = id(node)
        if token in seen:
            raise ValueError(
                f"points at concurrency {seen[token].concurrency} and "
                f"{point.concurrency} share one run directory"
            )
        seen[token] = point
    uncarried = [node for token, node in tree.items() if token not in seen]
    if uncarried:
        raise ValueError(
            f"run directory {uncarried[0].dir.name} has no carried measurement point"
        )


def _tree_root(submission: Submission) -> Node:
    """The validated submission's tree root (after `_validate_points`, nodes exist)."""
    node = submission.points[0].node
    if node is None:
        raise ValueError("point needs its point node")
    return node.root()


def _validate_artifacts(submission: Submission) -> None:
    """Fail loud on any artifact that failed to parse anywhere in the tree.

    A corrupt artifact must never read as an absent fact: that would silently
    switch `when`-gated checks off and pass the submission by omission.
    """
    root = _tree_root(submission)
    broken = [
        f"{node.dir / name}: {error}"
        for node in _all_nodes(root)
        for name, (value, error) in sorted(node.artifacts.items())
        if error is not None
    ]
    if broken:
        raise ValueError("unreadable artifacts: " + "; ".join(broken))


def _validate_agreement(submission: Submission) -> None:
    """Fail loud where a hand-carried fact disagrees with the parsed artifact.

    `Point` and the tree are two views of the same facts (the tree feeds rows,
    `Point` feeds derivations); a silent disagreement would make a check's row
    and its derivation judge different submissions. Compared symmetrically: a
    declaration missing from `point.yaml` disagrees with a carried declaration
    just as loudly as the other direction.
    """
    for point in submission.points:
        node = point.node
        if node is None:  # `_validate_points` has refused these; narrows the type
            continue
        config, _ = node.read("point.yaml")
        if not isinstance(config, Mapping):
            continue  # shape problems are `_validate_artifacts`' story
        declared_concurrencies: list[tuple[str, Any, Any]] = [
            ("concurrency", config.get("concurrency"), point.concurrency),
            ("offline", normalize_offline(config.get("offline")), point.offline),
        ]
        for field_name, declared, carried in declared_concurrencies:
            if declared != carried:
                raise ValueError(
                    f"point at {node.dir.name} carries {field_name}={carried!r} "
                    f"but its point.yaml declares {declared!r}"
                )
    parsed, _ = _tree_root(submission).read("system_desc.json")
    if isinstance(parsed, Mapping):
        for key, value in submission.system_desc.items():
            if key in parsed and parsed[key] != value:
                raise ValueError(
                    f"carried system_desc {key}={value!r} disagrees with "
                    f"system_desc.json ({parsed[key]!r})"
                )


def _validate_fold_needs(cohort: Cohort, submission: Submission) -> None:
    """Fail loud when a cohort names a scope fold the tree was built without.

    A forgotten `regions=`/`ruleset=` kwarg must be a load error, not rows
    resolving "missing" on a healthy upload. Which derivations need which fold
    is asked of `submission.REGION_DERIVATION_METHODS`, its single owner.
    """
    root = _tree_root(submission)
    folds = {"regions": root.regions, "ruleset": root.ruleset}
    for check in cohort.checks:
        for row in (*check.when, *check.assert_):
            for ref in (r for r in row.operands() if isinstance(r, Ref)):
                head = ref.parts[0]
                fold = head if head in folds else None
                if (
                    ref.parts[0] == "derived"
                    and len(ref.parts) > 1
                    and ref.parts[1] in REGION_DERIVATION_METHODS
                ):
                    fold = "regions"
                if fold is not None and folds[fold] is None:
                    raise ValueError(
                        f"check {check.id!r} names {ref} but the tree was built "
                        f"without {fold}"
                    )


def _all_nodes(node: Node) -> Iterator[Node]:
    """`node` and every node below it."""
    yield node
    for child in node.children:
        yield from _all_nodes(child)


def _collect_candidates(cohort: Cohort) -> tuple[Check, ...]:
    """Every check in the cohort with `needs` names validated against itself.

    An id referenced but absent from the cohort, or a `needs` cycle, raises
    ValueError at load -- a malformed cohort must fail identically for every
    submission, never only where the `when` rows happen to lead the walk.
    """
    by_id = {check.id: check for check in cohort.checks}
    for check in cohort.checks:
        for token in check.needs:
            if token not in by_id:
                raise ValueError(f"check {check.id!r} needs unknown check {token!r}")
    _raise_on_needs_cycles(by_id)
    return cohort.checks


def _raise_on_needs_cycles(by_id: Mapping[str, Check]) -> None:
    """ValueError on a `needs` cycle, found by DFS before any row runs."""
    finished: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in finished:
            return
        if name in visiting:
            raise ValueError(f"`needs` cycle through {name!r}")
        visiting.add(name)
        for token in by_id[name].needs:
            visit(token)
        visiting.discard(name)
        finished.add(name)

    for name in by_id:
        visit(name)


class _CheckState(Enum):
    """Whether a check's `assert` rows were evaluated and held at one node.

    `HOLDS`: its `when` applied there, its own chains vouched, every `assert`
    row held. `FAILS`: it ran there and a row failed. `NOT_APPLICABLE`: its
    `when` excluded the node -- it makes no claim there. `SILENCED`: its own
    `needs` chain refused to vouch, so its rows never ran. Keeping "no claim"
    and "cannot vouch" apart is what lets a `when`-gated upstream vouch exactly
    where it speaks instead of silencing whole subtrees.
    """

    HOLDS = "holds"
    FAILS = "fails"
    NOT_APPLICABLE = "not_applicable"
    SILENCED = "silenced"


class _GateResolver:
    """Per-(check, node) `needs` resolution over one submission, memoized.

    A check speaks at a node only where every named upstream chain **vouches**
    for it -- claims at least one node of its scope in the dependent node's
    chain-or-subtree and `HOLDS` at each (see `_chain_holds`). One `FAILS`
    node disavows; "no claim" (`NOT_APPLICABLE`/`SILENCED`) never vouches and
    never disavows. A `needs` cycle is refused at load
    (`_raise_on_needs_cycles`); the recursion trail here is the backstop.
    """

    def __init__(self, by_id: Mapping[str, Check], submission: Submission) -> None:
        self._by_id = by_id
        self._submission = submission
        self._memo: dict[tuple[str, str], _CheckState] = {}

    def to_run_at(self, check: Check, node: Node) -> bool:
        """Whether `check`'s rows are litigated at `node`: every `needs` chain HOLDS there."""
        return self._chain_holds(check, node, ())

    def _chain_holds(
        self, check: Check, node: Node, trail: tuple[tuple[str, str], ...]
    ) -> bool:
        """Whether every named upstream chain vouches for `node`.

        An upstream vouches when it **claims** at least one node of its scope in
        the dependent node's chain-or-subtree (`_chain_nodes`) and `HOLDS` at
        every node where it claims: `NOT_APPLICABLE` and `SILENCED` nodes are no
        claim at all, while one `FAILS` disavows. An empty vouch-set is no
        vouch -- an unverified premise never passes a dependent.
        """
        for token in check.needs:
            upstream = self._by_id[token]
            claims = [
                state
                for upstream_node in _chain_nodes(upstream.scope, node)
                if (state := self._state(upstream, upstream_node, trail))
                in (_CheckState.HOLDS, _CheckState.FAILS)
            ]
            if not claims or any(state is not _CheckState.HOLDS for state in claims):
                return False
        return True

    def _state(
        self, check: Check, node: Node, trail: tuple[tuple[str, str], ...]
    ) -> _CheckState:
        """`check`'s state at `node`: its chain first, then its `when`, then its rows."""
        key = (check.id, str(node.dir))
        if key in self._memo:
            return self._memo[key]
        if key in trail:
            raise ValueError(f"`needs` cycle through {check.id!r}")
        trail = (*trail, key)
        if not self._chain_holds(check, node, trail):
            state = _CheckState.SILENCED
        elif not _when_rows_hold(check, node, self._submission):
            state = _CheckState.NOT_APPLICABLE
        elif all(
            Verdict.evaluate(row, _scopes_rows_view(node, self._submission)).ok
            for row in check.assert_
        ):
            state = _CheckState.HOLDS
        else:
            state = _CheckState.FAILS
        self._memo[key] = state
        return state


_SCOPE_DEPTH = {
    RuleScopes.SUBMISSION: 0,
    RuleScopes.SYSTEM: 1,
    RuleScopes.CURVE: 2,
    RuleScopes.POINT: 3,
}


def _chain_nodes(scope: RuleScopes, node: Node) -> tuple[Node, ...]:
    """The `scope`-kind nodes a check at `node` depends on: up the chain, or down the subtree.

    Same-or-ancestor scope resolves to the one chain node (a point's curve,
    system and submission ancestors); a finer scope resolves to every such node
    below `node` (a system-scoped check vouches with all of its points). A
    missing ancestor raises ValueError (a load-level mistake); a subtree
    without any yields no nodes -- the chain claims nothing there.
    """
    if _SCOPE_DEPTH[scope] <= _SCOPE_DEPTH[node.kind]:
        probe: Node | None = node
        while probe is not None and probe.kind != scope:
            probe = probe.parent
        if probe is not None:
            return (probe,)
        raise ValueError(f"no {scope.value} node above {node.dir}")
    return tuple(_below(node, scope))


def _below(node: Node, scope: RuleScopes) -> Iterator[Node]:
    """`scope`-kind nodes anywhere in `node`'s subtree."""
    for child in node.children:
        if child.kind == scope:
            yield child
        else:
            yield from _below(child, scope)


def _row_findings(
    check: Check, node: Node, submission: Submission
) -> tuple[Finding, ...]:
    """The check's `assert` rows at one `when`-applicable node, one Finding each.

    Emits holds and failures alike -- the report side filters `not verdict.ok`
    for a failure summary; the audit trail keeps the record. A scalar `left`
    row that never names a `derived.<name>` runs no derivation.
    """
    return tuple(
        Finding.from_check_row(
            check,
            check.scope,
            node.dir,
            Verdict.evaluate(row, _scopes_rows_view(node, submission)),
        )
        for row in check.assert_
    )


def _scopes_rows_view(node: Node, submission: Submission) -> dict[str, Any]:
    """Row-resolvable scope map at `node`: the tree's per-node fields plus `derived`.

    The region-fold stays the tree's single source (fed by
    `SubmissionNode.from_directory(root, regions=...)` upstream) and reaches
    `derived.out_of_range_points` too -- one region derivation source;
    `derived` folds here because its values are submission-wide named facts
    (`Submission.derived()`, `derived.<name>` rows).
    """
    scopes: dict[str, Any] = node.scopes()
    scopes["derived"] = submission.derived(regions=node.root().regions)
    return scopes


def _scope_nodes(scope: RuleScopes, submission: Submission) -> tuple[Node, ...]:
    """Unique nodes of `scope` kind reached by walking each point's node chain.

    `Point` reaches the points' own `PointNode`s; `curve`/`system`/`submission`
    walk the parents (deduplicated because a curve's points share a curve
    node). A chain without the requested kind raises ValueError -- a
    load-level mistake, not a report item.
    """
    nodes: list[Node] = []
    if scope == RuleScopes.POINT:  # `_validate_points` has refused node-less points
        nodes = [point.node for point in submission.points if point.node is not None]
    else:
        for point in submission.points:
            probe: Node | None = point.node
            while probe is not None and probe.kind != scope:
                probe = probe.parent
            if probe is None:
                raise ValueError(
                    f"no {scope.value} node above point {point.concurrency}"
                )
            nodes.append(probe)
    return tuple({node.dir: node for node in nodes}.values())


def _when_rows_hold(check: Check, node: Node, submission: Submission) -> bool:
    """Every `when` row holds at this node (a `when`-free check always applies).

    A `when` row that failed to *evaluate* is refused loud: an unjudgeable gate
    must never read as a false predicate, which would switch the check -- and
    its dependents' vouches -- off by omission.
    """
    scopes = _scopes_rows_view(node, submission)
    for row in check.when:
        verdict = Verdict.evaluate(row, scopes)
        if verdict.broken:
            raise ValueError(
                f"check {check.id!r}: `when` row cannot be evaluated at "
                f"{node.dir}: {verdict.why}"
            )
        if not verdict.ok:
            return False
    return True
