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
`assert` row through `Verdict.evaluate`, returning the `Collected` record: the
row findings plus one `CheckRun` per check (the state counts the report words
its `ran at N`/`NEVER-RAN` lines from). A `needs` gate speaks per node: a check
runs only where every named upstream chain ran and held at the nodes that
vouch for it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from inference_endpoint.validation.check import Check, Cohort, Severity
from inference_endpoint.validation.compare import Ref, Verdict
from inference_endpoint.validation.submission import Point, Submission
from inference_endpoint.validation.tree import Node, RuleScopes


@dataclass(frozen=True)
class Finding:
    """One evaluated row's outcome at one applicable node: its `Verdict`, the
    row's own label, and the check's provenance (`rule`, `fix`)."""

    check_id: str
    severity: Severity
    scope: RuleScopes
    node_directory: Path
    verdict: Verdict
    rule: str
    row: str = ""
    fix: str | None = None

    @classmethod
    def from_check_row(
        cls,
        check: Check,
        scope,
        node_directory: Path,
        verdict: Verdict,
        row: str = "",
    ) -> Finding:
        """The `Finding` for one evaluated row.

        Example:

            finding = Finding.from_check_row(check, RuleScopes.POINT, dir, verdict, "1 eq 2")
            finding.check_id  # check.id

        Args:
            check (Check): The check the row belongs to.
            scope (RuleScopes | str): The check's scope (a `str` is coerced).
            node_directory (Path): The node the row was evaluated at.
            verdict (Verdict): The row's own verdict.
            row (str): The row's prose label (`row.describe()`). (Default: "")
        Returns:
            Finding: The record with the check's `rule`/`fix` attached.
        """
        return cls(
            check_id=check.id,
            severity=check.severity,
            scope=scope if isinstance(scope, RuleScopes) else RuleScopes(scope),
            node_directory=node_directory,
            verdict=verdict,
            rule=check.rule,
            row=row,
            fix=check.fix,
        )

    @classmethod
    def collect(cls, cohort: Cohort, submission: Submission) -> Collected:
        """The `Collected` record of one cohort run over one submission.

        One `Finding` per evaluated `assert` row at each applicable node (holds
        too), plus one `CheckRun` per check: the state counts the report's
        `ran at N instance(s)`/`NEVER-RAN` lines are worded from, so a check
        the `when` rows or a `needs` chain kept from running is told apart
        from one that ran and failed.

        A load-level check first (`_validate_*`: a submission the walk cannot
        cover, an unreadable artifact, or a checker root the cohort names but
        the tree lacks is an error -- silence must never be the verdict). Then
        `_collect_candidates` validates `needs` names (unknown -> ValueError)
        and `_GateResolver` resolves each `needs` chain per node. Each
        (check, node) is evaluated exactly once -- the chain first (a disavowed
        chain silences the check without evaluating anything), then its `when`,
        then its rows -- and the same record feeds gating and `_row_findings`.

        A check whose own rows fail still emits its Findings. Only its
        dependents stay silent, and only at the nodes whose upstreams did not
        hold there.
        """
        _validate_points(submission)
        _validate_artifacts(submission)
        _validate_declared_values(cohort, submission)
        _validate_ruleset_needs(cohort, submission)
        candidates = _collect_candidates(cohort)
        gate = _GateResolver({check.id: check for check in candidates}, submission)
        findings: list[Finding] = []
        runs: dict[str, CheckRun] = {}
        for check in candidates:
            ran_at = gated_off = silenced = 0
            for node in _scope_nodes(check.scope, submission):
                evaluation = gate.evaluation_at(check, node)
                if evaluation.state is _CheckState.NOT_APPLICABLE:
                    gated_off += 1
                    continue
                if evaluation.state is _CheckState.SILENCED:
                    silenced += 1
                    continue
                ran_at += 1
                findings.extend(_row_findings(check, node, evaluation))
            runs[check.id] = CheckRun(
                ran_at=ran_at, gated_off=gated_off, silenced=silenced
            )
        return Collected(tuple(findings), runs)


@dataclass(frozen=True)
class CheckRun:
    """One check's fate across the nodes `Finding.collect` offered it.

    State counts, not inference from emitted findings: a `when`-only check runs
    at every applicable node and emits none, and a check its `needs` chain
    silenced ran nowhere -- the report must tell these apart, and say which.

    Args:
        ran_at (int): Nodes where its `assert` rows were evaluated (holds and
            failures alike).
        gated_off (int): Nodes its `when` rows excluded.
        silenced (int): Nodes where its `needs` chain refused to vouch.
    """

    ran_at: int
    gated_off: int
    silenced: int


@dataclass(frozen=True)
class Collected:
    """One collection pass: the row findings and every check's run record.

    Args:
        findings (tuple[Finding, ...]): One per evaluated `assert` row at each
            applicable node (holds too).
        runs (Mapping[str, CheckRun]): One record per candidate check (cohort
            order), keyed by check id.
    """

    findings: tuple[Finding, ...]
    runs: Mapping[str, CheckRun]


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
    """The submission's tree root (the carried `tree`)."""
    return submission.tree


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


def _validate_declared_values(cohort: Cohort, submission: Submission) -> None:
    """Fail loud on a present fact value its declaration will not carry.

    One pass per node through `RefTable.validate_values` (type match, allowlist
    membership, declared-block shape): the cohort's `valid_refs.yaml` owns the
    values' contract, so this is a collect-time refusal, never a finding.
    """
    if cohort.refs is None:
        return
    for node in _all_nodes(_tree_root(submission)):
        fields = node.scopes().get(node.kind.value, {})
        try:
            cohort.refs.validate_values(node.kind.value, fields)
        except ValueError as exc:
            # The run directory names the offending node: a multi-pareto upload
            # has many `point.config.…` paths, one per run.
            raise ValueError(f"{node.dir}: {exc}") from exc


def _validate_ruleset_needs(cohort: Cohort, submission: Submission) -> None:
    """Fail loud when a cohort names the `ruleset` root the tree was built without.

    A forgotten `ruleset=` kwarg must be a load error, not rows resolving
    "missing" on a healthy upload. (`derived`/`regions` are per-pareto and
    always present at a pareto's nodes. Cohort load already refuses rows
    naming them anywhere else.)
    """
    root = _tree_root(submission)
    for check in cohort.checks:
        for row in (*check.when, *check.assert_):
            for ref in (r for r in row.operands() if isinstance(r, Ref)):
                if ref.parts[0] == "ruleset" and root.ruleset is None:
                    raise ValueError(
                        f"check {check.id!r} names {ref} but the tree was built "
                        "without ruleset"
                    )


def _all_nodes(node: Node) -> Iterator[Node]:
    """`node` and every node below it."""
    yield node
    for child in node.children:
        yield from _all_nodes(child)


def _collect_candidates(cohort: Cohort) -> tuple[Check, ...]:
    """Every check in the cohort with `needs` names validated against itself.

    Raises:
        ValueError: When an id is referenced but absent from the cohort, or a `needs`
            cycle exists, at load -- a malformed cohort must fail identically for every
            submission, never only where the `when` rows happen to lead the walk.
    """
    by_id = {check.id: check for check in cohort.checks}
    for check in cohort.checks:
        for token in check.needs:
            if token not in by_id:
                raise ValueError(f"check {check.id!r} needs unknown check {token!r}")
    # A `needs` cycle is refused here, by DFS before any row runs.
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
    return cohort.checks


class _CheckState(Enum):
    """Whether a check's `assert` rows were evaluated and held at one node.

    `HOLDS`: its `when` applied there, its own chains vouched, every `assert`
    row held. `FAILS`: it ran there and a row failed. `NOT_APPLICABLE`: its
    `when` excluded the node -- it makes no claim there. `SILENCED`: its own
    `needs` chain refused to vouch, so its rows never ran; a silenced check
    disavows its own dependents in turn. Keeping "no claim" and "cannot vouch"
    apart is what lets a `when`-gated upstream vouch exactly where it speaks
    instead of silencing whole subtrees.
    """

    HOLDS = "holds"
    FAILS = "fails"
    NOT_APPLICABLE = "not_applicable"
    SILENCED = "silenced"


@dataclass(frozen=True)
class _Evaluation:
    """One (check, node) evaluation: computed once, consumed by gate and report.

    Gating reads `state`; `_row_findings` renders `verdicts` -- the same row
    verdicts, so the two consumers can never disagree. `verdicts` is empty
    unless the check actually ran here (`HOLDS`/`FAILS`).

    Args:
        state (_CheckState): The check's state at the node.
        verdicts (tuple[Verdict, ...]): One verdict per `assert` row, in row
            order. Empty when the check did not run here. (Default: ())
    """

    state: _CheckState
    verdicts: tuple[Verdict, ...] = ()


class _GateResolver:
    """Per-(check, node) `needs` resolution and row evaluation, memoized.

    A check speaks at a node only where every named upstream chain **vouches**
    for it -- claims at least one node of its scope in the dependent node's
    chain-or-subtree and `HOLDS` at each (see `_chain_holds`). One `FAILS`
    node disavows, and so does a `SILENCED` one: a check whose own premises
    refused cannot vouch for anything downstream. A `when`-gated
    `NOT_APPLICABLE` node is no claim at all. Every (check, node) is evaluated
    exactly once (`_evaluate`) and every consumer reads that record. A `needs`
    cycle is refused at load (`_collect_candidates`'s DFS); the recursion trail
    here is the backstop.
    """

    def __init__(self, by_id: Mapping[str, Check], submission: Submission) -> None:
        self._by_id = by_id
        self._submission = submission
        self._memo: dict[tuple[str, str], _Evaluation] = {}
        self._scopes: dict[str, dict[str, Any]] = {}

    def evaluation_at(self, check: Check, node: Node) -> _Evaluation:
        """`check`'s one evaluation at `node` (chain, then `when`, then rows)."""
        return self._evaluate(check, node, ())

    def _chain_holds(
        self, check: Check, node: Node, trail: tuple[tuple[str, str], ...]
    ) -> bool:
        """Whether every named upstream chain vouches for `node`.

        An upstream vouches when it **claims** at least one node of its scope
        in the dependent node's chain-or-subtree (`_chain_nodes`) and `HOLDS`
        at every node where it claims. A `SILENCED` claim disavows like a
        `FAILS` one -- an indirect failure must not hide behind it -- while a
        `when`-gated `NOT_APPLICABLE` node is no claim at all. An empty
        vouch-set is no vouch: an unverified premise never passes a dependent.
        """
        for token in check.needs:
            upstream = self._by_id[token]
            claims = [
                state
                for upstream_node in _chain_nodes(upstream.scope, node)
                if (state := self._evaluate(upstream, upstream_node, trail).state)
                in (_CheckState.HOLDS, _CheckState.FAILS, _CheckState.SILENCED)
            ]
            if not claims or any(state is not _CheckState.HOLDS for state in claims):
                return False
        return True

    def _evaluate(
        self, check: Check, node: Node, trail: tuple[tuple[str, str], ...]
    ) -> _Evaluation:
        """`check`'s record at `node`: chain first, then `when`, then its rows."""
        key = (check.id, str(node.dir))
        if key in self._memo:
            return self._memo[key]
        if key in trail:
            raise ValueError(f"`needs` cycle through {check.id!r}")
        trail = (*trail, key)
        if not self._chain_holds(check, node, trail):
            evaluation = _Evaluation(_CheckState.SILENCED)
        else:
            scopes = self._row_scopes(node)
            if not _when_rows_hold(check, scopes, node.dir):
                evaluation = _Evaluation(_CheckState.NOT_APPLICABLE)
            else:
                verdicts = tuple(Verdict.evaluate(row, scopes) for row in check.assert_)
                state = (
                    _CheckState.HOLDS
                    if all(verdict.ok for verdict in verdicts)
                    else _CheckState.FAILS
                )
                evaluation = _Evaluation(state, verdicts)
        self._memo[key] = evaluation
        return evaluation

    def _row_scopes(self, node: Node) -> dict[str, Any]:
        """The row-resolvable scope map at `node`, built once per node."""
        key = str(node.dir)
        if key not in self._scopes:
            self._scopes[key] = _scopes_rows_view(node, self._submission)
        return self._scopes[key]


_SCOPE_DEPTH = {
    RuleScopes.SUBMISSION: 0,
    RuleScopes.SYSTEM: 1,
    RuleScopes.PARETO: 2,
    RuleScopes.POINT: 3,
}


def _chain_nodes(scope: RuleScopes, node: Node) -> tuple[Node, ...]:
    """The `scope`-kind nodes a check at `node` depends on: up the chain, or down the subtree.

    Same-or-ancestor scope resolves to the one chain node (a point's pareto,
    system and submission ancestors). A finer scope resolves to every such node
    below `node` (a system-scoped check vouches with all of its points). A
    subtree without any yields no nodes -- the chain claims nothing there.

    Raises:
        ValueError: When no ancestor of the requested `scope` kind exists above `node`
            (a load-level mistake).
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
    check: Check, node: Node, evaluation: _Evaluation
) -> tuple[Finding, ...]:
    """The check's evaluated rows at one node, one Finding each.

    Consumes the record `_GateResolver` already computed: holds and failures
    alike -- the report side filters `not verdict.ok` for a failure summary.
    The audit trail keeps the record.
    """
    return tuple(
        Finding.from_check_row(
            check, check.scope, node.dir, verdict, row=row.describe()
        )
        for row, verdict in zip(check.assert_, evaluation.verdicts, strict=True)
    )


def _scopes_rows_view(node: Node, submission: Submission) -> dict[str, Any]:
    """Row-resolvable scope map at `node`: the tree's fields plus the pareto's `derived`.

    `derived.<name>` rows address per-pareto named facts (`Pareto.derived()`),
    so the mapping is attached only where a pareto owns the node: nodes above
    every pareto (a multi-pareto upload's system/submission) carry no `derived`
    root, and cohort load refuses rows naming one there.
    """
    scopes: dict[str, Any] = node.scopes()
    pareto = submission.pareto_at(node)
    if pareto is not None:
        scopes["derived"] = pareto.derived()
    return scopes


def _scope_nodes(scope: RuleScopes, submission: Submission) -> tuple[Node, ...]:
    """Unique nodes of `scope` kind reached by walking each point's node chain.

    `Point` reaches the points' own `PointNode`s. `pareto`/`system`/`submission`
    walk the parents (deduplicated because a pareto's points share a pareto
    node).

    Raises:
        ValueError: When a chain lacks the requested kind -- a load-level mistake, not
            a report item.
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


def _when_rows_hold(check: Check, scopes: Mapping[str, Any], where: Path) -> bool:
    """Every `when` row holds at this node (a `when`-free check always applies).

    A `when` row that failed to *evaluate* is refused loud: an unjudgeable gate
    must never read as a false predicate, which would switch the check -- and
    its dependents' vouches -- off by omission. An operand whose optional fact
    the upload never declared is not unjudgeable: by schema the row does not
    hold, and `compare._missing_outcome` keeps that case distinct from an
    unjudgeable one.

    Args:
        check (Check): The check whose `when` rows gate.
        scopes (Mapping[str, Any]): The node's row-resolvable scope map.
        where (Path): The node's directory, for the refusal message.
    Returns:
        bool: Whether every `when` row holds.
    Raises:
        ValueError: When a `when` row cannot be evaluated.
    """
    for row in check.when:
        verdict = Verdict.evaluate(row, scopes)
        if verdict.broken:
            raise ValueError(
                f"check {check.id!r}: `when` row cannot be evaluated at "
                f"{where}: {verdict.why}"
            )
        if not verdict.ok:
            return False
    return True
