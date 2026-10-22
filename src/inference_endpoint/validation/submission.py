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

"""Measurement points, the per-pareto derivation unit, and the uploaded tree.

Endpoints Rules source (pinned):
https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md

`Point` carries one `r<N>` run's named facts per §8.3 (`concurrency`,
`offline`) plus `system_tps` (§4.1's aggregate metric, from the run's
`result_summary.json`) and the run's own `system_desc.json` fields (§8.2.1
places one descriptor per run directory). `Pareto` is one
`results/<system>/<model>/` -- the unit every policy derivation is computed
per (§5.6 caps "a single submission's pareto". §5.4 lets different paretos
choose different `c_min_override`s). `Submission` is the uploaded tree: its
paretos and the tree nodes rows resolve facts from.

Facts are read once, here (`Point.from_node`): the tree feeds rows and `Point`
feeds the derivations, and both come from the same parsed artifacts, so no
second copy can drift from the first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from inference_endpoint.validation.catalog import Catalog
from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.tree import Node, RuleScopes, SubmissionNode

RULES_URL = (
    "https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/"
    "endpoints_rules.md"
)

# §5.6: "the total number of measurement points on a single submission's
# pareto" may not exceed 32.
POINT_CAP = 32

# §5.7.2: the dedicated Offline run's system_tps must reach 0.98x the C_max
# point's.
OFFLINE_TPS_RATIO = 0.98

# §5.3: one mandatory point per concurrency region (Ultra Low, Low, Medium,
# High), three submitter's-choice points, and -- non-agentic -- one Offline
# run. §4.3's accuracy count N is the mandatory-point count (4 + the Offline
# slot); §5.3's minimum point count adds the three choice points.
MANDATORY_REGION_COUNT = 4
SUBMITTERS_CHOICE_COUNT = 3


def normalize_offline(declared: Any) -> Any:
    """The `offline` declaration as `Point.offline` carries it.

    Endpoints Policy Section 8.3's field is a three-state declaration:
    absent-or-`none` is "no declaration", so both spellings normalize together.

    Args:
        declared (Any): The raw `offline` value read from a run's `point.yaml`.
    Returns:
        Any: `"dedicated"` or `"elected"` when declared. `None` for
            absent-or-`none`.
    """
    return None if declared in (None, "none") else declared


def _mapping_artifact(node: Node, name: str) -> Mapping[str, Any]:
    """The parsed mapping artifact `name` of `node`, or a refusal naming it."""
    value, error = node.read(name)
    if error is not None or not isinstance(value, Mapping):
        raise ValueError(
            f"{node.dir / name} cannot carry its facts: {error or 'not a mapping'}"
        )
    return value


def _int_fact(source: Mapping[str, Any], name: str, where: str) -> int:
    """`source[name]` as the integer fact `where` documents."""
    value = source.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} needs an integer {name}")
    return value


@dataclass(frozen=True)
class Point:
    """One measurement point's named facts (Endpoints Policy §8.3 + §4.1).

    Args:
        offline (Literal["dedicated", "elected"] | None): The field's three-state
            declaration verbatim (`"dedicated"`, `"elected"`, or `None` for
            absent-or-`none`): a point with `offline is None` is neither an elected nor
            a dedicated Offline run.
        concurrency (int): The run's declared point concurrency (§8.3).
        system_tps (float): §4.1's aggregate metric in the run's
            `result_summary.json` (§8.1).
        system_desc (Mapping[str, Any]): This run's own `system_desc.json` fields
            (§8.2.1).
        node (Node | None): The run directory's tree node. `None` on a
            derivation-only point (built by hand for derivations over named values, not
            walked from the tree).
    """

    offline: Literal["dedicated", "elected"] | None
    concurrency: int
    system_tps: float
    system_desc: Mapping[str, Any] = field(hash=False)
    # A derivation-only Point carries no tree link; `Finding.collect` refuses
    # one without its point node.
    node: Node | None = None

    def __post_init__(self) -> None:
        if self.offline not in (None, "dedicated", "elected"):
            raise ValueError(
                f"offline must be 'dedicated', 'elected' or null, got {self.offline!r}"
            )

    @classmethod
    def from_node(cls, node: Node) -> Point:
        """The `Point` one `r<N>` run directory's artifacts declare.

        Per §8.2.1/§8.3 the run carries its own `system_desc.json` and
        `point.yaml`. `system_tps` is §4.1's aggregate metric in the run's
        `result_summary.json` (§8.1), not a `point.yaml` field.

        Args:
            node (Node): The `r<N>` run directory's tree node.
        Returns:
            Point: The run's named facts, with `node` carried on the point.
        Raises:
            ValueError: When a named source is missing or mistyped. Each named
                source is enforced here, at load: a missing or mistyped fact names its
                artifact rather than reaching a row as an absent value.
        """
        config = _mapping_artifact(node, "point.yaml")
        concurrency = _int_fact(config, "concurrency", f"{node.dir}/point.yaml")
        offline = normalize_offline(config.get("offline"))
        if offline not in (None, "dedicated", "elected"):
            raise ValueError(
                f"{node.dir}/point.yaml offline must be 'dedicated', 'elected' "
                f"or null, got {offline!r}"
            )
        system_desc = _mapping_artifact(node, "system_desc.json")
        _int_fact(
            system_desc, "max_supported_concurrency", f"{node.dir}/system_desc.json"
        )
        summary = _mapping_artifact(node, "result_summary.json")
        system_tps = summary.get("system_tps")
        if isinstance(system_tps, bool) or not isinstance(system_tps, int | float):
            raise ValueError(
                f"{node.dir}/result_summary.json needs a numeric system_tps"
            )
        return cls(
            offline=offline,
            concurrency=concurrency,
            system_tps=float(system_tps),
            system_desc=system_desc,
            node=node,
        )

    def point_power_kw(
        self,
        provisioned_power_kw: float,
        engaged_nodes: int,
        provisioned_nodes: int,
    ) -> float:
        """The Endpoints Policy Section 4.5.3 point denominator `provisioned_power_kw x Y/N`.

        Section 4.5.2/4.5.3's normalization denominator per measurement point:
        provisioned power scaled by the node fraction `Y_s / N_s` engaged at
        this point (whole nodes rounded up from accelerators). Rounded to two
        decimal places, once at the final figure (§4.5's Units-and-rounding
        rule). `engaged_nodes` (Y) and `provisioned_nodes` (N) are supplied by
        the caller until the fuller `nodes_used` carry lands on `Point` itself.

        Example of Appendix C.3's golden row:

            point = Point(offline=None, concurrency=32, system_tps=6000.0, system_desc={})
            point.point_power_kw(152.10, 16, 18)  # 135.2 == round(152.10 * 16 / 18, 2)

        With `engaged_nodes == provisioned_nodes` this reduces to `round(provisioned_power_kw, 2)`
        (the `152.10 x 18/18 == 152.1` row).

        Args:
            provisioned_power_kw (float): The system's provisioned power, in kW.
            engaged_nodes (int): `Y`: the nodes engaged at this point (whole
                nodes rounded up from accelerators).
            provisioned_nodes (int): `N`: the system's provisioned node count.
        Returns:
            float: The `provisioned_power_kw x Y/N` denominator, rounded to two
                decimal places.
        """
        return round(provisioned_power_kw * engaged_nodes / provisioned_nodes, 2)


@dataclass(frozen=True)
class Pareto:
    """One `results/<system>/<model>/`: the unit every policy derivation is per.

    Args:
        system_desc (Mapping[str, Any]): The declared facts shared by the pareto's
            points: `max_supported_concurrency` and `c_min_override`, both §8.3 fields.
        points (Sequence[Point]): The ordered `Point`s of one pareto.
        node (Node | None): The tree's pareto node when built from an upload
            (`Pareto.from_nodes`). `None` for a hand-built derivation-only Pareto
            (`Pareto(system_desc, points)`), whose points carry no node and which
            `Finding.collect` refuses to walk.
    """

    system_desc: Mapping[str, Any] = field(hash=False)
    points: Sequence[Point]
    # The tree's pareto node when built from an upload (`Pareto.from_nodes`);
    # `None` for a derivation-only hand-built Pareto.
    node: Node | None = None

    def __post_init__(self) -> None:
        # frozen dataclass: object.__setattr__ is the only sanctioned mutation.
        object.__setattr__(self, "points", tuple(self.points))
        self._validate_shape()

    @classmethod
    def from_nodes(cls, node: Node) -> Pareto:
        """The Pareto one `<system>/<model>` node's run directories declare.

        The pareto's §5.5 regions latch onto its tree node here, so the
        `regions` fold reaches every row below it (`Node.scopes` reads the
        nearest ancestor's fold).

        Args:
            node (Node): The `<system>/<model>` pareto node.
        Returns:
            Pareto: The pareto of the node's run directories, with its §5.5
                regions latched onto `node`.
        """
        points = tuple(Point.from_node(child) for child in node.children)
        pareto = cls(_shared_facts(points, node.dir), points, node=node)
        node.regions = pareto.regions()
        return pareto

    def _validate_shape(self) -> None:
        """Fail loud on shapes the derivations would trip over mid-report.

        These are rule violations, not parse errors (Endpoints Policy §9.1's
        Offline-point-present and Max-concurrency-declared rows). Each message
        names its row until they reach the report as findings.

        Raises:
            ValueError: When the pareto has no measurement point. When its points
                do not declare one integer `max_supported_concurrency` (§9.1, Max
                concurrency declared). When more than one point carries an `offline`
                declaration (§9.1, Offline point present). When an `elected` declaration
                sits off the C_max point (§9.1, Offline point present). When a dedicated
                Offline run lacks exactly one C_max point (§9.1, Offline ordering).
        """
        if not self.points:
            raise ValueError("a pareto needs at least one measurement point")
        c_max = self.system_desc.get("max_supported_concurrency")
        if (
            isinstance(c_max, bool)
            or not isinstance(c_max, int)
            or self.points[0].system_desc.get("max_supported_concurrency") != c_max
        ):
            raise ValueError(
                "a pareto's points must declare one integer "
                "max_supported_concurrency (Endpoints Policy §8.3; §9.1, "
                "Max concurrency declared)"
            )
        declared, elected, dedicated = self._offline_partition()
        if len(declared) > 1:
            raise ValueError(
                "exactly one point may carry an `offline` declaration "
                "(Endpoints Policy §9.1, Offline point present)"
            )
        if elected:
            point = elected[0]
            if point.concurrency != c_max:
                raise ValueError(
                    "an `elected` declaration must sit on the C_max point "
                    "(Endpoints Policy §9.1, Offline point present)"
                )
        if dedicated:
            rivals = [
                p
                for p in self.points
                if p is not dedicated[0] and p.concurrency == c_max
            ]
            if len(rivals) != 1:
                raise ValueError(
                    "a dedicated Offline run needs exactly one C_max point "
                    "(Endpoints Policy §9.1, Offline ordering)"
                )

    def _offline_partition(self) -> tuple[list[Point], list[Point], list[Point]]:
        """`(declared, elected, dedicated)` over the `offline` declaration field."""
        declared = [p for p in self.points if p.offline is not None]
        elected = [p for p in declared if p.offline == "elected"]
        dedicated = [p for p in declared if p.offline == "dedicated"]
        return declared, elected, dedicated

    def _c_max_point(self, exclude: Point | None = None) -> Point:
        """The unique non-`exclude` point at `concurrency == M` (validated at build)."""
        c_max = self.system_desc["max_supported_concurrency"]
        return next(
            p for p in self.points if p is not exclude and p.concurrency == c_max
        )

    def regions(self) -> Regions:
        """This pareto's §5.5 regions (`c_min_override`-aware, one source per pareto)."""
        return Regions.for_pareto(
            self.system_desc, [point.concurrency for point in self.points]
        )

    def point_cap(self) -> bool:
        """Whether Endpoints Policy §5.6's point cap holds (<= 32 points per pareto).

        `Pareto({...}, [...32 points]).point_cap()` -> True (verified:
        `32 <= 32`). 33 points -> False.

        Returns:
            bool: Whether the pareto holds at most `POINT_CAP` (32) measurement
                points. The dedicated Offline point counts.
        """
        return len(self.points) <= POINT_CAP

    def out_of_range_points(self, regions: Regions) -> tuple[Point, ...]:
        """Fixed-concurrency points outside `regions.coverage` (§9.1's Concurrency-in-range row).

        "Flag out-of-range points" (§9.1's Concurrency-in-range row of the
        compliance checklist) flags a point whose concurrency falls outside
        every valid region, the 10% High Concurrency margin included. A
        dedicated Offline point is exempt: it "is not a fixed-concurrency
        point" (§5.7.2) and its `concurrency` is the performance dataset's
        cardinality (§5.7), so no region judges it.

        Args:
            regions (Regions): The pareto's §5.5 `Regions`, taken rather than
                recomputed so one derivation source stays.
        Returns:
            tuple[Point, ...]: The fixed-concurrency points whose concurrency
                falls outside `regions.coverage`. Empty when every fixed-concurrency
                point is in range.
        """
        return tuple(
            p
            for p in self.points
            if p.offline != "dedicated" and p.concurrency not in regions.coverage()
        )

    def offline_ordering(self) -> bool | None:
        """Whether the dedicated Offline run meets Endpoints Policy §5.7.2's two clauses.

        `system_tps(Offline) >= OFFLINE_TPS_RATIO * system_tps` at the C_max
        point, and `concurrency(Offline) >= C_max` (the declared
        `max_supported_concurrency`). Mixed/multiple declarations and a missing
        C_max point are refused at build (`Pareto._validate_shape`), so this
        derivation's value is always meaningful.

        Returns:
            bool | None: True when both §5.7.2 clauses hold. False when either
                fails. `None` when not applicable: the point is `elected`, or no
                `offline` declaration is present at all (agentic benchmarks).
        """
        declared, elected, _ = self._offline_partition()
        if elected or not declared:
            return None
        offline_point = declared[0]
        c_max = self.system_desc["max_supported_concurrency"]
        c_max_point = self._c_max_point(exclude=offline_point)
        return (
            offline_point.system_tps >= OFFLINE_TPS_RATIO * c_max_point.system_tps
            and offline_point.concurrency >= c_max
        )

    def point_count_min(self) -> int:
        """Endpoints Policy §5.3's minimum point count this pareto needs.

        The four mandatory region points plus the three submitter's-choice
        points. The dedicated Offline run adds one more (`8`). A pareto whose
        C_max point is elected as the Offline result, and an agentic pareto
        (no `offline` declaration, §5.7), runs one fewer distinct run (`7`).

        Returns:
            int: `MANDATORY_REGION_COUNT` + `SUBMITTERS_CHOICE_COUNT` (7), plus
                one where a dedicated Offline run is declared (`8`). The elected or
                agentic shape stays at `7`.
        """
        base = MANDATORY_REGION_COUNT + SUBMITTERS_CHOICE_COUNT
        return base + (1 if any(p.offline == "dedicated" for p in self.points) else 0)

    def accuracy_run_count_requirement(self) -> int:
        """Endpoints Policy §4.3's accuracy result count `N` this pareto needs.

        "One per mandatory point": one per mandatory concurrency region (4)
        plus the Offline point where any `offline` run is declared
        (non-agentic), hence `5`. Agentic paretos declare none (`4`).

        Returns:
            int: `MANDATORY_REGION_COUNT` plus one where any `offline` run is
                declared (`5`). `MANDATORY_REGION_COUNT` alone for agentic paretos
                (`4`).
        """
        return MANDATORY_REGION_COUNT + (
            1 if any(p.offline is not None for p in self.points) else 0
        )

    def derived(self) -> Mapping[str, Any]:
        """Named derivations for `derived.<name>` row refs, as a lazy mapping.

        The boundary rule for derivations ("named derivations on their subject,
        exposed as fields `Ref` reads"): keys are `offline_ordering`,
        `point_cap`, `point_count_min`, `accuracy_run_count_requirement` and
        `out_of_range_points`, each resolved via its own named method at first
        access and memoized (`DerivedFields`). An unlisted name -> `KeyError`.

        Returns:
            Mapping[str, Any]: The `DerivedFields` view over this pareto: the
                five names above are its keys, and nothing runs until a row names one.
        """
        return DerivedFields(self)


@dataclass(frozen=True)
class Submission:
    """One uploaded tree: its paretos, and the nodes rows resolve facts from.

    `Submission.from_directory(root, ruleset=...)` walks
    `results/<system>/<model>/r<N>/` and builds one `Pareto` per
    `<system>/<model>` -- each run directory carrying its own
    `system_desc.json` (§8.2.1), `point.yaml` (§8.3) and
    `result_summary.json` (§8.1). `pareto_at(node)` locates the Pareto a tree
    node belongs to (nodes above every pareto have none: there is no
    single-pareto `derived`/`regions` view of a multi-pareto upload).

    Args:
        tree (Node): The uploaded tree. Its `<system>` children hold the
            `<model>` nodes the paretos hang from.
        paretos (Sequence[Pareto]): One `Pareto` per `<system>/<model>` of the
            tree, in tree order.
    """

    tree: Node
    paretos: Sequence[Pareto]

    def __post_init__(self) -> None:
        object.__setattr__(self, "paretos", tuple(self.paretos))

    @classmethod
    def from_directory(cls, root: Path, ruleset: Catalog | None = None) -> Submission:
        """Build one from its uploaded tree rather than by hand."""
        tree = SubmissionNode.from_directory(root, ruleset=ruleset)
        paretos: list[Pareto] = []
        for system in tree.children:
            for pareto_node in system.children:
                if not pareto_node.children:
                    raise ValueError(
                        f"{pareto_node.dir} has no run directories (r<N>) to check"
                    )
                paretos.append(Pareto.from_nodes(pareto_node))
        return cls(tree, paretos)

    @property
    def points(self) -> tuple[Point, ...]:
        """Every point of every pareto, in tree order (the runner's walk set)."""
        return tuple(point for pareto in self.paretos for point in pareto.points)

    def pareto_at(self, node: Node) -> Pareto | None:
        """The Pareto `node` belongs to. `None` for nodes above every pareto."""
        probe: Node | None = node
        while probe is not None and probe.kind is not RuleScopes.PARETO:
            probe = probe.parent
        if probe is None:
            return None
        return next((pareto for pareto in self.paretos if pareto.node is probe), None)


def _shared_facts(points: Sequence[Point], where: Path) -> Mapping[str, Any]:
    """The declared facts one pareto's points must agree on (§8.3/§5.4)."""
    if not points:
        raise ValueError(f"{where} has no run directories (r<N>) to check")
    first = points[0].system_desc
    for point in points[1:]:
        for key in ("max_supported_concurrency", "c_min_override"):
            if point.system_desc.get(key) != first.get(key):
                raise ValueError(
                    f"every point of {where} must declare the same {key} "
                    "(Endpoints Policy §8.3; §5.4, C_min override)"
                )
    return first


class DerivedFields(Mapping[str, Any]):
    """Lazy `derived.<name>` mapping: a derivation runs only when a row names it.

    `Ref.resolve` walks this like any scope root. `view["point_cap"]` ->
    `Pareto.point_cap()`'s value, memoized. `view["nope"]` -> `KeyError`, so
    `Ref.resolve` reports the field absent. Membership asks the known-names
    tables only and never runs a derivation, so one derivation's fail-loud
    `ValueError` reaches exactly the row that named it.

    Args:
        pareto (Pareto): The pareto whose named derivations this mapping exposes.
    """

    __slots__ = ("_pareto", "_cache")

    def __init__(self, pareto: Pareto) -> None:
        self._pareto = pareto
        self._cache: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        """Value of `derived.<name>`: runs its derivation once, memoized."""
        if name in self._cache:
            return self._cache[name]
        try:
            method = _DERIVATION_METHODS[name]
        except KeyError:
            raise KeyError(name) from None
        value = method(self._pareto)
        self._cache[name] = value
        return value

    def __contains__(self, name: object) -> bool:
        """Whether `name` is a known derivation name. Never runs one."""
        return name in _DERIVATION_METHODS

    def __iter__(self) -> Iterator[str]:
        """Every known derivation name (the `derived.<name>` key set)."""
        return iter(_DERIVATION_METHODS)

    def __len__(self) -> int:
        return len(_DERIVATION_METHODS)


_DERIVATION_METHODS: dict[str, Callable[[Pareto], Any]] = {
    "offline_ordering": Pareto.offline_ordering,
    "point_cap": Pareto.point_cap,
    "point_count_min": Pareto.point_count_min,
    "accuracy_run_count_requirement": Pareto.accuracy_run_count_requirement,
    "out_of_range_points": lambda pareto: pareto.out_of_range_points(pareto.regions()),
}

# The `derived.<name>` names a row may name (single owner: the table above).
DERIVATION_NAMES: tuple[str, ...] = tuple(_DERIVATION_METHODS)
