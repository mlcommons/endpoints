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

"""Submission and Point: the fields a compliance check computes against.

`Point` carries the three per-measurement fields the checklist's named shapes
read (Endpoints Policy Section 8.3's measurement point YAML Field table for
`offline` and `concurrency`; Section 4.1's Primary Metrics table for
`system_tps` = `total_output_tokens / elapsed_duration_seconds`):

https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md#83-measurement-point-yaml

plus its `PointNode` (W2 tree) so the wider artifact fields stay reachable
without duplicating them here. `Submission` pairs a system's `system_desc`
fields with its ordered points; it is the subject `Regions.for_submission`,
Point-cap, Offline-ordering and per-point Concurrency-in-range are computed
against.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from inference_endpoint.validation.regions import Regions
from inference_endpoint.validation.tree import PointNode


@dataclass(frozen=True)
class Point:
    """One measurement point's named fields, per Endpoints Policy Section 8.3.

    `offline` is the field's three-state declaration verbatim (`"dedicated"`,
    `"elected"`, or `None` for absent-or-`none`): a row with `Point.offline is
    None` is neither an elected nor a dedicated Offline run.
    """

    offline: Literal["dedicated", "elected"] | None
    concurrency: int
    system_tps: float
    node: PointNode

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
        decimal places, once at the final figure (`:897`'s Units and rounding
        rule). `engaged_nodes` (Y) and `provisioned_nodes` (N) are supplied by
        the caller until the fuller `nodes_used` carry lands on `Point` itself.

        `Point(offline=None, concurrency=32, system_tps=6000.0, node=None).point_power_kw(152.10, 16, 18)`
        -> `135.2` (the `:1906` golden row's `152.10 x 16/18`; verified
        `round(152.10 * 16 / 18, 2) == 135.2`). With `engaged_nodes ==
        provisioned_nodes` this reduces to `round(provisioned_power_kw, 2)`
        (the `:1907` row's `152.10 x 18/18 == 152.1`).
        """
        return round(provisioned_power_kw * engaged_nodes / provisioned_nodes, 2)


@dataclass(frozen=True)
class Submission:
    """A system's `system_desc` fields over its ordered measurement points."""

    system_desc: Mapping[str, Any]
    points: Sequence[Point]

    def __post_init__(self) -> None:
        # frozen dataclass: object.__setattr__ is the only sanctioned mutation.
        object.__setattr__(self, "points", tuple(self.points))

    def point_cap(self) -> bool:
        """Whether the Endpoints Policy Section 5.6 point cap holds (<= 32 points):
        https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md#56-maximum-point-cap

        `<= 32` counts every measurement point including the Offline point.
        `Submission([...32 points]).point_cap()` -> True (verified: `32 <= 32`);
        33 points -> False (`33 > 32`).
        """
        return len(self.points) <= 32

    def out_of_range_points(self, regions: Regions) -> tuple[Point, ...]:
        """Points whose `concurrency` falls outside `regions.coverage` (Section 5.4's row).

        Endpoints Policy Section 5.4's compliance-checklist row `Concurrency in
        range` says "each point's concurrency falls within a valid region
        (including the 10% High Concurrency margin) ... Flag out-of-range
        points":
        https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md#54-regions-of-interest

        `Submission([...p512, p9013]).out_of_range_points(regions_ex_a)` ->
        `(p9013,)`: Example A's coverage is `Interval(1, 9012)`, so 9013 (above
        the 10% margin) is the flagged point. Takes `Regions` rather than
        recomputing so one derivation source stays.
        """
        return tuple(p for p in self.points if p.concurrency not in regions.coverage())

    def offline_ordering(self) -> bool | None:
        """Whether the dedicated Offline point meets Endpoints Policy Section 5.7.2:
        https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md#572-relationship-to-maximum-supported-concurrency

        Both clauses: `system_tps(Offline) >= 0.98 * system_tps at the C_max
        point` and `concurrency(Offline) >= C_max` (= the declared
        `system_desc["max_supported_concurrency"]`). `None` means not
        applicable: the point is `elected`, or no `offline` declaration is
        present at all (agentic benchmarks). Mixed/multiple declarations raise
        ValueError (the "exactly one point carries `offline`" row is a separate
        check).

        `Submission.offline_ordering` example (values machine-verified):
        `0.98 * 5400.0 == 5292.0`, so a dedicated Offline point with
        `system_tps = 5292.0` and `concurrency = 8192` against an
        `M = 8192` system passes; `5291.0` fails the ratio clause alone.
        """
        declared = [p for p in self.points if p.offline is not None]
        elected = [p for p in declared if p.offline == "elected"]
        dedicated = [p for p in declared if p.offline == "dedicated"]
        if len(declared) > 1 or (elected and dedicated):
            raise ValueError("exactly one point may carry an `offline` declaration")
        if elected or not declared:
            return None
        offline_point = dedicated[0]
        c_max = self.system_desc["max_supported_concurrency"]
        c_max_points = [
            p for p in self.points if p is not offline_point and p.concurrency == c_max
        ]
        if len(c_max_points) != 1:
            raise ValueError("need exactly one C_max point (concurrency == M)")
        c_max_point = c_max_points[0]
        return (
            offline_point.system_tps >= 0.98 * c_max_point.system_tps
            and offline_point.concurrency >= c_max
        )

    def point_count_min(self) -> int:
        """The Endpoints Policy Section 5.3 minimum point count this shape needs.

        8 where a dedicated `offline` run is declared ("including a dedicated
        Offline run (non-agentic)"); 7 where the `C_max` point is elected as
        the Offline result, or for agentic benchmarks (which declare no
        `offline`, see Section 5.7): a submission then contains one fewer
        distinct run (`endpoints_rules.md` on §5.3's minimum).
        """
        return 8 if any(p.offline == "dedicated" for p in self.points) else 7

    def accuracy_run_count_requirement(self) -> int:
        """The Endpoints Policy Section 4.3 accuracy result count `N` this shape needs.

        `:1662`'s "Exactly `N` accuracy results are present -- 5 for
        non-agentic, 4 for agentic ([Section 4.3](#43-accuracy-metric)) --
        one per mandatory point, and none at a submitter's-choice point":
        5 where any `offline` run is declared (non-agentic), 4 otherwise.
        """
        return 5 if any(p.offline is not None for p in self.points) else 4

    def derived(self) -> Mapping[str, Any]:
        """Named derivations for `derived.<name>` row refs, as a lazy mapping.

        The charter's boundary rule item 1 ("named derivations on their
        subject, exposed as fields `Ref` reads"): keys are `offline_ordering`,
        `point_cap`, `point_count_min` and `accuracy_run_count_requirement`,
        each resolved via its own named method at first access and memoized
        (`DerivedFields`). `Submission({...}, points).derived()["point_cap"]`
        -> the cap verdict; an unlisted name -> `KeyError`.
        """
        return DerivedFields(self)


class DerivedFields(Mapping[str, Any]):
    """Lazy `derived.<name>` mapping: a derivation runs only when a row names it.

    `Ref.resolve` walks this like any scope root. `view["point_cap"]` ->
    `Submission.point_cap()`'s value, memoized; `view["nope"]` -> `KeyError`,
    so `Ref.resolve` reports the field absent. Membership asks the known-names
    table only and never runs a derivation, so one derivation's fail-loud
    `ValueError` (e.g. `derived.offline_ordering` with no unique `C_max`
    point) reaches exactly the row that named it.
    """

    __slots__ = ("_submission", "_cache")

    def __init__(self, submission: Submission) -> None:
        self._submission = submission
        self._cache: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        """Value of `derived.<name>`: runs `Submission.<name>()` once, memoized."""
        if name not in _DERIVATION_METHODS:
            raise KeyError(name)
        if name not in self._cache:
            self._cache[name] = _DERIVATION_METHODS[name](self._submission)
        return self._cache[name]

    def __contains__(self, name: object) -> bool:
        """Whether `name` is a known derivation name; never runs one."""
        return name in _DERIVATION_METHODS

    def __iter__(self) -> Iterator[str]:
        """Every known derivation name (the `derived_fields` key set)."""
        return iter(_DERIVATION_METHODS)

    def __len__(self) -> int:
        return len(_DERIVATION_METHODS)


_DERIVATION_METHODS: dict[str, Callable[[Submission], Any]] = {
    "offline_ordering": Submission.offline_ordering,
    "point_cap": Submission.point_cap,
    "point_count_min": Submission.point_count_min,
    "accuracy_run_count_requirement": Submission.accuracy_run_count_requirement,
}
