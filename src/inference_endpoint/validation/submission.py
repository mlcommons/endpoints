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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class Submission:
    """A system's `system_desc` fields over its ordered measurement points."""

    system_desc: Mapping[str, object]
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
