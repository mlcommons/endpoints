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

"""Concurrency regions of Endpoints Rules §5.5, exactly as its reference algorithm.

`Regions.for_bounds(c_min=..., c_max=...)` -> the five named regions of §5.5,
each an `Interval` whose `lo`/`hi` are the reference algorithm's `start`/`end`.
The three concurrency regions split `log2(C_max - C_min) / 3` with
round-half-to-even boundaries. `margin` is the 10% High Concurrency extension
to `ceil(1.10 * C_max)`. One `Regions` covers one pareto.

Endpoints Rules source:
https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from inference_endpoint.validation.ranges import Interval

# §5.4: `C_min` is a concurrency in [1, 32], derived or overridden.
C_MIN_LOWEST = 1
C_MIN_HIGHEST = 32

# §5.5: the 10% High Concurrency margin (`ceil(1.10 * C_max)`).
MARGIN_FACTOR = 1.10


class RegionKind(str, Enum):  # noqa: UP042 - repository wire-enum convention
    """Which region of Endpoints Policy Section 5.5 a concurrency value lands in.

    `values` are the five keys of that Section's `compute_regions` (the §5.4
    prose names `low_latency` the Ultra Low Concurrency region. The reference
    algorithm's key stays `low_latency`).
    """

    ULTRA_LOW_CONCURRENCY = "low_latency"
    LOW_CONCURRENCY = "low_concurrency"
    MED_CONCURRENCY = "med_concurrency"
    HIGH_CONCURRENCY = "high_concurrency"
    MARGIN = "margin"


@dataclass(frozen=True)
class Regions:
    """The five regions of Rules §5.5 over one pareto's `(C_min, C_max)` bounds.

    Args:
        low_latency (Interval): The Ultra Low Concurrency region, `1..C_min`. The
            §5.4 prose calls it "Ultra Low Concurrency" while §5.5's reference-algorithm
            key stays `low_latency`.
        low_concurrency (Interval): The Low Concurrency region, `C_min+1` through the
            first log-space boundary.
        med_concurrency (Interval): The Medium Concurrency region, up to the second
            log-space boundary.
        high_concurrency (Interval): The High Concurrency region, up to `C_max`.
        margin (Interval): The 10% High Concurrency margin, `C_max+1` through
            `ceil(1.10 * C_max)`.
    """

    low_latency: Interval
    low_concurrency: Interval
    med_concurrency: Interval
    high_concurrency: Interval
    margin: Interval

    @classmethod
    def for_pareto(
        cls, system_desc: Mapping[str, Any], concurrencies: Sequence[int]
    ) -> Regions:
        """Regions from one pareto's declared facts (`C_max` and the `C_min` derivation).

        `C_min` is `system_desc["c_min_override"]` where declared (§5.4), else the
        pareto's minimum point concurrency ("C_min is derived from the submission
        points", §5.5). Multiple points may sit in `[1, 32]`: §5.4 Example C's own
        seven-point set `{16, 24, 64, 96, 128, 256, 1000}` carries two, and with
        `c_min_override: 32` the points at 16 and 32 both sit in the Ultra Low
        Concurrency region (§5.4's override example).

        Args:
            system_desc (Mapping[str, Any]): The pareto's shared `system_desc.json`
                fields. `max_supported_concurrency` is the declared Maximum Supported
                Concurrency `M` (§8.3/§8.2 field name verbatim). `c_min_override`, where
                present, is the declared lower region boundary (§5.4).
            concurrencies (Sequence[int]): Every run directory's declared concurrency,
                in tree order.
        Returns:
            Regions: The five §5.5 regions over the resolved `(C_min, C_max)` bounds.
        Raises:
            ValueError: When `max_supported_concurrency` is missing or not an
                integer (§9.1, Max concurrency declared). When no points are given or a
                concurrency is not an integer. When a declared `c_min_override` is
                `null`, a non-integer, outside `[1, 32]`, or not equal to a submitted
                point's concurrency (§9.1, C_min override). When `for_bounds` refuses
                the resolved bounds.
        """
        c_max = system_desc.get("max_supported_concurrency")
        if not isinstance(c_max, int) or isinstance(c_max, bool):
            raise ValueError(
                "system_desc needs an integer max_supported_concurrency "
                "(Endpoints Policy §9.1, Max concurrency declared)"
            )
        if not concurrencies:
            raise ValueError("need at least one point to derive C_min")
        if not all(
            isinstance(x, int) and not isinstance(x, bool) for x in concurrencies
        ):
            raise ValueError("point concurrencies need to be integers")
        if "c_min_override" not in system_desc:
            return cls.for_bounds(c_min=min(concurrencies), c_max=c_max)
        declared = system_desc["c_min_override"]
        if (
            not isinstance(declared, int)
            or isinstance(declared, bool)
            or not C_MIN_LOWEST <= declared <= C_MIN_HIGHEST
            or declared not in concurrencies
        ):
            raise ValueError(
                f"c_min_override {declared!r} must be an integer in "
                f"[{C_MIN_LOWEST}, {C_MIN_HIGHEST}] equal to a submitted point's "
                "concurrency (Endpoints Policy §9.1, C_min override)"
            )
        return cls.for_bounds(c_min=declared, c_max=c_max)

    @classmethod
    def for_bounds(cls, *, c_min: int, c_max: int) -> Regions:
        """The Endpoints Policy §5.5 reference algorithm, bounds-checked.

        Example of §5.4's worked Example A:

            regions = Regions.for_bounds(c_min=32, c_max=8192)
            regions.low_latency  # Interval(lo=1, hi=32)
            regions.low_concurrency  # Interval(lo=33, hi=52)
            regions.med_concurrency  # Interval(lo=53, hi=437)
            regions.high_concurrency  # Interval(lo=438, hi=8192)
            regions.margin  # Interval(lo=8193, hi=9012)

        Example C's printed boundaries (17..26 and 27..117 at `c_min` 16, `c_max`
        1024) are the formula's (`round(16 + 2**6.652) == 117`).

        Args:
            c_min (int): The Ultra Low Concurrency region's upper bound, in `[1, 32]`
                (§5.4).
            c_max (int): The declared Maximum Supported Concurrency `M`, `> 32`. The
                High Concurrency region's upper bound.
        Returns:
            Regions: The five named regions, each an `Interval` whose `lo`/`hi` are
                the reference algorithm's `start` /`end`.
        Raises:
            ValueError: When `c_min` is outside `[1, 32]` (§9.1, C_min override)
                or `c_max <= 32` (§9.1, Max concurrency declared): the Rules admit no
                submission there.
        """
        if not C_MIN_LOWEST <= c_min <= C_MIN_HIGHEST:
            raise ValueError(
                f"C_min must be between {C_MIN_LOWEST} and {C_MIN_HIGHEST} "
                "(Endpoints Policy §9.1, C_min override)"
            )
        if c_max <= C_MIN_HIGHEST:
            raise ValueError(
                "Maximum Supported Concurrency must be > 32 "
                "(Endpoints Policy §9.1, Max concurrency declared)"
            )
        span = math.log2(c_max - c_min) / 3
        low_conc_end = round(c_min + 2**span)
        med_conc_end = round(c_min + 2 ** (2 * span))
        return cls(
            low_latency=Interval(1, c_min),
            low_concurrency=Interval(c_min + 1, low_conc_end),
            med_concurrency=Interval(low_conc_end + 1, med_conc_end),
            high_concurrency=Interval(med_conc_end + 1, c_max),
            margin=Interval(c_max + 1, math.ceil(MARGIN_FACTOR * c_max)),
        )

    def covers(self, concurrency: int) -> RegionKind | None:
        """The `RegionKind` a concurrency value lands in, or `None` outside every
        region.

        Endpoints Policy Section 5.4 requires each point's concurrency to fall in a
        valid region (including the 10% High Concurrency margin):
        https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md#54-regions-of-interest

        Example:

            regions = Regions.for_bounds(c_min=32, c_max=8192)
            regions.covers(32)  # RegionKind.ULTRA_LOW_CONCURRENCY
            regions.covers(9012)  # RegionKind.MARGIN
            regions.covers(0)  # None
            regions.covers(9013)  # None

        The result's truthiness is the predicate: `if regions.covers(x):`.

        Args:
            concurrency (int): The point's declared concurrency.
        Returns:
            RegionKind | None: The region containing `concurrency`, or `None` when no
                region does (the §9.1 Concurrency-in-range flag).
        """
        pairs = (
            (RegionKind.ULTRA_LOW_CONCURRENCY, self.low_latency),
            (RegionKind.LOW_CONCURRENCY, self.low_concurrency),
            (RegionKind.MED_CONCURRENCY, self.med_concurrency),
            (RegionKind.HIGH_CONCURRENCY, self.high_concurrency),
            (RegionKind.MARGIN, self.margin),
        )
        for kind, interval in pairs:
            if concurrency in interval:
                return kind
        return None

    def coverage(self) -> Interval:
        """`Interval(1, margin.hi)`: every concurrency a region covers, one interval.

        The five regions tile `[1, margin_end]` contiguously (a rounding collision
        gives a zero-width region, never a gap), so the §9.1 Concurrency-in-range
        row reduces to `concurrency in regions.coverage()`. The lower bound is 1,
        not `C_min`: §5.4's override example submits points below the declared
        `c_min_override` (points at 16 and 32 with the override at 32 are both in
        the Ultra Low Concurrency region), and every one of them is in range.

        Returns:
            Interval: `Interval(lo=1, hi=ceil(1.10 * C_max))`.
        """
        return Interval(1, self.margin.hi)
