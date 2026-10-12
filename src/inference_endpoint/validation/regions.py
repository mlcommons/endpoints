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

"""Concurrency regions of Endpoints Rules 5.5, exactly as its reference algorithm.

`Regions.for_bounds(C_max, C_min)` -> the five named regions of 5.5
(`low_latency`, `low_concurrency`, `med_concurrency`, `high_concurrency`,
`margin`), each an `Interval` whose `lo`/`hi` are the reference algorithm's
`start`/`end`. The three concurrency regions split `log2(C_max - C_min) / 3`
with round-half-to-even boundaries; `margin` is the 10% High Concurrency
extension to `ceil(1.10 * C_max)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from inference_endpoint.validation.ranges import Interval


@dataclass(frozen=True)
class Regions:
    """The five regions of Rules 5.5 over a submission's `(C_min, C_max)` bounds."""

    low_latency: Interval
    low_concurrency: Interval
    med_concurrency: Interval
    high_concurrency: Interval
    margin: Interval

    @classmethod
    def for_bounds(cls, c_max: int, c_min: int) -> Regions:
        """The Endpoints Policy Section 5.5 reference algorithm, bounds-checked:
        https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md#55-region-boundary-reference-algorithm

        `Regions.for_bounds(8192, 32)` -> the worked Example A of Endpoints
        Policy Section 5.4: low_latency 1..32, low 33..52, med 53..437, high
        438..8192, margin 8193..9012. Bounds outside `1 <= c_min <= 32` and
        `c_max > 32` raise ValueError: the Rules admit no such submission there.

        Example C of Endpoints Policy Section 5.4 prints 16 for its Low
        Concurrency start where `c_min + 1` and its own Med Concurrency start
        of 27 say 17, and prints `round(16 + 2^6.652) = 116` where the
        expression yields 117 (verified: `round(16 + 2**6.652) == 117`; only
        the truncated `2**6.65` gives 116). This port uses the algorithm's
        formula in both spots.
        """
        if not 1 <= c_min <= 32:
            raise ValueError("Minimum concurrency must be between 1 and 32 (inclusive)")
        if c_max <= 32:
            raise ValueError("Maximum Supported Concurrency must be > 32")
        span = math.log2(c_max - c_min) / 3
        low_conc_end = round(c_min + 2**span)
        med_conc_end = round(c_min + 2 ** (2 * span))
        return cls(
            low_latency=Interval(1, c_min),
            low_concurrency=Interval(c_min + 1, low_conc_end),
            med_concurrency=Interval(low_conc_end + 1, med_conc_end),
            high_concurrency=Interval(med_conc_end + 1, c_max),
            margin=Interval(c_max + 1, math.ceil(1.10 * c_max)),
        )
