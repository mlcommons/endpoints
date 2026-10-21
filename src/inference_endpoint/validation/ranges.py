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

"""Ranges: one interval type for every bounded value a check reads.

`Interval(2, 5)` is the closed 2..5. `Interval(3)` is 3 or more (no maximum,
`hi is None`). `Interval(0, 10, hi_inclusive=False)` is the half-open 0..10.
A scalar is `Interval(4, 4)`: `x in Interval(4, 4)` is `x == 4`. Used as the
`in` operator's range target (`operations.py`): an `in` row's
`{"lo": ..., "hi": ...}` literal, or a fold's value. Also the concurrency
regions of `regions.py` (whose `lo`/`hi` are the reference algorithm's
`start`/`end`) and accuracy score reaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Interval:
    """A `lo` with at most a `hi`, with per-side inclusivity. Membership is `in`.

    Example:

        5 in Interval(2, 5)  # True (closed)
        6 in Interval(2)  # True (open upper)
        10 in Interval(0, 10, hi_inclusive=False)  # False (half-open)
        4 in Interval(4, 4)  # True (scalar)
        3 in Interval(4, 4)  # False

    A value that cannot order against the bounds falls outside.

    Args:
        lo (float): The interval's lower bound.
        hi (float | None): The interval's upper bound. `None` means no maximum.
            (Default: None)
        lo_inclusive (bool): Whether `lo` itself is inside the interval.
            (Default: True)
        hi_inclusive (bool): Whether `hi` itself is inside the interval.
            (Default: True)
    """

    lo: float
    hi: float | None = None
    lo_inclusive: bool = True
    hi_inclusive: bool = True

    def __contains__(self, value: Any) -> bool:
        try:
            if self.lo_inclusive:
                above = value >= self.lo
            else:
                above = value > self.lo
            if self.hi is None:
                return bool(above)
            below = value <= self.hi if self.hi_inclusive else value < self.hi
        except TypeError:
            return False  # e.g. a string against a numeric range: outside, not an error
        return bool(above and below)

    @classmethod
    def for_accuracy_gate(cls, reference_score: float, tolerance: float) -> Interval:
        """The passing band of the accuracy gate: `[tolerance * reference_score, inf)`.

        Endpoints Policy Section 2.9.8 states the gate as
        `score >= factor x reference`, with `factor` and the reference score
        taken per benchmark definition:
        https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md#298-accuracy-gate

        Example:

            gate = Interval.for_accuracy_gate(41.39, 0.97)
            gate.lo  # 40.1483
            gate.hi is None  # True
            41.39 in gate  # True

        Args:
            reference_score (float): The benchmark definition's reference score
                (`reference` in `score >= factor x reference`).
            tolerance (float): The definition's `factor`, the fraction of the
                reference a passing result must reach.
        Returns:
            Interval: The band of every score `>= tolerance * reference_score`.
                `lo` is their product, `hi` is `None`.
        """
        return cls(tolerance * reference_score)
