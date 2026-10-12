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

"""W5 slice tests: the reference algorithm against its three worked examples.

The Examples A/B/C constants come from Endpoints Policy Section 5.4's worked
examples (A: C_min=32/C_max=8192, B: 1/256, C: 16/1024):
https://github.com/mlcommons/endpoints_policies/blob/d062b470abd2/endpoints_rules.md#54-regions-of-interest

A and B match the formula's computation exactly. Example C carries two
hand-typo slips of its own printed expression: it starts Low Concurrency at 16
where `C_min + 1` and its own Med start of 27 say 17, and it prints
`round(16 + 2^6.652) = 116` where the truncated-exponent coincidence is the
only route to 116 (`round(16 + 2**6.652) == 117`, verified by `python -c`;
`2*span = 6.65152`, never 6.65). Endpoints Policy Section 5.5 names the
pseudocode the authoritative computation, so C's expectations here are the
formula's; both printed slips are noted in `Regions.for_bounds`.
"""

import pytest

from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.regions import Regions


class TestWorkedExamples:
    """`Regions.for_bounds` reproduces 5.4's printed Examples A, B and C."""

    @pytest.mark.parametrize(
        ("c_max", "c_min", "expected"),
        [
            (
                8192,
                32,
                {
                    "low_latency": (1, 32),
                    "low_concurrency": (33, 52),
                    "med_concurrency": (53, 437),
                    "high_concurrency": (438, 8192),
                    "margin": (8193, 9012),
                },
            ),
            (
                256,
                1,
                {
                    "low_latency": (1, 1),
                    "low_concurrency": (2, 7),
                    "med_concurrency": (8, 41),
                    "high_concurrency": (42, 256),
                    "margin": (257, 282),
                },
            ),
            (
                1024,
                16,
                {
                    "low_latency": (1, 16),
                    "low_concurrency": (17, 26),
                    "med_concurrency": (27, 117),
                    "high_concurrency": (118, 1024),
                    "margin": (1025, 1127),
                },
            ),
        ],
    )
    def test_worked_example_boundaries(self, c_max, c_min, expected):
        regions = Regions.for_bounds(c_max, c_min)
        assert {
            name: (bounds.lo, bounds.hi)
            for name, bounds in {
                "low_latency": regions.low_latency,
                "low_concurrency": regions.low_concurrency,
                "med_concurrency": regions.med_concurrency,
                "high_concurrency": regions.high_concurrency,
                "margin": regions.margin,
            }.items()
        } == expected


class TestBounds:
    """The printed bounds-checks of the reference algorithm."""

    @pytest.mark.parametrize(
        ("c_max", "c_min"),
        [(32, 1), (256, 0), (256, 33), (0, 1)],
    )
    def test_bounds_outside_the_admitted_range_raise(self, c_max, c_min):
        with pytest.raises(ValueError):
            Regions.for_bounds(c_max, c_min)

    def test_c_max_of_33_is_admitted(self):
        assert Regions.for_bounds(33, 1).high_concurrency == Interval(12, 33)
