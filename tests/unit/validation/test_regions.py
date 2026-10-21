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
https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md#54-regions-of-interest

Section 5.5 names the pseudocode the authoritative computation and the three
printed examples match it exactly, so the expectations here are the formula's.
"""

import pytest
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.regions import (
    C_MIN_HIGHEST,
    C_MIN_LOWEST,
    MARGIN_FACTOR,
    RegionKind,
    Regions,
)


class TestWorkedExamples:
    """`Regions.for_bounds` reproduces 5.4's printed Examples A, B and C."""

    @pytest.mark.unit
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
        regions = Regions.for_bounds(c_min=c_min, c_max=c_max)
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

    @pytest.mark.unit
    def test_the_margin_is_the_ten_percent_high_concurrency_extension(self):
        assert Regions.for_bounds(c_min=32, c_max=8192).margin == Interval(8193, 9012)
        assert MARGIN_FACTOR == 1.10


class TestBounds:
    """The printed bounds-checks of the reference algorithm."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("c_min", "c_max"),
        [(1, 32), (0, 256), (33, 256), (1, 0)],
    )
    def test_bounds_outside_the_admitted_range_raise(self, c_min, c_max):
        with pytest.raises(
            ValueError,
            match=(
                "Maximum Supported Concurrency must be"
                "|C_min must be between|concurrencies need to be integers"
            ),
        ):
            Regions.for_bounds(c_min=c_min, c_max=c_max)

    @pytest.mark.unit
    def test_c_max_of_33_is_admitted(self):
        assert Regions.for_bounds(c_min=1, c_max=33).high_concurrency == Interval(
            12, 33
        )

    @pytest.mark.unit
    def test_declared_constants_frame_the_c_min_range(self):
        assert (C_MIN_LOWEST, C_MIN_HIGHEST) == (1, 32)


class TestForPareto:
    """`Regions.for_pareto`: `C_max` from `system_desc`, `C_min` derived or overridden."""

    @pytest.mark.unit
    def test_a_declared_override_wins_over_the_points(self):
        # Endpoints Policy 5.4's override example: with `c_min_override: 32`,
        # Low Concurrency starts at 33 and the point at 16 still sits in
        # the Ultra Low Concurrency region's coverage.
        system_desc = {"max_supported_concurrency": 8192, "c_min_override": 32}
        regions = Regions.for_pareto(system_desc, [16, 24, 32, 64, 1000])
        assert regions == Regions.for_bounds(c_min=32, c_max=8192)
        assert regions.low_concurrency.lo == 33
        assert 16 in regions.coverage()

    @pytest.mark.unit
    def test_an_absent_override_derives_c_min_from_the_minimum_point(self):
        regions = Regions.for_pareto(
            {"max_supported_concurrency": 1024}, [1000, 24, 16]
        )
        assert regions == Regions.for_bounds(c_min=16, c_max=1024)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "override", [C_MIN_LOWEST - 1, C_MIN_HIGHEST + 1, None, 40, "32"]
    )
    def test_an_invalid_override_is_refused_naming_the_rule_row(self, override):
        system_desc = {"max_supported_concurrency": 8192, "c_min_override": override}
        with pytest.raises(ValueError, match="C_min override"):
            Regions.for_pareto(system_desc, [16, 24, 32, 64, 1000])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("system_desc", "concurrencies"),
        [
            ({}, [32, 40]),
            ({"max_supported_concurrency": "8192"}, [32, 40]),
            ({"max_supported_concurrency": 8192}, [40, 80]),
            ({"max_supported_concurrency": 8192}, []),
            ({"max_supported_concurrency": 8192}, [32, 20.0, 40]),
        ],
    )
    def test_missing_or_ambiguous_facts_raise(self, system_desc, concurrencies):
        with pytest.raises(
            ValueError,
            match=(
                "max_supported_concurrency|must be between|need at least one point"
                "|concurrencies need to be integers"
            ),
        ):
            Regions.for_pareto(system_desc, concurrencies)


class TestCovers:
    """`Regions.covers`: which `RegionKind` covers a concurrency, or `None`."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            (32, RegionKind.ULTRA_LOW_CONCURRENCY),
            (52, RegionKind.LOW_CONCURRENCY),
            (53, RegionKind.MED_CONCURRENCY),
            (437, RegionKind.MED_CONCURRENCY),
            (438, RegionKind.HIGH_CONCURRENCY),
            (8192, RegionKind.HIGH_CONCURRENCY),
            (8193, RegionKind.MARGIN),
            (9012, RegionKind.MARGIN),
        ],
    )
    def test_kind_of_each_example_a_boundary(self, value, kind):
        assert Regions.for_bounds(c_min=32, c_max=8192).covers(value) is kind

    @pytest.mark.unit
    def test_outside_gives_none(self):
        regions = Regions.for_bounds(c_min=32, c_max=8192)
        assert regions.covers(0) is None
        assert regions.covers(9013) is None

    @pytest.mark.unit
    def test_the_ultra_low_wire_name_is_the_reference_algorithm_key(self):
        assert RegionKind.ULTRA_LOW_CONCURRENCY.value == "low_latency"
