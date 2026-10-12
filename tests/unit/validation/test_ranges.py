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

"""W5 slice tests: the unified `Interval` grid across every shape it absorbs."""

import pytest
from inference_endpoint.validation.ranges import Interval


class TestAccuracyGate:
    """`Interval.for_accuracy_gate`: the passing band of Endpoints Policy 2.9.8's gate."""

    def test_open_upper_threshold_band(self):
        band = Interval.for_accuracy_gate(41.39, 0.97)
        assert band == Interval(0.97 * 41.39)
        assert 0.97 * 41.39 in band
        assert (0.97 * 41.39 - 0.001) not in band
        assert 100.0 in band

    def test_tolerance_one_is_the_reference_score(self):
        assert Interval.for_accuracy_gate(10.0, 1.0) == Interval(10.0)

    def test_tolerance_zero_admits_nonnegative_scores(self):
        band = Interval.for_accuracy_gate(10.0, 0.0)
        assert 0 in band and 5 in band
        assert -5 not in band


class TestIntervalContains:
    """`value in interval`: closed, open upper, half-open, scalar, non-ordering."""

    @pytest.mark.parametrize(
        ("interval", "value", "inside"),
        [
            (Interval(2, 5), 2, True),
            (Interval(2, 5), 5, True),
            (Interval(2, 5), 6, False),
            (Interval(2, 5), 1, False),
            (Interval(2), 6, True),
            (Interval(2), 1, False),
            (Interval(0, 10, hi_inclusive=False), 10, False),
            (Interval(0, 10), 10, True),
            (Interval(2, 5, lo_inclusive=False), 2, False),
            (Interval(4, 4), 4, True),
            (Interval(4, 4), 3, False),
            (Interval(2, 5), "a", False),
        ],
    )
    def test_containment(self, interval, value, inside):
        assert (value in interval) is inside
