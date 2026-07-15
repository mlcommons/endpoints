# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MLPerf early-stopping percentile estimate."""

import math

import pytest

from inference_endpoint.metrics.early_stopping import (
    EarlyStoppingResult,
    es_percentile_estimate,
    find_min_passing,
)


@pytest.mark.unit
class TestFindMinPassing:
    def test_matches_loadgen_reference_values(self):
        # h_min(t=0, p99) closed form = ceil(ln(0.01)/ln(0.99)) = 459
        assert find_min_passing(0, 0.99) == 459 == math.ceil(math.log(0.01) / math.log(0.99))
        # p90 best case
        assert find_min_passing(0, 0.90) == 44
        # SingleStream estimate floor uses t=1
        assert find_min_passing(1, 0.99) == 661

    def test_monotonic_in_t(self):
        prev = -1
        for t in range(0, 20):
            h = find_min_passing(t, 0.99)
            assert h > prev
            prev = h


@pytest.mark.unit
class TestEsPercentileEstimate:
    def test_estimate_is_conservative(self):
        arr = [float(i) for i in range(10000)]  # ascending
        r = es_percentile_estimate(arr, 0.99)
        assert isinstance(r, EarlyStoppingResult)
        assert r.estimate is not None
        assert r.estimate >= r.empirical  # ES estimate never below empirical
        assert r.discarded == 77  # known discard count at n=10000, p99
        assert r.min_queries == 662

    def test_insufficient_samples_returns_none(self):
        arr = [float(i) for i in range(600)]  # below the p99 floor (662)
        r = es_percentile_estimate(arr, 0.99)
        assert r.estimate is None
        assert r.empirical is not None  # empirical is still reported
        assert r.min_queries == 662

    def test_just_above_floor_is_defined(self):
        arr = [float(i) for i in range(700)]  # >= 662
        r = es_percentile_estimate(arr, 0.99)
        assert r.estimate is not None

    def test_p90_floor_lower_than_p99(self):
        # p90 needs far fewer samples than p99
        r_ok = es_percentile_estimate([float(i) for i in range(100)], 0.90)
        assert r_ok.estimate is not None
        assert r_ok.min_queries == find_min_passing(1, 0.90) + 1

    def test_as_dict_shape(self):
        r = es_percentile_estimate([float(i) for i in range(10000)], 0.99)
        d = r.as_dict()
        assert d["sufficient"] is True
        assert set(d) == {
            "percentile", "confidence", "tolerance", "n", "estimate",
            "empirical", "sufficient", "min_queries", "discarded",
        }

    def test_empty_series(self):
        r = es_percentile_estimate([], 0.99)
        assert r.estimate is None and r.empirical is None and r.n == 0
