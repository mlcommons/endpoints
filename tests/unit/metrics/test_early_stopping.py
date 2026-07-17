# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MLPerf early-stopping percentile estimate."""

import math

import pytest
from inference_endpoint.metrics.early_stopping import (
    CONFIDENCE,
    ES_MIN_PERCENTILE,
    TOLERANCE,
    EarlyStoppingResult,
    EarlyStoppingSpec,
    es_percentile_estimate,
    es_percentiles_from_grid,
    find_min_passing,
    grid_percentile_key,
)


@pytest.mark.unit
class TestGridPercentileKey:
    def test_keys_mirror_the_report_grid_strings(self):
        # the compact report map must overlay the `percentiles` grid keys 1:1
        assert grid_percentile_key(0.5) == "50.0"
        assert grid_percentile_key(0.97) == "97.0"  # 0.97*100 float artifact handled
        assert grid_percentile_key(0.999) == "99.9"


@pytest.mark.unit
class TestSpecAndConstants:
    def test_loadgen_constants_not_knobs(self):
        # loadgen hardcodes confidence c = 0.99 and tolerance d = 0.0
        # (results.cc:157-158); neither is configuration.
        assert CONFIDENCE == 0.99
        assert TOLERANCE == 0.0

    def test_percentiles_derive_from_report_grid(self):
        # ES runs for every percentile the report grid shows at or above the
        # median (an ES bound is a tail certification; below-median entries are
        # not meaningful gates) — one source of truth, no separate ES set.
        grid = (99.9, 99.0, 97.0, 95.0, 90.0, 80.0, 75.0, 50.0, 25.0, 10.0, 5.0, 1.0)
        assert es_percentiles_from_grid(grid) == (
            0.5,
            0.75,
            0.8,
            0.9,
            0.95,
            0.97,
            0.99,
            0.999,
        )
        assert ES_MIN_PERCENTILE == 0.5

    def test_spec_defaults(self):
        spec = EarlyStoppingSpec()
        assert spec.percentiles is None  # None = derive from the series' grid
        assert spec.confidence == CONFIDENCE
        assert not hasattr(spec, "tolerance")


@pytest.mark.unit
class TestFindMinPassing:
    def test_matches_loadgen_reference_values(self):
        # h_min(t=0, p99) closed form = ceil(ln(0.01)/ln(0.99)) = 459
        assert (
            find_min_passing(0, 0.99)
            == 459
            == math.ceil(math.log(0.01) / math.log(0.99))
        )
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
        # tolerance is an algorithm constant, not configuration -> not reported.
        r = es_percentile_estimate([float(i) for i in range(10000)], 0.99)
        d = r.as_dict()
        assert d["sufficient"] is True
        assert set(d) == {
            "percentile",
            "confidence",
            "n",
            "estimate",
            "empirical",
            "sufficient",
            "min_queries",
            "discarded",
        }

    def test_median_estimate_is_conservative_upper_bound(self):
        # p50 is in the default report list: the estimate is a c-confidence upper
        # bound on the median, so it must sit above the empirical median.
        arr = [float(i) for i in range(10000)]
        r = es_percentile_estimate(arr, 0.5)
        assert r.estimate is not None
        assert r.estimate >= r.empirical

    def test_empty_series(self):
        r = es_percentile_estimate([], 0.99)
        assert r.estimate is None and r.empirical is None and r.n == 0

    def test_invalid_domain_raises(self):
        # p >= 1 would never terminate the doubling loop; c >= 1 only exits via
        # float underflow with a meaningless result — both must be rejected.
        with pytest.raises(ValueError):
            find_min_passing(1, 1.0)
        with pytest.raises(ValueError):
            find_min_passing(1, 0.0)
        with pytest.raises(ValueError):
            find_min_passing(1, 0.99, c=1.0)
        with pytest.raises(ValueError):
            find_min_passing(1, 0.99, d=0.99)  # tolerance must stay below p
        with pytest.raises(ValueError):
            es_percentile_estimate([1.0, 2.0], 1.5)
        with pytest.raises(ValueError):
            es_percentile_estimate([1.0, 2.0], 0.99, confidence=0.0)

    def test_empirical_matches_report_grid_convention(self):
        # `empirical` must use the same order statistic as the report's percentile
        # grid (np.percentile method="lower": index floor(p*(n-1))) so the block can
        # never disagree with the grid inside one result_summary.json. n=50 p99
        # discriminates: floor(0.99*49)=48, while ceil(0.99*50)-1=49.
        arr = [float(i) for i in range(50)]
        r = es_percentile_estimate(arr, 0.99)
        assert r.empirical == arr[48]
