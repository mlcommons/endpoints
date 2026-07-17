# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MLPerf early-stopping percentile estimate.

Each test pins one failure mode that has bitten (or would silently bite) in review:
wrong binomial math, a weakened statistical claim, an estimate/grid convention
divergence inside one summary, or an unbounded search on bad input.
"""

import pytest
from inference_endpoint.metrics.early_stopping import (
    CONFIDENCE,
    ES_MIN_PERCENTILE,
    TOLERANCE,
    EarlyStoppingSpec,
    es_percentile_estimate,
    es_targets_from_grid,
    find_min_passing,
)

pytestmark = pytest.mark.unit


def test_loadgen_reference_values():
    # Anchors the binomial math to LoadGen ground truth: any port drift changes
    # these exact numbers. h_min(t=0, p99) has the closed form
    # ceil(ln(1-c)/ln(p)) = 459; t=1/p99 = 661 is the SingleStream estimate floor.
    assert find_min_passing(0, 0.99) == 459
    assert find_min_passing(0, 0.90) == 44
    assert find_min_passing(1, 0.99) == 661
    # monotonicity in t is the precondition for the binary searches built on it
    values = [find_min_passing(t, 0.99) for t in range(6)]
    assert values == sorted(set(values))


def test_constants_and_spec_defaults():
    # LoadGen hardcodes c=0.99 and d=0.0 (results.cc:157-158) — weakening either
    # weakens the certified claim, so they must stay constants, and the spec must
    # default to grid derivation (percentiles=None), not a private list.
    assert CONFIDENCE == 0.99
    assert TOLERANCE == 0.0
    assert ES_MIN_PERCENTILE == 50.0  # grid convention (0-100)
    spec = EarlyStoppingSpec()
    assert spec.percentiles is None and spec.confidence == CONFIDENCE
    assert not hasattr(spec, "tolerance")


def test_grid_derivation_and_key_format():
    # The ES targets and map keys must overlay the report's percentile grid 1:1
    # (keys are str() of the ORIGINAL grid values — int grids key as "99"), in
    # the grid's own order, with values staying in the grid convention (0-100,
    # no conversion or rounding). Out-of-domain entries are excluded: 100.0
    # would crash the terminal snapshot (default-on!), below-median entries are
    # not tail certifications.
    grid = (100.0, 99.9, 99.0, 97.0, 95.0, 90.0, 80.0, 75.0, 50.0, 25.0, 1.0)
    targets = es_targets_from_grid(grid)
    assert list(targets) == [
        "99.9",
        "99.0",
        "97.0",
        "95.0",
        "90.0",
        "80.0",
        "75.0",
        "50.0",
    ]
    assert targets["99.9"] == 99.9 and targets["50.0"] == 50.0  # values as given
    assert list(es_targets_from_grid((99, 90, 50))) == ["99", "90", "50"]
    assert list(es_targets_from_grid(("99.0", "50.0"))) == ["99.0", "50.0"]


def test_estimate_reference_values_and_conservatism():
    # Anchors the end-to-end estimate on a known input: discard count, floor, and
    # the invariant that the estimate never under-reports the empirical value.
    arr = [float(i) for i in range(10000)]
    r = es_percentile_estimate(arr, 99.0)
    # t=77: the estimate is the 77th-highest sample; 76 samples sit above it
    assert (r.discarded, r.min_queries) == (76, 662)
    assert r.estimate == arr[10000 - 77]
    assert r.estimate >= r.empirical


def test_sufficiency_floor_boundary():
    # Below the floor the estimate must be None (never a fabricated bound), the
    # empirical value must still be reported, and n=0 must not crash. Just above
    # the floor the budget is t=1: nothing is discarded and the estimate IS the
    # maximum observed sample.
    below = es_percentile_estimate([float(i) for i in range(600)], 99.0)  # < 662
    assert below.estimate is None and below.empirical is not None
    arr = [float(i) for i in range(663)]
    at_floor = es_percentile_estimate(arr, 99.0)
    assert at_floor.estimate == arr[-1] and at_floor.discarded == 0
    empty = es_percentile_estimate([], 99.0)
    assert empty.estimate is None and empty.empirical is None and empty.n == 0


def test_betai_branches_and_cap_regime():
    # Pin both evaluation branches against closed forms — I_x(a,1) = x^a exercises
    # the direct branch, I_x(1,b) = 1-(1-x)^b at large x the reflection branch —
    # and pin that the iteration-cap regime (a ~ b, midpoint x) stays sane and
    # exception-free (accuracy there is deliberately NOT claimed; see _betacf).
    from inference_endpoint.metrics.early_stopping import _betai

    assert abs(_betai(3.0, 1.0, 0.2) - 0.2**3) < 1e-12  # direct branch
    assert abs(_betai(1.0, 3.0, 0.9) - (1.0 - 0.1**3)) < 1e-12  # reflection branch
    assert 0.4 < _betai(5e6, 5e6, 0.5) < 0.6  # cap regime: truncated but sane


def test_empirical_matches_report_grid_convention():
    # `empirical` must use the grid's order statistic (np.percentile
    # method="lower": floor(p*(n-1))) or the block can contradict the grid value
    # in the same summary. n=50/p99 discriminates: floor=48 vs ceil(p*n)-1=49.
    arr = [float(i) for i in range(50)]
    assert es_percentile_estimate(arr, 99.0).empirical == arr[48]


def test_invalid_domain_raises():
    # p >= 1 makes the doubling search non-terminating (a real hang, reproduced
    # in review); c >= 1 only exits via float underflow with a garbage result.
    for bad_call in (
        # kernel (fraction domain, loadgen parity)
        lambda: find_min_passing(1, 1.0),
        lambda: find_min_passing(1, 0.0),
        lambda: find_min_passing(1, 0.99, c=1.0),
        lambda: find_min_passing(1, 0.99, d=0.99),  # tolerance must stay below p
        # product surface (grid convention, 0-100)
        lambda: es_percentile_estimate([1.0, 2.0], 150.0),
        lambda: es_percentile_estimate([1.0, 2.0], 100.0),
        lambda: es_percentile_estimate([1.0, 2.0], 0.0),
        lambda: es_percentile_estimate([1.0, 2.0], 99.0, confidence=0.0),
    ):
        with pytest.raises(ValueError):
            bad_call()


def test_result_dict_shape():
    # The block dict is the post-hoc script's --json contract; tolerance is an
    # algorithm constant and must not be reported as if it were configuration.
    d = es_percentile_estimate([float(i) for i in range(10000)], 99.0).as_dict()
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
