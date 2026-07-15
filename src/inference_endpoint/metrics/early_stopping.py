# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLPerf-style early-stopping percentile estimate.

Ports the MLPerf LoadGen early-stopping binomial test (``loadgen/early_stopping.cc``,
``results.cc``) so a tail-latency percentile (p99 TTFT/TPOT for LLM Server, p90 for
SingleStream) can be reported as a *conservative, confidence-backed* estimate instead of the
raw empirical value. This mirrors LoadGen's SingleStream estimate: given ``n`` sorted latencies
and target percentile ``p``, report ``sorted[n - t]`` where ``t`` is the largest discard count
whose binomial-confidence bound still holds. The estimate is always >= the empirical percentile
and the true p-percentile is <= it at confidence ``c``.

The regularized incomplete beta uses a continued fraction (Numerical Recipes ``betai``); it
equals LoadGen's Gauss-hypergeometric ``beta_regularized`` but converges in tens of iterations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["EarlyStoppingResult", "es_percentile_estimate", "find_min_passing"]


def _betacf(a: float, b: float, x: float) -> float:
    eps, fpmin = 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = fpmin if abs(d) < fpmin else d
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) == LoadGen beta_regularized(x, a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _odds(h: int, t: int, p: float, d: float) -> float:
    # P(<= t over-latency in h + t total | over-latency rate = 1 - p + d)
    return _betai(h, 1 + t, p - d)


def find_min_passing(t: int, p: float, d: float = 0.0, c: float = 0.99) -> int:
    """Minimum ``h`` such that ``_odds(h, t, p, d) <= 1 - c``. ``_odds`` decreases in ``h``."""
    target = 1.0 - c
    lo, hi = 1, 2
    while _odds(hi, t, p, d) > target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if _odds(mid, t, p, d) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _discard_count(n: int, p: float, d: float, c: float) -> int:
    # Largest t >= 1 with n >= find_min_passing(t) + t; 0 if even t=1 needs more than n.
    if find_min_passing(1, p, d, c) + 1 > n:
        return 0
    lo, hi = 1, 2
    while find_min_passing(hi, p, d, c) + hi <= n:
        hi *= 2
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if find_min_passing(mid, p, d, c) + mid <= n:
            lo = mid
        else:
            hi = mid
    return lo


class EarlyStoppingResult:
    """Result of an early-stopping percentile estimate.

    ``estimate`` is None when there are too few samples to make any claim
    (``n < find_min_passing(1) + 1``); ``min_queries`` is that floor.
    """

    __slots__ = ("percentile", "confidence", "tolerance", "n", "estimate",
                 "empirical", "min_queries", "discarded")

    def __init__(self, percentile: float, confidence: float, tolerance: float, n: int,
                 estimate: float | None, empirical: float | None,
                 min_queries: int, discarded: int) -> None:
        self.percentile = percentile
        self.confidence = confidence
        self.tolerance = tolerance
        self.n = n
        self.estimate = estimate
        self.empirical = empirical
        self.min_queries = min_queries
        self.discarded = discarded

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "percentile": self.percentile,
            "confidence": self.confidence,
            "tolerance": self.tolerance,
            "n": self.n,
            "estimate": self.estimate,
            "empirical": self.empirical,
            "sufficient": self.estimate is not None,
            "min_queries": self.min_queries,
            "discarded": self.discarded,
        }


def es_percentile_estimate(
    sorted_latencies: Sequence[float],
    percentile: float,
    confidence: float = 0.99,
    tolerance: float = 0.0,
) -> EarlyStoppingResult:
    """Conservative early-stopping percentile estimate over an ascending-sorted latency series."""
    n = len(sorted_latencies)
    min_queries = find_min_passing(1, percentile, tolerance, confidence) + 1
    emp_idx = min(n - 1, max(0, math.ceil(percentile * n) - 1)) if n else 0
    empirical = sorted_latencies[emp_idx] if n else None
    if n < min_queries:
        return EarlyStoppingResult(percentile, confidence, tolerance, n, None,
                                   empirical, min_queries, 0)
    t = _discard_count(n, percentile, tolerance, confidence)
    return EarlyStoppingResult(percentile, confidence, tolerance, n,
                               sorted_latencies[n - t], empirical, min_queries, t)
