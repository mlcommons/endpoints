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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "CONFIDENCE",
    "ES_MIN_PERCENTILE",
    "TOLERANCE",
    "EarlyStoppingResult",
    "EarlyStoppingSpec",
    "es_percentile_estimate",
    "es_percentiles_from_grid",
    "find_min_passing",
]

# LoadGen hardcodes confidence c = 0.99 and tolerance d = 0.0 (results.cc:157-158). They are
# algorithm constants, not knobs: lowering c or raising d weakens the certified claim (d > 0
# certifies percentile p - d instead of p). Exposed only as defaulted arguments on the pure
# math for parity tests.
CONFIDENCE: Final[float] = 0.99
TOLERANCE: Final[float] = 0.0

# ES runs for every percentile the series' report grid shows at or above the median. The
# estimate is a tail certification (a conservative upper confidence bound), so below-median
# grid entries (p25/p10/...) are not meaningful gates and are skipped. Deriving from the grid
# keeps one source of truth: whatever percentiles a series reports, ES covers.
ES_MIN_PERCENTILE: Final[float] = 0.5


def es_percentiles_from_grid(grid_percentiles: Iterable[float]) -> tuple[float, ...]:
    """ES target percentiles (ascending fractions) from a report grid in 0-100 units.

    E.g. the default grid ``(99.9, 99.0, ..., 50.0, 25.0, ...)`` yields
    ``(0.5, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999)``.

    Rounded to 6 decimals: grid values carry at most a few decimals in 0-100 units, so
    this is lossless while avoiding float-division artifacts (99.9/100 != 0.999 exactly)
    in the self-describing ``percentile`` field of the report blocks.
    """
    return tuple(
        sorted(
            {
                round(p / 100.0, 6)
                for p in grid_percentiles
                if p / 100.0 >= ES_MIN_PERCENTILE
            }
        )
    )


@dataclass(frozen=True, slots=True)
class EarlyStoppingSpec:
    """Resolved early-stopping parameters used by the aggregator.

    ``percentiles=None`` (the default) derives the target set from each series' own
    report grid via ``es_percentiles_from_grid``; explicit tuples are for tests and
    offline analysis only — the config surface is a single ``enabled`` flag.
    """

    percentiles: tuple[float, ...] | None = None
    confidence: float = CONFIDENCE


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction kernel of the regularized incomplete beta.

    Numerical Recipes 3rd ed. §6.4 (``betacf``), evaluated with the modified Lentz
    algorithm: each loop iteration applies one even-numbered and one odd-numbered CF
    term, and stops when the running product changes by less than ``eps``.

    Magic numbers:
      - ``eps = 3e-16``: relative-convergence threshold, ~1.4x double-precision machine
        epsilon (2.22e-16) — one step above the noise floor, so the loop stops as soon
        as further terms cannot change the result.
      - ``fpmin = 1e-300``: substituted for near-zero intermediate denominators (Lentz's
        guard against division blow-up); any positive value far below the smallest
        normal double (~2.2e-308) times a typical term works.
      - ``400``: iteration cap. NR uses 100 with an error escape; our (a, b) grow with
        the discard search (hundreds to thousands) where the CF still converges in tens
        of iterations because ``_betai`` always evaluates on the fast-converging side of
        its symmetry split — 400 is a generous safety bound, not a tuning knob.
    """
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
    """Regularized incomplete beta I_x(a, b).

    Numerical Recipes 3rd ed. §6.4 (``betai``). Numerically equal to LoadGen's
    ``beta_regularized(x, a, b)`` (``loadgen/early_stopping.cc:42-45``, computed there
    from the Gauss hypergeometric 2F1 series, ``:25-35``) but converges in tens of CF
    iterations instead of thousands of series terms.

    ``ln_bt`` is log(x^a * (1-x)^b / B(a, b)) via ``lgamma`` so the prefactor cannot
    overflow for large a/b. The ``x < (a + 1) / (a + b + 2)`` test is NR's symmetry
    split: evaluate the continued fraction on whichever side of the identity
    I_x(a, b) = 1 - I_{1-x}(b, a) converges fastest.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _odds(h: int, t: int, p: float, d: float) -> float:
    """LoadGen ``odds()`` (``loadgen/early_stopping.cc:47-58``).

    P(<= t over-latency among h + t Bernoulli trials | over-latency rate = 1 - p + d),
    computed via the binomial-CDF <-> incomplete-beta identity: that probability equals
    I_{p-d}(h, t + 1).
    """
    return _betai(h, 1 + t, p - d)


def _validate_domain(p: float, d: float, c: float) -> None:
    # p >= 1 makes _odds identically 1.0 (the doubling search would never terminate);
    # c >= 1 makes the target 0.0 (only reachable via float underflow). Reject both.
    if not 0.0 < p < 1.0:
        raise ValueError(f"percentile must be in (0, 1), got {p}")
    if not 0.0 < c < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {c}")
    if not 0.0 <= d < p:
        raise ValueError(f"tolerance must be in [0, percentile), got {d}")


def find_min_passing(
    t: int, p: float, d: float = TOLERANCE, c: float = CONFIDENCE
) -> int:
    """Minimum ``h`` such that ``_odds(h, t, p, d) <= 1 - c``.

    LoadGen's ``MinPassingQueriesFinder`` (``loadgen/early_stopping.cc:62-114``): the
    smallest number of under-latency queries that, alongside ``t`` over-latency queries,
    lets you conclude at confidence ``c`` that the true p-percentile meets the bound.
    ``_odds`` is monotonically decreasing in ``h``, so exponential bracketing (double
    ``hi`` until it passes) followed by binary search finds the boundary exactly.
    """
    _validate_domain(p, d, c)
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
    """Largest ``t >= 1`` with ``n >= find_min_passing(t) + t``; 0 below the floor.

    This is the LoadGen SingleStream estimate construction (``results.cc:162-226``):
    discard the ``t`` highest samples such that the run of ``n`` queries would still
    pass the binomial test with those as its over-latency set — the (t+1)-th highest
    value is then a c-confidence upper bound on the true p-percentile. Same
    exponential-bracket + binary-search shape as ``find_min_passing`` (the passing
    margin ``n - (find_min_passing(t) + t)`` is monotone in ``t``).
    """
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

    __slots__ = (
        "percentile",
        "confidence",
        "n",
        "estimate",
        "empirical",
        "min_queries",
        "discarded",
    )

    def __init__(
        self,
        percentile: float,
        confidence: float,
        n: int,
        estimate: float | None,
        empirical: float | None,
        min_queries: int,
        discarded: int,
    ) -> None:
        self.percentile = percentile
        self.confidence = confidence
        self.n = n
        self.estimate = estimate
        self.empirical = empirical
        self.min_queries = min_queries
        self.discarded = discarded

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "percentile": self.percentile,
            "confidence": self.confidence,
            "n": self.n,
            "estimate": None if self.estimate is None else float(self.estimate),
            "empirical": None if self.empirical is None else float(self.empirical),
            "sufficient": self.estimate is not None,
            "min_queries": self.min_queries,
            "discarded": self.discarded,
        }


def es_percentile_estimate(
    sorted_latencies: Sequence[float],
    percentile: float,
    confidence: float = CONFIDENCE,
) -> EarlyStoppingResult:
    """Conservative early-stopping percentile estimate over an ascending-sorted latency series."""
    n = len(sorted_latencies)
    min_queries = find_min_passing(1, percentile, TOLERANCE, confidence) + 1
    # Same order statistic as the report's percentile grid (np.percentile with
    # method="lower": index floor(p*(n-1))) so the block's empirical can never
    # disagree with the grid value in the same result_summary.json.
    emp_idx = int(percentile * (n - 1)) if n else 0
    empirical = sorted_latencies[emp_idx] if n else None
    if n < min_queries:
        return EarlyStoppingResult(
            percentile, confidence, n, None, empirical, min_queries, 0
        )
    t = _discard_count(n, percentile, TOLERANCE, confidence)
    return EarlyStoppingResult(
        percentile,
        confidence,
        n,
        sorted_latencies[n - t],
        empirical,
        min_queries,
        t,
    )
