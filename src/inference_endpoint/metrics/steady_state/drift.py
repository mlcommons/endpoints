# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady-vs-drift detection for per-super-pass metric trajectories.

A run does not have a single steady state for *every* metric. Empirically QPS,
median TTFT, and e2e latency plateau, while p99 TTFT can drift monotonically
upward across the whole run (progressive tail degradation). Reporting one
"steady" value for a drifting metric is a false number, so before trusting the
adaptive-CoV window we classify each metric as ``steady`` or ``drifting``.

Two independent signals, combined:

- **Trend test** (``analyze_trend``): OLS slope of the per-super-pass series; a
  metric drifts if the total change across the run is both a large fraction of
  its typical value AND large relative to the super-pass-to-super-pass residual
  scatter (so a noisy-but-flat series is not mistaken for a trend).
- **CoV ensemble** (``ensemble_vote``): run the CoV stopping rule at several
  ``(window, bound)`` settings; concordance (most detectors converging near the
  same super-pass) corroborates a steady state, wide scatter corroborates drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median, pstdev

from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.stopping import rule_cov_converged
from inference_endpoint.metrics.steady_state.window import percentile_lower

# A metric drifts when the run-length change is >= REL_DRIFT of the typical value
# AND >= SNR times the residual scatter around the fitted line.
REL_DRIFT_THRESHOLD = 0.15
SNR_THRESHOLD = 2.0

# Default detector grid for the CoV ensemble (spans the useful window/bound range
# found in the ablation).
DEFAULT_ENSEMBLE = (
    (3, 0.03),
    (3, 0.05),
    (4, 0.05),
    (5, 0.08),
    (6, 0.10),
    (6, 0.15),
)


@dataclass(frozen=True, slots=True)
class MetricTrend:
    n: int
    first: float
    last: float
    slope: float  # OLS slope per super-pass
    total_change: float  # slope * (n - 1)
    rel_drift: float  # total_change / median(values), signed
    resid_std: float  # scatter around the fitted line
    snr: float  # |total_change| / resid_std
    verdict: str  # "steady" | "drifting" | "insufficient"


def analyze_trend(
    values: Sequence[float],
    rel_threshold: float = REL_DRIFT_THRESHOLD,
    snr_threshold: float = SNR_THRESHOLD,
) -> MetricTrend:
    """Classify a per-super-pass metric trajectory as steady or drifting."""
    n = len(values)
    if n < 4:
        f = values[0] if n else 0.0
        return MetricTrend(
            n, f, values[-1] if n else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "insufficient"
        )
    xbar = (n - 1) / 2.0
    ybar = sum(values) / n
    sxx = sum((x - xbar) ** 2 for x in range(n))
    sxy = sum((x - xbar) * (v - ybar) for x, v in enumerate(values))
    slope = sxy / sxx if sxx else 0.0
    intercept = ybar - slope * xbar
    resid = [v - (intercept + slope * x) for x, v in enumerate(values)]
    resid_std = pstdev(resid) if n > 1 else 0.0
    total_change = slope * (n - 1)
    med = median(values) or 1e-9
    rel_drift = total_change / med
    snr = abs(total_change) / (resid_std + 1e-12)
    drifting = abs(rel_drift) >= rel_threshold and snr >= snr_threshold
    return MetricTrend(
        n=n,
        first=values[0],
        last=values[-1],
        slope=slope,
        total_change=total_change,
        rel_drift=rel_drift,
        resid_std=resid_std,
        snr=snr,
        verdict="drifting" if drifting else "steady",
    )


def super_pass_metric(
    series: list[SuperPassRollup], kind: str, warmup: int = 1
) -> list[float]:
    """Per-super-pass metric trajectory over the post-warmup region.

    ``kind`` is one of ``ttft_p50``/``ttft_p99``/``lat_p50``/``lat_p99``/``qps``.
    """
    out: list[float] = []
    for sp in series[warmup:]:
        if kind == "qps":
            span_s = (sp.last_issue_ns - sp.first_issue_ns) / 1e9
            out.append(sp.n_issued / span_s if span_s > 0 else 0.0)
            continue
        source = sp.ttft_ns if kind.startswith("ttft") else sp.latency_ns
        if not source:
            continue
        p = 0.5 if kind.endswith("p50") else 0.99
        out.append(percentile_lower(sorted(source), p))
    return out


@dataclass(frozen=True, slots=True)
class EnsembleVote:
    n_detectors: int
    n_converged: int
    sp_ends: tuple[int, ...]  # sp_end each converging detector picked
    concordance: float  # 1 - normalized spread of sp_ends; 0.0 if <2 converged


def ensemble_vote(
    series: list[SuperPassRollup],
    configs: Sequence[tuple[int, float]] = DEFAULT_ENSEMBLE,
    warmup: int = 1,
) -> EnsembleVote:
    """Run the CoV rule at several (window, bound) settings; measure concordance."""
    ends: list[int] = []
    for window, bound in configs:
        region = rule_cov_converged(
            series, window=window, cov_bound=bound, warmup=warmup
        )
        if region is not None:
            ends.append(region[1])
    n_conv = len(ends)
    if n_conv < 2:
        concordance = 0.0
    else:
        spread = (max(ends) - min(ends)) / max(1, len(series) - warmup)
        concordance = max(0.0, 1.0 - spread)
    return EnsembleVote(len(configs), n_conv, tuple(ends), concordance)


def classify_run(
    series: list[SuperPassRollup],
    warmup: int = 1,
    kinds: Sequence[str] = ("qps", "ttft_p50", "ttft_p99", "lat_p50", "lat_p99"),
) -> dict[str, MetricTrend]:
    """Trend verdict per metric for a run's post-warmup super-pass series."""
    return {k: analyze_trend(super_pass_metric(series, k, warmup)) for k in kinds}
