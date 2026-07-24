#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady-vs-drift diagnosis for one run's per-super-pass metric trajectories.

Parses ``events.jsonl`` once, then for each metric (QPS, p50/p99 TTFT, p50/p99
latency) fits a trend and votes a CoV-detector ensemble to classify the metric as
``steady`` or ``drifting``. The point: a run can be steady in QPS/median yet have
no steady state for p99 TTFT — this flags that instead of reporting a false
steady value.

usage:
  uv run python scripts/steady_state_drift.py <events.jsonl> \
      --dataset-size N --concurrency N [--warmup 1]
"""

from __future__ import annotations

import argparse

from inference_endpoint.metrics.steady_state.drift import (
    classify_run,
    ensemble_vote,
    super_pass_metric,
)
from inference_endpoint.metrics.steady_state.series import (
    build_super_pass_series,
    coverage_status,
    super_pass_size,
)

_TTFT_LAT_S = {"ttft_p50", "ttft_p99", "lat_p50", "lat_p99"}


def _fmt(kind: str, v: float) -> str:
    return f"{v / 1e9:.3f}s" if kind in _TTFT_LAT_S else f"{v:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events")
    ap.add_argument("--dataset-size", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args(argv)

    series = build_super_pass_series(args.events, args.dataset_size, args.concurrency)
    if not series:
        raise SystemExit("FATAL: no performance-tracked samples in the events log")
    n_issued = sum(sp.n_issued for sp in series)
    status = coverage_status(n_issued, args.dataset_size, args.concurrency, args.warmup)
    n_full = n_issued // super_pass_size(args.dataset_size, args.concurrency)
    measured = series[:n_full] if status == "windowable" else series
    warmup = args.warmup if status == "windowable" else 0

    vote = ensemble_vote(measured, warmup=warmup)
    print(
        f"status={status}  full_super_passes={n_full}  measured={len(measured)}  warmup={warmup}"
    )
    print(
        f"CoV ensemble: {vote.n_converged}/{vote.n_detectors} converged  "
        f"sp_ends={vote.sp_ends}  concordance={vote.concordance:.2f}"
    )
    print("\n| metric | first | last | rel_drift | snr | verdict |")
    print("|---|---|---|---|---|---|")
    trends = classify_run(measured, warmup=warmup)
    for kind, t in trends.items():
        traj = super_pass_metric(measured, kind, warmup)
        _ = traj  # trajectory already folded into the trend; kept for parity
        print(
            f"| {kind} | {_fmt(kind, t.first)} | {_fmt(kind, t.last)} | "
            f"{t.rel_drift:+.2f} | {t.snr:.1f} | **{t.verdict}** |"
        )
    drifting = [k for k, t in trends.items() if t.verdict == "drifting"]
    print(f"\nDRIFTING metrics: {drifting or 'none'}")
    return trends


if __name__ == "__main__":
    main()
