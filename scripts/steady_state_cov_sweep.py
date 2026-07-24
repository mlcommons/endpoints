#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep the adaptive-CoV ``cov_bound`` (and optionally ``cov_window``) for one run.

Parses a run's ``events.jsonl`` into the super-pass series ONCE, then applies the
CoV stopping rule for every bound in the grid (pure functions over the cached
series — no re-parsing). Scores each choice against the full-series asymptote so
the knee (smallest region that lands near the asymptote) is visible.

usage:
  uv run python scripts/steady_state_cov_sweep.py <events.jsonl> \
      --dataset-size N --concurrency N [--warmup 1] [--cov-window 3] \
      [--bounds 0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20] [--target-err 0.05]
"""

from __future__ import annotations

import argparse

from inference_endpoint.metrics.steady_state.series import (
    build_super_pass_series,
    coverage_status,
    super_pass_size,
)
from inference_endpoint.metrics.steady_state.stopping import rule_cov_converged
from inference_endpoint.metrics.steady_state.window import windowed_metrics

_DEFAULT_BOUNDS = "0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20"


def _rel_err(est: float | None, ref: float | None) -> float | None:
    if est is None or ref is None or ref == 0:
        return None
    return abs(est - ref) / abs(ref)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events")
    ap.add_argument("--dataset-size", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--windows", default="3,4,5,6", help="cov_window grid")
    ap.add_argument("--bounds", default=_DEFAULT_BOUNDS)
    ap.add_argument(
        "--target-err",
        type=float,
        default=0.05,
        help="pick the cheapest (window,bound) whose p99-TTFT err is <= this",
    )
    args = ap.parse_args(argv)

    series = build_super_pass_series(args.events, args.dataset_size, args.concurrency)
    if not series:
        raise SystemExit("FATAL: no performance-tracked samples in the events log")

    n_issued = sum(sp.n_issued for sp in series)
    status = coverage_status(n_issued, args.dataset_size, args.concurrency, args.warmup)
    if status != "windowable":
        raise SystemExit(f"FATAL: run is {status}, not windowable — nothing to sweep")

    n_full = n_issued // super_pass_size(args.dataset_size, args.concurrency)
    measured = series[:n_full]

    # Ground truth = full windowable region after the warmup crop.
    asym = windowed_metrics(measured, args.warmup, len(measured))
    asym_qps = asym.qps
    asym_p99 = asym.ttft.get(0.99)
    print(
        f"windowable: {n_full} full super-passes | asymptote(warmup..{n_full}): "
        f"qps={asym_qps:,.2f}  ttft_p99={asym_p99}"
    )
    print(f"target p99-TTFT err <= {args.target_err}\n")

    windows = [int(x) for x in args.windows.split(",")]
    bounds = [float(x) for x in args.bounds.split(",")]
    print("| cov_window | cov_bound | region | super-passes | qps_err | ttft_p99_err |")
    print("|---|---|---|---|---|---|")
    rows = []
    for w in windows:
        for b in bounds:
            region = rule_cov_converged(
                measured, window=w, cov_bound=b, warmup=args.warmup
            )
            if region is None:
                print(f"| {w} | {b:g} | UNCONVERGED | - | - | - |")
                continue
            m = windowed_metrics(measured, region[0], region[1])
            sp = region[1] - region[0]
            qe = _rel_err(m.qps, asym_qps)
            pe = _rel_err(m.ttft.get(0.99), asym_p99)
            rows.append((w, b, region, sp, pe))
            qe_s = f"{qe:.4f}" if qe is not None else "-"
            pe_s = f"{pe:.4f}" if pe is not None else "-"
            print(f"| {w} | {b:g} | {region} | {sp} | {qe_s} | {pe_s} |")

    # Pick: cheapest (fewest super-passes) whose p99-TTFT err meets the target;
    # fall back to the min-error config if none qualify.
    qualifying = [r for r in rows if r[4] is not None and r[4] <= args.target_err]
    if qualifying:
        best = min(qualifying, key=lambda r: (r[3], r[4]))
        why = f"cheapest region within {args.target_err} p99 err"
    elif rows:
        best = min(rows, key=lambda r: (r[4] if r[4] is not None else 9e9))
        why = "min p99 err (none met target)"
    else:
        raise SystemExit("no config converged")
    print(
        f"\nBEST cov_window={best[0]} cov_bound={best[1]:g}  region={best[2]}  "
        f"super_passes={best[3]}  ttft_p99_err={best[4]:.4f}  ({why})"
    )
    return best


if __name__ == "__main__":
    main()
