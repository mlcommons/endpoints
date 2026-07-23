#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline steady-state windowing + A/B sweep from a run's events.jsonl.

Cold-path companion to ``scripts/early_stopping_estimate_from_events.py``. Buckets
performance-tracked samples into super-passes by issue order, computes the
issue-time steady-state window, and scores fixed-budget (A) vs adaptive-CoV (B)
stopping rules against the full-series asymptote. ``steady_state`` is the official
number; ``total`` is reported alongside so their divergence is visible. Note
``total`` here is the full-series *issue-time* window (warmup included), so the
difference from ``steady_state`` isolates the warmup crop — a drain-inclusive
wall-clock total is deferred to the Milestone-2 report wiring.

usage:
  uv run python scripts/steady_state_from_events.py <events.jsonl> \
      --dataset-size N --concurrency N [--warmup 1] [--k 3] \
      [--cov-window 3] [--cov-bound 0.05] [--tokenizer DIR] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    encode_lengths,
    load_reference_backend,
)
from inference_endpoint.metrics.steady_state.harness import sweep
from inference_endpoint.metrics.steady_state.series import (
    build_super_pass_series,
    coverage_status,
    super_pass_size,
)
from inference_endpoint.metrics.steady_state.window import windowed_metrics


def _make_counter(path):
    backend = load_reference_backend(path)
    if backend is None:
        raise SystemExit(f"FATAL: could not load tokenizer backend from {path}")
    return lambda texts: encode_lengths(backend, texts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events")
    ap.add_argument("--dataset-size", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cov-window", type=int, default=3)
    ap.add_argument("--cov-bound", type=float, default=0.05)
    ap.add_argument("--tokenizer")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    counter = _make_counter(args.tokenizer) if args.tokenizer else None
    series = build_super_pass_series(
        args.events, args.dataset_size, args.concurrency, count_tokens=counter
    )
    if not series:
        raise SystemExit("FATAL: no performance-tracked samples in the events log")

    n_issued = sum(sp.n_issued for sp in series)
    status = coverage_status(n_issued, args.dataset_size, args.concurrency, args.warmup)
    sp_samples = super_pass_size(args.dataset_size, args.concurrency)
    n_full = n_issued // sp_samples

    # windowable: measure whole super-passes after the warmup crop. Otherwise
    # best-effort — window everything (including any partial tail), no warmup
    # crop — and flag it low-confidence via ``status``.
    if status == "windowable":
        measured = series[:n_full]
        eff_warmup = args.warmup
    else:
        measured = series
        eff_warmup = 0

    total = windowed_metrics(series, 0, len(series))
    ref, scores = sweep(
        measured,
        k=args.k,
        cov_window=args.cov_window,
        cov_bound=args.cov_bound,
        warmup=eff_warmup,
    )

    print(
        f"status: {status}  |  issued={n_issued}  super_pass_samples={sp_samples}  "
        f"full_super_passes={n_full}  (warmup dropped: {eff_warmup})"
    )
    if status != "windowable":
        print(f"  WARNING: {status} — best-effort steady_state, low confidence")
    print(f"total     : qps={total.qps:,.2f}  ttft_p99={total.ttft.get(0.99)}")
    print(f"steady(ref): qps={ref.qps:,.2f}  ttft_p99={ref.ttft.get(0.99)}")
    print("\n| rule | super-passes | region | qps | qps_rel_err | ttft_p99_rel_err |")
    print("|---|---|---|---|---|---|")
    for s in scores:
        if s.region is None:
            print(f"| {s.name} | - | UNCONVERGED | - | - | - |")
            continue
        print(
            f"| {s.name} | {s.super_passes} | {s.region} | "
            f"{s.metrics.qps:,.2f} | {s.qps_rel_err:.4f} | {s.ttft_p99_rel_err} |"
        )

    doc = {
        "status": status,
        "windowable": status == "windowable",
        "n_issued": n_issued,
        "super_pass_samples": sp_samples,
        "n_full_super_passes": n_full,
        "warmup_applied": eff_warmup,
        "total": asdict(total),
        "steady_state": asdict(ref),
        "rules": [asdict(s) for s in scores],
    }
    if args.json_out:
        # WindowMetrics dicts use float percentile keys; JSON needs str keys.
        with open(args.json_out, "w") as f:
            json.dump(doc, f, indent=2, default=str)
        print(f"\nwritten to {args.json_out}")
    return doc


if __name__ == "__main__":
    main()
