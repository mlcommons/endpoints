#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline steady-state windowing + A/B sweep from a run's events.jsonl.

Cold-path companion to ``scripts/early_stopping_estimate_from_events.py``. Buckets
performance-tracked samples into super-passes by issue order, computes the
issue-time steady-state window, and scores fixed-budget (A) vs adaptive-CoV (B)
stopping rules against the full-series asymptote. ``steady_state`` is the official
number; ``total`` is reported alongside so their divergence is visible.

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
from inference_endpoint.metrics.steady_state.series import build_super_pass_series
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
    if len(series) <= args.warmup:
        raise SystemExit(
            f"FATAL: only {len(series)} super-passes; need > warmup ({args.warmup})"
        )

    total = windowed_metrics(series, 0, len(series))
    ref, scores = sweep(
        series,
        k=args.k,
        cov_window=args.cov_window,
        cov_bound=args.cov_bound,
        warmup=args.warmup,
    )

    print(f"super-passes: {len(series)}  (warmup dropped: {args.warmup})")
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
