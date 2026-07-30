#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline simulation of burst vs staged ("feathered") ramp for the load generator.

Companion analysis to ``docs/design/2026-07-29-feathered-starts-design.md``. This
is a *simulation only* — no live server, no strategy change. It models a
finite-rate inference server to quantify how a staged ramp (issuing ``N`` in
``S`` increments instead of all at ``t=0``) reshapes the offered-load curve,
shrinks the non-steady ramp region, and reduces the ramp-up TTFT spike, in both
``concurrency`` and ``max_throughput`` modes.

Server model (deliberately minimal, see the design doc for the full rationale and
its limits):

- A single first-token ("prefill") admission queue served FIFO at
  ``prefill_rate`` requests/sec when saturated. This is the finite server rate
  that turns a t=0 fill-burst into a queue: the k-th request admitted waits
  ~k / prefill_rate before its first token. This is the ramp-up TTFT inflation
  the steady-state windowing design targets.
- Per-request first-token compute adds a fixed ``prefill_service_s`` baseline.
- Once a request has its first token it "decodes" for a random time
  (lognormal, ``decode_mean_s`` / ``decode_cv`` — models OSL heterogeneity),
  then completes.
- ``concurrency`` mode holds ``N`` in flight: after the fill phase, each
  completion issues one replacement. ``max_throughput`` issues a fixed pool of
  queries with no replacement.

Calibration: with no measured burst-tail magnitude supplied, ``prefill_rate``
defaults so that a full ``t=0`` burst of ``--concurrency`` requests produces a
peak first-token wait on the order of the design doc's ``point_28080`` GB300
DeepSeek-R1 observation (max TTFT ~101.8 s at N=28080 ⇒ ~276 req/s). Override
``--prefill-rate`` to recalibrate to a Study-1 measured tail.

usage:
  uv run --with matplotlib python scripts/staggered_ramp_sim.py
  uv run python scripts/staggered_ramp_sim.py --mode concurrency \
      --concurrency 2048 --steps 1,2,4,8 --out-dir /tmp/ramp_sim
"""

from __future__ import annotations

import argparse
import math
import random
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerParams:
    """Finite-rate server model."""

    prefill_rate: float  # requests/sec brought to first-token when saturated
    prefill_service_s: float  # fixed per-request first-token baseline (steady TTFT)
    decode_mean_s: float  # mean decode (first-token -> complete) time
    decode_cv: float  # coefficient of variation of decode time (OSL spread)


@dataclass
class SimResult:
    label: str
    steps: int
    step_interval_s: float
    ttfts: list[float] = field(default_factory=list)  # per-request, admission-order
    ttft_issue_times: list[float] = field(
        default_factory=list
    )  # issue time of each ttft
    issue_times: list[float] = field(
        default_factory=list
    )  # every issue (may exceed ttfts)
    t_grid: list[float] = field(default_factory=list)  # sampled time axis
    inflight: list[int] = field(default_factory=list)  # in-flight(t)
    issued_cum: list[int] = field(default_factory=list)  # cumulative issued(t)
    # derived
    steady_ttft_s: float = 0.0
    peak_ttft_s: float = 0.0
    p99_ttft_s: float = 0.0
    ramp_end_s: float = 0.0
    ramp_nonsteady_count: int = 0
    total_issued: int = 0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _decode_time(rng: random.Random, mean_s: float, cv: float) -> float:
    """Lognormal decode time with the requested mean and coefficient of variation."""
    if cv <= 0.0:
        return mean_s
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    mu = math.log(mean_s) - 0.5 * sigma * sigma
    return rng.lognormvariate(mu, sigma)


def simulate(
    *,
    mode: str,
    concurrency: int,
    total_queries: int,
    steps: int,
    step_interval_s: float,
    server: ServerParams,
    dt: float,
    horizon_s: float,
    seed: int,
) -> SimResult:
    """Discrete-time simulation of one ramp schedule.

    ``steps == 1`` is the current production behavior: fill the whole target at
    ``t=0`` (BurstStrategy / ConcurrencyStrategy). ``steps > 1`` issues the
    target in ``steps`` equal increments spaced ``step_interval_s`` apart.
    """
    rng = random.Random(seed)
    is_conc = mode == "concurrency"
    fill_target = concurrency if is_conc else total_queries

    # Precompute staged fill events: (time, count) that together sum to fill_target.
    per_step = math.ceil(fill_target / steps)
    fill_events: list[tuple[float, int]] = []
    remaining = fill_target
    for k in range(steps):
        if remaining <= 0:
            break
        n = min(per_step, remaining)
        fill_events.append((k * step_interval_s, n))
        remaining -= n
    fill_event_idx = 0
    fill_done_target = fill_target  # total to issue during the fill phase

    prefill_queue: list[float] = []  # issue-times of requests awaiting first token
    decode_finish: list[float] = []  # absolute completion times of decoding requests

    ttfts: list[float] = []
    ttft_issue_times: list[float] = []
    issue_times: list[float] = []
    t_grid: list[float] = []
    inflight_series: list[int] = []
    issued_cum_series: list[int] = []

    issued_total = 0
    filled = 0  # counts fill-phase issues (fill complete when filled == fill_target)
    completed = 0

    def issue(now: float) -> None:
        nonlocal issued_total
        prefill_queue.append(now)
        issue_times.append(now)
        issued_total += 1

    t = 0.0
    prefill_budget_carry = 0.0
    n_steps_iter = int(math.ceil(horizon_s / dt))
    for _ in range(n_steps_iter):
        # 1) staged fill issuance due at or before `t`
        while fill_event_idx < len(fill_events) and fill_events[fill_event_idx][0] <= t:
            _, n = fill_events[fill_event_idx]
            for _i in range(n):
                issue(t)
                filled += 1
            fill_event_idx += 1

        # 2) admit from prefill queue at prefill_rate (FIFO). Carry fractional budget.
        prefill_budget_carry += server.prefill_rate * dt
        n_admit = int(prefill_budget_carry)
        if n_admit > 0:
            n_admit = min(n_admit, len(prefill_queue))
            prefill_budget_carry -= n_admit
            for _i in range(n_admit):
                issue_t = prefill_queue.pop(0)
                # TTFT = queue wait (now - issue) + fixed first-token compute.
                ttfts.append((t - issue_t) + server.prefill_service_s)
                ttft_issue_times.append(issue_t)
                decode_finish.append(
                    t + _decode_time(rng, server.decode_mean_s, server.decode_cv)
                )
        else:
            prefill_budget_carry -= 0.0

        # 3) process completions due at or before `t`
        if decode_finish:
            decode_finish.sort()
            due = bisect_right(decode_finish, t)
            if due > 0:
                del decode_finish[:due]
                completed += due
                # concurrency replacement: refill to N once the fill phase is done
                if is_conc and filled >= fill_done_target:
                    for _i in range(due):
                        issue(t)

        # 4) sample time series
        inflight = len(prefill_queue) + len(decode_finish)
        t_grid.append(t)
        inflight_series.append(inflight)
        issued_cum_series.append(issued_total)

        # stop early for max_throughput once everything has drained
        if not is_conc and filled >= fill_target and inflight == 0 and t > 0:
            break
        t += dt

    res = SimResult(
        label=f"{mode}/steps={steps}",
        steps=steps,
        step_interval_s=step_interval_s,
        ttfts=ttfts,
        ttft_issue_times=ttft_issue_times,
        issue_times=issue_times,
        t_grid=t_grid,
        inflight=inflight_series,
        issued_cum=issued_cum_series,
        total_issued=issued_total,
    )
    _derive_metrics(res, server)
    return res


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))
    return sorted_vals[idx]


def _derive_metrics(res: SimResult, server: ServerParams) -> None:
    """Compute steady baseline, spike, and non-steady ramp-region size.

    Steady baseline = the theoretical floor first-token latency
    (``prefill_service_s``); the non-steady region is the set of leading requests
    (in issue order) whose TTFT exceeds ``baseline * (1 + tol)``. ``ramp_end_s``
    is the issue time of the last such request.
    """
    if not res.ttfts:
        return
    baseline = server.prefill_service_s
    res.steady_ttft_s = baseline
    res.peak_ttft_s = max(res.ttfts)
    res.p99_ttft_s = _percentile(sorted(res.ttfts), 0.99)

    tol = 0.5  # 50% over the floor counts as "still ramping"
    threshold = baseline * (1.0 + tol)
    # Non-steady = leading run of issue-ordered requests above threshold. Use a
    # trailing tolerance so a lone late spike (drain jitter) doesn't extend it.
    last_hot = -1
    for i, v in enumerate(res.ttfts):
        if v > threshold:
            last_hot = i
    res.ramp_nonsteady_count = last_hot + 1
    res.ramp_end_s = res.ttft_issue_times[last_hot] if last_hot >= 0 else 0.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_row(res: SimResult, baseline: SimResult) -> str:
    def ratio(a: float, b: float) -> str:
        return f"{a / b:5.2f}x" if b else "  n/a"

    return (
        f"  steps={res.steps:>3}  int={res.step_interval_s:5.2f}s | "
        f"peak_ttft={res.peak_ttft_s:7.2f}s ({ratio(res.peak_ttft_s, baseline.peak_ttft_s)}) | "
        f"p99_ttft={res.p99_ttft_s:7.2f}s | "
        f"ramp_end={res.ramp_end_s:6.2f}s | "
        f"nonsteady={res.ramp_nonsteady_count:>7} "
        f"({ratio(float(res.ramp_nonsteady_count), float(baseline.ramp_nonsteady_count))}) | "
        f"issued={res.total_issued}"
    )


def _write_csv(results: list[SimResult], path: Path) -> None:
    lines = [
        "label,steps,step_interval_s,peak_ttft_s,p99_ttft_s,steady_ttft_s,"
        "ramp_end_s,ramp_nonsteady_count,total_issued"
    ]
    for r in results:
        lines.append(
            f"{r.label},{r.steps},{r.step_interval_s:.4f},{r.peak_ttft_s:.4f},"
            f"{r.p99_ttft_s:.4f},{r.steady_ttft_s:.4f},{r.ramp_end_s:.4f},"
            f"{r.ramp_nonsteady_count},{r.total_issued}"
        )
    path.write_text("\n".join(lines) + "\n")


def _maybe_plot(results: list[SimResult], mode: str, out_dir: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, (ax_load, ax_ttft) = plt.subplots(2, 1, figsize=(10, 9))
    for r in results:
        lbl = f"steps={r.steps}"
        ax_load.plot(r.t_grid, r.inflight, label=lbl, linewidth=1.3)
        # TTFT vs issue time (the ramp spike)
        ax_ttft.plot(r.ttft_issue_times, r.ttfts, label=lbl, linewidth=0.9, alpha=0.85)

    ax_load.set_title(f"In-flight (offered load) over time — {mode}")
    ax_load.set_xlabel("time (s)")
    ax_load.set_ylabel("in-flight requests")
    ax_load.legend()
    ax_load.grid(True, alpha=0.3)

    ax_ttft.set_title("TTFT by issue time (ramp-up spike)")
    ax_ttft.set_xlabel("issue time (s)")
    ax_ttft.set_ylabel("TTFT (s)")
    ax_ttft.legend()
    ax_ttft.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"ramp_sim_{mode}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_steps(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["concurrency", "max_throughput", "both"],
        default="both",
    )
    p.add_argument(
        "--concurrency", type=int, default=2048, help="target N (concurrency mode)"
    )
    p.add_argument(
        "--total-queries",
        type=int,
        default=8192,
        help="query pool size (max_throughput mode)",
    )
    p.add_argument(
        "--steps",
        type=_parse_steps,
        default=[1, 2, 4, 8, 16],
        help="comma-separated staged-ramp step counts; 1 = current t=0 burst",
    )
    p.add_argument(
        "--step-interval",
        type=float,
        default=0.0,
        help="seconds between staged increments; 0 = auto (one prefill-drain of a step)",
    )
    p.add_argument(
        "--prefill-rate",
        type=float,
        default=276.0,
        help="server first-token admission rate (req/s); calibrates burst-tail magnitude",
    )
    p.add_argument("--prefill-service", type=float, default=0.36, help="floor TTFT (s)")
    p.add_argument(
        "--decode-mean", type=float, default=12.0, help="mean decode time (s)"
    )
    p.add_argument(
        "--decode-cv", type=float, default=0.8, help="decode time CoV (OSL spread)"
    )
    p.add_argument("--dt", type=float, default=0.02, help="sim time step (s)")
    p.add_argument(
        "--horizon", type=float, default=0.0, help="sim horizon (s); 0 = auto"
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", type=Path, default=None, help="write CSV + plots here")
    args = p.parse_args()

    server = ServerParams(
        prefill_rate=args.prefill_rate,
        prefill_service_s=args.prefill_service,
        decode_mean_s=args.decode_mean,
        decode_cv=args.decode_cv,
    )

    modes = ["concurrency", "max_throughput"] if args.mode == "both" else [args.mode]
    out_dir = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Staggered-ramp simulation (OFFLINE — model only, not a live measurement)")
    print(
        f"server: prefill_rate={server.prefill_rate} req/s, "
        f"floor_ttft={server.prefill_service_s}s, "
        f"decode_mean={server.decode_mean_s}s (cv={server.decode_cv})"
    )
    print("=" * 78)

    for mode in modes:
        fill_target = args.concurrency if mode == "concurrency" else args.total_queries
        # auto step interval: time to drain one step's prefill queue.
        results: list[SimResult] = []
        for steps in args.steps:
            per_step = math.ceil(fill_target / steps)
            step_interval = args.step_interval
            if step_interval <= 0.0:
                step_interval = per_step / server.prefill_rate
            # auto horizon: fill span + a few decode times to observe steady state.
            horizon = args.horizon
            if horizon <= 0.0:
                fill_span = steps * step_interval
                horizon = fill_span + 6.0 * server.decode_mean_s
            results.append(
                simulate(
                    mode=mode,
                    concurrency=args.concurrency,
                    total_queries=args.total_queries,
                    steps=steps,
                    step_interval_s=step_interval,
                    server=server,
                    dt=args.dt,
                    horizon_s=horizon,
                    seed=args.seed,
                )
            )

        baseline = results[0]  # steps=1 burst is the reference
        print(f"\n--- mode: {mode} (fill_target={fill_target}) ---")
        print(f"  steady-state floor TTFT ~= {baseline.steady_ttft_s:.2f}s")
        for r in results:
            print(_fmt_row(r, baseline))

        if out_dir is not None:
            _write_csv(results, out_dir / f"ramp_sim_{mode}.csv")
            plot = _maybe_plot(results, mode, out_dir)
            if plot:
                print(f"  plot: {plot}")
            else:
                print("  (matplotlib not available — CSV written, plot skipped)")

    print(
        "\nNote: simulation only. Peak-TTFT and non-steady-count reductions are "
        "model outputs under the stated server assumptions, not measured results."
    )


if __name__ == "__main__":
    main()
