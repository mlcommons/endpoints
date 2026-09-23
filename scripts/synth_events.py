#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic benchmark ``events.jsonl`` generator with PLANTED ground truth.

Emits an ``events.jsonl`` whose wire schema matches what
``scripts/steady_state_diagnostics.py`` parses (``session.start_performance_tracking`` /
``session.stop_performance_tracking`` / ``sample.issued`` / ``sample.recv_first`` /
``sample.complete``), so the diagnostic's steady-state / drift detection can be validated
against a *known* steady region, warmup ramp, linear drift, staircase level-shift, and
closed-loop drain.

This is a tractable fluid/analytical simulator, NOT a discrete-event kernel:

  * ``concurrency`` (closed loop): C independent server "slots", each issuing the next
    request the instant the previous one completes -> exactly C in flight at all times.
  * ``poisson`` (open loop): Poisson-spaced arrivals at rate ``lambda`` (infinite server;
    requests overlap freely).

Per request the service model is::

    f(t)      = ramp(t) * drift(t) * staircase(t)          # multiplicative on the base
    TTFT_ns   = (base_ttft + prefill_per_tok*ISL) * f(t) * (1 + N(0, ttft_noise))
    d_ns      = base_tpot * f(t) * (1 + N(0, tpot_noise))  # per output token
    recv_first = issued + TTFT_ns
    complete   = recv_first + d_ns * (OSL - 1)

Output tokens are emitted as ``OSL`` single-word chunks (``"w "``) so a *whitespace*
tokenizer counts ``OSL - 1`` tokens for ``text_after_first_chunk`` (the diagnostic's TPOT
numerator excludes the first streamed chunk). The diagnostic is normally driven by a real
HF tokenizer; for synthetic validation feed it the whitespace word counter so 1 chunk ==
1 word == ~1 token, making TTFT/TPOT recover the planted base values.

The steady-state ground truth is the *base* metric level in the flat region: after the
ramp has decayed, with no drift and before any staircase step. It is written to a sidecar
``<out>.groundtruth.json``.

Determinism: all randomness flows through a single ``random.Random(seed)`` (default 42),
so a given seed reproduces byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Event-type wire constants (mirror scripts/steady_state_diagnostics.py).
EV_START = "session.start_performance_tracking"
EV_STOP = "session.stop_performance_tracking"
EV_ISSUED = "sample.issued"
EV_RECV_FIRST = "sample.recv_first"
EV_COMPLETE = "sample.complete"

# Sort priority for events sharing a timestamp: start first, then issue/recv/complete in
# causal order, stop last. Keeps the issued stream chronological so the diagnostic's
# issue-order super-pass bucketing is well defined.
_PRIORITY = {
    EV_START: 0,
    EV_ISSUED: 1,
    EV_RECV_FIRST: 2,
    EV_COMPLETE: 3,
    EV_STOP: 4,
}

# A minimum OSL of 2 guarantees at least one post-first-chunk token, so every sample
# contributes a TPOT sample (OSL == 1 would yield an empty text_after_first_chunk).
MIN_OSL = 2


@dataclass(slots=True)
class Planted:
    """Planted structure + base service levels (the ground truth)."""

    mode: str
    concurrency: int
    rate: float
    duration_s: float
    base_ttft_ns: float
    base_tpot_ns: float
    prefill_per_tok_ns: float
    ttft_noise: float
    tpot_noise: float
    osl_median: float
    osl_sigma: float
    osl_cap: int
    isl_median: float
    isl_sigma: float
    isl_cap: int
    ramp_s: float
    ramp_start: float
    drift_end: float
    step_at_s: float | None
    step_mag: float
    turns: int
    seed: int

    def factor(self, t_s: float) -> float:
        """Multiplicative service factor at wall-clock ``t_s`` seconds from tracking start.

        ramp decays from ``ramp_start`` at t=0 to exactly 1.0 at t=ramp_s (then stays 1);
        drift is linear 1.0 -> drift_end across the run; staircase steps by ``step_mag``
        at ``step_at_s``.
        """
        if self.ramp_s > 0 and self.ramp_start != 1.0 and t_s < self.ramp_s:
            ramp = self.ramp_start ** (1.0 - t_s / self.ramp_s)
        else:
            ramp = 1.0
        drift = (
            1.0 + (self.drift_end - 1.0) * (t_s / self.duration_s)
            if self.duration_s > 0
            else 1.0
        )
        stair = (
            self.step_mag
            if (self.step_at_s is not None and t_s >= self.step_at_s)
            else 1.0
        )
        return ramp * drift * stair


@dataclass(slots=True)
class _Gen:
    planted: Planted
    rng: random.Random
    start_ns: int
    events: list[dict[str, Any]] = field(default_factory=list)
    n_samples: int = 0
    last_complete_ns: int = 0
    _uuid: int = 0
    _conv: int = 0

    @property
    def duration_ns(self) -> int:
        return self.start_ns + int(self.planted.duration_s * 1e9)

    def _draw(self, median: float, sigma: float, cap: int, lo: int) -> int:
        v = self.rng.lognormvariate(math.log(median), sigma)
        return int(min(cap, max(lo, round(v))))

    def _emit_turn(self, issue_ns: int, conv_id: str | None, turn: int | None) -> int:
        """Emit one sample (issued/recv_first/complete); return its completion ns."""
        p = self.planted
        t_s = (issue_ns - self.start_ns) / 1e9
        f = p.factor(t_s)
        isl = self._draw(p.isl_median, p.isl_sigma, p.isl_cap, 1)
        osl = self._draw(p.osl_median, p.osl_sigma, p.osl_cap, MIN_OSL)
        ttft = (
            (p.base_ttft_ns + p.prefill_per_tok_ns * isl)
            * f
            * max(0.05, 1.0 + self.rng.gauss(0.0, p.ttft_noise))
        )
        d = p.base_tpot_ns * f * max(0.05, 1.0 + self.rng.gauss(0.0, p.tpot_noise))
        recv_ns = issue_ns + int(ttft)
        complete_ns = recv_ns + int(d * (osl - 1))

        uuid = f"s{self._uuid}"
        self._uuid += 1
        base: dict[str, Any] = {"sample_uuid": uuid}
        if conv_id is not None:
            base["conversation_id"] = conv_id
            base["turn"] = turn
        self._add(EV_ISSUED, issue_ns, base)
        self._add(EV_RECV_FIRST, recv_ns, base)
        chunks = ["w "] * osl  # OSL single-word chunks -> whitespace count == OSL
        self._add(EV_COMPLETE, complete_ns, base, data=["TextModelOutput", chunks])
        self.n_samples += 1
        self.last_complete_ns = max(self.last_complete_ns, complete_ns)
        return complete_ns

    def _add(
        self,
        et: str,
        ts: int,
        base: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> None:
        rec: dict[str, Any] = {"event_type": et, "timestamp_ns": ts}
        if base:
            rec.update(base)
        if data is not None:
            rec["data"] = data
        self.events.append(rec)

    def _run_conversation(self, start_ns: int) -> int:
        """Issue up to ``turns`` sequential turns from ``start_ns``; return end ns.

        Turns whose issue time has crossed the tracking window are not issued (the load
        stops offering at ``duration``); a turn issued just before the boundary still
        completes afterward -> the natural closed-loop drain.
        """
        p = self.planted
        conv_id = None if p.turns <= 1 else f"conv-{self._conv}"
        self._conv += 1
        t = start_ns
        for turn in range(1, p.turns + 1):
            if t >= self.duration_ns:
                break
            turn_id = None if p.turns <= 1 else turn
            t = self._emit_turn(t, conv_id, turn_id)
        return t

    def generate(self) -> None:
        p = self.planted
        self._add(EV_START, self.start_ns)
        if p.mode == "concurrency":
            for _ in range(p.concurrency):
                t = self.start_ns
                while t < self.duration_ns:
                    t = self._run_conversation(t)
        elif p.mode == "poisson":
            t = float(self.start_ns)
            while t < self.duration_ns:
                self._run_conversation(int(t))
                t += self.rng.expovariate(p.rate) * 1e9
        else:  # pragma: no cover - argparse restricts the choices
            raise ValueError(f"unknown mode: {p.mode}")
        self._add(EV_STOP, self.duration_ns)
        self.events.sort(key=lambda r: (r["timestamp_ns"], _PRIORITY[r["event_type"]]))


def build_events(planted: Planted, start_ns: int = 0) -> _Gen:
    gen = _Gen(planted=planted, rng=random.Random(planted.seed), start_ns=start_ns)
    gen.generate()
    return gen


def steady_window_s(planted: Planted) -> tuple[float, float]:
    """Wall-clock ``[lo, hi)`` seconds that is *truly* steady: after the ramp and before
    a staircase step (if any). Drift, if planted, is not removed -- the whole post-ramp
    span is returned and ``steady_is_flat`` records whether it is genuinely flat."""
    lo = planted.ramp_s
    hi = planted.duration_s
    if (
        planted.step_at_s is not None
        and planted.step_mag != 1.0
        and lo < planted.step_at_s < hi
    ):
        hi = planted.step_at_s
    return (lo, hi)


def groundtruth(planted: Planted, gen: _Gen) -> dict[str, Any]:
    lo, hi = steady_window_s(planted)
    has_step = (
        planted.step_at_s is not None
        and planted.step_mag != 1.0
        and planted.step_at_s < planted.duration_s
    )
    return {
        "mode": planted.mode,
        "concurrency": planted.concurrency if planted.mode == "concurrency" else None,
        "rate": planted.rate if planted.mode == "poisson" else None,
        "duration_s": planted.duration_s,
        "seed": planted.seed,
        "base_ttft_ns": planted.base_ttft_ns,
        "base_tpot_ns": planted.base_tpot_ns,
        "steady_per_user_tps": 1e9 / planted.base_tpot_ns,
        "ramp_s": planted.ramp_s,
        "ramp_start": planted.ramp_start,
        "drift_end": planted.drift_end,
        "step_at_s": planted.step_at_s,
        "step_mag": planted.step_mag,
        "osl_median": planted.osl_median,
        "osl_sigma": planted.osl_sigma,
        "osl_cap": planted.osl_cap,
        "isl_median": planted.isl_median,
        "isl_sigma": planted.isl_sigma,
        "ttft_noise": planted.ttft_noise,
        "tpot_noise": planted.tpot_noise,
        "turns": planted.turns,
        "n_samples": gen.n_samples,
        "run_wall_clock_s": (gen.last_complete_ns - gen.start_ns) / 1e9,
        "steady_window_s": [lo, hi],
        "steady_is_flat": (planted.drift_end == 1.0) and not has_step,
    }


def write_run(out_path: str, gen: _Gen, gt: dict[str, Any]) -> str:
    with open(out_path, "w") as fh:
        for rec in gen.events:
            fh.write(json.dumps(rec))
            fh.write("\n")
    sidecar = out_path + ".groundtruth.json"
    with open(sidecar, "w") as fh:
        json.dump(gt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return sidecar


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output events.jsonl path")
    ap.add_argument("--mode", choices=["concurrency", "poisson"], default="concurrency")
    ap.add_argument(
        "--concurrency", type=int, default=64, help="closed-loop in-flight C"
    )
    ap.add_argument(
        "--rate", type=float, default=50.0, help="poisson arrival rate lambda"
    )
    ap.add_argument("--duration-s", type=float, default=300.0, help="tracking duration")
    ap.add_argument("--seed", type=int, default=42)
    # base service levels
    ap.add_argument("--base-ttft-ms", type=float, default=50.0)
    ap.add_argument("--base-tpot-ms", type=float, default=10.0)
    ap.add_argument(
        "--prefill-per-tok-ns",
        type=float,
        default=0.0,
        help="per-ISL-token prefill added to TTFT (0 keeps base_ttft recoverable)",
    )
    # noise
    ap.add_argument("--ttft-noise", type=float, default=0.10)
    ap.add_argument("--tpot-noise", type=float, default=0.10)
    # OSL / ISL distributions (lognormal by median + sigma, capped)
    ap.add_argument("--osl-median", type=float, default=120.0)
    ap.add_argument("--osl-sigma", type=float, default=0.5)
    ap.add_argument("--osl-cap", type=int, default=8192)
    ap.add_argument("--isl-median", type=float, default=1024.0)
    ap.add_argument("--isl-sigma", type=float, default=0.5)
    ap.add_argument("--isl-cap", type=int, default=131072)
    # planted structure
    ap.add_argument(
        "--ramp-s", type=float, default=30.0, help="warmup ramp length (0=off)"
    )
    ap.add_argument("--ramp-start", type=float, default=3.0, help="ramp factor at t=0")
    ap.add_argument(
        "--drift-end", type=float, default=1.0, help="linear drift factor by run end"
    )
    ap.add_argument(
        "--step-at-s", type=float, default=None, help="staircase step time (None=off)"
    )
    ap.add_argument("--step-mag", type=float, default=1.15, help="staircase factor")
    ap.add_argument(
        "--turns", type=int, default=1, help=">1 emits multi-turn agentic trajectories"
    )
    return ap


def planted_from_args(args: argparse.Namespace) -> Planted:
    return Planted(
        mode=args.mode,
        concurrency=args.concurrency,
        rate=args.rate,
        duration_s=args.duration_s,
        base_ttft_ns=args.base_ttft_ms * 1e6,
        base_tpot_ns=args.base_tpot_ms * 1e6,
        prefill_per_tok_ns=args.prefill_per_tok_ns,
        ttft_noise=args.ttft_noise,
        tpot_noise=args.tpot_noise,
        osl_median=args.osl_median,
        osl_sigma=args.osl_sigma,
        osl_cap=args.osl_cap,
        isl_median=args.isl_median,
        isl_sigma=args.isl_sigma,
        isl_cap=args.isl_cap,
        ramp_s=args.ramp_s,
        ramp_start=args.ramp_start,
        drift_end=args.drift_end,
        step_at_s=args.step_at_s,
        step_mag=args.step_mag,
        turns=max(1, args.turns),
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    planted = planted_from_args(args)
    gen = build_events(planted)
    gt = groundtruth(planted, gen)
    sidecar = write_run(args.out, gen, gt)
    lo, hi = gt["steady_window_s"]
    print(
        f"wrote {args.out} ({gen.n_samples} samples, {len(gen.events)} events)\n"
        f"  mode={planted.mode} "
        f"{'C=' + str(planted.concurrency) if planted.mode == 'concurrency' else 'lambda=' + str(planted.rate)}"
        f" duration={planted.duration_s}s "
        f"run_wall_clock={gt['run_wall_clock_s']:.1f}s\n"
        f"  planted steady window: [{lo:.1f}s, {hi:.1f}s]  "
        f"flat={gt['steady_is_flat']}  "
        f"base_ttft={planted.base_ttft_ns / 1e6:.1f}ms "
        f"base_tpot={planted.base_tpot_ns / 1e6:.1f}ms "
        f"per_user_tps={gt['steady_per_user_tps']:.1f}\n"
        f"  groundtruth: {sidecar}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
