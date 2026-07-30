#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Study 3 — CoV-detector robustness + issue-to-issue tail-drop bias.

Two questions, both driven off the Phase-0 per-sample Parquet tables
(``~/skritch/endpoints-events-jsonl-artifacts/<point>/samples.parquet`` +
``run_meta.json``) — no re-parsing of the multi-GB ``events.jsonl`` files. The
Parquet ``super_pass`` column is the same issue-order bucketing that
``series.build_super_pass_series`` produces, so a per-super-pass ``SuperPassRollup``
series is reconstructed directly by grouping rows.

Q1 — tail-drop bias. The shipped window (``metrics/steady_state/window.py``) uses
an issue-to-issue throughput denominator (``last_issue - first_issue``), so
drain-tail completions (``complete_ns > last_issue_ns``, the tail-set definition
pinned in the coordination brief) never extend the throughput denominator but DO
enter the latency percentiles. This quantifies that asymmetry per point: how much
issue-to-issue throughput is overstated vs a wall-clock (issue->last-complete)
denominator, and how much the drain-tail inflates the reported latency/TTFT
percentiles.

Q2 — CoV convergence-detector robustness. Stress-tests
``stopping.rule_cov_converged`` / ``drift.ensemble_vote`` / ``drift.analyze_trend``:
(a) sensitivity to trailing-window length, CoV bound, percentile grid and metric
source set; (b) false-convergence on a synthetic slowly-drifting series; (c) noise
robustness under injected per-sample jitter; (d) cross-point generality including
the poisson point.

usage:
  uv run --with pyarrow --with numpy python scripts/steady_state_cov_robustness.py \
      [--base ~/skritch/endpoints-events-jsonl-artifacts] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from inference_endpoint.metrics.steady_state.drift import (
    DEFAULT_ENSEMBLE,
    analyze_trend,
    ensemble_vote,
)
from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.stopping import cov, rule_cov_converged
from inference_endpoint.metrics.steady_state.window import (
    percentile_lower,
    windowed_metrics,
)

POINTS = (
    "C8",
    "C140",
    "C1024",
    "C2048",
    "C7168",
    "C22528",
    "dsr1-c28k-tentative",
    "poisson-232386",
)
TENTATIVE = {"dsr1-c28k-tentative", "poisson-232386"}


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Point:
    name: str
    meta: dict
    issue_ns: np.ndarray
    complete_ns: np.ndarray
    ttft_ns: np.ndarray
    lifetime_ns: np.ndarray
    super_pass: np.ndarray

    @property
    def n_full(self) -> int:
        """Number of complete (non-partial) super-passes by issue count."""
        sps = self.meta["super_pass_size"]
        return self.meta["n_issued"] // sps


def load_point(base: str, name: str) -> Point:
    meta = json.load(open(f"{base}/{name}/run_meta.json"))
    t = pq.read_table(
        f"{base}/{name}/samples.parquet",
        columns=["super_pass", "issue_ns", "complete_ns", "ttft_ns", "lifetime_ns"],
    )
    return Point(
        name=name,
        meta=meta,
        issue_ns=t["issue_ns"].to_numpy(),
        complete_ns=t["complete_ns"].to_numpy(),
        ttft_ns=t["ttft_ns"].to_numpy().astype(float),
        lifetime_ns=t["lifetime_ns"].to_numpy().astype(float),
        super_pass=t["super_pass"].to_numpy(),
    )


def build_series(p: Point, drop_partial: bool = True) -> list[SuperPassRollup]:
    """Reconstruct the per-super-pass rollup series from the per-sample table.

    Mirrors ``series.build_super_pass_series`` (ttft/latency only; tpot null since
    out_tokens is absent from the COMPLETE payload). The trailing partial
    super-pass is dropped by default to match the ``measured = series[:n_full]``
    convention the shipped sweep scripts use.
    """
    nmax = int(p.super_pass.max()) + 1
    series = [SuperPassRollup(index=i) for i in range(nmax)]
    for i in range(nmax):
        m = p.super_pass == i
        r = series[i]
        r.n_issued = int(m.sum())
        r.first_issue_ns = int(p.issue_ns[m].min())
        r.last_issue_ns = int(p.issue_ns[m].max())
        r.ttft_ns = p.ttft_ns[m].tolist()
        r.latency_ns = p.lifetime_ns[m].tolist()
    if drop_partial:
        series = series[: p.n_full]
    return series


def _pct(a: np.ndarray, q: float) -> float:
    """percentile_lower semantics (matches window.py) over a numpy array."""
    a = np.sort(a)
    return float(a[int(q * (len(a) - 1))])


# --------------------------------------------------------------------------- #
# Q1 — issue-to-issue tail-drop bias
# --------------------------------------------------------------------------- #
def q1_tail_bias(p: Point) -> dict:
    first_issue = p.meta["first_issue_ns"]
    last_issue = p.meta["last_issue_ns"]
    last_complete = p.meta["last_complete_ns"]
    tail = p.complete_ns > last_issue  # pinned tail-set definition
    n = len(p.issue_ns)
    tn = int(tail.sum())

    iss_span = (last_issue - first_issue) / 1e9
    full_span = (last_complete - first_issue) / 1e9
    drain_span = (last_complete - last_issue) / 1e9
    # n_issued == n_complete on every corpus point, so the throughput
    # overstatement is purely the denominator ratio.
    thr_overstate_pct = (full_span / iss_span - 1.0) * 100.0

    def infl(arr_ns: np.ndarray, q: float) -> tuple[float, float, float]:
        allv = _pct(arr_ns, q) / 1e9
        notail = _pct(arr_ns[~tail], q) / 1e9
        pct = 100.0 * (allv - notail) / notail if notail else 0.0
        return allv, notail, pct

    lat_rows = {q: infl(p.lifetime_ns, q) for q in (0.5, 0.99, 0.999)}
    ttft_rows = {q: infl(p.ttft_ns, q) for q in (0.5, 0.99, 0.999)}

    # Is the drain-tail set itself latency-atypical? (Study-1 hairball framing.)
    tail_lat = p.lifetime_ns[tail] / 1e9
    steady_lat = p.lifetime_ns[~tail] / 1e9
    return {
        "point": p.name,
        "tentative": p.name in TENTATIVE,
        "n": n,
        "tail_n": tn,
        "tail_frac_pct": 100.0 * tn / n,
        "iss_span_s": iss_span,
        "full_span_s": full_span,
        "drain_span_s": drain_span,
        "drain_frac_pct": 100.0 * drain_span / full_span,
        "throughput_overstate_pct": thr_overstate_pct,
        "lat": lat_rows,
        "ttft": ttft_rows,
        "tail_lat_median_s": float(np.median(tail_lat)),
        "steady_lat_median_s": float(np.median(steady_lat)),
        "tail_lat_p99_s": _pct(p.lifetime_ns[tail], 0.99) / 1e9,
        "steady_lat_p99_s": _pct(p.lifetime_ns[~tail], 0.99) / 1e9,
    }


# --------------------------------------------------------------------------- #
# Q2 — CoV detector stress tests
# --------------------------------------------------------------------------- #
def _pct_across(sp: SuperPassRollup, source: str, q: float) -> float:
    vals = sorted(getattr(sp, source))
    return percentile_lower(vals, q) if vals else 0.0


def rule_cov_sources(
    series: list[SuperPassRollup],
    window: int,
    cov_bound: float,
    warmup: int,
    sources: tuple[str, ...],
    percentiles: tuple[float, ...],
) -> tuple[int, int] | None:
    """Generalized CoV rule: same logic as stopping.rule_cov_converged but with a
    configurable metric-source set, so the ensemble metric-set can be ablated.
    ``sources=("ttft_ns","latency_ns")`` + ``percentiles=(0.5,0.99)`` reproduces
    the shipped rule exactly."""
    n = len(series)
    for sp_end in range(warmup + window, n + 1):
        trailing = series[sp_end - window : sp_end]
        ok = True
        for source in sources:
            for q in percentiles:
                across = [_pct_across(sp, source, q) for sp in trailing]
                if cov(across) >= cov_bound:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return (warmup, sp_end)
    return None


def _rel_err(est: float | None, ref: float | None) -> float | None:
    if est is None or ref is None or ref == 0:
        return None
    return abs(est - ref) / abs(ref)


def q2a_sensitivity(series: list[SuperPassRollup], warmup: int = 1) -> dict:
    """Sweep window x bound x percentile-grid x metric-source-set.

    For each config record the converged region and its p99-TTFT / QPS rel-err vs
    the full post-warmup asymptote. Returns a compact summary + the raw grid.
    """
    if len(series) < warmup + 2:
        return {"windowable": False}
    asym = windowed_metrics(series, warmup, len(series))
    asym_p99 = asym.ttft.get(0.99)
    asym_qps = asym.qps

    windows = (2, 3, 4, 5, 6, 8)
    bounds = (0.02, 0.03, 0.05, 0.08, 0.10, 0.15)
    pgrids = {"p50": (0.5,), "p99": (0.99,), "p50+p99": (0.5, 0.99)}
    srcsets = {
        "ttft+lat": ("ttft_ns", "latency_ns"),
        "ttft": ("ttft_ns",),
        "lat": ("latency_ns",),
    }
    grid = []
    for wn in windows:
        for b in bounds:
            for pg_name, pg in pgrids.items():
                for ss_name, ss in srcsets.items():
                    reg = rule_cov_sources(series, wn, b, warmup, ss, pg)
                    if reg is None:
                        grid.append((wn, b, pg_name, ss_name, None, None, None))
                        continue
                    m = windowed_metrics(series, reg[0], reg[1])
                    pe = _rel_err(m.ttft.get(0.99), asym_p99)
                    qe = _rel_err(m.qps, asym_qps)
                    grid.append((wn, b, pg_name, ss_name, reg[1], pe, qe))
    return {
        "windowable": True,
        "n_super_passes": len(series),
        "warmup": warmup,
        "asym_p99_s": (asym_p99 / 1e9) if asym_p99 else None,
        "asym_qps": asym_qps,
        "grid": grid,
    }


def make_synthetic(
    n_sp: int,
    samples_per_sp: int,
    base_p50_s: float,
    drift_total_frac: float,
    noise_frac: float,
    seed: int = 0,
) -> list[SuperPassRollup]:
    """Synthetic super-pass series whose latency/TTFT drift linearly.

    ``drift_total_frac`` = total relative growth of the per-super-pass median from
    first to last super-pass (0.0 = flat). ``noise_frac`` = per-super-pass
    lognormal spread. TTFT and latency share the same shape (latency = 4x ttft) so
    the shipped 2-source rule sees a coherent drift.
    """
    rng = random.Random(seed)
    series: list[SuperPassRollup] = []
    for i in range(n_sp):
        frac = i / max(1, n_sp - 1)
        center = base_p50_s * (1.0 + drift_total_frac * frac)
        tt = []
        lat = []
        for _ in range(samples_per_sp):
            g = rng.gauss(0.0, noise_frac)
            v = center * math.exp(g)
            tt.append(v * 1e9)
            lat.append(v * 4.0 * 1e9)
        series.append(
            SuperPassRollup(
                index=i,
                n_issued=samples_per_sp,
                first_issue_ns=i * 1_000_000_000,
                last_issue_ns=i * 1_000_000_000 + samples_per_sp,
                ttft_ns=tt,
                latency_ns=lat,
            )
        )
    return series


def q2b_false_convergence() -> list[dict]:
    """Does the CoV rule declare steady on a series that is genuinely drifting?

    Sweep total drift 0..60% over 12 super-passes at low per-super-pass noise, run
    the shipped default rule (w3, b0.05) and a strict rule (w6, b0.03), and check
    the drift/trend gate (analyze_trend) that drift.py pairs with CoV.
    """
    out = []
    for drift in (0.0, 0.10, 0.20, 0.30, 0.45, 0.60):
        series = make_synthetic(
            n_sp=12,
            samples_per_sp=2000,
            base_p50_s=1.0,
            drift_total_frac=drift,
            noise_frac=0.02,
            seed=7,
        )
        reg_def = rule_cov_converged(series, window=3, cov_bound=0.05, warmup=1)
        reg_strict = rule_cov_converged(series, window=6, cov_bound=0.03, warmup=1)
        # trend gate over the p99-latency trajectory (post-warmup)
        traj = [_pct_across(sp, "latency_ns", 0.99) for sp in series[1:]]
        trend = analyze_trend(traj)
        out.append(
            {
                "drift_total_pct": drift * 100.0,
                "cov_default_w3_b05": reg_def,
                "cov_strict_w6_b03": reg_strict,
                "trend_rel_drift": trend.rel_drift,
                "trend_snr": trend.snr,
                "trend_verdict": trend.verdict,
            }
        )
    return out


def _jitter_series(
    series: list[SuperPassRollup], jitter_frac: float, seed: int
) -> list[SuperPassRollup]:
    rng = np.random.default_rng(seed)
    out = []
    for sp in series:
        tt = np.array(sp.ttft_ns)
        lat = np.array(sp.latency_ns)
        tt = tt * np.exp(rng.normal(0.0, jitter_frac, size=len(tt)))
        lat = lat * np.exp(rng.normal(0.0, jitter_frac, size=len(lat)))
        out.append(
            SuperPassRollup(
                index=sp.index,
                n_issued=sp.n_issued,
                first_issue_ns=sp.first_issue_ns,
                last_issue_ns=sp.last_issue_ns,
                ttft_ns=tt.tolist(),
                latency_ns=lat.tolist(),
            )
        )
    return out


def q2c_noise_robustness(
    series: list[SuperPassRollup], warmup: int = 1, n_seeds: int = 40
) -> dict:
    """Inject per-sample multiplicative jitter; measure verdict stability.

    Reports, per jitter level: converged fraction, median converged sp_end and its
    spread, and mean ensemble concordance across seeds.
    """
    base = rule_cov_converged(series, window=3, cov_bound=0.05, warmup=warmup)
    levels = (0.05, 0.10, 0.20)
    res = {"baseline_default_region": base, "levels": {}}
    for jf in levels:
        ends = []
        concords = []
        conv = 0
        for s in range(n_seeds):
            js = _jitter_series(series, jf, seed=1000 * int(jf * 100) + s)
            reg = rule_cov_converged(js, window=3, cov_bound=0.05, warmup=warmup)
            if reg is not None:
                conv += 1
                ends.append(reg[1])
            concords.append(ensemble_vote(js, warmup=warmup).concordance)
        res["levels"][jf] = {
            "converged_frac": conv / n_seeds,
            "end_median": float(np.median(ends)) if ends else None,
            "end_min": min(ends) if ends else None,
            "end_max": max(ends) if ends else None,
            "concordance_mean": float(np.mean(concords)),
        }
    return res


def q2d_cross_point(series: list[SuperPassRollup], warmup: int = 1) -> dict:
    reg = rule_cov_converged(series, window=3, cov_bound=0.05, warmup=warmup)
    vote = ensemble_vote(series, warmup=warmup)
    return {
        "n_super_passes": len(series),
        "default_region": reg,
        "ensemble_converged": vote.n_converged,
        "ensemble_detectors": vote.n_detectors,
        "ensemble_ends": vote.sp_ends,
        "concordance": vote.concordance,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _summarize_sensitivity(grid: list) -> dict:
    """Fraction of configs converging and their p99-err spread, per metric-set."""
    by_src: dict[str, list] = {}
    for _wn, _b, _pg, ss, end, pe, _qe in grid:
        by_src.setdefault(ss, []).append((end, pe))
    summ = {}
    for ss, rows in by_src.items():
        conv = [r for r in rows if r[0] is not None]
        pes = [r[1] for r in conv if r[1] is not None]
        summ[ss] = {
            "converged_frac": len(conv) / len(rows),
            "p99err_med": float(np.median(pes)) if pes else None,
            "p99err_p90": float(np.quantile(pes, 0.9)) if pes else None,
            "p99err_max": float(max(pes)) if pes else None,
        }
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--base",
        default=os.path.expanduser("~/skritch/endpoints-events-jsonl-artifacts"),
    )
    ap.add_argument("--json", default=None, help="write full results to this path")
    ap.add_argument("--points", default=",".join(POINTS))
    args = ap.parse_args(argv)
    points = args.points.split(",")

    results: dict = {"q1": {}, "q2": {}, "default_ensemble": list(DEFAULT_ENSEMBLE)}

    print("=" * 78)
    print("Q1 — issue-to-issue tail-drop bias (tail = complete_ns > last_issue_ns)")
    print("=" * 78)
    hdr = (
        f"{'point':>20} {'tail%':>6} {'drain%':>7} {'THR_over%':>9} "
        f"{'lat_p99_infl%':>13} {'lat_p999_infl%':>14} {'ttft_p99_infl%':>14}"
    )
    print(hdr)
    for name in points:
        p = load_point(args.base, name)
        q1 = q1_tail_bias(p)
        results["q1"][name] = q1
        star = "*" if q1["tentative"] else " "
        print(
            f"{name:>20}{star}{q1['tail_frac_pct']:>5.2f}%{q1['drain_frac_pct']:>6.2f}% "
            f"{q1['throughput_overstate_pct']:>+8.3f}% "
            f"{q1['lat'][0.99][2]:>+12.2f}% {q1['lat'][0.999][2]:>+13.2f}% "
            f"{q1['ttft'][0.99][2]:>+13.2f}%"
        )
    print("  (* = tentative config: dsr1-c28k / poisson)")
    print(
        "  THR_over% = issue-to-issue QPS overstatement vs issue->last-complete "
        "denominator"
    )
    print(
        "  *_infl% = drain-tail's inflation of the reported percentile "
        "(all - drop-tail)/drop-tail"
    )

    print("\n" + "=" * 78)
    print("Q2d — cross-point generality (shipped default: w=3, bound=0.05, warmup=1)")
    print("=" * 78)
    print(
        f"{'point':>20} {'n_sp':>5} {'default_region':>16} "
        f"{'ens_conv':>9} {'ends':>22} {'concord':>8}"
    )
    series_cache: dict[str, list[SuperPassRollup]] = {}
    for name in points:
        p = load_point(args.base, name)
        series = build_series(p)
        series_cache[name] = series
        d = q2d_cross_point(series)
        results["q2"].setdefault(name, {})["cross_point"] = d
        reg = str(d["default_region"])
        print(
            f"{name:>20} {d['n_super_passes']:>5} {reg:>16} "
            f"{d['ensemble_converged']:>1}/{d['ensemble_detectors']:>1} "
            f"{str(d['ensemble_ends']):>22} {d['concordance']:>7.2f}"
        )

    print("\n" + "=" * 78)
    print("Q2a — sensitivity: fraction of (w x bound x pgrid) configs that converge")
    print("       and their p99-TTFT rel-err vs asymptote, per metric-source set")
    print("=" * 78)
    for name in points:
        series = series_cache[name]
        sens = q2a_sensitivity(series, warmup=1)
        results["q2"][name]["sensitivity"] = sens
        if not sens.get("windowable"):
            print(f"{name:>20}  not windowable (n_sp<3)")
            continue
        summ = _summarize_sensitivity(sens["grid"])
        results["q2"][name]["sensitivity_summary"] = summ
        print(f"{name:>20}  (asym_qps={sens['asym_qps']:,.1f})")
        for ss in ("ttft+lat", "ttft", "lat"):
            s = summ[ss]
            pm = f"{s['p99err_med']:.3f}" if s["p99err_med"] is not None else "-"
            pmax = f"{s['p99err_max']:.3f}" if s["p99err_max"] is not None else "-"
            print(
                f"{'':>22}{ss:>10}: conv={s['converged_frac']:.2f}  "
                f"p99err med={pm} max={pmax}"
            )

    print("\n" + "=" * 78)
    print("Q2b — false convergence on a SYNTHETIC linearly-drifting series")
    print("       (12 super-passes, 2%/sp lognormal noise; drift = total growth)")
    print("=" * 78)
    fc = q2b_false_convergence()
    results["q2_synthetic_false_convergence"] = fc
    print(
        f"{'drift%':>7} {'cov_w3_b05':>12} {'cov_w6_b03':>12} "
        f"{'trend_reldrift':>14} {'trend_snr':>10} {'trend_verdict':>14}"
    )
    for r in fc:
        print(
            f"{r['drift_total_pct']:>6.0f}% {str(r['cov_default_w3_b05']):>12} "
            f"{str(r['cov_strict_w6_b03']):>12} {r['trend_rel_drift']:>+13.3f} "
            f"{r['trend_snr']:>10.1f} {r['trend_verdict']:>14}"
        )

    print("\n" + "=" * 78)
    print("Q2c — noise robustness: per-sample jitter, 40 seeds, default rule w3/b05")
    print("=" * 78)
    print(
        f"{'point':>20} {'jitter':>7} {'conv_frac':>10} "
        f"{'end_med':>8} {'end_range':>12} {'concord_mean':>12}"
    )
    for name in ("C1024", "C7168", "C22528", "poisson-232386"):
        if name not in series_cache:
            continue
        nr = q2c_noise_robustness(series_cache[name], warmup=1, n_seeds=40)
        results["q2"][name]["noise"] = nr
        for jf, lv in nr["levels"].items():
            rng = (
                f"[{lv['end_min']},{lv['end_max']}]"
                if lv["end_min"] is not None
                else "-"
            )
            em = f"{lv['end_median']:.0f}" if lv["end_median"] is not None else "-"
            print(
                f"{name:>20} {jf:>6.0%} {lv['converged_frac']:>10.2f} "
                f"{em:>8} {rng:>12} {lv['concordance_mean']:>12.2f}"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return results


if __name__ == "__main__":
    main()
