#!/usr/bin/env python3
"""Warmup strategy analysis plots for the concurrency sweep data.

Produces three figure sets:
1. Observed TTFT / TPOT / TPS over time per concurrency level
2. Extrapolated warmup scenarios (hybrid overlap vs drain-then-measure)
3. Performance impact summary comparing no-warmup, drain, and hybrid
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP_DIR = Path(__file__).parent
CONC_LEVELS = [1, 4, 16, 64, 256, 512, 1024]
WINDOW_S = 30  # rolling window for time-series metrics
WARMUP_FRACTION = 0.1  # warmup = 10% of total samples (matching concurrency)

# ── colour palette ──────────────────────────────────────────────────────
CMAP = plt.cm.viridis
COLORS = {c: CMAP(i / (len(CONC_LEVELS) - 1)) for i, c in enumerate(CONC_LEVELS)}


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════


def load_events(conc: int) -> list[dict]:
    path = SWEEP_DIR / f"concurrency_{conc}" / "events.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


CHARS_PER_TOKEN = 3.8  # rough estimate for English text; calibrated from OSL data


def build_sample_metrics(events: list[dict]) -> tuple[float, list[dict]]:
    """Return (test_start_ts, list of per-sample dicts sorted by issue time).

    Since events.jsonl only contains (data_load, issue, first_chunk, complete)
    per sample (no per-chunk events), we estimate output token count from the
    character length of the complete event's value field.
    """
    test_start_ts = None
    samples: dict[str, dict] = {}
    complete_values: dict[str, str] = {}

    for ev in events:
        if ev["event_type"] == "test_started":
            test_start_ts = ev["timestamp_ns"]
        uuid = ev["sample_uuid"]
        if not uuid:
            continue
        if uuid not in samples:
            samples[uuid] = {}
        samples[uuid][ev["event_type"]] = ev["timestamp_ns"]
        if ev["event_type"] == "complete":
            complete_values[uuid] = ev.get("value", "")

    out = []
    for uuid, evts in samples.items():
        issue = evts.get("loadgen_issue_called")
        complete = evts.get("complete")
        first_chunk = evts.get("first_chunk_received")
        if not issue or not complete:
            continue
        ttft = (first_chunk - issue) / 1e6 if first_chunk else None
        latency = (complete - issue) / 1e6

        # Estimate output tokens from response text length
        response_text = complete_values.get(uuid, "")
        n_output_tokens = max(1, int(len(response_text) / CHARS_PER_TOKEN))

        # TPOT = decode time / (output_tokens - 1)  [first token is TTFT]
        tpot = None
        if first_chunk and n_output_tokens > 1:
            decode_time_ms = (complete - first_chunk) / 1e6
            tpot = decode_time_ms / (n_output_tokens - 1)

        out.append(
            {
                "uuid": uuid,
                "issue_t": (issue - test_start_ts) / 1e9,
                "complete_t": (complete - test_start_ts) / 1e9,
                "ttft_ms": ttft,
                "latency_ms": latency,
                "tpot_ms": tpot,
                "n_output_tokens": n_output_tokens,
            }
        )
    out.sort(key=lambda x: x["issue_t"])
    return test_start_ts, out


def rolling_metrics(samples: list[dict], window_s: float = WINDOW_S):
    """Compute rolling TTFT, TPOT, TPS by completion time."""
    by_complete = sorted(samples, key=lambda s: s["complete_t"])
    if not by_complete:
        return [], [], [], []

    max_t = by_complete[-1]["complete_t"]
    bin_edges = np.arange(0, max_t + window_s, window_s)
    times, ttfts, tpots, tps_vals = [], [], [], []

    for i in range(len(bin_edges) - 1):
        t0, t1 = bin_edges[i], bin_edges[i + 1]
        bucket = [s for s in by_complete if t0 <= s["complete_t"] < t1]
        if not bucket:
            continue
        mid = (t0 + t1) / 2
        times.append(mid)

        valid_ttfts = [s["ttft_ms"] for s in bucket if s["ttft_ms"] is not None]
        valid_tpots = [s["tpot_ms"] for s in bucket if s["tpot_ms"] is not None]
        ttfts.append(np.median(valid_ttfts) if valid_ttfts else float("nan"))
        tpots.append(np.median(valid_tpots) if valid_tpots else float("nan"))

        total_tokens = sum(s["n_output_tokens"] for s in bucket)
        tps_vals.append(total_tokens / window_s)

    return times, ttfts, tpots, tps_vals


# ════════════════════════════════════════════════════════════════════════
# Figure 1: Observed metrics over time
# ════════════════════════════════════════════════════════════════════════


def plot_observed_metrics(all_data: dict):
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        "Observed Metrics Over Time by Concurrency Level\n(no warmup — raw benchmark data)",
        fontsize=14,
        fontweight="bold",
    )

    for conc in CONC_LEVELS:
        samples = all_data[conc]
        times, ttfts, tpots, tps_vals = rolling_metrics(samples)
        if not times:
            continue
        color = COLORS[conc]
        label = f"c={conc}"
        axes[0].plot(times, ttfts, color=color, label=label, alpha=0.85, linewidth=1.5)
        axes[1].plot(times, tpots, color=color, label=label, alpha=0.85, linewidth=1.5)
        axes[2].plot(
            times, tps_vals, color=color, label=label, alpha=0.85, linewidth=1.5
        )

    for ax_i, (ylabel, title, use_log) in enumerate(
        [
            ("Median TTFT (ms)", "Time to First Token", True),
            ("Median TPOT (ms)", "Time Per Output Token", True),
            ("TPS (tokens/s)", "Throughput (Tokens Per Second)", False),
        ]
    ):
        axes[ax_i].set_ylabel(ylabel)
        axes[ax_i].set_title(title)
        axes[ax_i].legend(loc="upper right", fontsize=8, ncol=2)
        axes[ax_i].grid(True, alpha=0.3)
        if use_log:
            ymin, ymax = axes[ax_i].get_ylim()
            if ymin > 0 and ymax > 0:
                axes[ax_i].set_yscale("log")
    axes[2].set_xlabel("Time from test start (s)")

    plt.tight_layout()
    fig.savefig(
        SWEEP_DIR / "plots" / "warmup_observed_metrics.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: warmup_observed_metrics.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 2: Extrapolated warmup scenarios
# ════════════════════════════════════════════════════════════════════════


def simulate_warmup_scenarios(samples: list[dict], conc: int):
    """Simulate three scenarios using the real data:

    1. No warmup (baseline): all samples counted, including cold start
    2. Drain-then-measure: warmup samples issued, wait for all to complete,
       then start perf. Simulated as: skip first N samples entirely,
       add a gap (drain time), shift remaining samples forward.
    3. Hybrid (overlap): warmup samples issued first, perf starts immediately
       after warmup issuance. START_PERFORMANCE_TRACKING fires when last
       warmup completes. Perf samples issued after that timestamp count.

    Returns dict of {scenario: {times, ttfts, tpots, tps_vals, perf_samples, total_duration}}.
    """
    n = len(samples)
    n_warmup = max(conc, int(n * WARMUP_FRACTION))  # at least conc samples for warmup
    n_warmup = min(n_warmup, n - 10)  # leave at least 10 for perf

    if n_warmup <= 0 or n < 20:
        return None

    # ── Scenario 1: No warmup (baseline) ─────────────────────────────
    baseline_times, baseline_ttfts, baseline_tpots, baseline_tps = rolling_metrics(
        samples
    )

    # ── Scenario 2: Drain-then-measure ───────────────────────────────
    # Use first n_warmup as warmup. After last warmup completes, gap, then perf.
    warmup_samples = samples[:n_warmup]
    perf_samples_drain = samples[n_warmup:]
    # Drain time = time from last warmup issue to last warmup complete
    last_warmup_complete = max(s["complete_t"] for s in warmup_samples)
    first_perf_issue = perf_samples_drain[0]["issue_t"] if perf_samples_drain else 0
    # In drain scenario, perf starts after drain completes
    # Shift perf sample times so they start after drain
    drain_gap = last_warmup_complete - first_perf_issue
    drain_shifted = []
    for s in perf_samples_drain:
        shifted = dict(s)
        shifted["issue_t"] = s["issue_t"] + drain_gap
        shifted["complete_t"] = s["complete_t"] + drain_gap
        drain_shifted.append(shifted)
    drain_times, drain_ttfts, drain_tpots, drain_tps = rolling_metrics(drain_shifted)
    # But the drain scenario means perf samples see a COLD batch scheduler again
    # because the batch emptied during drain. So the first perf samples would have
    # TTFT similar to the original first samples. We'll annotate this.

    # ── Scenario 3: Hybrid (overlap) ─────────────────────────────────
    # Warmup samples issued at their original times.
    # Perf samples also issued at their original times (overlap).
    # START_PERFORMANCE_TRACKING = last warmup completion time.
    # Only perf samples ISSUED after that timestamp count.
    start_perf_tracking = last_warmup_complete
    hybrid_perf = [s for s in perf_samples_drain if s["issue_t"] >= start_perf_tracking]
    # Re-base time to start_perf_tracking
    if not hybrid_perf:
        # All perf samples were issued before warmup completed — edge case
        hybrid_perf = perf_samples_drain[len(perf_samples_drain) // 2 :]
    hybrid_shifted = []
    t_offset = hybrid_perf[0]["issue_t"] if hybrid_perf else start_perf_tracking
    for s in hybrid_perf:
        shifted = dict(s)
        shifted["issue_t"] = s["issue_t"] - t_offset
        shifted["complete_t"] = s["complete_t"] - t_offset
        hybrid_shifted.append(shifted)
    hybrid_times, hybrid_ttfts, hybrid_tpots, hybrid_tps = rolling_metrics(
        hybrid_shifted
    )

    return {
        "no_warmup": {
            "times": baseline_times,
            "ttfts": baseline_ttfts,
            "tpots": baseline_tpots,
            "tps": baseline_tps,
            "n_perf": n,
            "total_duration": samples[-1]["complete_t"],
        },
        "drain": {
            "times": drain_times,
            "ttfts": drain_ttfts,
            "tpots": drain_tpots,
            "tps": drain_tps,
            "n_perf": len(perf_samples_drain),
            "total_duration": drain_shifted[-1]["complete_t"] if drain_shifted else 0,
            "drain_time": drain_gap,
        },
        "hybrid": {
            "times": hybrid_times,
            "ttfts": hybrid_ttfts,
            "tpots": hybrid_tpots,
            "tps": hybrid_tps,
            "n_perf": len(hybrid_perf),
            "total_duration": hybrid_shifted[-1]["complete_t"] if hybrid_shifted else 0,
            "overlap_excluded": len(perf_samples_drain) - len(hybrid_perf),
            "start_perf_tracking": start_perf_tracking,
        },
        "n_warmup": n_warmup,
    }


def plot_warmup_scenarios(all_data: dict):
    focus_concs = [16, 64, 256, 1024]
    fig, axes = plt.subplots(len(focus_concs), 3, figsize=(20, 4 * len(focus_concs)))
    fig.suptitle(
        "Warmup Strategy Comparison: No Warmup vs Drain vs Hybrid\n"
        f"(warmup = max(concurrency, 10% of samples); {WINDOW_S}s rolling window)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    scenario_colors = {"no_warmup": "#d62728", "drain": "#2ca02c", "hybrid": "#1f77b4"}
    scenario_labels = {
        "no_warmup": "No warmup",
        "drain": "Drain then measure",
        "hybrid": "Hybrid (overlap)",
    }
    scenario_styles = {"no_warmup": "--", "drain": "-.", "hybrid": "-"}

    for row, conc in enumerate(focus_concs):
        scenarios = simulate_warmup_scenarios(all_data[conc], conc)
        if scenarios is None:
            continue

        for col, (metric, ylabel, use_log) in enumerate(
            [
                ("ttfts", "Median TTFT (ms)", True),
                ("tpots", "Median TPOT (ms)", True),
                ("tps", "TPS (tokens/s)", False),
            ]
        ):
            ax = axes[row][col]
            for sname in ["no_warmup", "drain", "hybrid"]:
                sc = scenarios[sname]
                if sc["times"]:
                    ax.plot(
                        sc["times"],
                        sc[metric],
                        color=scenario_colors[sname],
                        linestyle=scenario_styles[sname],
                        label=scenario_labels[sname],
                        linewidth=1.8,
                        alpha=0.85,
                    )
            if use_log:
                ymin, ymax = ax.get_ylim()
                if ymin > 0 and ymax > 0:
                    ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(
                    ylabel.split("(")[0].strip(), fontsize=11, fontweight="bold"
                )
            if row == len(focus_concs) - 1:
                ax.set_xlabel("Time from measurement start (s)", fontsize=9)

            # Annotate concurrency on left column
            if col == 0:
                n_wu = scenarios["n_warmup"]
                ax.text(
                    0.02,
                    0.95,
                    f"c={conc}\nwarmup={n_wu}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "wheat",
                        "alpha": 0.5,
                    },
                )
            if row == 0 and col == 2:
                ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(
        SWEEP_DIR / "plots" / "warmup_scenario_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: warmup_scenario_comparison.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 3: Performance impact summary
# ════════════════════════════════════════════════════════════════════════


def plot_performance_impact(all_data: dict):
    """Bar charts showing the steady-state metric differences across strategies."""
    concs = [4, 16, 64, 128, 256, 512, 1024]
    results = {
        "no_warmup": {"ttft_p50": [], "tpot_p50": [], "tps": [], "n_perf": []},
        "drain": {"ttft_p50": [], "tpot_p50": [], "tps": [], "n_perf": []},
        "hybrid": {"ttft_p50": [], "tpot_p50": [], "tps": [], "n_perf": []},
    }
    valid_concs = []

    for conc in concs:
        if conc not in all_data:
            continue
        samples = all_data[conc]
        scenarios = simulate_warmup_scenarios(samples, conc)
        if scenarios is None:
            continue
        valid_concs.append(conc)

        for sname in ["no_warmup", "drain", "hybrid"]:
            sc = scenarios[sname]
            # Use metrics from the middle 80% of the measurement window to get steady state
            if not sc["times"]:
                for k in results[sname]:
                    results[sname][k].append(float("nan"))
                continue
            n_bins = len(sc["times"])
            start_idx = n_bins // 10
            end_idx = n_bins - n_bins // 10
            mid_ttfts = [
                sc["ttfts"][i]
                for i in range(start_idx, end_idx)
                if not math.isnan(sc["ttfts"][i])
            ]
            mid_tpots = [
                sc["tpots"][i]
                for i in range(start_idx, end_idx)
                if not math.isnan(sc["tpots"][i])
            ]
            mid_tps = [
                sc["tps"][i]
                for i in range(start_idx, end_idx)
                if not math.isnan(sc["tps"][i])
            ]

            results[sname]["ttft_p50"].append(
                np.median(mid_ttfts) if mid_ttfts else float("nan")
            )
            results[sname]["tpot_p50"].append(
                np.median(mid_tpots) if mid_tpots else float("nan")
            )
            results[sname]["tps"].append(
                np.median(mid_tps) if mid_tps else float("nan")
            )
            results[sname]["n_perf"].append(sc["n_perf"])

    x = np.arange(len(valid_concs))
    width = 0.25
    scenario_colors = {"no_warmup": "#d62728", "drain": "#2ca02c", "hybrid": "#1f77b4"}
    scenario_labels = {"no_warmup": "No warmup", "drain": "Drain", "hybrid": "Hybrid"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Performance Impact of Warmup Strategy\n(steady-state median from middle 80% of measurement window)",
        fontsize=14,
        fontweight="bold",
    )

    for _i, (metric, ylabel, ax, use_log) in enumerate(
        [
            ("ttft_p50", "Median TTFT (ms)", axes[0, 0], True),
            ("tpot_p50", "Median TPOT (ms)", axes[0, 1], True),
            ("tps", "TPS (tokens/s)", axes[1, 0], False),
            ("n_perf", "Perf samples counted", axes[1, 1], False),
        ]
    ):
        for j, sname in enumerate(["no_warmup", "drain", "hybrid"]):
            vals = results[sname][metric]
            offset = (j - 1) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=scenario_labels[sname],
                color=scenario_colors[sname],
                alpha=0.8,
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"c={c}" for c in valid_concs], fontsize=9)
        ax.set_xlabel("Concurrency Level")
        if use_log:
            ymin, ymax = ax.get_ylim()
            if ymin > 0 and ymax > 0:
                ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(
        SWEEP_DIR / "plots" / "warmup_performance_impact.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: warmup_performance_impact.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 4: Cold-start magnification — first N completions detail
# ════════════════════════════════════════════════════════════════════════


def plot_coldstart_detail(all_data: dict):
    """Per-sample TTFT for the first batch of completions at each concurrency."""
    focus_concs = [16, 64, 256, 1024]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Cold-Start Detail: TTFT of First Completions\n"
        "(each dot = one sample, ordered by completion time; shaded = warmup zone)",
        fontsize=14,
        fontweight="bold",
    )

    for idx, conc in enumerate(focus_concs):
        ax = axes[idx // 2][idx % 2]
        samples = all_data[conc]
        by_complete = sorted(samples, key=lambda s: s["complete_t"])

        n_show = min(len(by_complete), conc * 3)  # show up to 3x concurrency
        show = by_complete[:n_show]

        ttfts = [s["ttft_ms"] for s in show if s["ttft_ms"] is not None]
        complete_times = [s["complete_t"] for s in show if s["ttft_ms"] is not None]

        ax.scatter(complete_times, ttfts, s=8, alpha=0.6, color=COLORS[conc])

        # Mark the warmup zone (first `conc` completions)
        n_warmup = min(conc, len(show))
        if n_warmup > 0 and n_warmup < len(complete_times):
            warmup_end_t = complete_times[n_warmup - 1]
            ax.axvspan(
                0,
                warmup_end_t,
                alpha=0.1,
                color="orange",
                label=f"Warmup zone ({n_warmup} samples)",
            )
            ax.axvline(
                warmup_end_t, color="orange", linestyle="--", linewidth=1, alpha=0.7
            )

        # Steady-state reference line
        steady_ttfts = [
            s["ttft_ms"] for s in by_complete[conc * 2 :] if s["ttft_ms"] is not None
        ]
        if steady_ttfts:
            ss_median = np.median(steady_ttfts)
            ax.axhline(
                ss_median,
                color="green",
                linestyle=":",
                linewidth=1.5,
                label=f"Steady-state median ({ss_median:.0f}ms)",
            )

        ax.set_yscale("log")
        ax.set_ylabel("TTFT (ms)")
        ax.set_xlabel("Completion time from test start (s)")
        ax.set_title(f"Concurrency = {conc}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        SWEEP_DIR / "plots" / "warmup_coldstart_detail.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: warmup_coldstart_detail.png")


# ════════════════════════════════════════════════════════════════════════
# Figure 5: Warmup timeline diagram
# ════════════════════════════════════════════════════════════════════════


def plot_warmup_timeline(all_data: dict):
    """Visual timeline showing the three strategies for c=256."""
    conc = 256
    samples = all_data[conc]
    n = len(samples)
    n_warmup = max(conc, int(n * 0.1))

    warmup_samples = samples[:n_warmup]
    perf_samples = samples[n_warmup:]
    last_warmup_complete = max(s["complete_t"] for s in warmup_samples)

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(
        f"Warmup Strategy Timeline Comparison (c={conc}, {n_warmup} warmup samples)",
        fontsize=14,
        fontweight="bold",
    )

    for ax_idx, (title, strategy) in enumerate(
        [
            ("No Warmup (baseline)", "none"),
            ("Drain Then Measure", "drain"),
            ("Hybrid (overlap, recommended)", "hybrid"),
        ]
    ):
        ax = axes[ax_idx]

        if strategy == "none":
            # All samples are perf, show issue→complete bars
            for s in samples[:100]:  # show first 100
                color = "#d62728" if s["issue_t"] < 1.0 else "#1f77b4"
                ax.barh(
                    0,
                    s["complete_t"] - s["issue_t"],
                    left=s["issue_t"],
                    height=0.6,
                    color=color,
                    alpha=0.015,
                )
            # Mark measurement window
            ax.axvspan(0, samples[-1]["complete_t"], alpha=0.05, color="blue")
            ax.text(
                samples[-1]["complete_t"] / 2,
                0.7,
                "← All samples counted (includes cold start) →",
                ha="center",
                fontsize=9,
                style="italic",
            )

        elif strategy == "drain":
            # Warmup phase
            for s in warmup_samples[:100]:
                ax.barh(
                    0,
                    s["complete_t"] - s["issue_t"],
                    left=s["issue_t"],
                    height=0.6,
                    color="orange",
                    alpha=0.02,
                )
            ax.axvspan(0, last_warmup_complete, alpha=0.08, color="orange")
            ax.text(
                last_warmup_complete / 2,
                0.7,
                "Warmup (not counted)",
                ha="center",
                fontsize=9,
                color="darkorange",
                fontweight="bold",
            )
            # Drain gap
            drain_end = last_warmup_complete
            ax.axvline(drain_end, color="red", linewidth=2, linestyle="-")
            ax.text(
                drain_end + 2,
                0.7,
                "Drain complete →\nbatch empty!",
                fontsize=8,
                color="red",
                fontweight="bold",
            )
            # Perf phase (shifted)
            perf_start = drain_end
            ax.axvspan(
                perf_start,
                perf_start + perf_samples[-1]["complete_t"],
                alpha=0.05,
                color="blue",
            )
            ax.text(
                perf_start + 200,
                -0.7,
                "Perf (cold batch scheduler again!)",
                fontsize=9,
                color="blue",
                fontweight="bold",
            )

        elif strategy == "hybrid":
            # Warmup phase
            for s in warmup_samples[:100]:
                ax.barh(
                    0.3,
                    s["complete_t"] - s["issue_t"],
                    left=s["issue_t"],
                    height=0.3,
                    color="orange",
                    alpha=0.02,
                )
            ax.axvspan(0, last_warmup_complete, alpha=0.05, color="orange")
            ax.text(
                last_warmup_complete / 4,
                0.7,
                "Warmup issuing + completing",
                fontsize=9,
                color="darkorange",
            )
            # Perf starts immediately after warmup issuance (≈t=0 for concurrency)
            perf_issue_start = warmup_samples[-1]["issue_t"]
            for s in perf_samples[:100]:
                ax.barh(
                    -0.3,
                    s["complete_t"] - s["issue_t"],
                    left=s["issue_t"],
                    height=0.3,
                    color="#1f77b4",
                    alpha=0.02,
                )
            # Overlap zone (not counted)
            ax.axvspan(perf_issue_start, last_warmup_complete, alpha=0.08, color="gray")
            ax.text(
                (perf_issue_start + last_warmup_complete) / 2,
                -0.7,
                "Overlap\n(perf issued but excluded)",
                ha="center",
                fontsize=8,
                color="gray",
                fontweight="bold",
            )
            # START_PERFORMANCE_TRACKING
            ax.axvline(
                last_warmup_complete, color="green", linewidth=2.5, linestyle="-"
            )
            ax.text(
                last_warmup_complete + 5,
                0.7,
                f"START_PERF_TRACKING\nt={last_warmup_complete:.0f}s",
                fontsize=8,
                color="green",
                fontweight="bold",
            )
            # Counted perf window
            ax.axvspan(
                last_warmup_complete,
                perf_samples[-1]["complete_t"],
                alpha=0.05,
                color="blue",
            )
            ax.text(
                last_warmup_complete + 300,
                -0.7,
                "← Perf measured (steady state) →",
                fontsize=9,
                color="blue",
                fontweight="bold",
            )

        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([])
        ax.grid(True, alpha=0.2, axis="x")

    axes[-1].set_xlabel("Time from test start (s)")
    plt.tight_layout()
    fig.savefig(
        SWEEP_DIR / "plots" / "warmup_timeline_diagram.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved: warmup_timeline_diagram.png")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════


def main():
    (SWEEP_DIR / "plots").mkdir(exist_ok=True)

    print("Loading event data...")
    all_data = {}
    for conc in sorted(set(CONC_LEVELS + [4, 16, 64, 128, 256, 512, 1024])):
        path = SWEEP_DIR / f"concurrency_{conc}" / "events.jsonl"
        if not path.exists():
            continue
        _, samples = build_sample_metrics(load_events(conc))
        all_data[conc] = samples
        print(f"  c={conc}: {len(samples)} samples")

    print("\nGenerating plots...")
    plot_observed_metrics(all_data)
    plot_warmup_scenarios(all_data)
    plot_performance_impact(all_data)
    plot_coldstart_detail(all_data)
    plot_warmup_timeline(all_data)

    # Print summary table
    print("\n" + "=" * 90)
    print("WARMUP STRATEGY PERFORMANCE IMPACT SUMMARY")
    print("=" * 90)
    print(
        f"{'Conc':>6s} | {'Strategy':>12s} | {'TTFT p50':>10s} | {'TPOT p50':>10s} | {'TPS':>8s} | {'Perf N':>8s} | {'Excluded':>10s}"
    )
    print("-" * 90)

    for conc in [16, 64, 256, 512, 1024]:
        if conc not in all_data:
            continue
        scenarios = simulate_warmup_scenarios(all_data[conc], conc)
        if not scenarios:
            continue
        for sname, slabel in [
            ("no_warmup", "No warmup"),
            ("drain", "Drain"),
            ("hybrid", "Hybrid"),
        ]:
            sc = scenarios[sname]
            if not sc["times"]:
                continue
            n_bins = len(sc["times"])
            s, e = n_bins // 10, n_bins - n_bins // 10
            ttft = np.nanmedian([sc["ttfts"][i] for i in range(s, e)])
            tpot = np.nanmedian([sc["tpots"][i] for i in range(s, e)])
            tps = np.nanmedian([sc["tps"][i] for i in range(s, e)])
            excluded = sc.get("overlap_excluded", 0)
            print(
                f"{conc:>6d} | {slabel:>12s} | {ttft:>8.1f}ms | {tpot:>8.1f}ms | {tps:>8.0f} | {sc['n_perf']:>8d} | {excluded:>10d}"
            )
        print("-" * 90)


if __name__ == "__main__":
    main()
