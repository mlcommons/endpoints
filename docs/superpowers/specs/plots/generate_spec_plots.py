#!/usr/bin/env python3
"""Generate plots for the warmup strategy design spec.

Uses data from both the Apr 3 and Apr 8 sweeps to produce:
1. Cold-start TTFT scatter: per-sample TTFT vs issue time (4 concurrency levels)
2. Warmup scenario timeline: visual of hybrid overlap for c=256
3. TTFT stabilization: rolling avg TTFT showing where steady state begins
4. Assumption validation: side-by-side old vs new run comparison
5. Warmup impact projection: extrapolated metrics with/without warmup
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

SPEC_DIR = Path(__file__).parent
OLD_SWEEP = Path(
    "/Users/rkaleem/Documents/Work/Code/endpoints/gpt_oss_120b_sweep_20260403_073427"
)
NEW_SWEEP = Path("/Users/rkaleem/Documents/Work/Code/endpoints/gpt-oss-8APR26")
CHARS_PER_TOKEN = 3.8


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def median(lst):
    return statistics.median(lst) if lst else 0.0


def load_metrics(sweep_dir, conc):
    path = sweep_dir / f"concurrency_{conc}" / "events.jsonl"
    events = [json.loads(line) for line in open(path)]
    test_start_ts = None
    samples = {}
    complete_values = {}
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
        n_tokens = max(1, int(len(complete_values.get(uuid, "")) / CHARS_PER_TOKEN))
        tpot = None
        if first_chunk and n_tokens > 1:
            tpot = ((complete - first_chunk) / 1e6) / (n_tokens - 1)
        out.append(
            {
                "issue_t": (issue - test_start_ts) / 1e9,
                "complete_t": (complete - test_start_ts) / 1e9,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "latency_ms": latency,
                "n_output_tokens": n_tokens,
            }
        )
    out.sort(key=lambda x: x["issue_t"])
    return out


def rolling(samples, key, window_s=30):
    """Compute rolling metric by completion time."""
    by_complete = sorted(samples, key=lambda s: s["complete_t"])
    if not by_complete:
        return [], []
    max_t = by_complete[-1]["complete_t"]
    bin_edges = list(range(0, int(max_t) + window_s, window_s))
    times, vals = [], []
    for i in range(len(bin_edges) - 1):
        t0, t1 = bin_edges[i], bin_edges[i + 1]
        bucket = [s for s in by_complete if t0 <= s["complete_t"] < t1]
        if not bucket:
            continue
        valid = [s[key] for s in bucket if s[key] is not None]
        if not valid:
            continue
        times.append((t0 + t1) / 2)
        vals.append(median(valid))
    return times, vals


def rolling_tps(samples, window_s=30):
    by_complete = sorted(samples, key=lambda s: s["complete_t"])
    if not by_complete:
        return [], []
    max_t = by_complete[-1]["complete_t"]
    bin_edges = list(range(0, int(max_t) + window_s, window_s))
    times, vals = [], []
    for i in range(len(bin_edges) - 1):
        t0, t1 = bin_edges[i], bin_edges[i + 1]
        bucket = [s for s in by_complete if t0 <= s["complete_t"] < t1]
        if not bucket:
            continue
        total_tokens = sum(s["n_output_tokens"] for s in bucket)
        times.append((t0 + t1) / 2)
        vals.append(total_tokens / window_s)
    return times, vals


# ════════════════════════════════════════════════════════════════════════
# Plot 1: Cold-start TTFT scatter (new run, 4 concurrency levels)
# ════════════════════════════════════════════════════════════════════════
def plot_coldstart_scatter():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Cold-Start Effect: Per-Sample TTFT by Issue Time\n"
        "(Apr 8 run, 30-min window, shaded = warmup zone of N=concurrency samples)",
        fontsize=13,
        fontweight="bold",
    )
    colors = {16: "#2ca02c", 64: "#1f77b4", 256: "#ff7f0e", 1024: "#d62728"}

    for idx, conc in enumerate([16, 64, 256, 1024]):
        ax = axes[idx // 2][idx % 2]
        metrics = load_metrics(NEW_SWEEP, conc)
        n = len(metrics)

        # Plot all samples
        issue_times = [m["issue_t"] for m in metrics if m["ttft_ms"] is not None]
        ttfts = [m["ttft_ms"] for m in metrics if m["ttft_ms"] is not None]
        ax.scatter(
            issue_times, ttfts, s=4, alpha=0.3, color=colors[conc], rasterized=True
        )

        # Warmup zone
        warmup_end_t = metrics[min(conc, n - 1)]["issue_t"] + 0.5
        ax.axvspan(0, warmup_end_t, alpha=0.12, color="orange")

        # Steady-state reference
        steady_ttfts = [
            m["ttft_ms"] for m in metrics[n // 2 :] if m["ttft_ms"] is not None
        ]
        if steady_ttfts:
            ss_med = median(steady_ttfts)
            ax.axhline(ss_med, color="green", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.text(
                max(issue_times) * 0.7,
                ss_med * 1.3,
                f"Steady-state median: {ss_med:.0f} ms",
                fontsize=8,
                color="green",
            )

        ax.set_yscale("log")
        ax.set_ylabel("TTFT (ms)", fontsize=9)
        ax.set_xlabel("Issue time from test start (s)", fontsize=9)
        ax.set_title(f"Concurrency = {conc}  (N={n})", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.2)

        warmup_patch = mpatches.Patch(
            color="orange", alpha=0.3, label=f"Warmup zone ({conc} samples)"
        )
        ax.legend(handles=[warmup_patch], fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(SPEC_DIR / "fig1_coldstart_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig1_coldstart_scatter.png")


# ════════════════════════════════════════════════════════════════════════
# Plot 2: Rolling TTFT/TPOT/TPS over time (new run)
# ════════════════════════════════════════════════════════════════════════
def plot_rolling_metrics():
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Rolling Metrics Over Time by Concurrency Level\n"
        "(Apr 8 run, 30s rolling window, median per bucket)",
        fontsize=13,
        fontweight="bold",
    )
    concs = [1, 16, 64, 256, 512, 1024]
    cmap = plt.cm.viridis
    colors = {c: cmap(i / (len(concs) - 1)) for i, c in enumerate(concs)}

    for conc in concs:
        metrics = load_metrics(NEW_SWEEP, conc)
        label = f"c={conc}"
        color = colors[conc]

        t, v = rolling(metrics, "ttft_ms")
        if t:
            axes[0].plot(t, v, color=color, label=label, linewidth=1.5, alpha=0.85)

        t, v = rolling(metrics, "tpot_ms")
        if t:
            axes[1].plot(t, v, color=color, label=label, linewidth=1.5, alpha=0.85)

        t, v = rolling_tps(metrics)
        if t:
            axes[2].plot(t, v, color=color, label=label, linewidth=1.5, alpha=0.85)

    for ax_i, (ylabel, title, log) in enumerate(
        [
            ("Median TTFT (ms)", "Time to First Token", True),
            ("Median TPOT (ms)", "Time Per Output Token", True),
            ("TPS (tokens/s)", "Throughput", False),
        ]
    ):
        axes[ax_i].set_ylabel(ylabel, fontsize=10)
        axes[ax_i].set_title(title, fontsize=11, fontweight="bold")
        axes[ax_i].legend(loc="upper right", fontsize=8, ncol=2)
        axes[ax_i].grid(True, alpha=0.25)
        if log:
            ymin, ymax = axes[ax_i].get_ylim()
            if ymin > 0:
                axes[ax_i].set_yscale("log")

    axes[2].set_xlabel("Time from test start (s)", fontsize=10)
    plt.tight_layout()
    fig.savefig(SPEC_DIR / "fig2_rolling_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig2_rolling_metrics.png")


# ════════════════════════════════════════════════════════════════════════
# Plot 3: Warmup timeline diagram (c=256, new run)
# ════════════════════════════════════════════════════════════════════════
def plot_warmup_timeline():
    conc = 256
    metrics = load_metrics(NEW_SWEEP, conc)
    n_warmup = conc

    warmup = metrics[:n_warmup]
    last_warmup_complete = max(s["complete_t"] for s in warmup)
    run_end = metrics[-1]["complete_t"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(
        f"Warmup Strategy Timeline (c={conc}, {n_warmup} warmup samples, Apr 8 data)",
        fontsize=13,
        fontweight="bold",
    )

    strategies = [
        ("No Warmup (baseline)", "none"),
        ("Drain Then Measure", "drain"),
        ("Hybrid Overlap (recommended)", "hybrid"),
    ]

    for ax_idx, (title, strategy) in enumerate(strategies):
        ax = axes[ax_idx]

        if strategy == "none":
            ax.axvspan(0, run_end, alpha=0.06, color="blue")
            # Cold zone
            ax.axvspan(0, last_warmup_complete, alpha=0.08, color="red")
            ax.text(
                last_warmup_complete / 2,
                0.5,
                f"Cold-start zone\nTTFT up to {max(m['ttft_ms'] for m in warmup if m['ttft_ms']):.0f} ms",
                ha="center",
                fontsize=9,
                color="darkred",
                fontweight="bold",
                va="center",
            )
            ax.annotate(
                "All samples counted\n(includes cold start)",
                xy=(run_end / 2, -0.3),
                fontsize=9,
                ha="center",
                color="blue",
                style="italic",
            )

        elif strategy == "drain":
            ax.axvspan(0, last_warmup_complete, alpha=0.1, color="orange")
            ax.text(
                last_warmup_complete / 2,
                0.5,
                "Warmup\n(not counted)",
                ha="center",
                fontsize=9,
                color="darkorange",
                fontweight="bold",
                va="center",
            )
            ax.axvline(last_warmup_complete, color="red", linewidth=2.5)
            ax.annotate(
                f"Drain complete\nt={last_warmup_complete:.0f}s\nBatch empty!",
                xy=(last_warmup_complete, 0.7),
                xytext=(last_warmup_complete + 40, 0.7),
                fontsize=8,
                color="red",
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "red"},
            )
            ax.axvspan(
                last_warmup_complete,
                run_end + last_warmup_complete,
                alpha=0.06,
                color="blue",
            )
            ax.text(
                last_warmup_complete + 400,
                -0.3,
                "Perf starts cold again",
                fontsize=9,
                color="blue",
                fontweight="bold",
            )

        elif strategy == "hybrid":
            # Warmup issuing
            ax.axvspan(0, last_warmup_complete, alpha=0.06, color="orange")
            ax.text(
                last_warmup_complete / 4,
                0.6,
                "Warmup issuing\n+ completing",
                fontsize=8,
                color="darkorange",
                va="center",
            )
            # Overlap zone
            warmup_issue_end = warmup[-1]["issue_t"]
            ax.axvspan(warmup_issue_end, last_warmup_complete, alpha=0.12, color="gray")
            ax.text(
                (warmup_issue_end + last_warmup_complete) / 2,
                -0.4,
                "Overlap\n(perf issued,\nnot counted)",
                ha="center",
                fontsize=7,
                color="gray",
                fontweight="bold",
            )
            # START_PERFORMANCE_TRACKING
            ax.axvline(last_warmup_complete, color="green", linewidth=3)
            ax.annotate(
                f"START_PERF_TRACKING\nt={last_warmup_complete:.0f}s",
                xy=(last_warmup_complete, 0.7),
                xytext=(last_warmup_complete + 60, 0.7),
                fontsize=9,
                color="green",
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "green", "lw": 1.5},
            )
            # Measured perf window
            ax.axvspan(last_warmup_complete, run_end, alpha=0.06, color="blue")
            ax.text(
                (last_warmup_complete + run_end) / 2,
                -0.3,
                "Perf measured (steady state)",
                fontsize=9,
                color="blue",
                fontweight="bold",
                ha="center",
            )

        ax.set_ylabel(title, fontsize=9, fontweight="bold")
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.grid(True, alpha=0.15, axis="x")

    axes[-1].set_xlabel("Time from test start (s)", fontsize=10)
    plt.tight_layout()
    fig.savefig(SPEC_DIR / "fig3_warmup_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig3_warmup_timeline.png")


# ════════════════════════════════════════════════════════════════════════
# Plot 4: Cross-run validation (old vs new, key metrics)
# ════════════════════════════════════════════════════════════════════════
def plot_cross_run_validation():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Cross-Run Validation: Apr 3 (20 min) vs Apr 8 (30 min)\n"
        "Cold-start ratio, TPOT stability, and backfill TTFT ratio",
        fontsize=13,
        fontweight="bold",
    )
    concs = [16, 64, 256, 512, 1024]
    x = list(range(len(concs)))
    width = 0.35

    # (a) Cold-start TTFT ratio
    old_ratios, new_ratios = [], []
    for conc in concs:
        for sweep, out_list in [(OLD_SWEEP, old_ratios), (NEW_SWEEP, new_ratios)]:
            try:
                m = load_metrics(sweep, conc)
            except FileNotFoundError:
                out_list.append(0)
                continue
            n = len(m)
            cutoff = max(conc, n // 10)
            first = [s["ttft_ms"] for s in m[:cutoff] if s["ttft_ms"] is not None]
            rest = [s["ttft_ms"] for s in m[cutoff:] if s["ttft_ms"] is not None]
            ratio = mean(first) / mean(rest) if rest and mean(rest) > 0 else 1.0
            out_list.append(ratio)

    axes[0].bar(
        [i - width / 2 for i in x],
        old_ratios,
        width,
        label="Apr 3 (20 min)",
        color="#1f77b4",
        alpha=0.8,
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        new_ratios,
        width,
        label="Apr 8 (30 min)",
        color="#ff7f0e",
        alpha=0.8,
    )
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"c={c}" for c in concs])
    axes[0].set_ylabel("Cold-start TTFT / Steady-state TTFT")
    axes[0].set_title("Cold-Start TTFT Ratio", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2, axis="y")

    # (b) TPOT ratio (first 10% / rest)
    old_tpot, new_tpot = [], []
    for conc in concs:
        for sweep, out_list in [(OLD_SWEEP, old_tpot), (NEW_SWEEP, new_tpot)]:
            try:
                m = load_metrics(sweep, conc)
            except FileNotFoundError:
                out_list.append(1.0)
                continue
            n = len(m)
            cutoff = max(conc, n // 10)
            first = [s["tpot_ms"] for s in m[:cutoff] if s["tpot_ms"] is not None]
            rest = [s["tpot_ms"] for s in m[cutoff:] if s["tpot_ms"] is not None]
            ratio = mean(first) / mean(rest) if rest and mean(rest) > 0 else 1.0
            out_list.append(ratio)

    axes[1].bar(
        [i - width / 2 for i in x],
        old_tpot,
        width,
        label="Apr 3",
        color="#1f77b4",
        alpha=0.8,
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        new_tpot,
        width,
        label="Apr 8",
        color="#ff7f0e",
        alpha=0.8,
    )
    axes[1].axhline(
        1.0,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="No cold-start effect",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"c={c}" for c in concs])
    axes[1].set_ylabel("First 10% TPOT / Rest TPOT")
    axes[1].set_title("TPOT Stability (1.0 = no cold start)", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 2)
    axes[1].grid(True, alpha=0.2, axis="y")

    # (c) Backfill TTFT / Steady-state TTFT
    old_bf, new_bf = [], []
    for conc in concs:
        for sweep, out_list in [(OLD_SWEEP, old_bf), (NEW_SWEEP, new_bf)]:
            try:
                m = load_metrics(sweep, conc)
            except FileNotFoundError:
                out_list.append(1.0)
                continue
            n = len(m)
            post_burst = [s for s in m if s["issue_t"] > 1.0]
            backfill = post_burst[:conc] if post_burst else []
            steady = m[n // 2 :]
            bf_ttfts = [s["ttft_ms"] for s in backfill if s["ttft_ms"] is not None]
            ss_ttfts = [s["ttft_ms"] for s in steady if s["ttft_ms"] is not None]
            ratio = (
                mean(bf_ttfts) / mean(ss_ttfts)
                if ss_ttfts and mean(ss_ttfts) > 0 and bf_ttfts
                else 1.0
            )
            out_list.append(ratio)

    axes[2].bar(
        [i - width / 2 for i in x],
        old_bf,
        width,
        label="Apr 3",
        color="#1f77b4",
        alpha=0.8,
    )
    axes[2].bar(
        [i + width / 2 for i in x],
        new_bf,
        width,
        label="Apr 8",
        color="#ff7f0e",
        alpha=0.8,
    )
    axes[2].axhline(
        1.0,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="= Steady state",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"c={c}" for c in concs])
    axes[2].set_ylabel("Backfill TTFT / Steady-state TTFT")
    axes[2].set_title("Backfill Quality (1.0 = immediately steady)", fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(
        SPEC_DIR / "fig4_cross_run_validation.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved: fig4_cross_run_validation.png")


# ════════════════════════════════════════════════════════════════════════
# Plot 5: Projected warmup impact on reported metrics
# ════════════════════════════════════════════════════════════════════════
def plot_warmup_impact_projection():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Projected Impact of Hybrid Warmup on Reported Metrics\n"
        "(Apr 8 data: excluding samples issued before START_PERFORMANCE_TRACKING)",
        fontsize=13,
        fontweight="bold",
    )
    concs = [16, 64, 128, 256, 512, 1024]
    x = list(range(len(concs)))
    width = 0.35

    no_warmup_ttft, hybrid_ttft = [], []
    no_warmup_tpot, hybrid_tpot = [], []
    no_warmup_tps, hybrid_tps = [], []
    no_warmup_n, hybrid_n = [], []
    excluded_n = []

    for conc in concs:
        m = load_metrics(NEW_SWEEP, conc)
        n = len(m)
        n_warmup = conc

        warmup_samples = m[:n_warmup]
        perf_samples = m[n_warmup:]

        # Last warmup completion = START_PERFORMANCE_TRACKING
        last_warmup_complete = max(s["complete_t"] for s in warmup_samples)

        # No warmup: all samples counted
        all_ttfts = [s["ttft_ms"] for s in m if s["ttft_ms"] is not None]
        all_tpots = [s["tpot_ms"] for s in m if s["tpot_ms"] is not None]
        all_tokens = sum(s["n_output_tokens"] for s in m)
        total_time = m[-1]["complete_t"] - m[0]["issue_t"]
        no_warmup_ttft.append(median(all_ttfts))
        no_warmup_tpot.append(median(all_tpots))
        no_warmup_tps.append(all_tokens / total_time if total_time > 0 else 0)
        no_warmup_n.append(n)

        # Hybrid: only perf samples issued after last warmup completion
        hybrid_perf = [s for s in perf_samples if s["issue_t"] >= last_warmup_complete]
        if not hybrid_perf:
            hybrid_perf = perf_samples[len(perf_samples) // 2 :]

        h_ttfts = [s["ttft_ms"] for s in hybrid_perf if s["ttft_ms"] is not None]
        h_tpots = [s["tpot_ms"] for s in hybrid_perf if s["tpot_ms"] is not None]
        h_tokens = sum(s["n_output_tokens"] for s in hybrid_perf)
        h_time = (
            hybrid_perf[-1]["complete_t"] - hybrid_perf[0]["issue_t"]
            if hybrid_perf
            else 1
        )
        hybrid_ttft.append(median(h_ttfts) if h_ttfts else 0)
        hybrid_tpot.append(median(h_tpots) if h_tpots else 0)
        hybrid_tps.append(h_tokens / h_time if h_time > 0 else 0)
        hybrid_n.append(len(hybrid_perf))
        excluded_n.append(n - n_warmup - len(hybrid_perf))

    c_red = "#d62728"
    c_blue = "#1f77b4"

    # TTFT
    ax = axes[0, 0]
    ax.bar(
        [i - width / 2 for i in x],
        no_warmup_ttft,
        width,
        label="No warmup",
        color=c_red,
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        hybrid_ttft,
        width,
        label="Hybrid warmup",
        color=c_blue,
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concs])
    ax.set_ylabel("Median TTFT (ms)")
    ax.set_title("TTFT (lower = better)", fontweight="bold")
    ymin, ymax = ax.get_ylim()
    if ymin > 0:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # TPOT
    ax = axes[0, 1]
    ax.bar(
        [i - width / 2 for i in x],
        no_warmup_tpot,
        width,
        label="No warmup",
        color=c_red,
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        hybrid_tpot,
        width,
        label="Hybrid warmup",
        color=c_blue,
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concs])
    ax.set_ylabel("Median TPOT (ms)")
    ax.set_title("TPOT (expected: minimal change)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # TPS
    ax = axes[1, 0]
    ax.bar(
        [i - width / 2 for i in x],
        no_warmup_tps,
        width,
        label="No warmup",
        color=c_red,
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        hybrid_tps,
        width,
        label="Hybrid warmup",
        color=c_blue,
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concs])
    ax.set_ylabel("TPS (tokens/s)")
    ax.set_title("Throughput", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # Sample counts
    ax = axes[1, 1]
    ax.bar(x, hybrid_n, width * 2, label="Counted (hybrid)", color=c_blue, alpha=0.8)
    ax.bar(
        x,
        excluded_n,
        width * 2,
        bottom=hybrid_n,
        label="Excluded (overlap)",
        color="gray",
        alpha=0.4,
    )
    warmup_heights = [concs[i] for i in range(len(concs))]
    bottoms = [hybrid_n[i] + excluded_n[i] for i in range(len(concs))]
    ax.bar(
        x,
        warmup_heights,
        width * 2,
        bottom=bottoms,
        label="Warmup",
        color="orange",
        alpha=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concs])
    ax.set_ylabel("Sample count")
    ax.set_title("Sample Allocation", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate % excluded
    for i in range(len(concs)):
        total = no_warmup_n[i]
        pct = (excluded_n[i] / total) * 100 if total > 0 else 0
        ax.text(
            i,
            hybrid_n[i] + excluded_n[i] + warmup_heights[i] + 50,
            f"{pct:.0f}% excl",
            ha="center",
            fontsize=7,
            color="gray",
        )

    plt.tight_layout()
    fig.savefig(SPEC_DIR / "fig5_warmup_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig5_warmup_impact.png")


# ════════════════════════════════════════════════════════════════════════
def main():
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating spec plots...")
    plot_coldstart_scatter()
    plot_rolling_metrics()
    plot_warmup_timeline()
    plot_cross_run_validation()
    plot_warmup_impact_projection()
    print("Done.")


if __name__ == "__main__":
    main()
