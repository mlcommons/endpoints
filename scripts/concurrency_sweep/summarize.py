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

"""Summarize and tabulate metrics from a concurrency sweep results folder.

Usage:
    python scripts/concurrency_sweep/summarize.py <sweep_dir>
    python scripts/concurrency_sweep/summarize.py results/my_sweep/concurrency_sweep/

Outputs:
    - Formatted tables to stdout
    - <sweep_dir>/metrics_summary.csv
    - <sweep_dir>/metrics_summary.md
    - <sweep_dir>/metrics_summary.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def ns_to_ms(value: float) -> float:
    return round(value / 1e6, 2)


def parse_result_summary(result_file: Path) -> dict | None:
    try:
        data = json.loads(result_file.read_text())
    except Exception as e:
        print(f"Error reading {result_file}: {e}", file=sys.stderr)
        return None

    lat = data.get("latency") or {}
    ttft = data.get("ttft") or {}
    tpot = data.get("tpot") or {}
    osl = data.get("output_sequence_lengths") or {}

    def f(v: object) -> float:
        """Return v as float, falling back to 0.0 for None/missing."""
        return float(v) if v is not None else 0.0

    pct = lambda d, k: f(d.get("percentiles", {}).get(k))  # noqa: E731

    return {
        "qps": round(f(data.get("qps")), 2),
        "tps": round(f(data.get("tps")), 2),
        "latency_mean_ms": ns_to_ms(f(lat.get("avg"))),
        "latency_p50_ms": ns_to_ms(f(lat.get("median"))),
        "latency_p90_ms": ns_to_ms(pct(lat, "90")),
        "latency_p95_ms": ns_to_ms(pct(lat, "95")),
        "latency_p99_ms": ns_to_ms(pct(lat, "99")),
        "ttft_mean_ms": ns_to_ms(f(ttft.get("avg"))),
        "ttft_p50_ms": ns_to_ms(f(ttft.get("median"))),
        "ttft_p90_ms": ns_to_ms(pct(ttft, "90")),
        "ttft_p99_ms": ns_to_ms(pct(ttft, "99")),
        "tpot_mean_ms": ns_to_ms(f(tpot.get("avg"))),
        "tpot_p50_ms": ns_to_ms(f(tpot.get("median"))),
        "tpot_p90_ms": ns_to_ms(pct(tpot, "90")),
        "tpot_p99_ms": ns_to_ms(pct(tpot, "99")),
        "n_completed": data.get("n_samples_completed", 0),
        "duration_s": round(f(data.get("duration_ns")) / 1e9, 1),
        "avg_output_tokens": round(f(osl.get("avg")), 1),
    }


def collect_results(sweep_dir: Path) -> list[dict]:
    def _concurrency_key(p: Path) -> int:
        try:
            return int(p.name.split("_")[1])
        except (IndexError, ValueError):
            return -1

    rows = []
    for concurrency_dir in sorted(
        sweep_dir.glob("concurrency_*"), key=_concurrency_key
    ):
        try:
            concurrency = int(concurrency_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        result_file = concurrency_dir / "result_summary.json"
        if result_file.exists():
            metrics = parse_result_summary(result_file)
            if metrics:
                rows.append({"concurrency": concurrency, "status": "ok", **metrics})
            else:
                rows.append({"concurrency": concurrency, "status": "parse_error"})
        else:
            rows.append({"concurrency": concurrency, "status": "no_results"})

    return rows


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------


def print_table(rows: list[dict]) -> None:
    successful = [r for r in rows if r["status"] == "ok"]

    def section(title: str, columns: list[tuple[str, str]]) -> None:
        print(f"\n{title}")
        header_parts = [f"{'Conc':>6}"]
        for label, _ in columns:
            header_parts.append(f"{label:>12}")
        print("  ".join(header_parts))
        print("  ".join("-" * w for w in [6] + [12] * len(columns)))
        for r in rows:
            if r["status"] != "ok":
                vals = [f"{r['concurrency']:>6}", f"  {'-- ' + r['status']:>12}"]
                print("  ".join(vals))
                continue
            parts = [f"{r['concurrency']:>6}"]
            for _, key in columns:
                parts.append(f"{r[key]:>12}")
            print("  ".join(parts))

    section(
        "Throughput",
        [
            ("QPS", "qps"),
            ("TPS", "tps"),
            ("Completed", "n_completed"),
            ("Duration(s)", "duration_s"),
            ("AvgOutTok", "avg_output_tokens"),
        ],
    )
    section(
        "End-to-End Latency (ms)",
        [
            ("Mean", "latency_mean_ms"),
            ("P50", "latency_p50_ms"),
            ("P90", "latency_p90_ms"),
            ("P95", "latency_p95_ms"),
            ("P99", "latency_p99_ms"),
        ],
    )
    section(
        "Time to First Token / TTFT (ms)",
        [
            ("Mean", "ttft_mean_ms"),
            ("P50", "ttft_p50_ms"),
            ("P90", "ttft_p90_ms"),
            ("P99", "ttft_p99_ms"),
        ],
    )
    section(
        "Time Per Output Token / TPOT (ms)",
        [
            ("Mean", "tpot_mean_ms"),
            ("P50", "tpot_p50_ms"),
            ("P90", "tpot_p90_ms"),
            ("P99", "tpot_p99_ms"),
        ],
    )

    if successful:
        best_qps = max(successful, key=lambda r: r["qps"])
        best_lat = min(successful, key=lambda r: r["latency_p50_ms"])
        print(
            f"\nPeak throughput : {best_qps['qps']} QPS  (concurrency={best_qps['concurrency']})"
        )
        print(
            f"Best P50 latency: {best_lat['latency_p50_ms']} ms  (concurrency={best_lat['concurrency']})"
        )
        print(f"Successful runs : {len(successful)}/{len(rows)}")


# ---------------------------------------------------------------------------
# File outputs
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "concurrency",
    "status",
    "qps",
    "tps",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p99_ms",
    "n_completed",
    "duration_s",
    "avg_output_tokens",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def write_markdown(rows: list[dict], path: Path) -> None:
    successful = [r for r in rows if r["status"] == "ok"]

    def md_table(headers: list[str], col_keys: list[str], data: list[dict]) -> str:
        sep = "|" + "|".join("---" for _ in headers) + "|"
        hdr = "|" + "|".join(headers) + "|"
        lines = [hdr, sep]
        for r in data:
            cells = [str(r.get(k, r.get("status", "-"))) for k in col_keys]
            lines.append("|" + "|".join(cells) + "|")
        return "\n".join(lines)

    with path.open("w") as f:
        f.write("# Concurrency Sweep Results\n\n")

        f.write("## Throughput\n\n")
        f.write(
            md_table(
                [
                    "Concurrency",
                    "Status",
                    "QPS",
                    "TPS",
                    "Completed",
                    "Duration (s)",
                    "Avg Out Tokens",
                ],
                [
                    "concurrency",
                    "status",
                    "qps",
                    "tps",
                    "n_completed",
                    "duration_s",
                    "avg_output_tokens",
                ],
                rows,
            )
        )

        f.write("\n\n## End-to-End Latency (ms)\n\n")
        f.write(
            md_table(
                ["Concurrency", "Mean", "P50", "P90", "P95", "P99"],
                [
                    "concurrency",
                    "latency_mean_ms",
                    "latency_p50_ms",
                    "latency_p90_ms",
                    "latency_p95_ms",
                    "latency_p99_ms",
                ],
                [r for r in rows if r["status"] == "ok"],
            )
        )

        f.write("\n\n## Time to First Token / TTFT (ms)\n\n")
        f.write(
            md_table(
                ["Concurrency", "Mean", "P50", "P90", "P99"],
                [
                    "concurrency",
                    "ttft_mean_ms",
                    "ttft_p50_ms",
                    "ttft_p90_ms",
                    "ttft_p99_ms",
                ],
                [r for r in rows if r["status"] == "ok"],
            )
        )

        f.write("\n\n## Time Per Output Token / TPOT (ms)\n\n")
        f.write(
            md_table(
                ["Concurrency", "Mean", "P50", "P90", "P99"],
                [
                    "concurrency",
                    "tpot_mean_ms",
                    "tpot_p50_ms",
                    "tpot_p90_ms",
                    "tpot_p99_ms",
                ],
                [r for r in rows if r["status"] == "ok"],
            )
        )

        if successful:
            best_qps = max(successful, key=lambda r: r["qps"])
            best_lat = min(successful, key=lambda r: r["latency_p50_ms"])
            f.write("\n\n## Analysis\n\n")
            f.write(
                f"- **Peak throughput:** {best_qps['qps']} QPS at concurrency={best_qps['concurrency']}\n"
            )
            f.write(
                f"- **Best P50 latency:** {best_lat['latency_p50_ms']} ms at concurrency={best_lat['concurrency']}\n"
            )
            f.write(f"- **Successful runs:** {len(successful)}/{len(rows)}\n")


def write_plots(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        import matplotlib.ticker

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.", file=sys.stderr)
        return

    successful = [r for r in rows if r["status"] == "ok"]
    if not successful:
        print("No successful runs to plot.", file=sys.stderr)
        return

    x = [r["concurrency"] for r in successful]
    tps = [r["tps"] for r in successful]
    ttft_p99 = [r["ttft_p99_ms"] for r in successful]
    tpot_p50 = [r["tpot_p50_ms"] for r in successful]

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle("Concurrency Sweep Performance", fontsize=14, fontweight="bold")

    metrics = [
        (axes[0], tps, "TPS", "Tokens / s", "tab:blue"),
        (axes[1], ttft_p99, "TTFT P99 (ms)", "Latency (ms)", "tab:orange"),
        (axes[2], tpot_p50, "TPOT P50 (ms)", "Latency (ms)", "tab:green"),
    ]
    for ax, values, label, ylabel, color in metrics:
        ax.plot(x, values, marker="o", linewidth=2, color=color, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: str(int(v)))
        )
        ax.legend(loc="upper left")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Concurrency")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize metrics from a concurrency sweep results folder."
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Path to the concurrency sweep directory (contains concurrency_N/ subdirs).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to stdout only; do not write CSV, Markdown, or PNG files.",
    )
    args = parser.parse_args()

    sweep_dir: Path = args.sweep_dir
    if not sweep_dir.is_dir():
        print(f"Error: {sweep_dir} is not a directory", file=sys.stderr)
        return 1

    rows = collect_results(sweep_dir)
    if not rows:
        print(f"No concurrency_* subdirectories found in {sweep_dir}", file=sys.stderr)
        return 1

    print_table(rows)

    if not args.no_save:
        csv_path = sweep_dir / "metrics_summary.csv"
        md_path = sweep_dir / "metrics_summary.md"
        png_path = sweep_dir / "metrics_summary.png"
        write_csv(rows, csv_path)
        write_markdown(rows, md_path)
        write_plots(rows, png_path)
        print(f"\nSaved: {csv_path}")
        print(f"Saved: {md_path}")
        print(f"Saved: {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
