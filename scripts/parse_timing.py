#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Parse worker timing logs and print statistics + histograms.

Usage:
    # Batch mode (default): process all input, then print stats
    cat logfile.txt | python scripts/parse_timing.py
    python scripts/parse_timing.py < logfile.txt
    grep "timing:" logfile.txt | python scripts/parse_timing.py

    # Live mode: stream input, print rolling stats every N lines
    tail -f logfile.txt | python scripts/parse_timing.py --live
    tail -f logfile.txt | python scripts/parse_timing.py --live --interval 50
"""

import argparse
import re
import sys
from dataclasses import dataclass, field


@dataclass
class Stats:
    """Statistics for a timing metric."""

    name: str
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0

    @property
    def avg(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0

    @property
    def median(self) -> float:
        if not self.values:
            return 0
        sorted_vals = sorted(self.values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        return sorted_vals[n // 2]

    def percentile(self, p: float) -> float:
        """Get percentile value (p in 0-100)."""
        if not self.values:
            return 0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def std_dev(self) -> float:
        if len(self.values) < 2:
            return 0
        avg = self.avg
        variance = sum((x - avg) ** 2 for x in self.values) / len(self.values)
        return variance**0.5

    def histogram(self, bins: int = 10, width: int = 40) -> str:
        """Generate ASCII histogram."""
        if not self.values:
            return "  No data"

        min_val, max_val = self.min, self.max
        if min_val == max_val:
            return f"  All values = {min_val:.3f} ms"

        bin_width = (max_val - min_val) / bins
        counts = [0] * bins

        for val in self.values:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
            counts[bin_idx] += 1

        max_count = max(counts)
        lines = []

        for i, count in enumerate(counts):
            low = min_val + i * bin_width
            high = min_val + (i + 1) * bin_width
            bar_len = int(count / max_count * width) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(f"  [{low:8.3f}, {high:8.3f}) |{bar} {count}")

        return "\n".join(lines)

    def print_stats(self) -> None:
        """Print full statistics."""
        print(f"\n{'='*60}")
        print(f"{self.name}")
        print(f"{'='*60}")
        print(f"  Count:   {self.count}")
        print(f"  Min:     {self.min:.3f} ms")
        print(f"  Max:     {self.max:.3f} ms")
        print(f"  Avg:     {self.avg:.3f} ms")
        print(f"  Median:  {self.median:.3f} ms")
        print(f"  Std Dev: {self.std_dev:.3f} ms")
        print()
        print("  Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
            print(f"    p{p:5}: {self.percentile(p):.3f} ms")
        print()
        print("  Histogram:")
        print(self.histogram())


# All known timing metrics with descriptions
KNOWN_METRICS = {
    # Request path breakdown
    "recv_to_prepare": "d_recv_to_prepare (ZMQ recv → prep done)",
    "prepare_to_http": "d_prepare_to_http (prep done → HTTP send)",
    # Response path breakdown
    "http_to_headers": "d_http_to_headers (HTTP send → headers/TTFB)",
    "headers_to_first": "d_headers_to_first (headers → 1st chunk) [stream]",
    "first_to_last": "d_first_to_last (1st → last chunk) [stream]",
    "response_to_zmq": "d_response_to_zmq (response → ZMQ sent)",
    # Aggregate overhead
    "pre_overhead": "d_pre_overhead (ZMQ recv → HTTP send)",
    "post_overhead": "d_post_overhead (response → ZMQ sent)",
    # Full lifecycle
    "end_to_end": "d_end_to_end (ZMQ recv → ZMQ sent)",
}


def parse_timing_line(line: str) -> dict[str, float] | None:
    """Parse a timing log line and extract metrics.

    Converts microseconds from log to milliseconds for display.
    """
    # Pattern: d_recv_to_prepare=13.3us, d_prepare_to_http=45.2us, ...
    pattern = r"d_(\w+)=([\d.]+)us"
    matches = re.findall(pattern, line)

    if not matches:
        return None

    # Convert us to ms (divide by 1000)
    return {name: float(value) / 1000.0 for name, value in matches}


def create_stats_collectors() -> dict[str, Stats]:
    """Create stats collectors for all known metrics."""
    return {key: Stats(desc) for key, desc in KNOWN_METRICS.items()}


def print_summary(stats: dict[str, Stats], lines_parsed: int) -> None:
    """Print summary table of all metrics with data."""
    print(f"\n{'='*60}")
    print(f"Summary ({lines_parsed} entries, all values in milliseconds)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Min':>10} {'Avg':>10} {'p50':>10} {'p99':>10} {'Max':>10}")
    print("-" * 75)
    for stat in stats.values():
        if stat.count > 0:
            name = stat.name.split()[0]
            print(
                f"{name:<25} {stat.min:>10.3f} {stat.avg:>10.3f} "
                f"{stat.median:>10.3f} {stat.percentile(99):>10.3f} {stat.max:>10.3f}"
            )


def print_live_update(stats: dict[str, Stats], lines_parsed: int) -> None:
    """Print compact live update (single line per metric)."""
    # Clear screen and move cursor to top (ANSI escape codes)
    print("\033[2J\033[H", end="")
    print(f"=== Live Stats ({lines_parsed} entries, values in ms) ===\n")
    print(
        f"{'Metric':<25} {'Count':>8} {'Avg':>10} {'p50':>10} {'p99':>10} {'Max':>10}"
    )
    print("-" * 75)
    for stat in stats.values():
        if stat.count > 0:
            name = stat.name.split()[0]
            print(
                f"{name:<25} {stat.count:>8} {stat.avg:>10.3f} "
                f"{stat.median:>10.3f} {stat.percentile(99):>10.3f} {stat.max:>10.3f}"
            )
    print("\n(Ctrl+C to stop and print full stats)")
    sys.stdout.flush()


def run_batch_mode(stats: dict[str, Stats]) -> int:
    """Run in batch mode: read all input, then print full stats."""
    lines_parsed = 0
    for line in sys.stdin:
        metrics = parse_timing_line(line)
        if metrics:
            lines_parsed += 1
            for key, value in metrics.items():
                if key in stats:
                    stats[key].add(value)

    return lines_parsed


def run_live_mode(stats: dict[str, Stats], interval: int) -> int:
    """Run in live mode: print rolling stats every N lines."""
    lines_parsed = 0
    try:
        for line in sys.stdin:
            metrics = parse_timing_line(line)
            if metrics:
                lines_parsed += 1
                for key, value in metrics.items():
                    if key in stats:
                        stats[key].add(value)

                # Update display every interval lines
                if lines_parsed % interval == 0:
                    print_live_update(stats, lines_parsed)

    except KeyboardInterrupt:
        pass  # Ctrl+C to stop and print final stats

    return lines_parsed


def main():
    parser = argparse.ArgumentParser(
        description="Parse worker timing logs and print statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Batch mode (process all, then print)
    cat logfile.txt | python scripts/parse_timing.py
    grep "timing:" logfile.txt | python scripts/parse_timing.py

    # Live mode (rolling updates)
    tail -f logfile.txt | python scripts/parse_timing.py --live
    tail -f logfile.txt | python scripts/parse_timing.py --live --interval 50
        """,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live mode: stream input and show rolling stats",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Lines between live updates (default: 10)",
    )
    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Skip histograms in final output (faster for large datasets)",
    )
    args = parser.parse_args()

    # Initialize stats collectors
    stats = create_stats_collectors()

    # Run appropriate mode
    if args.live:
        lines_parsed = run_live_mode(stats, args.interval)
        print("\n" + "=" * 60)
        print("Final Statistics (Ctrl+C pressed)")
    else:
        lines_parsed = run_batch_mode(stats)

    if lines_parsed == 0:
        print("No timing data found in input.")
        print("Expected format: d_recv_to_prepare=X.Xus, d_prepare_to_http=X.Xus, ...")
        sys.exit(1)

    print(f"\nParsed {lines_parsed} timing entries\n")

    # Print stats for each metric
    for stat in stats.values():
        if stat.count > 0:
            stat.print_stats() if not args.no_histograms else None

    # Print summary table
    print_summary(stats, lines_parsed)


if __name__ == "__main__":
    main()
