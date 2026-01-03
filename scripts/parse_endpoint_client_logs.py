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
Parse worker timing logs and display rolling statistics.

Works with both worker.py (PreparedRequest) and worker_2.py (session.post).
Both emit timing_pre: and timing_post: log lines with identical metric names.

Usage:
    tail -f worker.log | python scripts/parse_endpoint_client_logs.py
"""

from __future__ import annotations

import os
import re
import statistics
import sys
import threading
from collections import defaultdict

# Regex patterns
TIMING_RE = re.compile(r"\[([^\]]+)\]\s+timing_(pre|post):\s+(.+)")
METRIC_RE = re.compile(r"d_(\w+)=([\d.]+)us")

# Metric categorization
PRE_METRICS = {"recv_to_prepare", "prepare_to_http", "pre_overhead"}
INFLIGHT_METRICS = {"http_to_headers", "headers_to_first", "first_to_last"}
POST_METRICS = {"response_to_zmq", "post_overhead"}
SUMMARY_METRICS = {"end_to_end"}

# Display options
REFRESH_INTERVAL = 0.33
WIDTH = 75
HEADER = f"{'Metric':<20} {'N':>8} {'Avg':>10} {'p50':>10} {'p99':>10} {'Max':>10}"


class Stats:
    """Thread-safe stats accumulator."""

    def __init__(self):
        self.pre: dict[str, list[float]] = defaultdict(list)
        self.post: dict[str, list[float]] = defaultdict(list)
        self.request_count = 0
        self.lock = threading.Lock()

    def add(self, phase: str, metrics: dict[str, float]) -> None:
        with self.lock:
            target = self.pre if phase == "pre" else self.post
            for name, value in metrics.items():
                target[name].append(value)
            if phase == "pre":
                self.request_count += 1

    def snapshot(self) -> tuple[dict, dict, int]:
        with self.lock:
            return dict(self.pre), dict(self.post), self.request_count


def parse_line(line: str) -> tuple[str, dict[str, float]] | None:
    """Parse timing line. Returns (phase, metrics) or None."""
    match = TIMING_RE.search(line)
    if not match:
        return None

    phase = match.group(2)
    metrics = {
        m.group(1): float(m.group(2)) for m in METRIC_RE.finditer(match.group(3))
    }
    return phase, metrics


def format_row(label: str, values: list[float]) -> str:
    """Format a stats row with N, Avg, p50, p99, Max (values in ms)."""
    n = len(values)
    avg = statistics.mean(values) / 1000.0  # us -> ms
    if n >= 2:
        sorted_vals = sorted(values)
        p50 = statistics.median(values) / 1000.0
        p99 = sorted_vals[int(n * 0.99)] / 1000.0
        max_val = sorted_vals[-1] / 1000.0
        return f"  {label:<18} {n:>8} {avg:>10.3f} {p50:>10.3f} {p99:>10.3f} {max_val:>10.3f}"
    return f"  {label:<18} {n:>8} {avg:>10.3f} {'-':>10} {'-':>10} {'-':>10}"


def print_stats(pre: dict, post: dict, count: int) -> None:
    """Print stats to terminal."""
    sys.stdout.write("\033[2J\033[H")  # Clear screen

    print(f"{'=' * WIDTH}")
    print(f"  WORKER TIMING STATS  |  Requests: {count}")
    print(f"{'=' * WIDTH}")
    print(HEADER)
    print(f"{'=' * WIDTH}")

    def section(
        title: str, data: dict, metrics: set, total_key: str | None = None
    ) -> None:
        print(f"\n  [{title}]")

        has_data = False
        for name in metrics:
            if name in data and data[name] and name != total_key:
                print(format_row(name, data[name]))
                has_data = True

        if not has_data:
            print("    (no data)")
            return

        if total_key and total_key in data and data[total_key]:
            print(format_row("  → subtotal", data[total_key]))

    # PRE-SEND: client overhead before HTTP request leaves
    section("PRE-SEND", pre, PRE_METRICS, "pre_overhead")

    # IN-FLIGHT: time receiving HTTP response from server
    section("IN-FLIGHT", post, INFLIGHT_METRICS)

    # POST-RECV: client overhead after response received
    section("POST-RECV", post, POST_METRICS, "post_overhead")

    # SUMMARY: end-to-end
    if "end_to_end" in post and post["end_to_end"]:
        print(f"\n{'─' * WIDTH}")
        print(format_row("END-TO-END TOTAL", post["end_to_end"]))

    print(f"{'=' * WIDTH}")
    sys.stdout.flush()


def display_loop(stats: Stats, stop: threading.Event) -> None:
    """Background thread: refresh display every REFRESH_INTERVAL."""
    while not stop.wait(REFRESH_INTERVAL):
        pre, post, count = stats.snapshot()
        if count > 0:
            print_stats(pre, post, count)


def main() -> None:
    stats = Stats()
    stop = threading.Event()

    # Start display thread
    display_thread = threading.Thread(
        target=display_loop, args=(stats, stop), daemon=True
    )
    display_thread.start()

    print("Waiting for timing data on stdin...")

    try:
        for line in sys.stdin:
            result = parse_line(line)
            if result:
                phase, metrics = result
                stats.add(phase, metrics)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()

    # Final stats
    pre, post, count = stats.snapshot()
    if count > 0:
        print("\n\nFinal Statistics:")
        print_stats(pre, post, count)


if __name__ == "__main__":
    if os.isatty(sys.stdin.fileno()):
        print(__doc__)
        sys.exit(0)
    main()
