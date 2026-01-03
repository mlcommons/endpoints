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
Parse verbose outputs to report inference-endpoint http-client overhead.

Usage:
    inference-endpoint -vvv benchmark offline ... --report-dir output/ 2>&1 | python scripts/parse_endpoint_client_logs.py --report-dir output/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

# Regex patterns
TIMING_RE = re.compile(r"\[([^\]]+)\]\s+timing_(pre|post):\s+(.+)")
METRIC_RE = re.compile(r"d_(\w+)=([\d.]+)ms")
TIMESTAMP_RE = re.compile(r"(t_\w+)=(\d+)")
REPORT_DIR_RE = re.compile(r"Saved:\s+(.+?/reports_[^/]+)")


class MetricSection(NamedTuple):
    """A section of metrics to display in the timing report."""

    name: str  # Section header (e.g., "PRE-SEND")
    metrics: set[str]  # Metric names in this section
    subtotal_key: str | None  # Metric to use as subtotal, or None


# Metric categories for display (ordered)
SECTIONS = [
    MetricSection(
        "PRE-SEND",
        {"recv_to_prepare", "pool_acquire", "http_send", "pre_overhead"},
        "pre_overhead",
    ),
    MetricSection(
        "IN-FLIGHT",
        {"task_overhead", "http_to_headers", "headers_to_first", "first_to_last"},
        None,
    ),
    MetricSection("POST-RECV", {"response_to_zmq", "post_overhead"}, "post_overhead"),
]


def percentiles(values: list[float]) -> tuple[float, float, float, float, float]:
    """Return (avg, p50, p99, p999, max) for a list of values."""
    if not values:
        return (0, 0, 0, 0, 0)
    s = sorted(values)
    n = len(s)
    return (
        statistics.mean(values),
        statistics.median(values),
        s[int(n * 0.99)] if n >= 2 else s[0],
        s[min(int(n * 0.999), n - 1)] if n >= 2 else s[0],
        s[-1],
    )


def fmt_row(label: str, values: list[float], width: int = 18) -> str:
    """Format a stats row."""
    n = len(values)
    if n == 0:
        return ""
    avg, p50, p99, p999, mx = percentiles(values)
    return f"  {label:<{width}} {n:>8} {avg:>12.4f} {p50:>12.4f} {p99:>12.4f} {p999:>12.4f} {mx:>12.4f}"


class Stats:
    """Thread-safe stats accumulator."""

    def __init__(self):
        self.pre: dict[str, list[float]] = defaultdict(list)
        self.post: dict[str, list[float]] = defaultdict(list)
        self.timestamps: dict[
            str, dict[str, float]
        ] = {}  # query_id -> {t_recv, t_zmq_sent}
        self.count = 0
        self.lock = threading.Lock()
        self.report_dir: Path | None = None

    def add(self, phase: str, query_id: str, metrics: dict[str, float]) -> None:
        with self.lock:
            target = self.pre if phase == "pre" else self.post
            for name, value in metrics.items():
                if name.startswith("t_"):
                    self.timestamps.setdefault(query_id, {})[name] = value
                else:
                    target[name].append(value)
            if phase == "pre":
                self.count += 1


def parse_line(line: str) -> tuple[str, str, dict[str, float]] | None:
    """Parse timing line -> (query_id, phase, metrics) or None."""
    m = TIMING_RE.search(line)
    if not m:
        return None
    metrics = {m.group(1): float(m.group(2)) for m in METRIC_RE.finditer(m.group(3))}
    metrics.update(
        {m.group(1): float(m.group(2)) for m in TIMESTAMP_RE.finditer(m.group(3))}
    )
    return m.group(1), m.group(2), metrics


def load_events(report_dir: Path) -> dict[str, dict[str, int]]:
    """Load events.json -> {sample_uuid: {event_type: timestamp_ns}}."""
    events_file = report_dir / "events.json"
    if not events_file.exists():
        return {}
    samples: dict[str, dict[str, int]] = {}
    with open(events_file) as f:
        for line in f:
            if not (line := line.strip()):
                continue
            e = json.loads(line)
            if (uuid := e.get("sample_uuid")) and (
                ts := e.get("timestamp_ns")
            ) is not None:
                samples.setdefault(uuid, {})[e.get("event_type")] = ts
    return samples


def print_live(stats: Stats) -> None:
    """Print live stats (clears screen)."""
    pre, post, count = stats.pre, stats.post, stats.count
    sys.stdout.write("\033[2J\033[H")
    print("=" * 102)
    print(f"  WORKER TIMING  |  Requests: {count}")
    print("=" * 102)

    if count == 0:
        print("\n  Waiting for data...")
        sys.stdout.flush()
        return

    header = f"  {'Metric':<18} {'N':>8} {'Avg':>12} {'p50':>12} {'p99':>12} {'p99.9':>12} {'Max':>12}"
    print(header)
    print("=" * 102)

    for section in SECTIONS:
        data = pre if section.name == "PRE-SEND" else post
        rows = [
            fmt_row(m, data.get(m, []))
            for m in section.metrics
            if m != section.subtotal_key and data.get(m)
        ]
        if rows:
            print(f"\n  [{section.name}]")
            for r in rows:
                print(r)
            if section.subtotal_key and data.get(section.subtotal_key):
                print(fmt_row("→ subtotal", data[section.subtotal_key]))

    if e2e := post.get("end_to_end"):
        print(f"\n{'─' * 102}")
        print(fmt_row("END-TO-END TOTAL", e2e))

    print("=" * 102)
    sys.stdout.flush()


def print_analysis(stats: Stats, events: dict[str, dict[str, int]]) -> None:
    """Print overhead breakdown."""
    pre, post = stats.pre, stats.post

    # Calculate metrics from events
    ttft = [
        (e["first_chunk_received"] - e["loadgen_issue_called"]) / 1e6
        for e in events.values()
        if "first_chunk_received" in e and "loadgen_issue_called" in e
    ]
    latency = [
        (e["complete"] - e["loadgen_issue_called"]) / 1e6
        for e in events.values()
        if "complete" in e and "loadgen_issue_called" in e
    ]

    # IPC delays (all timestamps are monotonic_ns)
    ipc_send, ipc_recv = [], []
    for qid, ts in stats.timestamps.items():
        if (
            qid in events
            and (t_recv := ts.get("t_recv"))
            and (t_sent := ts.get("t_zmq_sent"))
        ):
            e = events[qid]
            if main_send := e.get("loadgen_issue_called"):
                ipc_send.append((t_recv - main_send) / 1e6)
            if main_recv := e.get("complete"):
                ipc_recv.append((main_recv - t_sent) / 1e6)

    e2e = post.get("end_to_end", [])
    server = (
        [
            sum(x)
            for x in zip(
                post.get("http_to_headers", []),
                post.get("headers_to_first", []),
                post.get("first_to_last", []),
                strict=False,
            )
        ]
        if all(
            k in post for k in ("http_to_headers", "headers_to_first", "first_to_last")
        )
        else []
    )

    def row(label: str, vals: list[float], pct_base: float = 0) -> str:
        if not vals:
            return f"  {label:<32} {'N/A':>12}"
        avg, p50, p99, p999, mx = percentiles(vals)
        pct = f"{(avg / pct_base) * 100:>7.3f}%" if pct_base else ""
        return f"  {label:<32} {avg:>12.4f} {p50:>12.4f} {p99:>12.4f} {pct}"

    e2e_avg = statistics.mean(e2e) if e2e else 0

    print("\n" + "=" * 102)
    print(f"  OVERHEAD BREAKDOWN  |  Samples: {len(latency) or stats.count}")
    print("=" * 102)
    print(f"  {'Metric':<32} {'Avg(ms)':>12} {'p50(ms)':>12} {'p99(ms)':>12} {'%':>8}")
    print("─" * 102)

    print("  [LOAD GENERATOR]")
    print(row("  TTFT", ttft))
    print(row("  Latency", latency))

    print("\n  [WORKER]")
    print(row("  Pre-Overhead", pre.get("pre_overhead", []), e2e_avg))
    print(row("  Server Time", server, e2e_avg))
    print(row("  Post-Overhead", post.get("post_overhead", []), e2e_avg))
    print(row("  E2E", e2e))

    if ipc_send or ipc_recv:
        print("\n  [IPC]")
        print(row("  Send (main→worker)", ipc_send))
        print(row("  Recv (worker→main)", ipc_recv))
        if ipc_send and ipc_recv:
            total = [s + r for s, r in zip(ipc_send, ipc_recv, strict=False)]
            print(row("  Total", total))

    print("=" * 102)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, help="Report directory")
    args = parser.parse_args()

    if os.isatty(sys.stdin.fileno()):
        print(__doc__)
        return

    stats = Stats()
    stats.report_dir = args.report_dir
    stop = threading.Event()

    # Live display thread
    def display_loop():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        while not stop.wait(1.0):
            with stats.lock:
                print_live(stats)

    threading.Thread(target=display_loop, daemon=True).start()

    # Parse stdin
    try:
        for line in sys.stdin:
            if result := parse_line(line):
                stats.add(result[1], result[0], result[2])
            if not stats.report_dir and (m := REPORT_DIR_RE.search(line)):
                stats.report_dir = Path(m.group(1))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()

    # Final output
    if stats.count == 0:
        print("\nNo timing data found")
        return

    with stats.lock:
        print_live(stats)

    events = load_events(stats.report_dir) if stats.report_dir else {}
    print_analysis(stats, events)


if __name__ == "__main__":
    main()
