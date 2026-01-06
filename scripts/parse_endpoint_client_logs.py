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
Parse timing data to report inference-endpoint http-client overhead.

Usage:
    # Option 1: Read from timing JSONL files (when --report-dir was specified during benchmark)
    # Timing files are at: {report-dir}/endpoint_client/timing_worker_*.jsonl
    inference-endpoint benchmark offline ... --report-dir outputs/
    python scripts/parse_endpoint_client_logs.py --report-dir outputs/

    # Option 2: Pipe verbose log output (requires -vvv for TRACE level timing)
    # Use this when --report-dir was NOT specified during benchmark
    inference-endpoint benchmark offline ... -vvv 2>&1 | python scripts/parse_endpoint_client_logs.py
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

    name: str  # Section header with description (e.g., "PRE-SEND (recv → http_send)")
    metrics: set[str]  # Breakdown metric names (shown indented)
    total_key: str | None  # Metric for section total (shown as header row), or None


# Metric categories for display (ordered)
# total_key: shown as the section header row (recv to end of section)
# metrics: breakdown components shown indented below
SECTIONS = [
    MetricSection(
        "PRE-SEND (recv → http_payload_send)",
        {
            "recv_to_bytes",
            "bytes_to_http_payload",
            "tcp_conn_pool",
            "http_payload_send",
        },
        "pre_overhead",  # Total: t_http - t_recv
    ),
    MetricSection(
        "IN-FLIGHT (http_payload_send → response)",
        {
            "task_overhead",
            "http_to_headers",
            "headers_to_first_chunk",
            "first_to_last_chunk",
        },
        "in_flight_time",  # Total: t_response - t_http
    ),
    MetricSection(
        "POST-RECV (response → query_result_sent)",
        {"query_result_sent"},
        "post_overhead",  # Total: t_zmq_sent - t_response
    ),
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


class TableFormatter:
    """
    Dynamic table formatter that adjusts column widths based on content.

    Collects all rows first, then formats with consistent column widths.
    """

    STAT_COLS = ("N", "Avg", "p50", "p99", "p99.9", "Max")
    MIN_STAT_WIDTH = 10  # Minimum width for stat columns

    def __init__(self, label_header: str = "Metric"):
        self.rows: list[tuple[str, int, tuple[float, ...]]] = []  # (label, n, stats)
        self.label_header = label_header

    def add_row(self, label: str, values: list[float]) -> None:
        """Add a row to the table."""
        if not values:
            return
        stats = percentiles(values)
        self.rows.append((label, len(values), stats))

    def _fmt_num(self, val: float) -> str:
        """Format a number, using compact notation for large values."""
        if abs(val) >= 1_000_000:
            return f"{val:.2e}"
        elif abs(val) >= 1000:
            return f"{val:.2f}"
        else:
            return f"{val:.4f}"

    def _compute_widths(self) -> tuple[int, list[int]]:
        """Compute column widths based on content."""
        if not self.rows:
            return len(self.label_header), [self.MIN_STAT_WIDTH] * 6

        # Label column width
        label_width = max(len(self.label_header), max(len(r[0]) for r in self.rows))

        # Stat column widths (N + 5 percentile columns)
        stat_widths = [len(h) for h in self.STAT_COLS]  # Start with header widths

        for _, n, stats in self.rows:
            # N column
            stat_widths[0] = max(stat_widths[0], len(str(n)))
            # Percentile columns
            for i, val in enumerate(stats):
                stat_widths[i + 1] = max(stat_widths[i + 1], len(self._fmt_num(val)))

        # Apply minimum widths
        stat_widths = [max(w, self.MIN_STAT_WIDTH) for w in stat_widths]

        return label_width, stat_widths

    def format(self) -> list[str]:
        """Format all rows with consistent column widths."""
        if not self.rows:
            return []

        label_width, stat_widths = self._compute_widths()
        total_width = label_width + 2 + sum(w + 1 for w in stat_widths) + 2

        lines = []

        # Header
        lines.append("=" * total_width)
        header_parts = [f"  {self.label_header:<{label_width}}"]
        for col, w in zip(self.STAT_COLS, stat_widths, strict=False):
            header_parts.append(f"{col:>{w}}")
        lines.append(" ".join(header_parts))
        lines.append("=" * total_width)

        return lines

    def format_row(self, label: str, n: int, stats: tuple[float, ...]) -> str:
        """Format a single data row."""
        label_width, stat_widths = self._compute_widths()

        parts = [f"  {label:<{label_width}}"]
        parts.append(f"{n:>{stat_widths[0]}}")
        for val, w in zip(stats, stat_widths[1:], strict=False):
            parts.append(f"{self._fmt_num(val):>{w}}")
        return " ".join(parts)

    def format_all(self) -> list[str]:
        """Format header and all data rows."""
        lines = self.format()
        for label, n, stats in self.rows:
            lines.append(self.format_row(label, n, stats))
        return lines

    def get_separator(self, char: str = "─") -> str:
        """Get a separator line of the correct width."""
        label_width, stat_widths = self._compute_widths()
        total_width = label_width + 2 + sum(w + 1 for w in stat_widths) + 2
        return char * total_width


# Legacy function for simple cases
def fmt_row(label: str, values: list[float], width: int = 30) -> str:
    """Format a stats row (legacy, use TableFormatter for dynamic widths)."""
    n = len(values)
    if n == 0:
        return ""
    avg, p50, p99, p999, mx = percentiles(values)

    def fmt(v: float) -> str:
        if abs(v) >= 1_000_000:
            return f"{v:>12.2e}"
        elif abs(v) >= 1000:
            return f"{v:>12.2f}"
        else:
            return f"{v:>12.4f}"

    return (
        f"  {label:<{width}} {n:>8}{fmt(avg)}{fmt(p50)}{fmt(p99)}{fmt(p999)}{fmt(mx)}"
    )


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


def load_timing_jsonl(report_dir: Path) -> Stats:
    """
    Load timing data from worker JSONL files.

    Reads timing_worker_*.jsonl files from report_dir.
    Each line is a JSON object with: query_id, worker_id, phase, metrics.

    Returns a Stats object populated with the timing data.
    """
    stats = Stats()
    stats.report_dir = report_dir

    timing_files = list(report_dir.glob("timing_worker_*.jsonl"))
    if not timing_files:
        return stats

    for timing_file in timing_files:
        with open(timing_file) as f:
            for line in f:
                if not (line := line.strip()):
                    continue
                try:
                    entry = json.loads(line)
                    query_id = entry.get("query_id", "")
                    phase = entry.get("phase", "")
                    metrics = entry.get("metrics", {})
                    stats.add(phase, query_id, metrics)
                except json.JSONDecodeError:
                    continue

    return stats


def print_live(stats: Stats) -> None:
    """Print live stats (clears screen)."""
    pre, post, count = stats.pre, stats.post, stats.count
    sys.stdout.write("\033[2J\033[H")

    if count == 0:
        print("=" * 60)
        print(f"  WORKER TIMING  |  Requests: {count}")
        print("=" * 60)
        print("\n  Waiting for data...")
        sys.stdout.flush()
        return

    # Build table structure: collect all rows first for dynamic column sizing
    # Structure: list of (section_name, total_values, [(metric_name, values), ...])
    table_data: list[tuple[str, list[float] | None, list[tuple[str, list[float]]]]] = []

    for section in SECTIONS:
        data = pre if "PRE-SEND" in section.name else post

        total_values = data.get(section.total_key, []) if section.total_key else None
        breakdown = [
            (m, data.get(m, []))
            for m in sorted(section.metrics)  # Sort for consistent ordering
            if m != section.total_key and data.get(m)
        ]

        if total_values or breakdown:
            table_data.append(
                (section.name, total_values if total_values else None, breakdown)
            )

    # Add end-to-end if available
    e2e = post.get("end_to_end", [])

    # Create formatter and add all rows to compute widths
    fmt = TableFormatter()
    for section_name, total_values, breakdown in table_data:
        if total_values:
            fmt.add_row(section_name, total_values)
        for metric_name, values in breakdown:
            fmt.add_row(f"  {metric_name}", values)
    if e2e:
        fmt.add_row("END-TO-END TOTAL", e2e)

    # Print title
    sep = fmt.get_separator("=")
    print(sep)
    print(f"  WORKER TIMING  |  Requests: {count}")

    # Print header
    for line in fmt.format():
        print(line)

    # Print sections
    for section_name, total_values, breakdown in table_data:
        print()
        if total_values:
            stats_tuple = percentiles(total_values)
            print(fmt.format_row(section_name, len(total_values), stats_tuple))
        else:
            print(f"  [{section_name}]")

        for metric_name, values in breakdown:
            stats_tuple = percentiles(values)
            print(fmt.format_row(f"  {metric_name}", len(values), stats_tuple))

    # Print end-to-end total
    if e2e:
        print()
        print(fmt.get_separator("─"))
        stats_tuple = percentiles(e2e)
        print(fmt.format_row("END-TO-END TOTAL", len(e2e), stats_tuple))

    print(sep)
    sys.stdout.flush()


def print_analysis(stats: Stats, events: dict[str, dict[str, int]]) -> None:
    """Print overhead breakdown with p99.9 and max for worst-case analysis."""
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

    def fmt_num(v: float) -> str:
        """Format number, using compact notation for large values."""
        if abs(v) >= 1_000_000:
            return f"{v:.2e}"
        elif abs(v) >= 1000:
            return f"{v:.2f}"
        else:
            return f"{v:.4f}"

    def row(label: str, vals: list[float], pct_base: float = 0, indent: int = 0) -> str:
        prefix = "  " + "  " * indent
        if not vals:
            return f"{prefix}{label:<{32 - indent * 2}} {'N/A':>12}"
        avg, p50, p99, p999, mx = percentiles(vals)
        pct = f"{(avg / pct_base) * 100:>6.2f}%" if pct_base else ""
        return (
            f"{prefix}{label:<{32 - indent * 2}} "
            f"{fmt_num(avg):>12} {fmt_num(p50):>12} {fmt_num(p99):>12} "
            f"{fmt_num(p999):>12} {fmt_num(mx):>12} {pct}"
        )

    e2e_avg = statistics.mean(e2e) if e2e else 0
    width = 120

    print("\n" + "=" * width)
    print(f"  OVERHEAD BREAKDOWN  |  Samples: {len(latency) or stats.count}")
    print("=" * width)
    print(
        f"  {'Metric':<32} {'Avg(ms)':>12} {'p50(ms)':>12} {'p99(ms)':>12} "
        f"{'p99.9(ms)':>12} {'Max(ms)':>12} {'%':>7}"
    )
    print("─" * width)

    print("  [LOAD GENERATOR]")
    print(row("TTFT", ttft, indent=1))
    print(row("Latency", latency, indent=1))

    print("\n  [WORKER]")
    print(row("Pre-Overhead", pre.get("pre_overhead", []), e2e_avg, indent=1))

    # Pre-overhead breakdown
    pre_breakdown = [
        ("recv_to_bytes", pre.get("recv_to_bytes", [])),
        ("bytes_to_http_payload", pre.get("bytes_to_http_payload", [])),
        ("tcp_conn_pool", pre.get("tcp_conn_pool", [])),
        ("http_payload_send", pre.get("http_payload_send", [])),
    ]
    for name, vals in pre_breakdown:
        if vals:
            print(row(name, vals, e2e_avg, indent=2))

    print(row("Server Time", server, e2e_avg, indent=1))

    # Server time breakdown (in-flight: http_send -> response)
    server_breakdown = [
        ("task_overhead", post.get("task_overhead", [])),  # Task creation -> task wake
        ("http_to_headers", post.get("http_to_headers", [])),
        ("headers_to_first", post.get("headers_to_first", [])),
        ("first_to_last", post.get("first_to_last", [])),
    ]
    for name, vals in server_breakdown:
        if vals:
            print(row(name, vals, e2e_avg, indent=2))

    print(row("Post-Overhead", post.get("post_overhead", []), e2e_avg, indent=1))

    # Post-overhead breakdown (response -> zmq_sent)
    post_breakdown = [
        ("query_result_sent", post.get("query_result_sent", [])),
    ]
    for name, vals in post_breakdown:
        if vals:
            print(row(name, vals, e2e_avg, indent=2))

    print("─" * width)
    print(row("E2E", e2e, indent=1))

    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Report directory (timing files in endpoint_client/, events in events.json)",
    )
    args = parser.parse_args()

    # Timing files are in {report-dir}/endpoint_client/
    timing_dir = args.report_dir / "endpoint_client" if args.report_dir else None

    # Check if stdin is piped (not a TTY)
    stdin_is_tty = os.isatty(sys.stdin.fileno())

    if stdin_is_tty:
        # No stdin input - read from timing JSONL files
        if not timing_dir:
            print(__doc__)
            return

        stats = load_timing_jsonl(timing_dir)
        if stats.count == 0:
            print(f"No timing data found in {timing_dir}")
            print("Looking for: timing_worker_*.jsonl files")
            return

        with stats.lock:
            print_live(stats)

        # Load events.json from report_dir for IPC calculations
        events = load_events(args.report_dir) if args.report_dir else {}
        print_analysis(stats, events)
        return

    # Legacy mode: parse timing from verbose log output on stdin
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
        print("\nNo timing data found in stdin")
        print(
            "Hint: Use -vvv flag with inference-endpoint to enable TRACE-level timing logs"
        )
        return

    with stats.lock:
        print_live(stats)

    events = load_events(stats.report_dir) if stats.report_dir else {}
    print_analysis(stats, events)


if __name__ == "__main__":
    main()
