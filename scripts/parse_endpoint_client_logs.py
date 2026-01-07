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
import logging
import os
import re
import statistics
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeAlias

# Type aliases for complex types (using TypeAlias for Python 3.10 compatibility)
EventsMap: TypeAlias = dict[str, dict[str, int]]  # noqa: UP040
MetricsDict: TypeAlias = dict[str, float]  # noqa: UP040
TimestampsDict: TypeAlias = dict[str, dict[str, float]]  # noqa: UP040

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Constants and Regex Patterns
# =============================================================================

# Regex for parsing timing log lines from verbose output
TIMING_RE = re.compile(r"\[([^\]]+)\]\s+timing_(pre|post):\s+(.+)")
# Regex for extracting duration metrics (d_*=Xms)
METRIC_RE = re.compile(r"d_(\w+)=([\d.]+)ms")
# Regex for extracting timestamp values (t_*=X)
TIMESTAMP_RE = re.compile(r"(t_\w+)=(\d+)")
# Regex for extracting report directory from log output
REPORT_DIR_RE = re.compile(r"Saved:\s+(.+?/reports_[^/]+)")


class SectionPhase(Enum):
    """Phase indicator for metric sections."""

    PRE = "pre"
    POST = "post"


@dataclass(slots=True, frozen=True)
class MetricSection:
    """A section of metrics to display in the timing report."""

    name: str  # Section header with description
    metrics: frozenset[str]  # Breakdown metric names (shown indented)
    phase: SectionPhase  # Which stats dict to use (pre or post)
    total_key: str | None = None  # Metric for section total, or None


# Metric categories for display (ordered)
# total_key: shown as the section header row (recv to end of section)
# metrics: breakdown components shown indented below
SECTIONS: tuple[MetricSection, ...] = (
    MetricSection(
        name="PRE-SEND (recv → http_payload_send)",
        metrics=frozenset(
            {
                "recv_to_bytes",
                "bytes_to_http_payload",
                "tcp_conn_pool",
                "http_payload_send",
            }
        ),
        phase=SectionPhase.PRE,
        total_key="pre_overhead",  # Total: t_http - t_recv
    ),
    MetricSection(
        name="IN-FLIGHT (http_payload_send → response)",
        metrics=frozenset(
            {
                # Client overhead (t_http → t_task_awake):
                "task_overhead",
                # Server TTFB + response (t_http → t_headers):
                "http_to_headers",
                # Streaming breakdown:
                "headers_to_first_chunk",
                "first_to_last_chunk",
            }
        ),
        phase=SectionPhase.POST,
        total_key="in_flight_time",  # Total: t_response - t_http
    ),
    MetricSection(
        name="POST-RECV (response → query_result_sent)",
        metrics=frozenset({"query_result_sent"}),
        phase=SectionPhase.POST,
        total_key="post_overhead",  # Total: t_zmq_sent - t_response
    ),
)


# =============================================================================
# Statistics Utilities
# =============================================================================


def compute_percentiles(
    values: list[float],
) -> tuple[float, float, float, float, float]:
    """
    Compute statistical percentiles for a list of values.

    Args:
        values: List of numeric values to analyze.

    Returns:
        Tuple of (average, p50, p99, p99.9, max) values.
        Returns all zeros if the input list is empty.
    """
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    sorted_values = sorted(values)
    n = len(sorted_values)
    return (
        statistics.mean(values),
        statistics.median(values),
        sorted_values[int(n * 0.99)] if n >= 2 else sorted_values[0],
        sorted_values[min(int(n * 0.999), n - 1)] if n >= 2 else sorted_values[0],
        sorted_values[-1],
    )


# =============================================================================
# Table Formatting
# =============================================================================


class TableFormatter:
    """
    Dynamic table formatter that adjusts column widths based on content.

    Collects all rows first, then formats with consistent column widths.
    """

    STAT_COLS: tuple[str, ...] = ("N", "Avg", "p50", "p99", "p99.9", "Max")
    MIN_STAT_WIDTH: int = 10  # Minimum width for stat columns

    def __init__(self, label_header: str = "Metric") -> None:
        """
        Initialize the table formatter.

        Args:
            label_header: Header text for the label column.
        """
        self._rows: list[tuple[str, int, tuple[float, ...]]] = []
        self._label_header = label_header

    def add_row(self, label: str, values: list[float]) -> None:
        """
        Add a row to the table.

        Args:
            label: Row label text.
            values: List of values to compute statistics for.
        """
        if not values:
            return
        stats = compute_percentiles(values)
        self._rows.append((label, len(values), stats))

    def _format_number(self, val: float) -> str:
        """Format a number, using compact notation for large values."""
        if abs(val) >= 1_000_000:
            return f"{val:.2e}"
        elif abs(val) >= 1000:
            return f"{val:.2f}"
        else:
            return f"{val:.4f}"

    def _compute_widths(self) -> tuple[int, list[int]]:
        """
        Compute column widths based on content.

        Returns:
            Tuple of (label_width, list of stat column widths).
        """
        if not self._rows:
            return len(self._label_header), [self.MIN_STAT_WIDTH] * 6

        # Label column width
        label_width = max(len(self._label_header), max(len(r[0]) for r in self._rows))

        # Stat column widths (N + 5 percentile columns)
        stat_widths = [len(h) for h in self.STAT_COLS]  # Start with header widths

        for _, n, stats in self._rows:
            # N column
            stat_widths[0] = max(stat_widths[0], len(str(n)))
            # Percentile columns
            for i, val in enumerate(stats):
                stat_widths[i + 1] = max(
                    stat_widths[i + 1], len(self._format_number(val))
                )

        # Apply minimum widths
        stat_widths = [max(w, self.MIN_STAT_WIDTH) for w in stat_widths]

        return label_width, stat_widths

    def format_header(self) -> list[str]:
        """
        Format the table header.

        Returns:
            List of header lines including separators.
        """
        if not self._rows:
            return []

        label_width, stat_widths = self._compute_widths()
        total_width = label_width + 2 + sum(w + 1 for w in stat_widths) + 2

        lines = []

        # Header
        lines.append("=" * total_width)
        header_parts = [f"  {self._label_header:<{label_width}}"]
        for col, w in zip(self.STAT_COLS, stat_widths, strict=False):
            header_parts.append(f"{col:>{w}}")
        lines.append(" ".join(header_parts))
        lines.append("=" * total_width)

        return lines

    def format_row(self, label: str, n: int, stats: tuple[float, ...]) -> str:
        """
        Format a single data row.

        Args:
            label: Row label text.
            n: Number of samples.
            stats: Tuple of percentile statistics.

        Returns:
            Formatted row string.
        """
        label_width, stat_widths = self._compute_widths()

        parts = [f"  {label:<{label_width}}"]
        parts.append(f"{n:>{stat_widths[0]}}")
        for val, w in zip(stats, stat_widths[1:], strict=False):
            parts.append(f"{self._format_number(val):>{w}}")
        return " ".join(parts)

    def format_all(self) -> list[str]:
        """
        Format header and all data rows.

        Returns:
            List of all formatted lines.
        """
        lines = self.format_header()
        for label, n, stats in self._rows:
            lines.append(self.format_row(label, n, stats))
        return lines

    def get_separator(self, char: str = "─") -> str:
        """
        Get a separator line of the correct width.

        Args:
            char: Character to use for the separator.

        Returns:
            Separator string of the appropriate width.
        """
        label_width, stat_widths = self._compute_widths()
        total_width = label_width + 2 + sum(w + 1 for w in stat_widths) + 2
        return char * total_width


# =============================================================================
# Stats Accumulator
# =============================================================================


@dataclass
class Stats:
    """Thread-safe statistics accumulator for timing metrics."""

    pre: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    post: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    timestamps: TimestampsDict = field(default_factory=dict)
    count: int = 0
    report_dir: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, phase: str, query_id: str, metrics: MetricsDict) -> None:
        """
        Add metrics for a query.

        Args:
            phase: Either "pre" or "post".
            query_id: Unique identifier for the query.
            metrics: Dictionary of metric name to value.
        """
        with self._lock:
            target = self.pre if phase == "pre" else self.post
            for name, value in metrics.items():
                if name.startswith("t_"):
                    self.timestamps.setdefault(query_id, {})[name] = value
                else:
                    target[name].append(value)
            if phase == "pre":
                self.count += 1

    def get_stats_for_phase(self, phase: SectionPhase) -> dict[str, list[float]]:
        """
        Get the stats dictionary for a given phase.

        Args:
            phase: The section phase.

        Returns:
            The pre or post stats dictionary.
        """
        return self.pre if phase == SectionPhase.PRE else self.post


# =============================================================================
# Parsing Functions
# =============================================================================


def parse_timing_line(line: str) -> tuple[str, str, MetricsDict] | None:
    """
    Parse a timing log line into structured data.

    Args:
        line: A log line potentially containing timing data.

    Returns:
        Tuple of (query_id, phase, metrics) if parsing succeeds, None otherwise.
    """
    match = TIMING_RE.search(line)
    if not match:
        return None

    query_id = match.group(1)
    phase = match.group(2)
    raw_metrics = match.group(3)

    metrics: MetricsDict = {}
    # Extract duration metrics (d_*=Xms)
    for m in METRIC_RE.finditer(raw_metrics):
        metrics[m.group(1)] = float(m.group(2))
    # Extract timestamp values (t_*=X)
    for m in TIMESTAMP_RE.finditer(raw_metrics):
        metrics[m.group(1)] = float(m.group(2))

    return query_id, phase, metrics


def load_events(report_dir: Path) -> EventsMap:
    """
    Load events.json from the report directory.

    Args:
        report_dir: Path to the report directory.

    Returns:
        Dictionary mapping sample_uuid to {event_type: timestamp_ns}.
    """
    events_file = report_dir / "events.json"
    if not events_file.exists():
        logger.debug("Events file not found: %s", events_file)
        return {}

    samples: EventsMap = {}
    try:
        with open(events_file) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    uuid = event.get("sample_uuid")
                    timestamp = event.get("timestamp_ns")
                    if uuid and timestamp is not None:
                        event_type = event.get("event_type")
                        samples.setdefault(uuid, {})[event_type] = timestamp
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Invalid JSON at line %d in %s: %s", line_num, events_file, e
                    )
    except OSError as e:
        logger.error("Failed to read events file %s: %s", events_file, e)
        return {}

    logger.debug("Loaded %d samples from events.json", len(samples))
    return samples


def load_timing_jsonl(timing_dir: Path) -> Stats:
    """
    Load timing data from worker JSONL files.

    Reads timing_worker_*.jsonl files from the timing directory.
    Each line is a JSON object with: query_id, worker_id, phase, metrics.

    Args:
        timing_dir: Path to directory containing timing_worker_*.jsonl files.

    Returns:
        Stats object populated with the timing data.
    """
    stats = Stats(report_dir=timing_dir)

    timing_files = list(timing_dir.glob("timing_worker_*.jsonl"))
    if not timing_files:
        logger.warning("No timing files found in %s", timing_dir)
        return stats

    logger.debug("Found %d timing files", len(timing_files))

    for timing_file in timing_files:
        try:
            with open(timing_file) as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        query_id = entry.get("query_id", "")
                        phase = entry.get("phase", "")
                        metrics = entry.get("metrics", {})
                        stats.add(phase, query_id, metrics)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Invalid JSON at line %d in %s: %s",
                            line_num,
                            timing_file,
                            e,
                        )
        except OSError as e:
            logger.error("Failed to read timing file %s: %s", timing_file, e)

    return stats


# =============================================================================
# Output Functions
# =============================================================================


def print_live_stats(stats: Stats) -> None:
    """
    Print live stats display (clears screen).

    Args:
        stats: Stats object containing the accumulated metrics.
    """
    post, count = stats.post, stats.count
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
        data = stats.get_stats_for_phase(section.phase)

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
    formatter = TableFormatter()
    for section_name, total_values, breakdown in table_data:
        if total_values:
            formatter.add_row(section_name, total_values)
        for metric_name, values in breakdown:
            formatter.add_row(f"  {metric_name}", values)
    if e2e:
        formatter.add_row("END-TO-END TOTAL", e2e)

    # Print title
    sep = formatter.get_separator("=")
    print(sep)
    print(f"  WORKER TIMING  |  Requests: {count}")

    # Print header
    for line in formatter.format_header():
        print(line)

    # Print sections
    for section_name, total_values, breakdown in table_data:
        print()
        if total_values:
            stats_tuple = compute_percentiles(total_values)
            print(formatter.format_row(section_name, len(total_values), stats_tuple))
        else:
            print(f"  [{section_name}]")

        for metric_name, values in breakdown:
            stats_tuple = compute_percentiles(values)
            print(formatter.format_row(f"  {metric_name}", len(values), stats_tuple))

    # Print end-to-end total
    if e2e:
        print()
        print(formatter.get_separator("─"))
        stats_tuple = compute_percentiles(e2e)
        print(formatter.format_row("END-TO-END TOTAL", len(e2e), stats_tuple))

    print(sep)
    sys.stdout.flush()


def print_analysis(stats: Stats, events: EventsMap) -> None:
    """
    Print overhead breakdown with p99.9 and max for worst-case analysis.

    Args:
        stats: Stats object containing the accumulated metrics.
        events: Events map from load_events().
    """
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
        avg, p50, p99, p999, mx = compute_percentiles(vals)
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
        # Client overhead (t_http → t_task_awake):
        ("task_overhead", post.get("task_overhead", [])),
        # Server TTFB (t_http → t_headers):
        ("http_to_headers", post.get("http_to_headers", [])),
        # Streaming response breakdown:
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


# =============================================================================
# Mode Handlers
# =============================================================================


def run_file_mode(report_dir: Path) -> int:
    """
    Run the parser in file mode, reading from timing JSONL files.

    Args:
        report_dir: Path to the report directory.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    timing_dir = report_dir / "endpoint_client"

    if not timing_dir.exists():
        logger.error("Timing directory not found: %s", timing_dir)
        print(f"No timing data found in {timing_dir}")
        print("Looking for: timing_worker_*.jsonl files")
        return 1

    stats = load_timing_jsonl(timing_dir)
    if stats.count == 0:
        print(f"No timing data found in {timing_dir}")
        print("Looking for: timing_worker_*.jsonl files")
        return 1

    with stats._lock:
        print_live_stats(stats)

    # Load events.json from report_dir for IPC calculations
    events = load_events(report_dir)
    print_analysis(stats, events)
    return 0


def run_stdin_mode(report_dir: Path | None) -> int:
    """
    Run the parser in stdin mode, reading from piped log output.

    Args:
        report_dir: Optional path to report directory for events.json.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    stats = Stats(report_dir=report_dir)
    stop = threading.Event()

    # Live display thread
    def display_loop() -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        while not stop.wait(1.0):
            with stats._lock:
                print_live_stats(stats)

    display_thread = threading.Thread(target=display_loop, daemon=True)
    display_thread.start()

    # Parse stdin
    try:
        for line in sys.stdin:
            result = parse_timing_line(line)
            if result:
                stats.add(result[1], result[0], result[2])
            if not stats.report_dir:
                match = REPORT_DIR_RE.search(line)
                if match:
                    stats.report_dir = Path(match.group(1))
    except KeyboardInterrupt:
        logger.debug("Interrupted by user")
    finally:
        stop.set()

    # Final output
    if stats.count == 0:
        print("\nNo timing data found in stdin")
        print(
            "Hint: Use -vvv flag with inference-endpoint to enable TRACE-level timing logs"
        )
        return 1

    with stats._lock:
        print_live_stats(stats)

    events = load_events(stats.report_dir) if stats.report_dir else {}
    print_analysis(stats, events)
    return 0


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """
    Main entry point for the timing parser.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Report directory (timing files in endpoint_client/, events in events.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Check if stdin is piped (not a TTY)
    stdin_is_tty = os.isatty(sys.stdin.fileno())

    if stdin_is_tty:
        # No stdin input - read from timing JSONL files
        if not args.report_dir:
            print(__doc__)
            return 0
        return run_file_mode(args.report_dir)
    else:
        # Legacy mode: parse timing from verbose log output on stdin
        return run_stdin_mode(args.report_dir)


if __name__ == "__main__":
    sys.exit(main())
