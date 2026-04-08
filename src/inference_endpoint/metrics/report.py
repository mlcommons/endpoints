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

"""Benchmark report: summary statistics, display, and JSON serialization."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec.json
import numpy as np

from inference_endpoint.async_utils.services.metrics_aggregator.kv_store import (
    BasicKVStoreReader,
    SeriesStats,
)
from inference_endpoint.utils.version import get_version_info

from ..utils import monotime_to_datetime

# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

_DEFAULT_PERCENTILES = (99.9, 99, 97, 95, 90, 80, 75, 50, 25, 10, 5, 1)


def compute_summary(
    stats: SeriesStats,
    percentiles: tuple[float, ...] = _DEFAULT_PERCENTILES,
    n_histogram_buckets: int = 10,
) -> dict[str, Any]:
    """Compute rollup statistics from pre-computed SeriesStats.

    Scalar stats (total, min, max, avg, std_dev) are derived from the
    incrementally maintained rollups in SeriesStats. Numpy is only used
    for percentiles and histograms, which require the raw values.

    Returns a dict with: total, min, max, avg, std_dev, median,
    percentiles (dict), and histogram (buckets + counts).
    """
    if stats.count == 0:
        return {
            "total": 0,
            "min": 0,
            "max": 0,
            "median": 0.0,
            "avg": 0.0,
            "std_dev": 0.0,
            "percentiles": {str(p): 0.0 for p in percentiles},
            "histogram": {"buckets": [], "counts": []},
        }

    # Scalar stats from pre-computed rollups (no numpy needed)
    avg = stats.total / stats.count
    # Bessel's correction (ddof=1) for sample standard deviation
    if stats.count > 1:
        n = stats.count
        std_dev = math.sqrt((stats.sum_sq - stats.total**2 / n) / (n - 1))
    else:
        std_dev = 0.0

    # Percentiles and histogram require raw values
    # Don't force float64 — numpy preserves int for uint64 series,
    # so percentile(method="lower") returns actual observed values
    # in their original type.
    arr = np.array(stats.values)
    arr.sort()

    # Inject 50th percentile for median if not already requested
    need_median = 50 not in percentiles
    all_percentiles = (*percentiles, 50) if need_median else percentiles

    perc_values = np.percentile(arr, all_percentiles, method="lower")
    perc_dict = {
        str(p): v.item() for p, v in zip(all_percentiles, perc_values, strict=True)
    }
    median = perc_dict.pop("50") if need_median else perc_dict["50"]

    bounds = np.histogram_bin_edges(arr, bins=n_histogram_buckets)
    counts, _ = np.histogram(arr, bins=bounds)
    hist_buckets = [
        (float(bounds[i]), float(bounds[i + 1])) for i in range(len(bounds) - 1)
    ]

    return {
        "total": stats.total,
        "min": stats.min_val,
        "max": stats.max_val,
        "median": median,
        "avg": avg,
        "std_dev": std_dev,
        "percentiles": perc_dict,
        "histogram": {"buckets": hist_buckets, "counts": counts.tolist()},
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Report(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """Summarized benchmark report."""

    version: str
    git_sha: str | None
    test_started_at: int
    n_samples_issued: int
    n_samples_completed: int
    n_samples_failed: int
    duration_ns: int | None

    # Per-metric rollup dicts (output of compute_summary)
    ttft: dict[str, Any]
    tpot: dict[str, Any]
    latency: dict[str, Any]
    output_sequence_lengths: dict[str, Any]

    def qps(self) -> float | None:
        if self.duration_ns is None or self.duration_ns <= 0:
            return None
        return self.n_samples_completed / (self.duration_ns / 1e9)

    def tps(self) -> float | None:
        if self.duration_ns is None or self.duration_ns <= 0:
            return None
        if not self.output_sequence_lengths:
            return None
        total = self.output_sequence_lengths.get("total", 0)
        return total / (self.duration_ns / 1e9)

    @classmethod
    def from_kv_reader(cls, reader: BasicKVStoreReader) -> Report:
        """Build a Report from the current KVStore state.

        Reads counters and series from the reader, computes rollup summaries
        (percentiles, histograms) for each series metric, and returns a Report.

        Works identically for live metrics (mid-test) and final reports
        (post-drain). The caller decides when to call.
        """
        snap = reader.snapshot()

        def _counter(key: str) -> int:
            val = snap.get(key)
            return int(val) if isinstance(val, int) else 0

        def _summarize(key: str) -> dict:
            val = snap.get(key)
            if isinstance(val, SeriesStats) and val.count > 0:
                return compute_summary(val)
            return {}

        version_info = get_version_info()
        duration_ns = _counter("tracked_duration_ns")

        return cls(
            version=str(version_info.get("version", "unknown")),
            git_sha=version_info.get("git_sha"),
            test_started_at=0,  # TODO: add test_started_at counter to aggregator
            n_samples_issued=_counter("tracked_samples_issued"),
            n_samples_completed=_counter("tracked_samples_completed"),
            # TODO: Add tracked_samples_failed to MetricCounterKey.
            # For now, total_samples_failed is the best available.
            n_samples_failed=_counter("total_samples_failed"),
            duration_ns=duration_ns if duration_ns > 0 else None,
            ttft=_summarize("ttft_ns"),
            tpot=_summarize("tpot_ns"),
            latency=_summarize("sample_latency_ns"),
            output_sequence_lengths=_summarize("osl"),
        )

    def to_json(self, save_to: os.PathLike | None = None) -> bytes:
        json_bytes = msgspec.json.format(msgspec.json.encode(self), indent=2)
        if save_to is not None:
            with Path(save_to).open("wb") as f:
                f.write(json_bytes)
        return json_bytes

    def display(
        self,
        fn: Callable[[str], None] = print,
        summary_only: bool = False,
        newline: str = "",
    ) -> None:
        fn(f"----------------- Summary -----------------{newline}")
        fn(f"Version: {self.version}{newline}")
        if self.git_sha:
            fn(f"Git SHA: {self.git_sha}{newline}")
        if self.test_started_at > 0:
            approx = monotime_to_datetime(self.test_started_at)
            fn(f"Test started at: {approx.strftime('%Y-%m-%d %H:%M:%S')}{newline}")
        fn(f"Total samples issued: {self.n_samples_issued}{newline}")
        fn(f"Total samples completed: {self.n_samples_completed}{newline}")
        fn(f"Total samples failed: {self.n_samples_failed}{newline}")
        if self.duration_ns is not None:
            fn(f"Duration: {self.duration_ns / 1e9:.2f} seconds{newline}")
        else:
            fn(f"Duration: N/A{newline}")

        if (qps := self.qps()) is not None:
            fn(f"QPS: {qps:.2f}{newline}")
        else:
            fn(f"QPS: N/A{newline}")

        if (tps := self.tps()) is not None:
            fn(f"TPS: {tps:.2f}{newline}")

        if summary_only:
            fn(f"----------------- End of Summary -----------------{newline}")
            return

        fn(f"\n------------------- Latency Breakdowns -------------------{newline}")

        for section_name, metric_dict, unit, scale_factor in [
            ("TTFT", self.ttft, "ms", 1e-6),
            ("TPOT", self.tpot, "ms", 1e-6),
            ("Latency", self.latency, "ms", 1e-6),
            ("Output sequence lengths", self.output_sequence_lengths, "tokens", 1.0),
        ]:
            if not metric_dict:
                continue
            fn(f"{section_name}:{newline}")
            _display_metric(
                metric_dict,
                fn=fn,
                unit=unit,
                scale_factor=scale_factor,
                newline=newline,
            )
            fn(f"{newline}")

        fn(f"----------------- End of Report -----------------{newline}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_metric(
    metric_dict: dict[str, Any],
    fn: Callable[[str], None],
    unit: str = "",
    max_bar_length: int = 30,
    scale_factor: float = 1.0,
    newline: str = "",
) -> None:
    for name, key in [
        ("Min", "min"),
        ("Max", "max"),
        ("Median", "median"),
        ("Avg.", "avg"),
        ("Std Dev.", "std_dev"),
    ]:
        fn(f"  {name}: {metric_dict[key] * scale_factor:.2f} {unit}{newline}")

    fn(f"\n  Histogram:{newline}")
    buckets = metric_dict["histogram"]["buckets"]
    counts = metric_dict["histogram"]["counts"]

    if buckets:
        bucket_strs = [
            f"  [{lo * scale_factor:.2f}, {hi * scale_factor:.2f}"
            + ("]" if i == len(buckets) - 1 else ")")
            for i, (lo, hi) in enumerate(buckets)
        ]
        max_count = max(counts)
        normalize = max_bar_length / max_count if max_count > 0 else 1
        max_label = max(len(s) for s in bucket_strs)

        for label, count in zip(bucket_strs, counts, strict=True):
            bar = "#" * int(count * normalize)
            fn(f"  {label:>{max_label}} |{bar} {count}{newline}")

    fn(f"\n  Percentiles:{newline}")
    for p, val in metric_dict.get("percentiles", {}).items():
        fn(f"  {p:>6}: {val * scale_factor:.2f} {unit}{newline}")
