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

from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    CounterStat,
    MetricsSnapshot,
    SeriesStat,
    SessionState,
)
from inference_endpoint.utils.version import get_version_info

from ..utils import monotime_to_datetime


def _series_to_metric_dict(stat: SeriesStat) -> dict[str, Any]:
    """Convert a wire ``SeriesStat`` into the dict shape ``display()`` expects.

    Derives ``avg``, ``std_dev``, and ``median`` from the rollups +
    percentiles. ``median`` falls back to the bucket-midpoint search if
    the producer didn't emit p50.
    """
    if stat.count == 0:
        return {}

    avg = stat.total / stat.count if stat.count > 0 else 0.0
    if stat.count > 1:
        n = stat.count
        var_num = stat.sum_sq - stat.total * stat.total / n
        std_dev = math.sqrt(var_num / (n - 1)) if var_num > 0 else 0.0
    else:
        std_dev = 0.0

    # Median: prefer p50 from the producer, fall back to (min+max)/2 so
    # ``display()`` still has a numeric value to format.
    perc = stat.percentiles
    if "50" in perc:
        median: float = perc["50"]
    elif "50.0" in perc:
        median = perc["50.0"]
    else:
        median = (stat.min + stat.max) / 2

    return {
        "total": stat.total,
        "min": stat.min,
        "max": stat.max,
        "median": median,
        "avg": avg,
        "std_dev": std_dev,
        "percentiles": dict(stat.percentiles),
        "histogram": {
            "buckets": [(lo, hi) for (lo, hi), _ in stat.histogram],
            "counts": [c for _, c in stat.histogram],
        },
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
    # True iff the snapshot was state=COMPLETE AND n_pending_tasks==0.
    # False signals partial async metrics — either drain timed out
    # (state=COMPLETE, n_pending_tasks>0) or no COMPLETE snapshot was
    # received and we fell back to a live/draining snapshot.
    complete: bool

    # Per-metric rollup dicts (output of _series_to_metric_dict)
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
    def from_snapshot(cls, snap: MetricsSnapshot) -> Report:
        """Build a Report from a MetricsSnapshot.

        Counters are looked up by name; series are converted to the
        dict shape that ``display()`` expects. Percentiles / histograms
        are passed straight through from the snapshot.
        """
        counters: dict[str, int | float] = {}
        series: dict[str, SeriesStat] = {}
        for stat in snap.metrics:
            if isinstance(stat, CounterStat):
                counters[stat.name] = stat.value
            elif isinstance(stat, SeriesStat):
                series[stat.name] = stat

        def _counter(key: str) -> int:
            val = counters.get(key, 0)
            return int(val)

        def _series_dict(key: str) -> dict[str, Any]:
            stat = series.get(key)
            if stat is None or stat.count == 0:
                return {}
            return _series_to_metric_dict(stat)

        version_info = get_version_info()
        duration_ns = _counter("tracked_duration_ns")

        return cls(
            version=str(version_info.get("version", "unknown")),
            git_sha=version_info.get("git_sha"),
            test_started_at=0,  # TODO: surface session_started_ns via snapshot
            n_samples_issued=_counter("tracked_samples_issued"),
            n_samples_completed=_counter("tracked_samples_completed"),
            n_samples_failed=_counter("tracked_samples_failed"),
            duration_ns=duration_ns if duration_ns > 0 else None,
            complete=(
                snap.state == SessionState.COMPLETE and snap.n_pending_tasks == 0
            ),
            ttft=_series_dict("ttft_ns"),
            tpot=_series_dict("tpot_ns"),
            latency=_series_dict("sample_latency_ns"),
            output_sequence_lengths=_series_dict("osl"),
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

        if not self.complete:
            fn(
                f"WARNING: Some async metrics may be incomplete "
                f"(drain timeout){newline}"
            )

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
