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

from inference_endpoint.utils.version import get_version_info

from ..utils import monotime_to_datetime


def _series_to_metric_dict(stat: dict[str, Any]) -> dict[str, Any]:
    """Convert a series-stat dict into the shape ``display()`` expects.

    Input is the dict form produced by ``snapshot_to_dict``. Derives
    ``avg``, ``std_dev``, and ``median`` from the rollups + percentiles.
    ``median`` falls back to ``(min + max) / 2`` if the producer didn't
    emit p50.

    All field reads use ``.get(...)`` with sensible defaults so a
    truncated / partial dict (e.g. an INTERRUPTED snapshot) produces an
    honest empty rollup instead of crashing.
    """
    count = stat.get("count", 0)
    if count == 0:
        return {}

    total = stat.get("total", 0)
    sum_sq = stat.get("sum_sq", 0)
    s_min = stat.get("min", 0)
    s_max = stat.get("max", 0)

    avg = total / count if count > 0 else 0.0
    if count > 1:
        n = count
        # Integer-aggregate series (latency in ns) can have very large
        # sum_sq and total values; the naive `sum_sq - total^2 / n`
        # form loses precision when total^2 / n is close to sum_sq.
        # Use the exact integer form `n*sum_sq - total^2` when inputs
        # are int, falling back to the float form otherwise.
        if isinstance(total, int) and isinstance(sum_sq, int):
            var_num_int = n * sum_sq - total * total
            std_dev = math.sqrt(max(0, var_num_int)) / math.sqrt(n * (n - 1))
        else:
            var_num = sum_sq - total * total / n
            std_dev = math.sqrt(max(0.0, var_num / (n - 1)))
    else:
        std_dev = 0.0

    # p50 is contractually required on every registered series — see
    # ``MetricsRegistry.register_series``, which rejects registrations
    # whose percentiles tuple omits 50.0. The midrange fallback below
    # only fires for hand-crafted snapshot dicts that bypass the
    # registration path (e.g. a manually-edited JSON file), in which
    # case the midrange is wrong-but-displayable rather than crashing.
    perc = stat.get("percentiles", {})
    if "50" in perc:
        median: float = perc["50"]
    elif "50.0" in perc:
        median = perc["50.0"]
    else:
        # Approximate-only fallback for non-registry-produced dicts.
        median = (s_min + s_max) / 2

    histogram = stat.get("histogram", [])
    return {
        "total": total,
        "min": s_min,
        "max": s_max,
        "median": median,
        "avg": avg,
        "std_dev": std_dev,
        "percentiles": dict(perc),
        "histogram": {
            "buckets": [(rng[0], rng[1]) for rng, _ in histogram],
            "counts": [c for _, c in histogram],
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
    # The terminal SessionState that produced this report. Surfaced as a
    # raw string so ``display()`` can render an INTERRUPTED indicator
    # without re-parsing the source dict, and so JSON round-trips don't
    # depend on Report importing the SessionState enum.
    state: str
    # True iff state=="complete" AND n_pending_tasks==0. False signals
    # partial async metrics — drain timed out (state=="complete",
    # n_pending_tasks>0), the run was interrupted (state=="interrupted"),
    # or no final snapshot was found and we fell back to a live tick.
    complete: bool

    # Per-metric rollup dicts (output of _series_to_metric_dict)
    ttft: dict[str, Any]
    tpot: dict[str, Any]
    latency: dict[str, Any]
    output_sequence_lengths: dict[str, Any]

    # Derived throughput, computed once in from_snapshot so the serialized
    # report (result_summary.json) is self-complete. qps is None without a
    # duration; tps is also None when no OSL was recorded (non-streaming or
    # tokenizer unavailable).
    qps: float | None = None
    tps: float | None = None

    # RNG seeds for this run (scheduler/dataloader/warmup, from config). Carried
    # so result_summary.json is self-validating: a reproducible run is identified
    # by its seeds. These are config, not a measured metric, so the from_snapshot
    # caller supplies them rather than reading them from the metrics snapshot.
    seeds: dict[str, int] | None = None

    @classmethod
    def from_snapshot(
        cls, snap: dict[str, Any], *, seeds: dict[str, int] | None = None
    ) -> Report:
        """Build a Report from a snapshot dict.

        ``seeds`` (optional) carries the run's RNG seeds from config into the
        report so result_summary.json is self-validating; it is keyword-only
        because it is config, not part of the metrics snapshot.

        Input is the dict form produced by
        ``inference_endpoint.async_utils.services.metrics_aggregator.snapshot
        .snapshot_to_dict``, which is also the shape persisted to
        ``final_snapshot.json``. Consumers can therefore feed
        ``json.loads(path.read_bytes())`` straight in without an
        intermediate Struct decode — this is deliberate, because the
        wire ``MetricsSnapshot`` uses ``array_like=True`` for compact
        msgpack and decoding a dict back into an array-like Struct is
        ergonomically painful (msgspec's decoders follow the Struct's
        ``array_like`` flag).

        All field reads use ``.get(...)`` with defaults that produce an
        honest "incomplete" report on missing fields instead of crashing:
        missing ``state`` defaults to ``"interrupted"`` (worst-case),
        missing counters / series to zero / empty.
        """
        counters: dict[str, int | float] = {}
        series: dict[str, dict[str, Any]] = {}
        for stat in snap.get("metrics", []):
            stat_type = stat.get("type")
            name = stat.get("name", "")
            if not name:
                continue
            if stat_type == "counter":
                counters[name] = stat.get("value", 0)
            elif stat_type == "series":
                series[name] = stat

        def _counter(key: str) -> int:
            return int(counters.get(key, 0))

        def _series_dict(key: str) -> dict[str, Any]:
            stat = series.get(key)
            if stat is None or stat.get("count", 0) == 0:
                return {}
            return _series_to_metric_dict(stat)

        version_info = get_version_info()
        raw_duration_ns = _counter("tracked_duration_ns")
        duration_ns = raw_duration_ns if raw_duration_ns > 0 else None
        n_completed = _counter("tracked_samples_completed")
        osl = _series_dict("osl")

        # Derived throughput. qps needs a duration; tps additionally needs OSL.
        if duration_ns is None:
            qps = tps = None
        else:
            duration_s = duration_ns / 1e9
            qps = n_completed / duration_s
            tps = (osl.get("total", 0) / duration_s) if osl else None

        # Default missing state to "interrupted" — a malformed / partial
        # snapshot dict is treated as worst-case (run did not reach a
        # clean completion). Drives complete=False and the interrupted
        # indicator in display().
        state = snap.get("state", "interrupted")
        n_pending_tasks = snap.get("n_pending_tasks", 0)

        return cls(
            version=str(version_info.get("version", "unknown")),
            git_sha=version_info.get("git_sha"),
            test_started_at=0,  # TODO: surface session_started_ns via snapshot
            n_samples_issued=_counter("tracked_samples_issued"),
            n_samples_completed=n_completed,
            n_samples_failed=_counter("tracked_samples_failed"),
            duration_ns=duration_ns,
            state=state,
            complete=(state == "complete" and n_pending_tasks == 0),
            ttft=_series_dict("ttft_ns"),
            tpot=_series_dict("tpot_ns"),
            latency=_series_dict("sample_latency_ns"),
            output_sequence_lengths=osl,
            qps=qps,
            tps=tps,
            seeds=seeds,
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
        if self.state == "interrupted":
            fn(
                "WARNING: run was interrupted (SIGTERM/SIGINT) — "
                f"metrics below are best-effort partial data.{newline}"
            )
        elif not self.complete:
            fn(
                "WARNING: report is incomplete (drain timed out or no "
                f"final snapshot received) — some async metrics may be missing.{newline}"
            )
        fn(f"Version: {self.version}{newline}")
        if self.git_sha:
            fn(f"Git SHA: {self.git_sha}{newline}")
        if self.seeds:
            seed_str = ", ".join(f"{k}={v}" for k, v in self.seeds.items())
            fn(f"Seeds: {seed_str}{newline}")
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

        if self.qps is not None:
            fn(f"QPS: {self.qps:.2f}{newline}")
        else:
            fn(f"QPS: N/A{newline}")

        if (tps := self.tps) is not None:
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
    # ``_scrub_nonfinite`` (snapshot.py) maps producer-side NaN/±Inf to
    # ``None`` so the persisted JSON stays strict. Any of the named
    # scalars / percentile values below can therefore be ``None`` —
    # render an ``N/A`` indicator instead of crashing on
    # ``None * scale_factor``.
    def _scaled(v: Any) -> str:
        if v is None:
            return "N/A"
        return f"{v * scale_factor:.2f}"

    for name, key in [
        ("Min", "min"),
        ("Max", "max"),
        ("Median", "median"),
        ("Avg.", "avg"),
        ("Std Dev.", "std_dev"),
    ]:
        fn(f"  {name}: {_scaled(metric_dict[key])} {unit}{newline}")

    fn(f"\n  Histogram:{newline}")
    buckets = metric_dict["histogram"]["buckets"]
    counts = metric_dict["histogram"]["counts"]

    if buckets:
        bucket_strs = [
            f"  [{_scaled(lo)}, {_scaled(hi)}" + ("]" if i == len(buckets) - 1 else ")")
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
        fn(f"  {p:>6}: {_scaled(val)} {unit}{newline}")
