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
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final

import msgspec.json
import msgspec.structs

from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricCounterKey,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    build_token_series_dict,
)
from inference_endpoint.evaluation.accuracy_results import average_accuracy
from inference_endpoint.utils.version import get_version_info

from ..utils import monotime_to_datetime

# Aggregator series name -> result_summary.json field. Single source of truth for the
# summary's latency sections: ``Report.from_snapshot`` builds its fields from this, and
# ``scripts/early_stopping_estimate_from_events.py`` uses it to key its post-hoc output the same way.
SERIES_TO_SUMMARY_FIELD: Final[dict[str, str]] = {
    "ttft_ns": "ttft",
    "tpot_ns": "tpot",
    "sample_latency_ns": "latency",
}


def place_early_stopping_percentiles(
    metric_dict: dict[str, Any], esp: dict[str, float | None]
) -> dict[str, Any]:
    """Return ``metric_dict`` with the ES map directly after ``percentiles``.

    Single source of the map's position in every summary-shaped dict — used by
    the report builder and by ``scripts/early_stopping_estimate_from_events.py``
    when augmenting a historical ``result_summary.json``. Replaces any existing
    placement; appends at the end if the dict has no ``percentiles`` key.
    """
    out: dict[str, Any] = {}
    for key, value in metric_dict.items():
        if key == "early_stopping_percentiles":
            continue
        out[key] = value
        if key == "percentiles":
            out["early_stopping_percentiles"] = esp
    if "early_stopping_percentiles" not in out:
        out["early_stopping_percentiles"] = esp
    return out


_FINISH_REASON_COUNTERS = (
    MetricCounterKey.TRACKED_FINISH_REASON_STOP,
    MetricCounterKey.TRACKED_FINISH_REASON_LENGTH,
    MetricCounterKey.TRACKED_FINISH_REASON_TOOL_CALLS,
    MetricCounterKey.TRACKED_FINISH_REASON_CONTENT_FILTER,
    MetricCounterKey.TRACKED_FINISH_REASON_FUNCTION_CALL,
    MetricCounterKey.TRACKED_FINISH_REASON_OTHER,
)


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
        # No data -> no rollups, but an enabled-ES empty series still self-describes
        # (all-null map) in the summary instead of looking feature-off.
        esp = stat.get("early_stopping_percentiles")
        return {"early_stopping_percentiles": esp} if esp is not None else {}

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
    metric: dict[str, Any] = {
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
    # Early-stopping estimate map — present only when the feature is enabled
    # (COMPLETE snapshots for TTFT/TPOT/latency); sits right after `percentiles`.
    early_stopping_percentiles = stat.get("early_stopping_percentiles")
    if early_stopping_percentiles is not None:
        metric = place_early_stopping_percentiles(metric, early_stopping_percentiles)
    return metric


def series_metric_dict(values: Iterable[int]) -> dict[str, Any]:
    """Build a series rollup (same shape as the perf ``output_sequence_lengths``
    block) from raw integer values, off the hot path.

    Delegates the token-series construction to the aggregator's
    ``build_token_series_dict`` — the single source of the HDR bounds / sig-figs /
    bucket count — then applies the same ``_series_to_metric_dict`` shaping the
    perf report uses, so an accuracy-phase OSL block is byte-for-byte the same
    shape (avg/min/max/median/std_dev/percentiles/histogram) as the performance
    one. Returns ``{}`` when ``values`` is empty.
    """
    return _series_to_metric_dict(build_token_series_dict(values))


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
    # Legacy MLPerf LoadGen Server "completed" window (poisson only): first
    # issued request -> completion of the last-issued request
    # (final_query_all_samples_done_time analog; see mlcommons/inference
    # loadgen/results.cc). Not None iff QPS/TPS were computed over this window;
    # None means the endpoints-native full-run window was used. Recorded so
    # result_summary.json is self-describing about which view it holds.
    # TODO(vir): deprecate once endpoints has a formal tail-cutting mechanism.
    legacy_loadgen_window_duration_ns: int | None = None

    # Derived throughput, computed once in from_snapshot so the serialized
    # report (result_summary.json) is self-complete. qps is None without a
    # duration; tps is also None when no OSL was recorded (non-streaming or
    # tokenizer unavailable).
    qps: float | None = None
    tps: float | None = None
    finish_reason_counts: dict[str, int] = msgspec.field(default_factory=dict)

    # Run configuration (load_pattern, warmup, and the scheduler/dataloader RNG
    # seeds), from config. Carried so result_summary.json is self-describing and a
    # valid run is identified by its settings. Config, not a measured metric, so
    # the from_snapshot caller supplies it rather than reading it from the metrics
    # snapshot. (Resolved/effective runtime settings — sample count + ordering,
    # which can differ per audit phase — are deferred to a follow-up.)
    run_config: dict[str, Any] | None = None

    # Per-dataset accuracy entries (one per scored dataset), attached after
    # scoring in finalize_benchmark. Accuracy is not in the metrics snapshot, so
    # from_snapshot leaves this empty; runs without configured scoring keep it
    # empty. Each entry carries score + sample counts and, for multi-subset
    # scorers, a BFCL-shaped breakdown. Display-only.
    accuracy: list[dict[str, Any]] = msgspec.field(default_factory=list)

    @classmethod
    def from_snapshot(
        cls,
        snap: dict[str, Any],
        *,
        run_config: dict[str, Any] | None = None,
        use_legacy_loadgen_qps_metrics: bool = True,
    ) -> Report:
        """Build a Report from a snapshot dict.

        ``run_config`` (optional, keyword-only) carries the run's configuration
        (load_pattern, warmup, and the scheduler/dataloader RNG seeds) into the
        report so result_summary.json is self-describing; it is config, not part
        of the metrics snapshot.

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

        The snapshot always carries BOTH ``tracked_duration_ns`` and
        ``legacy_loadgen_window_duration_ns``, so it stays config-agnostic and
        fully reinterpretable either way. Which window the reported QPS/TPS use
        is decided by the run config (``use_legacy_loadgen_qps_metrics``,
        recorded in ``config.yaml`` and in this Report's serialized JSON), not
        by the snapshot.
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

        # Legacy MLPerf LoadGen Server "completed" window (poisson only): first
        # issued request -> completion of the last-issued request
        # (final_query_all_samples_done_time analog; see mlcommons/inference
        # loadgen/results.cc).
        # TODO(vir): deprecate once endpoints has a formal tail-cutting mechanism.
        raw_loadgen_window_ns = _counter("legacy_loadgen_window_duration_ns")

        # Derived throughput, computed once so result_summary.json is
        # self-complete. The legacy LoadGen window drives the headline QPS/TPS
        # only when it is enabled, available, AND there are >=2 completions
        # (QPS = (completed-1)/window is undefined below 2). If any of those
        # fail, BOTH QPS and TPS fall back to the native window so they always
        # share one window, and legacy_loadgen_window_duration_ns stays None so
        # the serialized report honestly records which view it holds.
        use_legacy_window = (
            use_legacy_loadgen_qps_metrics
            and raw_loadgen_window_ns > 0
            and n_completed >= 2
        )
        legacy_loadgen_window_duration_ns = (
            raw_loadgen_window_ns if use_legacy_window else None
        )
        if use_legacy_window:
            window_s = raw_loadgen_window_ns / 1e9
            qps = (n_completed - 1) / window_s
            tps = (osl.get("total", 0) / window_s) if osl else None
        elif duration_ns is not None:
            duration_s = duration_ns / 1e9
            qps = n_completed / duration_s
            tps = (osl.get("total", 0) / duration_s) if osl else None
        else:
            qps = None
            tps = None

        # Default missing state to "interrupted" — a malformed / partial
        # snapshot dict is treated as worst-case (run did not reach a
        # clean completion). Drives complete=False and the interrupted
        # indicator in display().
        state = snap.get("state", "interrupted")
        n_pending_tasks = snap.get("n_pending_tasks", 0)
        finish_reason_prefix = "tracked_finish_reason_"
        finish_reason_counts = {
            reason.value.removeprefix(finish_reason_prefix): int(
                counters.get(reason.value, 0)
            )
            for reason in _FINISH_REASON_COUNTERS
        }

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
            **{
                field: _series_dict(series)
                for series, field in SERIES_TO_SUMMARY_FIELD.items()
            },
            output_sequence_lengths=osl,
            legacy_loadgen_window_duration_ns=legacy_loadgen_window_duration_ns,
            qps=qps,
            tps=tps,
            finish_reason_counts=finish_reason_counts,
            run_config=run_config,
        )

    def to_json(self, save_to: os.PathLike | None = None) -> bytes:
        # result_summary.json is the performance report — accuracy lives only in
        # the dedicated accuracy report (accuracy/accuracy_results.json). The
        # accuracy field stays on the struct so report.txt / the console summary
        # still render it; it is just dropped from this serialized perf summary.
        payload = {
            k: v for k, v in msgspec.structs.asdict(self).items() if k != "accuracy"
        }
        json_bytes = msgspec.json.format(msgspec.json.encode(payload), indent=2)
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
        if self.run_config:
            fn(f"Run config:{newline}")
            for section, params in self.run_config.items():
                if isinstance(params, dict):
                    inner = ", ".join(f"{k}={v}" for k, v in params.items())
                    fn(f"  {section}: {inner}{newline}")
                else:
                    fn(f"  {section}: {params}{newline}")
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
        else:
            fn(f"TPS: N/A{newline}")

        if self.accuracy:
            fn(f"Accuracy:{newline}")
            for entry in self.accuracy:
                name = entry.get("dataset_name", "?")
                score = entry.get("score")
                if score is None:
                    score_str = "N/A"
                elif isinstance(score, bool):
                    score_str = str(score)
                elif isinstance(score, int | float):
                    score_str = f"{score:.4g}"
                else:
                    score_str = str(score)
                unit = entry.get("unit_samples")
                repeats = entry.get("num_repeats")
                total = entry.get("total_samples")
                dur = entry.get("duration_s")
                dur_str = (
                    f", duration={dur:.2f}s" if isinstance(dur, int | float) else ""
                )
                fn(
                    f"  {name}: {score_str} "
                    f"(unit={unit}, repeats={repeats}, total={total}{dur_str}){newline}"
                )
                rc = entry.get("response_counts")
                if isinstance(rc, dict):
                    fn(
                        f"    responses: {rc.get('scored', 0)}/{rc.get('issued', 0)} "
                        f"scored ({rc.get('empty', 0)} empty, "
                        f"{rc.get('missing', 0)} missing){newline}"
                    )
                osl = entry.get("output_sequence_lengths")
                if isinstance(osl, dict):
                    fn(
                        f"    output tokens (avg/min/max): "
                        f"{osl.get('avg', 0):.1f}/{osl.get('min')}/{osl.get('max')}"
                        f"{newline}"
                    )
                breakdown = entry.get("breakdown")
                if isinstance(breakdown, dict):
                    for sub, sub_score in (
                        breakdown.get("subset_scores") or {}
                    ).items():
                        fn(f"    {sub}: {sub_score:.2f}%{newline}")
                if entry.get("complete") is False:
                    fn(f"    (incomplete){newline}")
            avg = average_accuracy(self.accuracy)
            if avg is not None:
                fn(f"  Average: {avg:.4g}{newline}")
            if any("osl_tokenize_s" in e for e in self.accuracy):
                osl_tok = sum(e.get("osl_tokenize_s", 0.0) for e in self.accuracy)
                fn(f"  OSL tokenization: {osl_tok:.3g}s{newline}")

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

    # .get(): an empty-but-ES-carrying metric dict has no rollups/histogram.
    for name, key in [
        ("Min", "min"),
        ("Max", "max"),
        ("Median", "median"),
        ("Avg.", "avg"),
        ("Std Dev.", "std_dev"),
    ]:
        fn(f"  {name}: {_scaled(metric_dict.get(key))} {unit}{newline}")

    fn(f"\n  Histogram:{newline}")
    histogram = metric_dict.get("histogram") or {"buckets": [], "counts": []}
    buckets = histogram["buckets"]
    counts = histogram["counts"]

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

    es_map = metric_dict.get("early_stopping_percentiles")
    if es_map:
        fn(
            f"\n  Early-stopping percentile estimates (N/A = insufficient samples):{newline}"
        )
        for k, v in es_map.items():
            fn(f"  {k:>6}: {_scaled(v)} {unit}{newline}")
