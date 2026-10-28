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

import logging
import math
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final, NamedTuple

import msgspec.json
import msgspec.structs

from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricCounterKey,
    MetricSeriesKey,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    build_token_series_dict,
)
from inference_endpoint.core.types import TokenCountSource, basis_key
from inference_endpoint.evaluation.accuracy_results import (
    samples_weighted_average_accuracy,
)
from inference_endpoint.metrics.steady_state_diagnostics import SteadyState
from inference_endpoint.metrics.token_count_basis import missing_usage_message
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


class TokenMetric(NamedTuple):
    """A token-count metric reported on both bases, by its base names."""

    series: str  # aggregator series
    field: str  # result_summary.json key
    label: str  # report.txt section
    unit: str  # report.txt unit
    scale: float  # series value -> unit


# The token-count metrics. The primary basis reports them under these names;
# the other basis under the names ``basis_key`` gives it (``tpot_server``).
# Existing tooling parses report.txt by section label, so labels stay fixed.
TOKEN_METRICS: Final = (
    TokenMetric(
        MetricSeriesKey.ISL.value,
        "input_sequence_lengths",
        "Input sequence lengths",
        "tokens",
        1.0,
    ),
    TokenMetric(
        MetricSeriesKey.OSL.value,
        "output_sequence_lengths",
        "Output sequence lengths",
        "tokens",
        1.0,
    ),
    TokenMetric(MetricSeriesKey.TPOT_NS.value, "tpot", "TPOT", "ms", 1e-6),
)


logger = logging.getLogger(__name__)


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


def _steady_state_of(snap: dict) -> SteadyState | None:
    """Decode the steady-state verdict from a snapshot dict.

    ``final_snapshot.json`` is read with ``json.loads``, so the verdict arrives
    as plain dicts and must be converted.

    A schema mismatch is dropped rather than raised because the verdict is a
    diagnostic. Dropping it makes the absence visible. Raising would suppress
    the report, and coercing could print a plausible wrong number.
    """
    raw = snap.get("steady_state")
    if raw is None:
        return None
    try:
        return msgspec.convert(raw, type=SteadyState)
    except msgspec.ValidationError:
        logger.warning(
            "Steady-state verdict did not match the expected schema; dropping it",
            exc_info=True,
        )
        return None


class OtherBasisMetrics(msgspec.Struct, frozen=True, kw_only=True):  # type: ignore[call-arg]
    """The token-count metrics of a run's non-primary basis.

    A block is None when the basis recorded no sample. A recorded block
    carries ``"partial": True`` when the basis missed some samples: the server
    basis any sample without usable usage, the client basis any tokenization
    still pending. ``tps`` is None for a partial basis, since a throughput over
    an incomplete token total misleads. ``Report.to_json`` writes each field
    under its ``basis_key`` name.
    """

    input_sequence_lengths: dict[str, Any] | None
    output_sequence_lengths: dict[str, Any] | None
    tpot: dict[str, Any] | None
    tps: float | None

    @property
    def partial(self) -> bool:
        """True when a recorded block covers only some samples."""
        return any(
            block is not None and block.get("partial", False)
            for block in (
                self.input_sequence_lengths,
                self.output_sequence_lengths,
                self.tpot,
            )
        )


class OslPaired(msgspec.Struct, frozen=True, kw_only=True):  # type: ignore[call-arg]
    """OSL over the samples counted on both bases.

    ``mean_diff`` is the mean per-sample (client - server) difference;
    ``ratio`` is client over server tokens, None when the server total is 0.
    """

    n_samples: int
    client_tokens: int
    server_tokens: int
    mean_diff: float
    ratio: float | None


class Report(msgspec.Struct, frozen=True, kw_only=True):  # type: ignore[call-arg]
    """Summarized benchmark report."""

    version: str
    git_sha: str | None
    test_started_at: int
    n_samples_issued: int
    # Terminal responses; failed samples are a subset of this count.
    n_samples_completed: int
    n_samples_failed: int
    duration_ns: int | None
    # The terminal SessionState that produced this report. Surfaced as a
    # raw string so ``display()`` can render an INTERRUPTED indicator
    # without re-parsing the source dict, and so JSON round-trips don't
    # depend on Report importing the SessionState enum.
    state: str
    # True iff the aggregator completed, every issued sample reached a terminal
    # outcome with valid counter ordering, and the primary basis counted every
    # successful sample: on the client basis no tokenization work remains, on
    # the server basis every successful tracked sample reported usable usage.
    # The other basis falling short leaves this True and marks its blocks
    # partial. False signals dropped samples, a primary-basis shortfall, an
    # interrupted run, or a live-snapshot fallback.
    complete: bool

    # Per-metric rollup dicts (output of _series_to_metric_dict). tpot and the
    # sequence-length blocks count tokens on the token_count_source basis.
    ttft: dict[str, Any]
    tpot: dict[str, Any]
    latency: dict[str, Any]
    output_sequence_lengths: dict[str, Any]
    input_sequence_lengths: dict[str, Any]
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
    # duration; tps is also None when no OSL was recorded. tps counts tokens
    # on the token_count_source basis; other_basis.tps is the other basis
    # over the same window.
    qps: float | None = None
    tps: float | None = None
    e2e_avg_interactivity: float | None = None
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

    # OSL over EVERY completed turn, including turns issued after the
    # performance window closed. ``output_sequence_lengths`` above covers only
    # the window, whose length depends on concurrency — so it is not
    # comparable across Pareto points, while this block is. Counted on both
    # bases like the window: the client basis with the tokenizer through
    # ``extract_tokenization_input``, the server basis from each turn's usable
    # ``completion_tokens``. The primary basis takes ``output_sequence_lengths``/
    # ``n_turns_counted``/``n_empty`` (and ``n_errors`` when it is the client
    # basis); the other basis sits beside it under ``basis_key`` names, its
    # series None when it counted nothing and carrying ``"partial": True`` when
    # it fell short. ``n_undecodable``/``n_missing``/``n_missing_usage`` are
    # basis-independent. Attached at finalize; None only when the block was
    # skipped (a client-primary run without a tokenizer, or an
    # unreadable/missing performance sample map). A run that scanned turns but
    # counted none returns a block with ``n_turns_counted == 0``. ``partial``
    # follows the primary basis only: True when any turn was errored,
    # undecodable, missing (a perf UUID with no COMPLETE record), or, on the
    # server basis, lacked usable usage — the mean is then over a subset. A run
    # is invalid for the OSL accuracy gate when this is None OR
    # ``n_turns_counted == 0`` OR ``partial`` OR not complete.
    # See mlcommons/endpoints#500.
    output_sequence_lengths_full_run: dict[str, Any] | None = None

    # Steady-window verdict carried on the metrics snapshot. None when
    # collection was off or the collected series do not describe the run. Kept in
    # ``to_json`` (unlike ``accuracy``), so it reaches result_summary.json.
    steady_state: SteadyState | None = None

    # The primary basis: every unsuffixed token count above (tpot, the
    # sequence-length blocks, tps, e2e_avg_interactivity, the early-stopping
    # maps, the steady block) is on it, and only its shortfall can clear
    # ``complete``.
    token_count_source: TokenCountSource
    # Successful tracked samples whose response reported no usable usage, and
    # the subset with empty output: the server basis's shortfall.
    n_samples_missing_usage: int = 0
    n_samples_missing_usage_empty_output: int = 0
    # Tokenization tasks still pending at the final snapshot: the client
    # basis's shortfall.
    client_token_tasks_pending: int = 0
    # The token-count metrics on the other basis.
    other_basis: OtherBasisMetrics
    # None when no sample was counted on both bases.
    osl_paired: OslPaired | None = None

    @property
    def n_samples_succeeded(self) -> int:
        return max(0, self.n_samples_completed - self.n_samples_failed)

    @property
    def n_samples_dropped(self) -> int:
        return max(0, self.n_samples_issued - self.n_samples_completed)

    def incomplete_reason(self) -> str | None:
        """The cause of an incomplete report, or None when it is complete.

        The run state and dropped samples come first, then the primary basis's
        shortfall: an interrupted run or a live-snapshot fallback holds partial
        counters, so its state is the cause.
        """
        if self.complete:
            return None
        if self.state == "complete" and self.n_samples_dropped == 0:
            if self.token_count_source is TokenCountSource.CLIENT:
                return "Metrics tokenization did not finish"
            if self.n_samples_missing_usage > 0:
                return missing_usage_message(
                    self.n_samples_missing_usage,
                    self.n_samples_missing_usage_empty_output,
                    self.n_samples_succeeded,
                    primary=TokenCountSource.SERVER,
                )
        return (
            "Benchmark report is incomplete "
            f"(state={self.state}, n_samples_dropped={self.n_samples_dropped})"
        )

    def other_basis_partial_reason(self) -> str | None:
        """Why the other basis's blocks are partial, or None when none is.

        Information, not a failure: only the primary basis decides
        ``complete``.
        """
        if not self.other_basis.partial:
            return None
        primary = self.token_count_source
        if primary.other is TokenCountSource.SERVER:
            return missing_usage_message(
                self.n_samples_missing_usage,
                self.n_samples_missing_usage_empty_output,
                self.n_samples_succeeded,
                primary=primary,
            )
        return (
            f"{self.client_token_tasks_pending} client tokenization tasks were "
            "still pending; the *_client blocks are partial and the run is not "
            "affected"
        )

    @classmethod
    def from_snapshot(
        cls,
        snap: dict[str, Any],
        *,
        token_count_source: TokenCountSource,
        run_config: dict[str, Any] | None = None,
        use_legacy_loadgen_qps_metrics: bool = True,
    ) -> Report:
        """Build a Report from a snapshot dict.

        ``run_config`` (optional, keyword-only) carries the run's configuration
        (load_pattern, warmup, and the scheduler/dataloader RNG seeds) into the
        report so result_summary.json is self-describing; it is config, not part
        of the metrics snapshot.

        ``token_count_source`` is the run's primary token-count basis, which
        the snapshot does not record. Its series keep their base names and its
        shortfall alone decides ``complete``. The other basis's series, named
        by ``basis_key``, fill ``other_basis``.

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
            # count==0 handling lives in _series_to_metric_dict, which preserves
            # the all-null early_stopping_percentiles map for enabled-but-empty
            # series — short-circuiting here would silently drop it.
            return _series_to_metric_dict(stat) if stat is not None else {}

        version_info = get_version_info()
        raw_duration_ns = _counter("tracked_duration_ns")
        duration_ns = raw_duration_ns if raw_duration_ns > 0 else None
        n_issued = _counter("tracked_samples_issued")
        n_failed = _counter("tracked_samples_failed")
        n_completed = _counter("tracked_samples_completed")
        n_missing = _counter("tracked_samples_missing_usage")
        n_missing_empty = _counter("tracked_samples_missing_usage_empty_output")
        n_pending_tasks = snap.get("n_pending_tasks", 0)
        # The primary basis keeps the base series names.
        isl = _series_dict(MetricSeriesKey.ISL.value)
        osl = _series_dict(MetricSeriesKey.OSL.value)

        primary = token_count_source
        other = primary.other
        # Whether each basis missed some successful samples: the client basis
        # while tokenization is pending, the server basis for each sample
        # without usable usage.
        fell_short = {
            TokenCountSource.CLIENT: n_pending_tasks > 0,
            TokenCountSource.SERVER: n_missing > 0,
        }

        def _other_block(metric: TokenMetric) -> dict[str, Any] | None:
            stat = series.get(basis_key(metric.series, other, primary=primary))
            if stat is None or stat.get("count", 0) == 0:
                return None
            block = _series_to_metric_dict(stat)
            if fell_short[other]:
                block["partial"] = True
            return block

        other_blocks = {metric.field: _other_block(metric) for metric in TOKEN_METRICS}
        other_osl = other_blocks["output_sequence_lengths"]

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
        # fail, QPS and both bases' TPS fall back to the native window so they
        # always share one window, and legacy_loadgen_window_duration_ns stays
        # None so the serialized report honestly records which view it holds.
        use_legacy_window = (
            use_legacy_loadgen_qps_metrics
            and raw_loadgen_window_ns > 0
            and n_completed >= 2
        )
        legacy_loadgen_window_duration_ns = (
            raw_loadgen_window_ns if use_legacy_window else None
        )
        window_s: float | None
        qps: float | None
        if use_legacy_window:
            window_s = raw_loadgen_window_ns / 1e9
            qps = (n_completed - 1) / window_s
        elif duration_ns is not None:
            window_s = duration_ns / 1e9
            qps = n_completed / window_s
        else:
            window_s = qps = None
        tps = osl.get("total", 0) / window_s if osl and window_s is not None else None
        other_tps = (
            other_osl["total"] / window_s
            if other_osl is not None and not fell_short[other] and window_s is not None
            else None
        )

        n_paired = _counter("osl_paired_samples")
        osl_paired: OslPaired | None = None
        if n_paired > 0:
            paired_client = _counter("osl_paired_client_tokens")
            paired_server = _counter("osl_paired_server_tokens")
            osl_paired = OslPaired(
                n_samples=n_paired,
                client_tokens=paired_client,
                server_tokens=paired_server,
                mean_diff=(paired_client - paired_server) / n_paired,
                ratio=paired_client / paired_server if paired_server else None,
            )

        total_sample_latency_ns = series.get("sample_latency_ns", {}).get("total", 0)
        # OSL is recorded only for successful responses, while sample latency
        # includes every terminal request. Without a success-correlated latency
        # sum, any tracked failure would make the two sides cover different
        # samples, so omit the derived metric instead of publishing a biased
        # ratio.
        e2e_avg_interactivity = (
            osl.get("total", 0) / (total_sample_latency_ns / 1e9)
            if osl and total_sample_latency_ns > 0 and n_failed == 0
            else None
        )

        # Default missing state to "interrupted" — a malformed / partial
        # snapshot dict is treated as worst-case (run did not reach a
        # clean completion). Drives complete=False and the interrupted
        # indicator in display().
        state = snap.get("state", "interrupted")
        finish_reason_prefix = "tracked_finish_reason_"
        finish_reason_counts = {
            reason.value.removeprefix(finish_reason_prefix): int(
                counters.get(reason.value, 0)
            )
            for reason in _FINISH_REASON_COUNTERS
        }

        accounting_valid = 0 <= n_failed <= n_completed <= n_issued
        return cls(
            version=str(version_info.get("version", "unknown")),
            git_sha=version_info.get("git_sha"),
            test_started_at=0,  # TODO: surface session_started_ns via snapshot
            n_samples_issued=n_issued,
            n_samples_completed=n_completed,
            n_samples_failed=n_failed,
            duration_ns=duration_ns,
            state=state,
            complete=(
                state == "complete"
                and accounting_valid
                and n_completed == n_issued
                and not fell_short[primary]
            ),
            **{
                field: _series_dict(series)
                for series, field in SERIES_TO_SUMMARY_FIELD.items()
            },
            input_sequence_lengths=isl,
            output_sequence_lengths=osl,
            legacy_loadgen_window_duration_ns=legacy_loadgen_window_duration_ns,
            qps=qps,
            tps=tps,
            e2e_avg_interactivity=e2e_avg_interactivity,
            finish_reason_counts=finish_reason_counts,
            run_config=run_config,
            steady_state=_steady_state_of(snap),
            token_count_source=primary,
            n_samples_missing_usage=n_missing,
            n_samples_missing_usage_empty_output=n_missing_empty,
            client_token_tasks_pending=n_pending_tasks,
            other_basis=OtherBasisMetrics(**other_blocks, tps=other_tps),
            osl_paired=osl_paired,
        )

    def to_json(self, save_to: os.PathLike | None = None) -> bytes:
        # result_summary.json is the performance report — accuracy lives only in
        # the dedicated accuracy report (accuracy/accuracy_results.json). The
        # accuracy field stays on the struct so report.txt / the console summary
        # still render it; it is just dropped from this serialized perf summary.
        payload = msgspec.structs.asdict(self)
        # The other basis's fields go flat under their ``basis_key`` names.
        primary = self.token_count_source
        for name, value in msgspec.structs.asdict(payload.pop("other_basis")).items():
            payload[basis_key(name, primary.other, primary=primary)] = value
        payload["n_samples_succeeded"] = self.n_samples_succeeded
        payload["n_samples_dropped"] = self.n_samples_dropped
        payload.pop("accuracy")
        json_bytes = msgspec.json.format(msgspec.json.encode(payload), indent=2)
        if save_to is not None:
            with Path(save_to).open("wb") as f:
                f.write(json_bytes)
        return json_bytes

    def _display_steady_state(self, fn: Callable[[str], None], newline: str) -> None:
        """Render the actionable steady-window summary, if there is a verdict.

        The full verdict stays in result_summary.json: histograms, CoV table,
        and trend detail. The terminal display prints only the numbers a reader
        acts on.
        """
        ss = self.steady_state
        if ss is None:
            return
        if not ss.found:
            fn(f"Steady state: not found ({ss.reason}){newline}")
            return
        window, tps = ss.window, ss.tps
        if window is None or tps is None:
            return
        held_s = (window.end_ns - window.start_ns) / 1e9
        fn(
            f"Steady state: super-passes {window.sp_lo}..."
            f"{window.sp_hi - 1} (post-warmup), "
            f"{window.n_samples} samples over {held_s:.0f}s{newline}"
        )
        fn(
            f"  Steady TPS: {tps.system:.2f} system, "
            f"{tps.per_user:.2f} per user{newline}"
        )
        if ss.drifting_up:
            fn(
                f"  WARNING: {', '.join(ss.drifting_up)} drifting up over the "
                f"rest of the run{newline}"
            )
        if ss.anomaly.detected:
            fn(
                f"  ANOMALY: level shift at super-pass "
                f"{ss.anomaly.change_point_sp}, TPOT "
                f"{ss.anomaly.delta_pct:+.1f}% toward end of run"
                f"{newline}"
            )

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
            # A client-primary report keeps one fixed wording for report.txt
            # readers; a server-primary one names the cause.
            if self.token_count_source is TokenCountSource.CLIENT:
                warning = (
                    "report is incomplete (drain timed out or no final snapshot "
                    "received) — some async metrics may be missing."
                )
            else:
                warning = f"report is incomplete: {self.incomplete_reason()}"
            fn(f"WARNING: {warning}{newline}")
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
        fn(f"Total samples succeeded: {self.n_samples_succeeded}{newline}")
        fn(f"Total samples failed: {self.n_samples_failed}{newline}")
        fn(f"Total samples dropped: {self.n_samples_dropped}{newline}")
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

        if (interactivity := self.e2e_avg_interactivity) is not None:
            fn(f"E2E average interactivity: {interactivity:.2f} tokens/s{newline}")
        else:
            fn(f"E2E average interactivity: N/A{newline}")

        self._display_steady_state(fn, newline)

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
            avg = samples_weighted_average_accuracy(self.accuracy)
            if avg is not None:
                fn(f"  Average: {avg:.4g}{newline}")
            if any("osl_tokenize_s" in e for e in self.accuracy):
                osl_tok = sum(e.get("osl_tokenize_s", 0.0) for e in self.accuracy)
                fn(f"  OSL tokenization: {osl_tok:.3g}s{newline}")

        fr = self.output_sequence_lengths_full_run
        n_counted = (fr or {}).get("n_turns_counted", 0)
        fr_counts = ""
        if fr:
            fr_counts = (
                f"({fr.get('n_empty', 0)} empty, {fr.get('n_errors', 0)} errored, "
                f"{fr.get('n_undecodable', 0)} undecodable, "
                f"{fr.get('n_missing', 0)} missing"
            )
            # Missing usage decides the block only on a server primary; a
            # client-primary line reads as a tokenizer-only run's.
            if self.token_count_source is TokenCountSource.SERVER:
                fr_counts += f", {fr.get('n_missing_usage', 0)} missing usage"
            fr_counts += ")"
        if fr and n_counted > 0:
            osl = fr.get("output_sequence_lengths", {})
            fn(
                f"  OSL per-turn mean (accuracy, all turns): "
                f"{osl.get('avg', 0):.1f} tokens over {n_counted} turns "
                f"{fr_counts}{newline}"
            )
            if fr.get("partial"):
                fn(
                    f"    (PARTIAL — dropped/missing turns; "
                    f"NOT valid for the OSL accuracy gate){newline}"
                )
            if not self.complete:
                fn(
                    f"    (run incomplete — NOT valid for the OSL accuracy gate){newline}"
                )
        elif fr:
            # Block present but no countable turns — do not print a fake 0.0 mean.
            fn(
                f"  OSL per-turn mean (accuracy): no countable turns "
                f"{fr_counts}{newline}"
            )
        elif self.output_sequence_lengths and self.complete:
            # A complete performance run (windowed OSL present) but no full-run
            # block: the accuracy OSL field is missing, so the run is not usable
            # for that gate.
            # Only a client-primary run needs a tokenizer for the block; the
            # block is also skipped without an event log or performance map.
            cause = (
                "no tokenizer or unreadable sample map"
                if self.token_count_source is TokenCountSource.CLIENT
                else "no event log or unreadable sample map"
            )
            fn(f"  OSL per-turn mean (accuracy): not computed ({cause}){newline}")
        elif not self.complete:
            fn(
                f"  OSL per-turn mean (accuracy): not computed — run incomplete{newline}"
            )

        # The other basis follows the primary's summary rows, which stay
        # contiguous for readers that expect them in a fixed order.
        primary = self.token_count_source
        other = primary.other
        fn(f"Other token basis ({other.value}):{newline}")
        fn(f"  Token count source: {primary.value}{newline}")
        if (other_tps := self.other_basis.tps) is not None:
            fn(f"  TPS ({other.value}): {other_tps:.2f}{newline}")
        if (paired := self.osl_paired) is not None:
            ratio_str = f"{paired.ratio:.4f}" if paired.ratio is not None else "N/A"
            fn(
                f"  OSL paired (client - server): mean {paired.mean_diff:.2f} "
                f"tokens/sample, ratio {ratio_str} over {paired.n_samples} "
                f"samples{newline}"
            )
        if (partial_reason := self.other_basis_partial_reason()) is not None:
            fn(f"  {partial_reason}{newline}")

        if summary_only:
            fn(f"----------------- End of Summary -----------------{newline}")
            return

        fn(f"\n------------------- Latency Breakdowns -------------------{newline}")

        token_sections = {
            metric.field: (
                metric.label,
                getattr(self, metric.field),
                metric.unit,
                metric.scale,
            )
            for metric in TOKEN_METRICS
        }
        sections: list[tuple[str, dict[str, Any], str, float]] = [
            ("TTFT", self.ttft, "ms", 1e-6),
            token_sections["tpot"],
            ("Latency", self.latency, "ms", 1e-6),
            token_sections["input_sequence_lengths"],
            token_sections["output_sequence_lengths"],
            (
                "Output sequence lengths (full run, all turns)",
                (self.output_sequence_lengths_full_run or {}).get(
                    "output_sequence_lengths", {}
                ),
                "tokens",
                1.0,
            ),
        ]
        for metric in TOKEN_METRICS:
            block = getattr(self.other_basis, metric.field)
            if block is not None:
                sections.append(
                    (
                        _other_basis_title(metric.label, other, block),
                        block,
                        metric.unit,
                        metric.scale,
                    )
                )
        full_run_other = (self.output_sequence_lengths_full_run or {}).get(
            basis_key("output_sequence_lengths", other, primary=primary)
        ) or {}
        sections.append(
            (
                _other_basis_title(
                    "Output sequence lengths (full run, all turns)",
                    other,
                    full_run_other,
                ),
                full_run_other,
                "tokens",
                1.0,
            )
        )

        for section_name, metric_dict, unit, scale_factor in sections:
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


def _other_basis_title(
    label: str, other: TokenCountSource, block: dict[str, Any]
) -> str:
    """An other-basis section title, marked when its block misses some samples."""
    title = f"{label} ({other.value})"
    return f"{title} (partial)" if block.get("partial") else title


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
