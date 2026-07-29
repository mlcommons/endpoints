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

"""Tests for ``Report.from_snapshot`` and display helpers.

Reports are built from a ``MetricsSnapshot`` produced by a populated
``MetricsRegistry``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import msgspec.structs
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricCounterKey,
)
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricSeriesKey,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshot,
    SeriesStat,
    SessionState,
    snapshot_to_dict,
)
from inference_endpoint.metrics.report import Report, series_metric_dict

# 1 hour in ns — same as the aggregator's default bound for time-series.
_NS_HIGH = 3_600_000_000_000


@pytest.mark.unit
class TestSeriesMetricDict:
    """Direct coverage for the accuracy-OSL rollup builder (perf-shape parity)."""

    _PERF_KEYS = {
        "avg",
        "min",
        "max",
        "median",
        "std_dev",
        "total",
        "percentiles",
        "histogram",
    }

    def test_empty_returns_empty(self):
        assert series_metric_dict([]) == {}

    def test_keys_match_perf_block(self):
        # Same key set as the perf report's output_sequence_lengths.
        assert self._PERF_KEYS <= set(series_metric_dict([2, 4, 6]))

    def test_basic_stats(self):
        d = series_metric_dict([2, 4, 6])
        assert d["avg"] == 4.0
        assert d["min"] == 2
        assert d["max"] == 6
        assert d["total"] == 12
        assert d["median"] == 4

    def test_single_value(self):
        d = series_metric_dict([7])
        assert d["avg"] == 7.0
        assert d["min"] == d["max"] == 7
        assert d["std_dev"] == 0.0
        assert d["median"] == 7

    def test_all_equal_values(self):
        # min == max must not degenerate the log-spaced histogram edges.
        d = series_metric_dict([3, 3, 3])
        assert d["min"] == d["max"] == 3
        assert d["avg"] == 3.0
        assert d["std_dev"] == 0.0


def _make_registry(n_samples: int = 50) -> MetricsRegistry:
    """A registry populated with the metrics ``Report.from_snapshot`` reads.

    Only the metrics consumed by ``Report.from_snapshot`` are registered:
    the tracked counters (issued/completed/failed/duration) and the four
    series surfaced on the report (ttft_ns, sample_latency_ns, osl,
    tpot_ns). ISL/chunk_delta_ns are intentionally not registered to
    keep the test data minimal — ``Report.from_snapshot`` ignores them.
    """
    registry = MetricsRegistry()
    for key in MetricCounterKey.__members__.values():
        registry.register_counter(key.value)
    registry.register_series(
        MetricSeriesKey.SAMPLE_LATENCY_NS.value,
        hdr_low=1,
        hdr_high=_NS_HIGH,
        sig_figs=3,
        n_histogram_buckets=10,
        percentiles=(50.0, 90.0, 99.0),
    )
    registry.register_series(
        MetricSeriesKey.TTFT_NS.value,
        hdr_low=1,
        hdr_high=_NS_HIGH,
        sig_figs=3,
        n_histogram_buckets=10,
        percentiles=(50.0, 90.0, 99.0),
    )
    registry.register_series(
        MetricSeriesKey.OSL.value,
        hdr_low=1,
        hdr_high=10_000_000,
        sig_figs=3,
        n_histogram_buckets=10,
        percentiles=(50.0, 90.0, 99.0),
    )
    registry.register_series(
        MetricSeriesKey.TPOT_NS.value,
        hdr_low=1,
        hdr_high=_NS_HIGH,
        sig_figs=3,
        n_histogram_buckets=10,
        percentiles=(50.0, 90.0, 99.0),
        dtype=float,
    )

    if n_samples > 0:
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_ISSUED.value, n_samples)
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_COMPLETED.value, n_samples)
        registry.set_counter(MetricCounterKey.TRACKED_DURATION_NS.value, 10_000_000_000)
        for i in range(n_samples):
            registry.record(MetricSeriesKey.TTFT_NS.value, 1_000_000 + i * 10_000)
            registry.record(
                MetricSeriesKey.SAMPLE_LATENCY_NS.value, 5_000_000 + i * 50_000
            )
            registry.record(MetricSeriesKey.OSL.value, 100 + i)

    return registry


def _set_loadgen_window(registry: MetricsRegistry, *, duration_ns: int) -> None:
    """Populate the legacy LoadGen window counter a loadgen-view Report reads."""
    registry.set_counter(
        MetricCounterKey.LEGACY_LOADGEN_WINDOW_DURATION_NS.value, duration_ns
    )


def _build_report(
    registry: MetricsRegistry,
    *,
    state: SessionState = SessionState.COMPLETE,
    n_pending_tasks: int = 0,
    use_legacy_loadgen_qps_metrics: bool = True,
) -> Report:
    """Build a Report from a snapshot dict (matches the consumer contract).

    ``Report.from_snapshot`` consumes the dict form produced by
    ``snapshot_to_dict``; that's also the shape persisted to
    ``final_snapshot.json``. We deliberately route through the dict
    here so the tests exercise the same path the production consumer
    does (loaded JSON file → Report).
    """
    snap = registry.build_snapshot(state=state, n_pending_tasks=n_pending_tasks)
    return Report.from_snapshot(
        snapshot_to_dict(snap),
        use_legacy_loadgen_qps_metrics=use_legacy_loadgen_qps_metrics,
    )


# ---------------------------------------------------------------------------
# from_snapshot — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromSnapshot:
    def test_empty_registry(self):
        registry = _make_registry(n_samples=0)
        report = _build_report(registry)

        assert report.n_samples_issued == 0
        assert report.n_samples_completed == 0
        assert report.n_samples_failed == 0
        assert report.duration_ns is None
        # No duration -> from_snapshot leaves throughput unset.
        assert report.qps is None
        assert report.tps is None
        # Series with count==0 should produce empty dicts.
        assert report.ttft == {}
        assert report.latency == {}
        assert report.output_sequence_lengths == {}
        assert report.tpot == {}

    def test_with_metrics(self):
        registry = _make_registry(n_samples=50)
        # Native view (completed / tracked_duration) so QPS/TPS are computable
        # from tracked_duration_ns alone (no loadgen window counter set here).
        report = _build_report(registry, use_legacy_loadgen_qps_metrics=False)

        assert report.n_samples_issued == 50
        assert report.n_samples_completed == 50
        assert report.duration_ns == 10_000_000_000
        assert report.qps == pytest.approx(5.0)

        assert "min" in report.ttft
        assert "percentiles" in report.ttft
        assert "histogram" in report.ttft
        assert report.ttft["min"] > 0
        assert report.latency["min"] > 0
        # No TPOT recordings in the registry → empty dict.
        assert report.tpot == {}
        # OSL data was written → tps is computable.
        assert report.tps is not None

    def test_run_config_keyword_only_passthrough(self):
        """run_config is config, not a snapshot metric: None unless the caller
        supplies it, and carried verbatim into the report when it does."""
        registry = _make_registry(n_samples=5)
        snap = snapshot_to_dict(
            registry.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        assert Report.from_snapshot(snap).run_config is None
        run_config = {
            "load_pattern": {"type": "poisson", "target_qps": 14.75},
            "warmup": {"enabled": False, "warmup_random_seed": 42},
            "scheduler_random_seed": 42,
            "dataloader_random_seed": 42,
        }
        assert (
            Report.from_snapshot(snap, run_config=run_config).run_config == run_config
        )

    def test_failed_uses_tracked_counter(self):
        """``n_samples_failed`` reads from ``tracked_samples_failed``, not
        ``total_samples_failed``. The two diverge when an ERROR fires for
        an untracked sample (warmup window) — only the tracked count
        flows into the Report.
        """
        registry = _make_registry(n_samples=10)
        registry.increment(MetricCounterKey.TOTAL_SAMPLES_FAILED.value, 3)
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_FAILED.value, 1)
        report = _build_report(registry)
        assert report.n_samples_failed == 1

    def test_finish_reason_counts_include_zeros(self):
        registry = _make_registry(n_samples=2)
        registry.increment(MetricCounterKey.TRACKED_FINISH_REASON_STOP.value, 1)
        registry.increment(MetricCounterKey.TRACKED_FINISH_REASON_LENGTH.value, 1)

        report = _build_report(registry)

        expected_counts = {
            "stop": 1,
            "length": 1,
            "tool_calls": 0,
            "content_filter": 0,
            "function_call": 0,
            "other": 0,
        }
        assert report.finish_reason_counts == expected_counts
        assert json.loads(report.to_json())["finish_reason_counts"] == expected_counts

    def test_complete_flag_true_when_state_complete_and_no_pending(self):
        registry = _make_registry(n_samples=5)
        report = _build_report(registry, state=SessionState.COMPLETE, n_pending_tasks=0)
        assert report.complete is True

    def test_complete_flag_false_when_drain_timeout(self):
        """COMPLETE state but n_pending_tasks > 0 → drain timed out, report
        is partial.
        """
        registry = _make_registry(n_samples=5)
        report = _build_report(registry, state=SessionState.COMPLETE, n_pending_tasks=2)
        assert report.complete is False

    def test_complete_flag_false_when_state_live(self):
        """LIVE/DRAINING snapshots produce reports with ``complete=False``."""
        registry = _make_registry(n_samples=5)
        report = _build_report(registry, state=SessionState.LIVE, n_pending_tasks=0)
        assert report.complete is False


# ---------------------------------------------------------------------------
# Display + JSON serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReportDisplayAndSerialize:
    def test_display_summary(self):
        registry = _make_registry(n_samples=10)
        report = _build_report(registry)

        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)

        assert "Summary" in output
        assert "QPS:" in output
        assert "TPS:" in output
        assert "End of Summary" in output

    def test_from_snapshot_leaves_accuracy_empty(self):
        """Accuracy isn't in the metrics snapshot; from_snapshot leaves it []."""
        report = _build_report(_make_registry(n_samples=5))
        assert report.accuracy == []

    def test_display_tps_na_when_unmeasured(self):
        """TPS renders an explicit N/A (symmetric with QPS) with no duration/OSL."""
        report = _build_report(_make_registry(n_samples=0))
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        assert "TPS: N/A" in "\n".join(lines)

    def test_display_accuracy_section(self):
        """Each accuracy entry renders score + sample counts, plus per-subset
        breakdown lines and the incomplete marker when present."""
        report = _build_report(_make_registry(n_samples=10))
        entry = {
            "dataset_name": "gptoss",
            "score": 82.3,
            "unit_samples": 1283,
            "num_repeats": 1,
            "total_samples": 1283,
            "duration_s": 12.5,
            "complete": False,
            "response_counts": {
                "issued": 1283,
                "scored": 1280,
                "empty": 2,
                "missing": 1,
            },
            "output_sequence_lengths": {"avg": 650.4, "min": 12, "max": 4096},
            "osl_tokenize_s": 1.234,
            "breakdown": {
                "overall_accuracy": 82.3,
                "subset_scores": {"aime25": 70.0, "livecodebench": 60.0},
                "total_samples": 1283,
                "complete": False,
            },
        }
        report = msgspec.structs.replace(report, accuracy=[entry])

        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)

        assert "Accuracy:" in output
        assert (
            "gptoss: 82.3 (unit=1283, repeats=1, total=1283, duration=12.50s)" in output
        )
        assert "aime25: 70.00%" in output
        assert "(incomplete)" in output
        assert "responses: 1280/1283 scored (2 empty, 1 missing)" in output
        assert "output tokens (avg/min/max): 650.4/12/4096" in output
        # Cross-component mean (one component here, so it equals its score).
        assert "Average: 82.3" in output
        # Total accuracy-path tokenization time (summed per-entry osl_tokenize_s).
        assert "OSL tokenization: 1.23s" in output

    def test_display_full(self):
        registry = _make_registry(n_samples=10)
        report = _build_report(registry)

        lines: list[str] = []
        report.display(fn=lines.append, summary_only=False)
        output = "\n".join(lines)

        assert "Latency Breakdowns" in output
        assert "TTFT" in output
        assert "Histogram" in output
        assert "Percentiles" in output

    def test_to_json(self):
        registry = _make_registry(n_samples=5)
        report = _build_report(registry)

        data = json.loads(report.to_json())
        assert data["n_samples_completed"] == 5
        assert "ttft" in data

    def test_to_json_excludes_accuracy(self):
        """Accuracy lives only in the dedicated accuracy report; result_summary.json
        stays purely performance. The field remains on the struct so report.txt /
        the console summary still render it."""
        report = _build_report(_make_registry(n_samples=10))
        report = msgspec.structs.replace(
            report,
            accuracy=[{"dataset_name": "d", "score": 0.5, "total_samples": 10}],
        )
        data = json.loads(report.to_json())
        assert "accuracy" not in data
        # ...but the human-readable render still shows it.
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        assert any("Accuracy:" in ln for ln in lines)

    def test_to_json_serializes_qps_and_tps(self):
        """result_summary.json is self-complete: qps/tps are serialized so
        consumers don't recompute them from duration + counts."""
        report = _build_report(_make_registry(n_samples=50))
        data = json.loads(report.to_json())
        assert data["qps"] == pytest.approx(5.0)  # 50 completed / 10s
        assert data["tps"] == pytest.approx(report.tps)
        assert data["tps"] > 0  # OSL was recorded, so TPS is computable

    def test_to_json_qps_tps_null_without_duration(self):
        """No duration -> qps/tps serialize as null, not omitted or crashing."""
        data = json.loads(_build_report(_make_registry(n_samples=0)).to_json())
        assert data["qps"] is None
        assert data["tps"] is None

    def test_to_json_and_display_carry_run_config(self):
        """result_summary.json + report.txt carry the run's config so a run is
        self-describing/reproducible; absent run_config serializes as null."""
        registry = _make_registry(n_samples=5)
        snap = snapshot_to_dict(
            registry.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        run_config = {
            "load_pattern": {"type": "poisson", "target_qps": 14.75},
            "scheduler_random_seed": 42,
            "dataloader_random_seed": 42,
        }
        report = Report.from_snapshot(snap, run_config=run_config)
        assert json.loads(report.to_json())["run_config"] == run_config

        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        assert any("Run config:" in ln for ln in lines)
        assert any("load_pattern:" in ln and "poisson" in ln for ln in lines)

        # Absent run_config -> null, not omitted.
        assert json.loads(Report.from_snapshot(snap).to_json())["run_config"] is None

    def test_to_json_save(self, tmp_path: Path):
        registry = _make_registry(n_samples=5)
        report = _build_report(registry)

        out_path = tmp_path / "report.json"
        report.to_json(save_to=out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_bytes())
        assert data["n_samples_completed"] == 5

    def test_display_no_started_at(self):
        """test_started_at=0 should not display a timestamp."""
        report = Report(
            version="test",
            git_sha=None,
            test_started_at=0,
            n_samples_issued=0,
            n_samples_completed=0,
            n_samples_failed=0,
            duration_ns=None,
            state="complete",
            complete=True,
            ttft={},
            tpot={},
            latency={},
            output_sequence_lengths={},
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)
        assert "Test started at" not in output

    def test_display_warns_when_incomplete(self):
        """Reports with ``complete=False`` surface a WARNING in display()."""
        report = Report(
            version="test",
            git_sha=None,
            test_started_at=0,
            n_samples_issued=10,
            n_samples_completed=10,
            n_samples_failed=0,
            duration_ns=1_000_000_000,
            state="complete",  # drain-timeout case: complete state, n_pending>0
            complete=False,
            ttft={},
            tpot={},
            latency={},
            output_sequence_lengths={},
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)
        assert "WARNING" in output or "incomplete" in output.lower()

    def test_display_warns_when_interrupted(self):
        """Reports with ``state == "interrupted"`` surface a distinct WARNING."""
        report = Report(
            version="test",
            git_sha=None,
            test_started_at=0,
            n_samples_issued=10,
            n_samples_completed=5,
            n_samples_failed=0,
            duration_ns=1_000_000_000,
            state="interrupted",
            complete=False,
            ttft={},
            tpot={},
            latency={},
            output_sequence_lengths={},
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)
        assert "interrupted" in output.lower()
        assert "SIGTERM" in output or "signal" in output.lower()


# ---------------------------------------------------------------------------
# Direct dict construction — Report.from_snapshot accepts arbitrary dicts
# (matches the JSON-file → consumer path; defaults absorb partial input).
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromSnapshotDict:
    def test_minimal_dict_yields_empty_report(self):
        """A snapshot dict with no metrics produces a Report whose counters
        are 0 and whose series dicts are empty. ``duration_ns`` is None
        because ``tracked_duration_ns`` is missing.
        """
        snap = {
            "counter": 1,
            "timestamp_ns": 0,
            "state": "complete",
            "n_pending_tasks": 0,
            "metrics": [],
        }
        report = Report.from_snapshot(snap)
        assert report.n_samples_issued == 0
        assert report.n_samples_completed == 0
        assert report.n_samples_failed == 0
        assert report.duration_ns is None
        assert report.state == "complete"
        assert report.complete is True
        assert report.ttft == {}

    def test_empty_dict_defaults_to_interrupted_incomplete(self):
        """A dict missing every key (e.g. corrupt file, truncated read)
        produces a non-crashing Report tagged interrupted and incomplete.
        Defaults: state→interrupted, counters→0, series→empty.
        """
        report = Report.from_snapshot({})
        assert report.state == "interrupted"
        assert report.complete is False
        assert report.n_samples_issued == 0
        assert report.ttft == {}

    def test_interrupted_state_round_trips_to_report(self):
        """An INTERRUPTED snapshot dict produces a Report flagged as such."""
        snap = {
            "counter": 1,
            "timestamp_ns": 0,
            "state": "interrupted",
            "n_pending_tasks": 5,
            "metrics": [
                {"type": "counter", "name": "tracked_samples_issued", "value": 100},
                {"type": "counter", "name": "tracked_samples_completed", "value": 80},
            ],
        }
        report = Report.from_snapshot(snap)
        assert report.state == "interrupted"
        assert report.complete is False
        # Partial counters still surface through.
        assert report.n_samples_issued == 100
        assert report.n_samples_completed == 80

    def test_missing_metric_type_is_skipped_not_crashed(self):
        """A malformed metric entry (no 'type' field) is skipped rather
        than crashing the whole report build.
        """
        snap = {
            "state": "complete",
            "n_pending_tasks": 0,
            "metrics": [
                {"name": "orphan_no_type", "value": 99},  # missing 'type'
                {"type": "counter", "name": "tracked_samples_issued", "value": 5},
            ],
        }
        report = Report.from_snapshot(snap)
        assert report.n_samples_issued == 5

    def test_display_handles_scrubbed_nan_percentiles(self):
        """``_scrub_nonfinite`` maps producer-side NaN/Inf to ``None`` so the
        snapshot JSON stays strict. ``Report.display()`` is called from
        ``finalize_benchmark`` outside the report-build try/except — a
        ``None * scale_factor`` crash there takes down the whole run.

        Asserts: display() does not raise and renders an N/A indicator
        for the scrubbed values.
        """
        snap = {
            "counter": 1,
            "timestamp_ns": 0,
            "state": "complete",
            "n_pending_tasks": 0,
            "metrics": [
                {
                    "type": "counter",
                    "name": "tracked_samples_issued",
                    "value": 5,
                },
                {
                    "type": "counter",
                    "name": "tracked_samples_completed",
                    "value": 5,
                },
                {
                    "type": "counter",
                    "name": "tracked_duration_ns",
                    "value": 1_000_000_000,
                },
                {
                    "type": "series",
                    "name": "ttft_ns",
                    "count": 5,
                    "total": 5_000_000,
                    "min": 1_000_000,
                    "max": 1_500_000,
                    "sum_sq": 5_005_000_000_000,
                    # All percentile values scrubbed from NaN → None.
                    "percentiles": {"50.0": None, "90.0": None, "99.0": None},
                    "histogram": [[[1_000_000.0, 1_500_000.0], 5]],
                },
            ],
        }
        report = Report.from_snapshot(snap)

        lines: list[str] = []
        # Currently crashes with TypeError on val * scale_factor.
        report.display(fn=lines.append, summary_only=False)
        output = "\n".join(lines)
        assert "TTFT" in output
        # Scrubbed values surface as a sentinel rather than crashing.
        assert "N/A" in output


# ---------------------------------------------------------------------------
# use_legacy_loadgen_qps_metrics: legacy MLPerf LoadGen "completed" (default) vs
# endpoints' native throughput, with native fallback when the window is
# unavailable.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadgenQpsMetrics:
    def test_default_uses_loadgen_window(self):
        """Default: QPS = (completed-1)/W, TPS = tokens/W, where W is the
        legacy LoadGen window.
        """
        registry = _make_registry(n_samples=50)
        _set_loadgen_window(registry, duration_ns=8_000_000_000)
        report = _build_report(registry)
        assert report.legacy_loadgen_window_duration_ns == 8_000_000_000
        # (50 - 1) / 8 s
        assert report.qps == pytest.approx(49 / 8.0)
        total = report.output_sequence_lengths["total"]
        assert report.tps == pytest.approx(total / 8.0)

    def test_disabled_uses_native(self):
        """--no-use-legacy-loadgen-qps-metrics → native completed/duration and
        tokens/duration, ignoring the legacy LoadGen window.
        """
        registry = _make_registry(n_samples=50)
        _set_loadgen_window(registry, duration_ns=8_000_000_000)
        report = _build_report(registry, use_legacy_loadgen_qps_metrics=False)
        # Native view selected → window not recorded on the report.
        assert report.legacy_loadgen_window_duration_ns is None
        # Native: 50 / 10 s.
        assert report.qps == pytest.approx(5.0)
        total = report.output_sequence_lengths["total"]
        assert report.tps == pytest.approx(total / 10.0)

    def test_falls_back_to_native_when_window_unavailable(self):
        """loadgen with absent/zero window → native fallback (not None), so the
        default never silently drops the headline.
        """
        registry = _make_registry(n_samples=50)  # no window counter set
        report = _build_report(registry)
        assert report.legacy_loadgen_window_duration_ns is None
        assert report.qps == pytest.approx(5.0)
        total = report.output_sequence_lengths["total"]
        assert report.tps == pytest.approx(total / 10.0)

    def test_loadgen_qps_falls_back_when_completed_lt_2(self):
        """Fewer than 2 completions → native QPS (the (completed-1)/W form is
        undefined for a single sample).
        """
        registry = _make_registry(n_samples=1)
        _set_loadgen_window(registry, duration_ns=1_000_000_000)
        report = _build_report(registry)
        # Both QPS and TPS fall back to the native window (10s) — they must
        # share one window, and the legacy field must be None so the serialized
        # report does not mislabel which view it holds.
        assert report.qps == pytest.approx(0.1)
        total = report.output_sequence_lengths["total"]
        assert report.tps == pytest.approx(total / 10.0)
        assert report.legacy_loadgen_window_duration_ns is None


@pytest.mark.unit
def test_scrub_nonfinite_round_trip_yields_none():
    """End-to-end: a registry that records a non-finite series value
    produces a snapshot dict whose percentile entries are ``None`` (not
    NaN literals). Anchors the producer-side invariant the display-time
    None-guard depends on.
    """
    series = SeriesStat(
        name="ttft_ns",
        count=1,
        total=0.0,
        min=0.0,
        max=0.0,
        sum_sq=0.0,
        percentiles={
            "50.0": float("nan"),
            "90.0": float("inf"),
            "99.0": float("-inf"),
        },
        histogram=[],
    )
    snap = MetricsSnapshot(
        counter=1,
        timestamp_ns=0,
        state=SessionState.COMPLETE,
        n_pending_tasks=0,
        metrics=[series],
    )
    d = snapshot_to_dict(snap)
    perc = d["metrics"][0]["percentiles"]
    assert perc == {"50.0": None, "90.0": None, "99.0": None}
    # And the result must be strict-JSON serializable.
    json.dumps(d, allow_nan=False)
    # Sanity: original NaN was indeed non-finite.
    assert not math.isfinite(float("nan"))
