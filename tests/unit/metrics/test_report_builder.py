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
from typing import Any

import msgspec.structs
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricCounterKey,
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
from inference_endpoint.core.types import TokenCountSource, basis_key
from inference_endpoint.metrics.report import (
    OtherBasisMetrics,
    Report,
    series_metric_dict,
)
from inference_endpoint.metrics.steady_state_diagnostics import (
    Anomaly,
    SteadyState,
    SteadyWindow,
    TpsBlock,
)

from tests.test_helpers import DEFAULT_SOURCE, make_report

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


def _make_registry(n_samples: int = 50, *, isl: bool = True) -> MetricsRegistry:
    """A registry populated with the metrics ``Report.from_snapshot`` reads.

    Only the metrics consumed by ``Report.from_snapshot`` are registered:
    the tracked counters (issued/completed/failed/duration) and the five
    series surfaced on the report (ttft_ns, sample_latency_ns, isl, osl,
    tpot_ns). ``chunk_delta_ns`` is intentionally not registered to keep the
    test data minimal — ``Report.from_snapshot`` ignores it.
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
    if isl:
        registry.register_series(
            MetricSeriesKey.ISL.value,
            hdr_low=1,
            hdr_high=10_000_000,
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
            if isl:
                registry.record(MetricSeriesKey.ISL.value, 50 + i)
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
    token_count_source: TokenCountSource = DEFAULT_SOURCE,
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
        token_count_source=token_count_source,
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
        assert report.e2e_avg_interactivity is None
        # Series with count==0 should produce empty dicts.
        assert report.ttft == {}
        assert report.latency == {}
        assert report.input_sequence_lengths == {}
        assert report.output_sequence_lengths == {}
        assert report.tpot == {}

    def test_isl_series_absent(self):
        # settings.metrics_isl=false: the aggregator never registers the
        # series, so the snapshot lacks it entirely (not merely count==0).
        report = _build_report(_make_registry(n_samples=50, isl=False))

        assert report.input_sequence_lengths == {}
        assert report.output_sequence_lengths["total"] > 0

    def test_with_metrics(self):
        registry = _make_registry(n_samples=50)
        # Native view (completed / tracked_duration) so QPS/TPS are computable
        # from tracked_duration_ns alone (no loadgen window counter set here).
        report = _build_report(registry, use_legacy_loadgen_qps_metrics=False)

        assert report.n_samples_issued == 50
        assert report.n_samples_completed == 50
        assert report.duration_ns == 10_000_000_000
        assert report.qps == pytest.approx(5.0)
        assert report.e2e_avg_interactivity == pytest.approx(20_000.0)

        assert "min" in report.ttft
        assert "percentiles" in report.ttft
        assert "histogram" in report.ttft
        assert report.ttft["min"] > 0
        assert report.latency["min"] > 0
        isl = report.input_sequence_lengths
        osl = report.output_sequence_lengths
        # The fixture records distinct ranges (ISL: 50..99; OSL: 100..149).
        # Assert the full scalar rollups so this fails if the report accidentally
        # maps the OSL snapshot series into input_sequence_lengths.
        assert isl["total"] == sum(range(50, 100))
        assert isl["min"] == 50
        assert isl["max"] == 99
        assert isl["median"] == 74.0
        assert isl["avg"] == 74.5
        assert isl["std_dev"] == pytest.approx(math.sqrt(212.5))
        assert osl["total"] == sum(range(100, 150))
        assert osl["min"] == 100
        assert osl["max"] == 149
        assert osl["median"] == 124.0
        assert osl["avg"] == 124.5
        assert osl["std_dev"] == pytest.approx(math.sqrt(212.5))
        # Both series retain the same schema, but preserve their own percentile
        # values and histogram bucket edges from the snapshot.
        assert set(isl) == set(osl)
        assert isl["percentiles"] != osl["percentiles"]
        assert isl["histogram"]["buckets"] != osl["histogram"]["buckets"]
        assert sum(isl["histogram"]["counts"]) == 50
        assert sum(osl["histogram"]["counts"]) == 50
        # No TPOT recordings in the registry → empty dict.
        assert report.tpot == {}
        # OSL data was written → tps is computable.
        assert report.tps is not None

    def test_e2e_avg_interactivity_is_unavailable_with_failed_requests(self):
        """Do not divide successful-output tokens by mixed-success latency."""
        registry = _make_registry(n_samples=0)
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_ISSUED.value, 2)
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_COMPLETED.value, 2)
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_FAILED.value)
        registry.set_counter(MetricCounterKey.TRACKED_DURATION_NS.value, 10_000_000_000)
        # Both terminal requests contribute latency, but only the successful
        # response contributes output tokens.
        registry.record(MetricSeriesKey.SAMPLE_LATENCY_NS.value, 1_000_000_000)
        registry.record(MetricSeriesKey.SAMPLE_LATENCY_NS.value, 9_000_000_000)
        registry.record(MetricSeriesKey.OSL.value, 100)

        report = _build_report(registry, use_legacy_loadgen_qps_metrics=False)

        assert report.latency["total"] == 10_000_000_000
        assert report.output_sequence_lengths["total"] == 100
        assert report.e2e_avg_interactivity is None

    def test_run_config_keyword_only_passthrough(self):
        """run_config is config, not a snapshot metric: None unless the caller
        supplies it, and carried verbatim into the report when it does."""
        registry = _make_registry(n_samples=5)
        snap = snapshot_to_dict(
            registry.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        assert (
            Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE).run_config
            is None
        )
        run_config = {
            "load_pattern": {"type": "poisson", "target_qps": 14.75},
            "warmup": {"enabled": False, "warmup_random_seed": 42},
            "scheduler_random_seed": 42,
            "dataloader_random_seed": 42,
        }
        assert (
            Report.from_snapshot(
                snap, run_config=run_config, token_count_source=DEFAULT_SOURCE
            ).run_config
            == run_config
        )

    def test_additive_accounting_uses_tracked_failures_and_rejects_drops(self):
        registry = _make_registry(n_samples=10)
        registry.increment(MetricCounterKey.TOTAL_SAMPLES_FAILED.value, 3)
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_FAILED.value, 2)
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_ISSUED.value, 2)

        report = _build_report(registry, state=SessionState.COMPLETE, n_pending_tasks=0)

        assert (
            report.n_samples_succeeded,
            report.n_samples_failed,
            report.n_samples_dropped,
        ) == (8, 2, 2)
        assert report.n_samples_issued == (
            report.n_samples_succeeded
            + report.n_samples_failed
            + report.n_samples_dropped
        )
        assert report.complete is False
        payload = json.loads(report.to_json())
        assert (payload["n_samples_succeeded"], payload["n_samples_dropped"]) == (8, 2)

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

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_complete_flag_on_drain_timeout_follows_primary(self, primary):
        """COMPLETE state but n_pending_tasks > 0 → the tokenizer drain timed
        out. Only a client-primary report is then partial: on the server basis
        the pending work feeds the *_client blocks alone.
        """
        registry = _make_registry(n_samples=5)
        report = _build_report(
            registry,
            state=SessionState.COMPLETE,
            n_pending_tasks=2,
            token_count_source=primary,
        )
        assert report.complete is (primary is TokenCountSource.SERVER)

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
        assert "E2E average interactivity: 20000.00 tokens/s" in output
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
        output = "\n".join(lines)
        assert "TPS: N/A" in output
        assert "E2E average interactivity: N/A" in output

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

    def test_client_primary_sections_render_pinned_lines(self):
        """The rendering of the primary sections is pinned: order, label, unit
        and scale (TPOT is recorded in ns and shown in ms)."""
        registry = _make_registry(n_samples=3)
        for value in (2_000_000.0, 3_000_000.0, 4_000_000.0):
            registry.record(MetricSeriesKey.TPOT_NS.value, value)
        report = _build_report(registry, token_count_source=TokenCountSource.CLIENT)

        lines: list[str] = []
        report.display(fn=lines.append)
        breakdowns = lines[
            lines.index(
                "\n------------------- Latency Breakdowns -------------------"
            ) :
        ]
        assert [
            line
            for line in breakdowns
            if (line[:1].isalpha() and line.endswith(":")) or line.startswith("  Avg.:")
        ] == [
            "TTFT:",
            "  Avg.: 1.01 ms",
            "TPOT:",
            "  Avg.: 3.00 ms",
            "Latency:",
            "  Avg.: 5.05 ms",
            "Input sequence lengths:",
            "  Avg.: 51.00 tokens",
            "Output sequence lengths:",
            "  Avg.: 101.00 tokens",
        ]

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
        assert data["e2e_avg_interactivity"] == pytest.approx(20_000.0)

    def test_to_json_qps_tps_null_without_duration(self):
        """No duration -> qps/tps serialize as null, not omitted or crashing."""
        data = json.loads(_build_report(_make_registry(n_samples=0)).to_json())
        assert data["qps"] is None
        assert data["tps"] is None
        assert data["e2e_avg_interactivity"] is None

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
        report = Report.from_snapshot(
            snap, run_config=run_config, token_count_source=DEFAULT_SOURCE
        )
        assert json.loads(report.to_json())["run_config"] == run_config

        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        assert any("Run config:" in ln for ln in lines)
        assert any("load_pattern:" in ln and "poisson" in ln for ln in lines)

        # Absent run_config -> null, not omitted.
        report = Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE)
        assert json.loads(report.to_json())["run_config"] is None

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
        report = make_report(
            n_samples_issued=0, n_samples_completed=0, duration_ns=None
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)
        assert "Test started at" not in output

    def test_display_warns_when_incomplete(self):
        """Reports with ``complete=False`` surface a WARNING in display()."""
        # Drain-timeout case: complete state, n_pending>0.
        report = make_report(duration_ns=1_000_000_000, complete=False)
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)
        assert "WARNING" in output or "incomplete" in output.lower()

    def _report_with_full_run_osl(
        self, fr, *, osl_windowed=None, complete=True, primary=DEFAULT_SOURCE
    ):
        return make_report(
            duration_ns=1_000_000_000,
            state="complete" if complete else "interrupted",
            complete=complete,
            output_sequence_lengths=osl_windowed or {},
            output_sequence_lengths_full_run=fr,
            token_count_source=primary,
        )

    def test_display_full_run_osl_mean(self):
        """A block with counted turns prints the accuracy per-turn mean."""
        report = self._report_with_full_run_osl(
            {
                "output_sequence_lengths": {"avg": 412.5},
                "n_turns_counted": 1055,
                "n_empty": 3,
                "n_errors": 0,
            }
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "".join(lines)
        assert (
            "OSL per-turn mean (accuracy, all turns): 412.5 tokens over 1055 turns"
            in output
        )

    def test_display_full_run_osl_no_countable_turns(self):
        """An all-empty/all-error block prints counts, not a fabricated 0.0 mean."""
        report = self._report_with_full_run_osl(
            {
                "output_sequence_lengths": {},
                "n_turns_counted": 0,
                "n_empty": 5,
                "n_errors": 2,
            }
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "".join(lines)
        assert "no countable turns" in output
        assert "0.0 tokens" not in output

    def test_display_full_run_osl_partial_is_flagged(self):
        """A partial block (errored/missing turns) is labeled NOT valid for the gate."""
        report = self._report_with_full_run_osl(
            {
                "output_sequence_lengths": {"avg": 400.0},
                "n_turns_counted": 1,
                "n_empty": 0,
                "n_errors": 1,
                "n_undecodable": 0,
                "n_missing": 1,
                "partial": True,
            }
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "".join(lines)
        assert "1 missing" in output
        assert "PARTIAL" in output
        assert "NOT valid for the OSL accuracy gate" in output

    @pytest.mark.parametrize(
        ("primary", "expected"),
        [
            # Fixed wording that report.txt readers match on. A missing
            # tokenizer can only cost a client-primary run its block.
            (
                TokenCountSource.CLIENT,
                "  OSL per-turn mean (accuracy): not computed "
                "(no tokenizer or unreadable sample map)",
            ),
            (
                TokenCountSource.SERVER,
                "  OSL per-turn mean (accuracy): not computed "
                "(no event log or unreadable sample map)",
            ),
        ],
        ids=["client", "server"],
    )
    def test_display_full_run_osl_missing_on_complete_performance_run(
        self, primary, expected
    ):
        """A COMPLETE perf run (windowed OSL present) with no full-run block
        blames the sample map (and, client-primary, the tokenizer), not
        incompleteness."""
        report = self._report_with_full_run_osl(
            None, osl_windowed={"avg": 100.0}, primary=primary
        )
        lines = _display_lines(report, summary_only=True)
        assert expected in lines
        assert not any("run incomplete" in line for line in lines)

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_display_full_run_osl_incomplete_run_with_windowed_osl(self, primary):
        """An INTERRUPTED perf run (windowed OSL present, no block) must report
        incompleteness — not a false tokenizer/sample-map diagnosis."""
        report = self._report_with_full_run_osl(
            None, osl_windowed={"avg": 100.0}, complete=False, primary=primary
        )
        output = "".join(_display_lines(report, summary_only=True))
        assert "run incomplete" in output
        assert "unreadable sample map" not in output

    @pytest.mark.parametrize(
        ("primary", "counts"),
        [
            # Missing usage does not decide a client-primary block, so its
            # counts line omits it.
            (
                TokenCountSource.CLIENT,
                "(1 empty, 0 errored, 0 undecodable, 0 missing)",
            ),
            (
                TokenCountSource.SERVER,
                "(1 empty, 0 errored, 0 undecodable, 0 missing, 1 missing usage)",
            ),
        ],
    )
    def test_display_full_run_osl_counts_by_primary(self, primary, counts):
        """The summary lines name missing usage only on a server primary."""
        fr = {
            "output_sequence_lengths": {"avg": 5.0},
            "n_turns_counted": 2,
            "n_empty": 1,
            "n_errors": 0,
            "n_undecodable": 0,
            "n_missing": 0,
            "n_missing_usage": 1,
            "partial": False,
        }
        uncounted = {**fr, "output_sequence_lengths": {}, "n_turns_counted": 0}
        lines: list[str] = []
        for block in (fr, uncounted):
            self._report_with_full_run_osl(block, primary=primary).display(
                fn=lines.append, summary_only=True
            )
        assert [line for line in lines if "OSL per-turn mean" in line] == [
            f"  OSL per-turn mean (accuracy, all turns): 5.0 tokens over 2 turns "
            f"{counts}",
            f"  OSL per-turn mean (accuracy): no countable turns {counts}",
        ]

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    @pytest.mark.parametrize("other_block", [None, {"avg": 3.0}])
    def test_display_full_run_osl_by_basis(self, primary, other_block):
        """The other basis's full-run breakdown is shown from its suffixed key."""
        fr = {
            "output_sequence_lengths": {"avg": 5.0},
            "n_turns_counted": 2,
            "n_empty": 1,
            "n_errors": 0,
            "n_undecodable": 0,
            "n_missing": 0,
            "n_missing_usage": 1,
            basis_key(
                "output_sequence_lengths", primary.other, primary=primary
            ): other_block,
        }
        lines: list[str] = []
        self._report_with_full_run_osl(fr, primary=primary).display(fn=lines.append)
        out = "".join(lines)
        label = (
            f"Output sequence lengths (full run, all turns) ({primary.other.value}):"
        )
        assert (label in out) is (other_block is not None)
        assert other_block is None or "  Avg.: 3.00 tokens" in out

    def test_display_full_run_osl_no_usage_at_all(self):
        """A server block whose turns all lacked usage shows the count, not a mean."""
        lines: list[str] = []
        self._report_with_full_run_osl(
            {
                "output_sequence_lengths": {},
                "n_turns_counted": 0,
                "n_missing_usage": 4,
                "partial": True,
            },
            primary=TokenCountSource.SERVER,
        ).display(fn=lines.append, summary_only=True)
        assert (
            "no countable turns "
            "(0 empty, 0 errored, 0 undecodable, 0 missing, 4 missing usage)"
        ) in "".join(lines)

    def test_display_warns_when_interrupted(self):
        """Reports with ``state == "interrupted"`` surface a distinct WARNING."""
        report = make_report(
            n_samples_completed=5,
            duration_ns=1_000_000_000,
            state="interrupted",
            complete=False,
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
        report = Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE)
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
        report = Report.from_snapshot({}, token_count_source=DEFAULT_SOURCE)
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
        report = Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE)
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
        report = Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE)
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
        report = Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE)

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


@pytest.mark.unit
class TestSteadyStateOnTheReport:
    """The verdict rides the snapshot into result_summary.json and report.txt."""

    @staticmethod
    def _verdict(**over) -> SteadyState:
        """A complete verdict; ``over`` replaces individual fields."""
        base = SteadyState(
            found=True,
            reason=None,
            superpass_size=500,
            n_super_passes=12,
            warmup=1,
            window=SteadyWindow(
                sp_lo=1,
                sp_hi=9,
                n_super_passes=8,
                n_samples=4000,
                start_ns=0,
                end_ns=700_000_000_000,
                plateau_index=0,
                n_plateaus=1,
                skipped_short=0,
            ),
            ttft=None,
            tpot=None,
            osl=None,
            latency=None,
            tps=TpsBlock(
                per_user=42.0,
                per_user_ci=[41.0, 43.0],
                system=1234.5,
                system_ci=[1200.0, 1270.0],
            ),
            cov={},
            cov_basis=None,
            anomaly=Anomaly(
                detected=True,
                change_point_sp=7,
                delta_pct=12.5,
                pettitt=None,
                plateaus=[[1, 9]],
            ),
            short_window=None,
            global_trend={},
            drifting_up=["ttft_p50"],
        )
        return msgspec.structs.replace(base, **over) if over else base

    def _report(self, verdict):
        registry = _make_registry(n_samples=5)
        snap = snapshot_to_dict(
            registry.build_snapshot(
                state=SessionState.COMPLETE, n_pending_tasks=0, steady_state=verdict
            )
        )
        return Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE)

    def test_absent_by_default(self):
        assert self._report(None).steady_state is None

    def test_read_from_the_snapshot_and_kept_in_to_json(self):
        verdict = self._verdict()
        report = self._report(verdict)
        assert report.steady_state == verdict
        assert json.loads(report.to_json())["steady_state"] == msgspec.to_builtins(
            verdict
        )

    def test_a_scrubbed_verdict_is_dropped_rather_than_rendered_wrong(self):
        """``snapshot_to_dict`` scrubs non-finite floats to ``None``.

        A ``None`` where the schema promises a number means the verdict no
        longer describes anything, so it is dropped and the run keeps its
        report. Coercing instead would print a plausible wrong number --
        ``cov()`` divides by a mean that can be zero, so this is reachable.
        """
        raw = msgspec.to_builtins(self._verdict())
        raw["tps"]["system"] = None  # what _scrub_deep leaves behind
        registry = _make_registry(n_samples=5)
        snap = snapshot_to_dict(
            registry.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        snap["steady_state"] = raw
        report = Report.from_snapshot(snap, token_count_source=DEFAULT_SOURCE)
        assert report.steady_state is None
        lines: list[str] = []
        report.display(lines.append)  # must not raise
        assert "Steady state" not in "".join(lines)

    def test_display_renders_the_headline(self):
        lines: list[str] = []
        self._report(self._verdict()).display(lines.append)
        text = "".join(lines)
        assert "Steady state: super-passes 1...8 (post-warmup), 4000 samples" in text
        assert "over 700s" in text
        assert "Steady TPS: 1234.50 system, 42.00 per user" in text
        assert "ttft_p50 drifting up" in text
        assert "ANOMALY: level shift at super-pass 7, TPOT +12.5%" in text

    def test_display_reports_a_missing_window(self):
        lines: list[str] = []
        verdict = self._verdict(
            found=False, reason="no admissible steady plateau", window=None, tps=None
        )
        self._report(verdict).display(lines.append)
        assert "Steady state: not found (no admissible steady plateau)" in "".join(
            lines
        )


# ---------------------------------------------------------------------------
# Token-count bases: the primary keeps the base names, the other is suffixed
# ---------------------------------------------------------------------------

# Snapshot inputs that leave one basis short of some samples: the server basis
# through samples without usable usage, the client basis through tokenization
# still pending at the final snapshot.
_SHORTFALL: dict[TokenCountSource, dict[str, int]] = {
    TokenCountSource.SERVER: {"n_missing": 2, "n_missing_empty": 1},
    TokenCountSource.CLIENT: {"n_pending": 3},
}
_TOKEN_SERIES = ("isl", "osl", "tpot_ns")
_SERVER_USAGE_COUNTERS = frozenset(
    key.value
    for key in (
        MetricCounterKey.TRACKED_SAMPLES_MISSING_USAGE,
        MetricCounterKey.TRACKED_SAMPLES_MISSING_USAGE_EMPTY_OUTPUT,
        MetricCounterKey.OSL_PAIRED_SAMPLES,
        MetricCounterKey.OSL_PAIRED_CLIENT_TOKENS,
        MetricCounterKey.OSL_PAIRED_SERVER_TOKENS,
    )
)
# Report fields that describe server usage, which a snapshot without server
# series or usage counters cannot fill.
_SERVER_USAGE_FIELDS = frozenset(
    {
        "other_basis",
        "osl_paired",
        "n_samples_missing_usage",
        "n_samples_missing_usage_empty_output",
    }
)
_NO_OTHER_BASIS = OtherBasisMetrics(
    input_sequence_lengths=None, output_sequence_lengths=None, tpot=None, tps=None
)


def _basis_snapshot(
    primary: TokenCountSource,
    *,
    other_series: bool = True,
    other_records: bool = True,
    n_pending: int = 0,
    n_missing: int = 0,
    n_missing_empty: int = 0,
    paired: tuple[int, int, int] = (0, 0, 0),
    loadgen_window_ns: int = 0,
    state: SessionState = SessionState.COMPLETE,
) -> dict[str, Any]:
    """A 5-sample snapshot dict with the other basis's series beside the primary's.

    The primary basis records ISL 50..54 and OSL 100..104 (total 510) over a
    10 s tracked window. The other basis records ISL 40..44 (total 210), OSL
    90..94 (total 460) and a 2 ms TPOT per sample.
    """
    registry = _make_registry(n_samples=5)
    if other_series:
        isl, osl, tpot = (
            basis_key(base, primary.other, primary=primary) for base in _TOKEN_SERIES
        )
        for name, dtype in ((isl, int), (osl, int), (tpot, float)):
            registry.register_series(
                name,
                hdr_low=1,
                hdr_high=_NS_HIGH,
                n_histogram_buckets=10,
                percentiles=(50.0, 90.0, 99.0),
                dtype=dtype,
            )
        for i in range(5 if other_records else 0):
            registry.record(isl, 40 + i)
            registry.record(osl, 90 + i)
            registry.record(tpot, 2_000_000.0)
    for key, value in (
        (MetricCounterKey.TRACKED_SAMPLES_MISSING_USAGE, n_missing),
        (MetricCounterKey.TRACKED_SAMPLES_MISSING_USAGE_EMPTY_OUTPUT, n_missing_empty),
        (MetricCounterKey.OSL_PAIRED_SAMPLES, paired[0]),
        (MetricCounterKey.OSL_PAIRED_CLIENT_TOKENS, paired[1]),
        (MetricCounterKey.OSL_PAIRED_SERVER_TOKENS, paired[2]),
    ):
        registry.set_counter(key.value, value)
    if loadgen_window_ns:
        _set_loadgen_window(registry, duration_ns=loadgen_window_ns)
    return snapshot_to_dict(
        registry.build_snapshot(state=state, n_pending_tasks=n_pending)
    )


def _basis_report(primary: TokenCountSource, **snapshot_over: Any) -> Report:
    return Report.from_snapshot(
        _basis_snapshot(primary, **snapshot_over), token_count_source=primary
    )


def _display_lines(report: Report, **display_over: Any) -> list[str]:
    lines: list[str] = []
    report.display(fn=lines.append, **display_over)
    return lines


def _with_full_run(report: Report, *, other_partial: bool) -> Report:
    """``report`` with a full-run OSL block on both bases.

    The primary basis counts 2 turns (mean 6.0) and one empty turn; the other
    basis counts 1 turn, marked partial when ``other_partial``.
    """
    primary = report.token_count_source
    other_series = series_metric_dict([5])
    if other_partial:
        other_series["partial"] = True
    block = {
        "output_sequence_lengths": series_metric_dict([3, 9]),
        "n_turns_counted": 2,
        "n_empty": 1,
        "n_errors": 0,
        "n_undecodable": 0,
        "n_missing": 0,
        "partial": False,
        "n_missing_usage": 0,
        basis_key("output_sequence_lengths", primary.other, primary=primary): (
            other_series
        ),
    }
    return msgspec.structs.replace(report, output_sequence_lengths_full_run=block)


@pytest.mark.unit
class TestTokenCountSourceRoles:
    @pytest.mark.parametrize(
        ("state", "n_pending"),
        [
            (SessionState.COMPLETE, 0),
            (SessionState.COMPLETE, 3),
            (SessionState.INTERRUPTED, 0),
        ],
    )
    def test_client_primary_unsuffixed_fields_ignore_server_series(
        self, state, n_pending
    ):
        """S+ adds the *_server series and usage counters to S; under the client
        basis that changes only the fields describing server usage, and the
        display only gains lines."""
        client = TokenCountSource.CLIENT
        s_plus = _basis_snapshot(
            client,
            n_missing=2,
            n_missing_empty=1,
            paired=(4, 380, 400),
            n_pending=n_pending,
            state=state,
        )
        server_only = _SERVER_USAGE_COUNTERS | {
            basis_key(base, TokenCountSource.SERVER, primary=client)
            for base in _TOKEN_SERIES
        }
        s = {
            **s_plus,
            "metrics": [m for m in s_plus["metrics"] if m["name"] not in server_only],
        }
        base = Report.from_snapshot(s, token_count_source=client)
        plus = Report.from_snapshot(s_plus, token_count_source=client)

        fields = [f for f in Report.__struct_fields__ if f not in _SERVER_USAGE_FIELDS]
        assert {f: getattr(plus, f) for f in fields} == {
            f: getattr(base, f) for f in fields
        }
        assert plus.incomplete_reason() == base.incomplete_reason()
        assert base.other_basis == _NO_OTHER_BASIS != plus.other_basis
        # Every line of base's display appears in plus's, in the same order.
        plus_lines = iter(_display_lines(plus))
        assert all(line in plus_lines for line in _display_lines(base))

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_other_basis_json_keys_follow_naming_rule(self, primary):
        data = json.loads(_basis_report(primary).to_json())
        other = primary.other
        for field in (
            "input_sequence_lengths",
            "output_sequence_lengths",
            "tpot",
            "tps",
        ):
            assert basis_key(field, other, primary=primary) in data, field
        assert [key for key in data if key.endswith(f"_{primary.value}")] == []
        assert "other_basis" not in data
        assert (
            data[basis_key("output_sequence_lengths", other, primary=primary)]["total"]
            == 460
        )
        assert data[basis_key("tps", other, primary=primary)] == pytest.approx(46.0)
        assert data["token_count_source"] == primary.value
        assert {
            "n_samples_missing_usage",
            "n_samples_missing_usage_empty_output",
            "client_token_tasks_pending",
            "osl_paired",
        } <= data.keys()

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    @pytest.mark.parametrize(
        ("window_ns", "tps", "other_tps"),
        [(0, 51.0, 46.0), (5_000_000_000, 102.0, 92.0)],
    )
    def test_other_basis_tps_shares_the_tps_window(
        self, primary, window_ns, tps, other_tps
    ):
        r = _basis_report(primary, loadgen_window_ns=window_ns)
        other = r.other_basis
        assert r.tps == pytest.approx(tps)
        assert other.tps == pytest.approx(other_tps)
        assert other.input_sequence_lengths["total"] == 210
        assert other.output_sequence_lengths["total"] == 460
        assert other.tpot["min"] == pytest.approx(2_000_000.0)
        assert "partial" not in other.output_sequence_lengths
        # From the primary OSL.
        assert r.e2e_avg_interactivity == pytest.approx(20_000.0)

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_other_block_partial_inside_dict_and_tps_none(self, primary):
        r = _basis_report(primary, **_SHORTFALL[primary.other])
        other = r.other_basis
        for block in (
            other.input_sequence_lengths,
            other.output_sequence_lengths,
            other.tpot,
        ):
            assert block["partial"] is True
        assert other.tps is None
        assert "partial" not in r.output_sequence_lengths
        assert r.tps == pytest.approx(51.0)
        data = json.loads(r.to_json())
        assert (
            data[basis_key("output_sequence_lengths", primary.other, primary=primary)][
                "partial"
            ]
            is True
        )
        assert data[basis_key("tps", primary.other, primary=primary)] is None

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    @pytest.mark.parametrize("other_series", [False, True], ids=["absent", "empty"])
    def test_other_block_none_when_nothing_recorded(self, primary, other_series):
        """No recorded sample means no block, even when the basis fell short."""
        r = _basis_report(
            primary,
            other_series=other_series,
            other_records=False,
            **_SHORTFALL[primary.other],
        )
        assert r.other_basis == _NO_OTHER_BASIS
        assert r.other_basis_partial_reason() is None
        data = json.loads(r.to_json())
        for field in (
            "input_sequence_lengths",
            "output_sequence_lengths",
            "tpot",
            "tps",
        ):
            assert data[basis_key(field, primary.other, primary=primary)] is None
        label = f"({primary.other.value})"
        assert [line for line in _display_lines(r) if label in line] == [
            f"Other token basis {label}:"
        ]

    @pytest.mark.parametrize(
        ("primary", "short", "complete"),
        [
            (TokenCountSource.CLIENT, TokenCountSource.SERVER, True),
            (TokenCountSource.SERVER, TokenCountSource.CLIENT, True),
            (TokenCountSource.CLIENT, TokenCountSource.CLIENT, False),
            (TokenCountSource.SERVER, TokenCountSource.SERVER, False),
        ],
    )
    def test_complete_follows_primary_only(self, primary, short, complete):
        assert _basis_report(primary, **_SHORTFALL[short]).complete is complete

    @pytest.mark.parametrize(
        ("primary", "reason"),
        [
            (
                TokenCountSource.CLIENT,
                "2 of 5 successful tracked samples returned no server token "
                "usage; the *_server blocks are partial and the run is not "
                "affected",
            ),
            (
                TokenCountSource.SERVER,
                "3 client tokenization tasks were still pending; the *_client "
                "blocks are partial and the run is not affected",
            ),
        ],
    )
    def test_partial_other_basis_is_shown_as_information(self, primary, reason):
        r = _basis_report(primary, **_SHORTFALL[primary.other])
        assert r.other_basis_partial_reason() == reason
        out = "\n".join(_display_lines(r, summary_only=True))
        assert reason in out
        assert "WARNING" not in out

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_other_basis_sections_are_labelled_with_the_basis(self, primary):
        other = primary.other.value
        out = "\n".join(_display_lines(_basis_report(primary, paired=(4, 380, 400))))
        for text in (
            "Input sequence lengths:",
            "Output sequence lengths:",
            f"Input sequence lengths ({other}):",
            f"Output sequence lengths ({other}):",
            f"TPOT ({other}):",
            f"Token count source: {primary.value}",
            f"TPS ({other}): 46.00",
            "OSL paired (client - server): mean -5.00 tokens/sample, "
            "ratio 0.9500 over 4 samples",
        ):
            assert text in out

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    def test_other_basis_lines_follow_the_primary_summary(self, primary):
        other = primary.other.value
        report = _basis_report(
            primary, paired=(4, 380, 400), **_SHORTFALL[primary.other]
        )
        lines = _display_lines(
            _with_full_run(report, other_partial=False), summary_only=True
        )
        heading = lines.index(f"Other token basis ({other}):")
        assert lines[heading + 1 : -1] == [
            f"  Token count source: {primary.value}",
            "  OSL paired (client - server): mean -5.00 tokens/sample, "
            "ratio 0.9500 over 4 samples",
            f"  {report.other_basis_partial_reason()}",
        ]
        assert lines[heading - 1].startswith(
            "  OSL per-turn mean (accuracy, all turns)"
        )
        assert lines[-1] == "----------------- End of Summary -----------------"
        assert f"  TPS ({other}): 46.00" in _display_lines(
            _basis_report(primary), summary_only=True
        )

    def test_client_primary_summary_is_mains_up_to_the_other_basis_heading(self):
        report = msgspec.structs.replace(
            _with_full_run(
                _basis_report(
                    TokenCountSource.CLIENT,
                    paired=(4, 380, 400),
                    **_SHORTFALL[TokenCountSource.SERVER],
                ),
                other_partial=True,
            ),
            version="9.9.9",
            git_sha="abc1234",
            steady_state=TestSteadyStateOnTheReport._verdict(),
            accuracy=[
                {
                    "dataset_name": "gpqa",
                    "score": 0.5,
                    "unit_samples": 5,
                    "num_repeats": 1,
                    "total_samples": 5,
                }
            ],
        )
        lines = _display_lines(report, summary_only=True)
        heading = lines.index("Other token basis (server):")
        # Byte-identical to main's display() for this snapshot: 5 samples over
        # a 10 s window, primary OSL 100..104, latency 5.00..5.20 ms.
        assert lines[:heading] == [
            "----------------- Summary -----------------",
            "Version: 9.9.9",
            "Git SHA: abc1234",
            "Total samples issued: 5",
            "Total samples succeeded: 5",
            "Total samples failed: 0",
            "Total samples dropped: 0",
            "Duration: 10.00 seconds",
            "QPS: 0.50",
            "TPS: 51.00",
            "E2E average interactivity: 20000.00 tokens/s",
            "Steady state: super-passes 1...8 (post-warmup), 4000 samples over 700s",
            "  Steady TPS: 1234.50 system, 42.00 per user",
            "  WARNING: ttft_p50 drifting up over the rest of the run",
            "  ANOMALY: level shift at super-pass 7, TPOT +12.5% toward end of run",
            "Accuracy:",
            "  gpqa: 0.5 (unit=5, repeats=1, total=5)",
            "  Average: 0.5",
            "  OSL per-turn mean (accuracy, all turns): 6.0 tokens over 2 turns "
            "(1 empty, 0 errored, 0 undecodable, 0 missing)",
        ]

    @pytest.mark.parametrize("primary", list(TokenCountSource))
    @pytest.mark.parametrize("short", [False, True], ids=["whole", "partial"])
    def test_partial_other_basis_sections_are_marked(self, primary, short):
        other = primary.other.value
        report = _with_full_run(
            _basis_report(primary, **(_SHORTFALL[primary.other] if short else {})),
            other_partial=short,
        )
        titles = [line for line in _display_lines(report) if line.endswith(":")]
        mark = " (partial)" if short else ""
        for label in (
            "Input sequence lengths",
            "Output sequence lengths",
            "TPOT",
            "Output sequence lengths (full run, all turns)",
        ):
            assert f"{label} ({other}){mark}:" in titles
        for label in (
            "Input sequence lengths",
            "Output sequence lengths",
            "Output sequence lengths (full run, all turns)",
        ):
            assert f"{label}:" in titles
        assert sum("(partial)" in title for title in titles) == (4 if short else 0)

    def test_osl_paired_json_shape(self):
        golden = {
            "n_samples": 4,
            "client_tokens": 380,
            "server_tokens": 400,
            "mean_diff": -5.0,
            "ratio": 0.95,
        }
        paired = json.loads(
            _basis_report(DEFAULT_SOURCE, paired=(4, 380, 400)).to_json()
        )["osl_paired"]
        assert paired == golden
        assert list(paired) == list(golden)
        assert _basis_report(DEFAULT_SOURCE).osl_paired is None
        no_server_tokens = _basis_report(DEFAULT_SOURCE, paired=(2, 10, 0))
        assert json.loads(no_server_tokens.to_json())["osl_paired"]["ratio"] is None

    def test_from_snapshot_requires_basis(self):
        with pytest.raises(TypeError, match="token_count_source"):
            Report.from_snapshot(_basis_snapshot(DEFAULT_SOURCE))


_USAGE = "3 of 10 successful tracked samples returned no server token usage"
# The CLI flag, then the YAML key: `benchmark from-config` takes no such flag.
_FIX = (
    "set --token-count-source client if this server cannot report usage "
    "(YAML: settings.token_count_source: client)"
)
_MISSING = {
    "complete": False,
    "token_count_source": TokenCountSource.SERVER,
    "n_samples_missing_usage": 3,
}
# Report fields that leave each basis short when it is primary, and the cause
# ``incomplete_reason`` then names.
_PRIMARY_SHORTFALL: dict[TokenCountSource, tuple[dict[str, int], str]] = {
    TokenCountSource.CLIENT: (
        {"client_token_tasks_pending": 4},
        "Metrics tokenization did not finish",
    ),
    TokenCountSource.SERVER: ({"n_samples_missing_usage": 3}, f"{_USAGE}; {_FIX}"),
}


@pytest.mark.unit
@pytest.mark.parametrize("primary", list(TokenCountSource))
def test_incomplete_reason_names_state_then_drops_then_primary(primary):
    """State, then dropped samples, then the primary basis's shortfall."""
    shortfall, reason = _PRIMARY_SHORTFALL[primary]
    incomplete = {"complete": False, "token_count_source": primary, **shortfall}
    assert make_report(**incomplete).incomplete_reason() == reason
    assert make_report(**incomplete, n_samples_completed=8).incomplete_reason() == (
        "Benchmark report is incomplete (state=complete, n_samples_dropped=2)"
    )
    assert make_report(**incomplete, state="interrupted").incomplete_reason() == (
        "Benchmark report is incomplete (state=interrupted, n_samples_dropped=0)"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({}, None),
        (
            {"complete": False, "state": "live"},
            "Benchmark report is incomplete (state=live, n_samples_dropped=0)",
        ),
        # Every server count is whole, so there is no shortfall to name.
        (
            {"complete": False, "token_count_source": TokenCountSource.SERVER},
            "Benchmark report is incomplete (state=complete, n_samples_dropped=0)",
        ),
    ],
)
def test_incomplete_reason_without_a_primary_shortfall(over, expected):
    assert make_report(**over).incomplete_reason() == expected


@pytest.mark.unit
def test_display_names_the_cause_not_a_drain_timeout():
    lines: list[str] = []
    make_report(**_MISSING).display(fn=lines.append, summary_only=True)
    out = "\n".join(lines)
    assert f"WARNING: report is incomplete: {_USAGE}" in out
    assert "drain timed out" not in out and "tokenization did not finish" not in out


@pytest.mark.unit
@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        # Fixed wording that report.txt readers match on.
        (
            TokenCountSource.CLIENT,
            "WARNING: report is incomplete (drain timed out or no final "
            "snapshot received) — some async metrics may be missing.",
        ),
        (
            TokenCountSource.SERVER,
            f"WARNING: report is incomplete: {_USAGE}; {_FIX}",
        ),
    ],
    ids=["client", "server"],
)
def test_display_incomplete_warning_wording_by_primary(primary, expected):
    shortfall, _ = _PRIMARY_SHORTFALL[primary]
    report = make_report(
        duration_ns=1_000_000_000,
        complete=False,
        token_count_source=primary,
        **shortfall,
    )
    assert expected in _display_lines(report, summary_only=True)
