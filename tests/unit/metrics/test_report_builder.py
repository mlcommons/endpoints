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
from pathlib import Path

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
    SessionState,
)
from inference_endpoint.metrics.report import Report

# 1 hour in ns — same as the aggregator's default bound for time-series.
_NS_HIGH = 3_600_000_000_000


def _make_registry(n_samples: int = 50) -> MetricsRegistry:
    """A registry populated with the metrics ``Report.from_snapshot`` reads.

    Only the metrics consumed by ``Report.from_snapshot`` are registered:
    the tracked counters (issued/completed/failed/duration) and the four
    series surfaced on the report (ttft_ns, sample_latency_ns, osl,
    tpot_ns). ISL/chunk_delta_ns are intentionally not registered to
    keep the test data minimal — ``Report.from_snapshot`` ignores them.
    """
    registry = MetricsRegistry()
    for key in MetricCounterKey:
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


def _build_report(
    registry: MetricsRegistry,
    *,
    state: SessionState = SessionState.COMPLETE,
    n_pending_tasks: int = 0,
) -> Report:
    """Build a Report from a snapshot of ``registry`` at ``state``."""
    snap = registry.build_snapshot(state=state, n_pending_tasks=n_pending_tasks)
    return Report.from_snapshot(snap)


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
        assert report.qps() is None
        # Series with count==0 should produce empty dicts.
        assert report.ttft == {}
        assert report.latency == {}
        assert report.output_sequence_lengths == {}
        assert report.tpot == {}

    def test_with_metrics(self):
        registry = _make_registry(n_samples=50)
        report = _build_report(registry)

        assert report.n_samples_issued == 50
        assert report.n_samples_completed == 50
        assert report.duration_ns == 10_000_000_000
        assert report.qps() == pytest.approx(5.0)

        assert "min" in report.ttft
        assert "percentiles" in report.ttft
        assert "histogram" in report.ttft
        assert report.ttft["min"] > 0
        assert report.latency["min"] > 0
        # No TPOT recordings in the registry → empty dict.
        assert report.tpot == {}
        # OSL data was written → tps() is computable.
        assert report.tps() is not None

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
        assert "End of Summary" in output

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

    def test_to_json_save(self, tmp_path: Path):
        registry = _make_registry(n_samples=5)
        report = _build_report(registry)

        out_path = tmp_path / "report.json"
        report.to_json(save_to=out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_bytes())
        assert data["n_samples_completed"] == 5

    def test_qps_none_without_duration(self):
        report = Report(
            version="test",
            git_sha=None,
            test_started_at=0,
            n_samples_issued=100,
            n_samples_completed=100,
            n_samples_failed=0,
            duration_ns=None,
            complete=True,
            ttft={},
            tpot={},
            latency={},
            output_sequence_lengths={},
        )
        assert report.qps() is None
        assert report.tps() is None

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


# ---------------------------------------------------------------------------
# Direct snapshot construction (no registry) — explicit wire shape coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromSnapshotDirect:
    def test_minimal_snapshot_yields_empty_report(self):
        """A snapshot with no metrics produces a Report whose counters are 0
        and whose series dicts are empty. ``duration_ns`` is None because
        ``tracked_duration_ns`` is missing.
        """
        snap = MetricsSnapshot(
            counter=1,
            timestamp_ns=0,
            state=SessionState.COMPLETE,
            n_pending_tasks=0,
            metrics=[],
        )
        report = Report.from_snapshot(snap)
        assert report.n_samples_issued == 0
        assert report.n_samples_completed == 0
        assert report.n_samples_failed == 0
        assert report.duration_ns is None
        assert report.complete is True
        assert report.ttft == {}
