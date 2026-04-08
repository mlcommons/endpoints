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

"""Tests for report.py and report_builder.py."""

import json
from pathlib import Path

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.kv_store import (
    BasicKVStore,
    BasicKVStoreReader,
    SeriesStats,
)
from inference_endpoint.metrics.report import Report, compute_summary

# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeSummary:
    def test_empty(self):
        s = compute_summary(SeriesStats())
        assert s["total"] == 0
        assert s["min"] == 0
        assert s["histogram"]["buckets"] == []

    def test_single_value(self):
        s = compute_summary(SeriesStats([42.0], dtype=float))
        assert s["min"] == 42.0
        assert s["max"] == 42.0
        assert s["avg"] == 42.0
        assert s["std_dev"] == 0.0

    def test_multiple_values(self):
        s = compute_summary(SeriesStats([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float))
        assert s["min"] == 1.0
        assert s["max"] == 5.0
        assert s["total"] == 15.0
        assert s["avg"] == 3.0
        assert s["median"] == 3.0
        assert len(s["histogram"]["buckets"]) > 0
        assert len(s["percentiles"]) > 0

    def test_percentiles(self):
        values = list(range(1, 101))  # 1..100
        s = compute_summary(
            SeriesStats([float(v) for v in values], dtype=float),
            percentiles=(50, 90, 99),
        )
        assert s["percentiles"]["50"] == pytest.approx(50.5, abs=1)
        assert s["percentiles"]["90"] == pytest.approx(90.1, abs=1)
        assert s["percentiles"]["99"] == pytest.approx(99.01, abs=1)


# ---------------------------------------------------------------------------
# Helper: create a populated KVStore writer + reader
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, n_samples: int = 50):
    """Create a writer with typical benchmark data and return (writer, reader)."""
    store_dir = tmp_path / "kv"
    w = BasicKVStore(store_dir)

    # Counter keys matching MetricCounterKey enum
    for key in [
        "total_samples_issued",
        "total_samples_completed",
        "total_samples_failed",
        "tracked_samples_issued",
        "tracked_samples_completed",
        "tracked_duration_ns",
        "total_duration_ns",
    ]:
        w.create_key(key, "counter")
    for key in ["ttft_ns", "sample_latency_ns", "osl", "isl", "chunk_delta_ns"]:
        w.create_key(key, "series")
    w.create_key("tpot_ns", "series", dtype=float)

    w.update("tracked_samples_issued", n_samples)
    w.update("tracked_samples_completed", n_samples)
    w.update("total_samples_failed", 0)
    if n_samples > 0:
        w.update("tracked_duration_ns", 10_000_000_000)

    for i in range(n_samples):
        w.update("ttft_ns", 1_000_000 + i * 10_000)
        w.update("sample_latency_ns", 5_000_000 + i * 50_000)
        w.update("osl", 100 + i)

    r = BasicKVStoreReader(store_dir)
    for key in [
        "total_samples_issued",
        "total_samples_completed",
        "total_samples_failed",
        "tracked_samples_issued",
        "tracked_samples_completed",
        "tracked_duration_ns",
        "total_duration_ns",
    ]:
        r.register_key(key, "counter")
    for key in ["ttft_ns", "sample_latency_ns", "osl", "isl", "chunk_delta_ns"]:
        r.register_key(key, "series")
    r.register_key("tpot_ns", "series", dtype=float)

    return w, r


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildReport:
    def test_empty_store(self, tmp_path: Path):
        w, r = _make_store(tmp_path, n_samples=0)
        report = Report.from_kv_reader(r)

        assert report.n_samples_issued == 0
        assert report.duration_ns is None
        assert report.qps() is None
        assert report.ttft == {}
        assert report.latency == {}

        r.close()
        w.close()

    def test_with_metrics(self, tmp_path: Path):
        w, r = _make_store(tmp_path, n_samples=50)
        report = Report.from_kv_reader(r)

        assert report.n_samples_issued == 50
        assert report.n_samples_completed == 50
        assert report.duration_ns == 10_000_000_000
        assert report.qps() == pytest.approx(5.0)

        assert "min" in report.ttft
        assert "percentiles" in report.ttft
        assert "histogram" in report.ttft
        assert report.ttft["min"] > 0
        assert report.latency["min"] > 0
        assert report.tpot == {}  # No TPOT values written
        assert report.tps() is not None  # OSL data present

        r.close()
        w.close()


# ---------------------------------------------------------------------------
# Report display and serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReport:
    def test_display_summary(self, tmp_path: Path):
        w, r = _make_store(tmp_path, n_samples=10)
        report = Report.from_kv_reader(r)

        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)

        assert "Summary" in output
        assert "QPS:" in output
        assert "End of Summary" in output

        r.close()
        w.close()

    def test_display_full(self, tmp_path: Path):
        w, r = _make_store(tmp_path, n_samples=10)
        report = Report.from_kv_reader(r)

        lines: list[str] = []
        report.display(fn=lines.append, summary_only=False)
        output = "\n".join(lines)

        assert "Latency Breakdowns" in output
        assert "TTFT" in output
        assert "Histogram" in output
        assert "Percentiles" in output

        r.close()
        w.close()

    def test_to_json(self, tmp_path: Path):
        w, r = _make_store(tmp_path, n_samples=5)
        report = Report.from_kv_reader(r)

        data = json.loads(report.to_json())
        assert data["n_samples_completed"] == 5
        assert "ttft" in data

        r.close()
        w.close()

    def test_to_json_save(self, tmp_path: Path):
        w, r = _make_store(tmp_path, n_samples=5)
        report = Report.from_kv_reader(r)

        out_path = tmp_path / "report.json"
        report.to_json(save_to=out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_bytes())
        assert data["n_samples_completed"] == 5

        r.close()
        w.close()

    def test_qps_none_without_duration(self):
        report = Report(
            version="test",
            git_sha=None,
            test_started_at=0,
            n_samples_issued=100,
            n_samples_completed=100,
            n_samples_failed=0,
            duration_ns=None,
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
            ttft={},
            tpot={},
            latency={},
            output_sequence_lengths={},
        )
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)
        assert "Test started at" not in output
