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

"""Tests for finalize_benchmark run_metadata.json generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from inference_endpoint.commands.benchmark.execute import (
    BenchmarkContext,
    BenchmarkResult,
    ResponseCollector,
    _build_run_metadata,
    _metric_pct,
    _metric_stat,
    _ns_to_ms,
    finalize_benchmark,
)
from inference_endpoint.config.schema import OfflineBenchmarkConfig, TestMode
from inference_endpoint.load_generator.session import SessionResult
from inference_endpoint.metrics.report import Report

_OFFLINE_KWARGS = {
    "endpoint_config": {"endpoints": ["http://localhost:8000"]},
    "model_params": {"name": "test-model"},
    "datasets": [{"path": "test.jsonl"}],
}


def _make_ctx(tmp_path: Path) -> BenchmarkContext:
    config = OfflineBenchmarkConfig(**_OFFLINE_KWARGS)
    return BenchmarkContext(
        config=config,
        test_mode=TestMode.PERF,
        report_dir=tmp_path,
        tokenizer_name=None,
        dataloader=MagicMock(),
        rt_settings=MagicMock(),
        total_samples=1,
        eval_configs=[],
    )


def _make_bench(tmp_path: Path, report: Report | None = None) -> BenchmarkResult:
    t = 1_000_000_000
    session = SessionResult(
        session_id="test-session",
        phase_results=[],
        start_time_ns=t,
        end_time_ns=t + 1_000_000_000,
    )
    return BenchmarkResult(
        session=session,
        collector=ResponseCollector(),
        report=report,
        tmpfs_dir=tmp_path,
    )


# ns values per percentile bucket, chosen so ms conversion gives round numbers.
_PCT_NS = {
    "50.0": 500_000_000,
    "90.0": 900_000_000,
    "95.0": 950_000_000,
    "99.0": 990_000_000,
    "99.9": 999_000_000,
}

_SERIES_DICT = {
    "percentiles": _PCT_NS,
    "min": 100_000_000,
    "max": 1_000_000_000,
    "avg": 600_000_000,
}


def _make_populated_report() -> MagicMock:
    """Return a Report mock with all fields accessed by _build_run_metadata."""
    report = MagicMock()
    report.tps.return_value = 200.0
    report.qps.return_value = 20.0
    report.n_samples_completed = 95
    report.n_samples_issued = 100
    report.n_samples_failed = 5
    report.duration_ns = 10_000_000_000
    report.output_sequence_lengths = {"total": 2000}
    report.ttft = dict(_SERIES_DICT)
    report.tpot = dict(_SERIES_DICT)
    report.latency = dict(_SERIES_DICT)
    return report


class TestRunMetadataWritten:
    @pytest.mark.unit
    def test_run_metadata_json_created_after_run(self, tmp_path: Path) -> None:
        """run_metadata.json must be written to report_dir after finalize_benchmark."""
        ctx = _make_ctx(tmp_path)
        bench = _make_bench(tmp_path, report=_make_populated_report())

        finalize_benchmark(ctx, bench)

        metadata_path = tmp_path / "run_metadata.json"
        assert metadata_path.exists(), "run_metadata.json was not created"
        data = json.loads(metadata_path.read_text())
        assert "run_date" in data
        assert "qps" in data

    @pytest.mark.unit
    def test_run_metadata_written_even_without_report(self, tmp_path: Path) -> None:
        """run_metadata.json must be written even when no Report is available."""
        ctx = _make_ctx(tmp_path)
        bench = _make_bench(tmp_path, report=None)

        finalize_benchmark(ctx, bench)

        metadata_path = tmp_path / "run_metadata.json"
        assert metadata_path.exists()
        data = json.loads(metadata_path.read_text())
        # qps comes from the session-timing fallback (0 issued / 1 s = 0.0), not None
        assert data["qps"] == 0.0
        assert data["system_tps"] is None

    @pytest.mark.unit
    def test_results_json_written_even_if_metadata_write_fails(
        self, tmp_path: Path
    ) -> None:
        """results.json must be written even when run_metadata.json write fails."""
        ctx = _make_ctx(tmp_path)
        bench = _make_bench(tmp_path, report=_make_populated_report())

        results_path = tmp_path / "results.json"
        metadata_path = tmp_path / "run_metadata.json"

        from inference_endpoint.utils import write_json_atomic as real_write_json_atomic

        def selective_write_json_atomic(path, payload):
            if path == metadata_path:
                raise OSError("disk full")
            return real_write_json_atomic(path, payload)

        with patch(
            "inference_endpoint.commands.benchmark.execute.write_json_atomic",
            side_effect=selective_write_json_atomic,
        ):
            finalize_benchmark(ctx, bench)

        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert "results" in data
        assert not metadata_path.exists()


@pytest.mark.unit
class TestBuildRunMetadata:
    def test_percentile_fields_populated_from_report(self) -> None:
        """All percentile fields must be non-None when report uses float-string keys."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        metadata = _build_run_metadata(ctx, _make_populated_report(), qps=20.0)

        for field, expected_ms in [
            ("measured_latency_ttft_p50", 500.0),
            ("measured_latency_ttft_p90", 900.0),
            ("measured_latency_ttft_p95", 950.0),
            ("measured_latency_ttft_p99", 990.0),
            ("measured_latency_ttft_p999", 999.0),
            ("measured_latency_tpot_p50", 500.0),
            ("measured_latency_tpot_p90", 900.0),
            ("measured_latency_tpot_p95", 950.0),
            ("measured_latency_tpot_p99", 990.0),
            ("measured_latency_tpot_p999", 999.0),
            ("measured_latency_request_p50", 500.0),
            ("measured_latency_request_p90", 900.0),
            ("measured_latency_request_p95", 950.0),
            ("measured_latency_request_p99", 990.0),
            ("measured_latency_request_p999", 999.0),
        ]:
            assert metadata[field] == pytest.approx(
                expected_ms
            ), f"{field} unexpected value (float-string key lookup failed)"

        assert metadata["ttft"] == pytest.approx(990.0)

    def test_scalar_fields_populated_from_report(self) -> None:
        """measured_run_duration, measured_total_output_tokens, and measured_total_requests
        must carry the values derived from the report, not None."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        metadata = _build_run_metadata(ctx, _make_populated_report(), qps=20.0)

        # duration_ns=10_000_000_000 → 10.0 s
        assert metadata["measured_run_duration"] == pytest.approx(10.0)
        # output_sequence_lengths={"total": 2000} → 2000
        assert metadata["measured_total_output_tokens"] == 2000
        # n_samples_completed=95 (not n_samples_issued=100)
        assert metadata["measured_total_requests"] == 95

    def test_none_fields_when_no_report(self) -> None:
        """All measured fields must be None when report is None."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        metadata = _build_run_metadata(ctx, None, qps=None)

        assert metadata["qps"] is None
        assert metadata["system_tps"] is None
        assert metadata["measured_total_requests"] is None
        assert metadata["measured_run_duration"] is None
        assert metadata["measured_total_output_tokens"] is None
        assert metadata["measured_latency_ttft_p99"] is None

    def test_infrastructure_fields_start_as_none(self) -> None:
        """Fields populated by external tooling must start as None."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        metadata = _build_run_metadata(ctx, None, qps=None)

        for field in (
            "disaggregated",
            "tensor_parallel",
            "pipeline_parallel",
            "data_parallel",
            "expert_parallel",
            "batch",
            "config_summary",
            "config_summary_notes",
            "tps_utilization",
            "link_config",
            "link_logs",
        ):
            assert metadata[field] is None, f"{field} must start as None"

    def test_tps_per_user_computed_from_concurrency(self) -> None:
        """tps_per_user = system_tps / concurrency when both are set."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = 10

        report = _make_populated_report()
        report.tps.return_value = 200.0

        metadata = _build_run_metadata(ctx, report, qps=20.0)

        assert metadata["tps_per_user"] == pytest.approx(20.0)

    def test_tps_per_user_none_when_no_concurrency(self) -> None:
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        metadata = _build_run_metadata(ctx, _make_populated_report(), qps=20.0)

        assert metadata["tps_per_user"] is None

    def test_tps_per_user_none_when_system_tps_is_none(self) -> None:
        """tps_per_user must be None when system_tps is None, even if concurrency is set."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = 10

        report = _make_populated_report()
        report.tps.return_value = None

        metadata = _build_run_metadata(ctx, report, qps=20.0)

        assert metadata["tps_per_user"] is None

    def test_qps_passed_through_from_caller(self) -> None:
        """qps in metadata is exactly the value finalize_benchmark computed — no re-derivation."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        metadata = _build_run_metadata(ctx, _make_populated_report(), qps=42.5)

        assert metadata["qps"] == pytest.approx(42.5)

    def test_integer_string_percentile_keys_return_none(self) -> None:
        """Integer-string keys ("99", "50") must not resolve — documents registry key format."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        report = _make_populated_report()
        report.ttft = {
            "percentiles": {"50": 500_000_000, "99": 990_000_000},
            "min": 100_000_000,
            "max": 1_000_000_000,
            "avg": 600_000_000,
        }
        report.tpot = {}
        report.latency = {}

        metadata = _build_run_metadata(ctx, report, qps=20.0)

        assert metadata["measured_latency_ttft_p50"] is None
        assert metadata["measured_latency_ttft_p99"] is None

    def test_run_date_is_iso_format(self) -> None:
        """run_date must be a valid ISO 8601 string."""
        from datetime import datetime

        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None

        metadata = _build_run_metadata(ctx, None, qps=None)

        # Will raise ValueError if the string is not a valid ISO datetime.
        parsed = datetime.fromisoformat(metadata["run_date"])
        assert parsed.tzinfo is not None


@pytest.mark.unit
class TestQpsCrossFileConsistency:
    def test_both_files_agree_on_qps_zero_when_report_qps_returns_none(
        self, tmp_path: Path
    ) -> None:
        """results.json and run_metadata.json must both carry qps=0.0 when report.qps() is None."""
        ctx = _make_ctx(tmp_path)
        report = _make_populated_report()
        report.qps.return_value = None
        bench = _make_bench(tmp_path, report=report)

        finalize_benchmark(ctx, bench)

        results = json.loads((tmp_path / "results.json").read_text())
        metadata = json.loads((tmp_path / "run_metadata.json").read_text())

        assert results["results"]["qps"] == 0.0
        assert metadata["qps"] == 0.0

    def test_both_files_agree_on_qps_when_duration_ns_is_none(
        self, tmp_path: Path
    ) -> None:
        """Both files must carry the same qps when report.duration_ns is None (degenerate run).

        When duration_ns is None, finalize_benchmark falls to the SessionResult fallback
        and computes qps from session timing. run_metadata.json must use that same value,
        not re-derive it from report.qps() which returns None in this case.
        """
        ctx = _make_ctx(tmp_path)
        report = _make_populated_report()
        report.duration_ns = None
        report.qps.return_value = None
        bench = _make_bench(tmp_path, report=report)

        finalize_benchmark(ctx, bench)

        results = json.loads((tmp_path / "results.json").read_text())
        metadata = json.loads((tmp_path / "run_metadata.json").read_text())

        # The session window is 1 s (end - start = 1e9 ns) and phase_results=[] so
        # total_issued=0; both files must carry 0.0, not diverge.
        assert results["results"]["qps"] == 0.0
        assert metadata["qps"] == 0.0


@pytest.mark.unit
class TestMetricHelpers:
    def test_ns_to_ms_converts(self) -> None:
        assert _ns_to_ms(1_000_000) == pytest.approx(1.0)
        assert _ns_to_ms(0) == pytest.approx(0.0)
        assert _ns_to_ms(None) is None

    def test_metric_stat_returns_none_on_empty_dict(self) -> None:
        assert _metric_stat({}, "min") is None

    def test_metric_pct_returns_none_on_missing_key(self) -> None:
        assert _metric_pct({"percentiles": {"50.0": 500_000_000}}, "99.0") is None

    def test_metric_pct_converts_correctly(self) -> None:
        assert _metric_pct(
            {"percentiles": {"99.0": 990_000_000}}, "99.0"
        ) == pytest.approx(990.0)
