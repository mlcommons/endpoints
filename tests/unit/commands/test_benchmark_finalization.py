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

"""Tests that sys info failure does not block results.json in finalize_benchmark."""

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
    finalize_benchmark,
)
from inference_endpoint.config.schema import OfflineBenchmarkConfig, TestMode
from inference_endpoint.exceptions import ExecutionError
from inference_endpoint.load_generator.session import SessionResult
from inference_endpoint.metrics.report import Report

_OFFLINE_KWARGS = {
    "endpoint_config": {"endpoints": ["http://localhost:8000"]},
    "model_params": {"name": "test-model"},
    "datasets": [{"path": "test.jsonl"}],
    "system_info": {
        "ssh_ids": ["alice@10.0.0.1"],
        "accelerator_backend": "cuda",
    },
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


class TestRunMetadataWriteNonBlocking:
    @pytest.mark.unit
    def test_write_failure_does_not_abort_finalize(self, tmp_path: Path) -> None:
        """A write error on run_metadata.json must not propagate out of finalize_benchmark."""
        ctx = _make_ctx(tmp_path)
        bench = _make_bench(tmp_path)

        with (
            patch(
                "inference_endpoint.commands.benchmark.execute._build_run_metadata",
                return_value={},
            ),
            patch("builtins.open", side_effect=OSError("disk full")),
            patch(
                "inference_endpoint.sys_info.capture.capture_system_info",
                side_effect=ExecutionError("skipped"),
            ),
        ):
            # Must not raise even though open() fails for every file write.
            finalize_benchmark(ctx, bench)

    @pytest.mark.unit
    def test_results_json_written_even_if_metadata_write_fails(
        self, tmp_path: Path
    ) -> None:
        """results.json must be written even when run_metadata.json write fails.

        The real _build_run_metadata runs with a populated Report so the full
        metadata-building path (percentile lookups, ms conversion) is exercised
        before the write fails.
        """
        ctx = _make_ctx(tmp_path)
        bench = _make_bench(tmp_path, report=_make_populated_report())

        results_path = tmp_path / "results.json"
        metadata_path = tmp_path / "run_metadata.json"

        real_open = open

        def selective_open(path, *args, **kwargs):
            if str(path) == str(metadata_path):
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with (
            patch("builtins.open", side_effect=selective_open),
            patch(
                "inference_endpoint.sys_info.capture.capture_system_info",
                side_effect=ExecutionError("skipped"),
            ),
        ):
            finalize_benchmark(ctx, bench)

        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert "results" in data
        assert not metadata_path.exists()


class TestSysInfoFailureNonBlocking:
    @pytest.mark.unit
    def test_execution_error_does_not_prevent_results_write(
        self, tmp_path: Path
    ) -> None:
        """ExecutionError from capture_system_info must not prevent results.json."""
        ctx = _make_ctx(tmp_path)
        bench = _make_bench(tmp_path)

        with (
            patch(
                "inference_endpoint.commands.benchmark.execute._build_run_metadata",
                return_value={},
            ),
            patch(
                "inference_endpoint.sys_info.capture.capture_system_info",
                side_effect=ExecutionError("ssh timeout"),
            ),
        ):
            finalize_benchmark(ctx, bench)

        results_path = tmp_path / "results.json"
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert "results" in data

    @pytest.mark.unit
    def test_unexpected_exception_does_not_prevent_results_write(
        self, tmp_path: Path
    ) -> None:
        """An unexpected exception from capture_system_info must not prevent results.json."""
        ctx = _make_ctx(tmp_path)
        bench = _make_bench(tmp_path)

        with (
            patch(
                "inference_endpoint.commands.benchmark.execute._build_run_metadata",
                return_value={},
            ),
            patch(
                "inference_endpoint.sys_info.capture.capture_system_info",
                side_effect=RuntimeError("unexpected crash"),
            ),
        ):
            finalize_benchmark(ctx, bench)

        results_path = tmp_path / "results.json"
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert "results" in data


# One nanosecond value per percentile bucket (in ns), chosen so the ms
# conversion produces a round number that is easy to assert on.
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
    """Return a Report mock with registry-format float-string percentile keys.

    Sets every attribute that finalize_benchmark and _build_run_metadata access
    directly so no MagicMock auto-attribute leaks into arithmetic comparisons.
    """
    report = MagicMock()
    report.tps.return_value = 200.0
    report.qps.return_value = 20.0
    report.n_samples_completed = 100
    report.n_samples_issued = 100
    report.n_samples_failed = 0
    report.duration_ns = 10_000_000_000
    report.output_sequence_lengths = {"total": 2000}
    report.ttft = dict(_SERIES_DICT)
    report.tpot = dict(_SERIES_DICT)
    report.latency = dict(_SERIES_DICT)
    return report


@pytest.mark.unit
class TestBuildRunMetadata:
    def _make_report(self) -> MagicMock:
        return _make_populated_report()

    def test_percentile_keys_resolve_with_float_string_format(self) -> None:
        """All percentile fields must be non-None when the report uses registry-format
        float-string keys (e.g. "99.0", "50.0") rather than integer strings."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None
        ctx.config.system_info = None

        metadata = _build_run_metadata(ctx, self._make_report())

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
            ), f"{field} was None — float-string percentile key lookup failed"

        # Top-level ttft field is the p99.
        assert metadata["ttft"] == pytest.approx(990.0)

    def test_integer_string_keys_silently_return_none(self) -> None:
        """Confirm that integer-string keys ("99", "50") do NOT resolve —
        documenting the registry key format requirement."""
        ctx = MagicMock()
        ctx.config.settings.load_pattern.target_concurrency = None
        ctx.config.system_info = None

        report = self._make_report()
        # Use integer-string keys (old broken format).
        report.ttft = {
            "percentiles": {"50": 500_000_000, "99": 990_000_000},
            "min": 100_000_000,
            "max": 1_000_000_000,
            "avg": 600_000_000,
        }
        report.tpot = {}
        report.latency = {}

        metadata = _build_run_metadata(ctx, report)

        assert metadata["measured_latency_ttft_p50"] is None
        assert metadata["measured_latency_ttft_p99"] is None
