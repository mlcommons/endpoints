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
    finalize_benchmark,
)
from inference_endpoint.config.schema import OfflineBenchmarkConfig, TestMode
from inference_endpoint.exceptions import ExecutionError
from inference_endpoint.load_generator.session import SessionResult

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


def _make_bench(tmp_path: Path) -> BenchmarkResult:
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
        report=None,
        tmpfs_dir=tmp_path,
    )


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
