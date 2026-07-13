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

"""Whole-run watchdog (settings.timeouts.run_timeout_s) integration tests.

Locking invariant: a fired run watchdog must never produce a COMPLETE
report. The watchdog SIGTERMs the metrics aggregator (whose handler writes
an INTERRUPTED final snapshot) before stopping the session, and
``run_benchmark`` exits non-zero via ``ExecutionError``.
"""

import json

import pytest
from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    Dataset,
    DatasetType,
    EndpointConfig,
    LoadPattern,
    LoadPatternType,
    ModelParams,
    Settings,
    StreamingMode,
    TestMode,
    TestType,
    WarmupConfig,
)
from inference_endpoint.config.timeouts import Timeouts
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.exceptions import ExecutionError


@pytest.mark.integration
def test_run_timeout_produces_interrupted_report(
    mock_http_echo_server, ds_dataset_path, tmp_path
):
    """run_timeout_s firing mid-run aborts with an INTERRUPTED report."""
    config = BenchmarkConfig(
        type=TestType.ONLINE,
        endpoint_config=EndpointConfig(endpoints=[mock_http_echo_server.url]),
        model_params=ModelParams(name="echo-server", streaming=StreamingMode.OFF),
        datasets=[Dataset(path=str(ds_dataset_path), type=DatasetType.PERFORMANCE)],
        report_dir=tmp_path,
        settings=Settings(
            load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=5),
            client=HTTPClientConfig(
                num_workers=1, warmup_connections=0, max_connections=10
            ),
            # Long workload so only the watchdog can end the run.
            timeouts=Timeouts(min_duration_ms=120_000, run_timeout_s=2.0),
            warmup=WarmupConfig(enabled=False),
        ),
    )

    with pytest.raises(ExecutionError, match="Run timeout"):
        run_benchmark(config, TestMode.PERF)

    snapshot_path = tmp_path / "metrics" / "final_snapshot.json"
    assert snapshot_path.exists(), "aggregator must still write a final snapshot"
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["state"] == "interrupted"

    # Locking invariant: a fired run watchdog must never yield a COMPLETE report.
    summary = json.loads((tmp_path / "result_summary.json").read_text())
    assert summary["complete"] is False
