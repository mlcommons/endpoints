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

"""Integration tests for benchmark commands against echo server."""

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
    RuntimeConfig,
    Settings,
    StreamingMode,
    TestMode,
    TestType,
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig

_TEST_SETTINGS = Settings(
    runtime=RuntimeConfig(min_duration_ms=0),
    load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    client=HTTPClientConfig(num_workers=1, warmup_connections=0, max_connections=10),
)


def _config(endpoint_url: str, dataset_path: str, **overrides) -> BenchmarkConfig:
    """Build a minimal BenchmarkConfig for testing."""
    defaults = {
        "type": TestType.OFFLINE,
        "endpoint_config": EndpointConfig(endpoints=[endpoint_url]),
        "model_params": ModelParams(name="echo-server", streaming=StreamingMode.OFF),
        "datasets": [Dataset(path=dataset_path, type=DatasetType.PERFORMANCE)],
        "settings": _TEST_SETTINGS,
    }
    return BenchmarkConfig(**(defaults | overrides))


def _poisson_settings(target_qps: float, duration_s: int = 2) -> Settings:
    return Settings(
        runtime=RuntimeConfig(min_duration_ms=duration_s * 1000),
        load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=target_qps),
        client=HTTPClientConfig(
            num_workers=1, warmup_connections=0, max_connections=10
        ),
    )


class TestBenchmarkCommandIntegration:
    """Integration tests for benchmark commands with echo server."""

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_offline_benchmark(
        self, mock_http_echo_server, ds_jsonl_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_jsonl_dataset_path,
            model_params=ModelParams(name="echo-server", streaming=streaming),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text
        assert "QPS:" in caplog.text
        assert "MaxThroughputScheduler" in caplog.text

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_online_benchmark(
        self, mock_http_echo_server, ds_jsonl_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_jsonl_dataset_path,
            type=TestType.ONLINE,
            model_params=ModelParams(name="echo-server", streaming=streaming),
            settings=_poisson_settings(target_qps=50),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text
        assert "PoissonDistributionScheduler" in caplog.text
        assert "50" in caplog.text

    @pytest.mark.integration
    def test_results_json_output(
        self, mock_http_echo_server, ds_jsonl_dataset_path, tmp_path
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_jsonl_dataset_path,
            report_dir=tmp_path,
        )
        run_benchmark(config, TestMode.PERF)

        results_path = tmp_path / "results.json"
        assert results_path.exists()
        results = json.loads(results_path.read_text())
        assert "config" in results
        assert results["results"]["total"] > 0
        assert results["results"]["successful"] >= 0

    @pytest.mark.integration
    def test_mode_logging(self, mock_http_echo_server, ds_jsonl_dataset_path, caplog):
        config = _config(
            mock_http_echo_server.url,
            ds_jsonl_dataset_path,
            type=TestType.ONLINE,
            settings=_poisson_settings(target_qps=20),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Mode:" in caplog.text
        assert "QPS: 20" in caplog.text
        assert "Responses: False" in caplog.text
