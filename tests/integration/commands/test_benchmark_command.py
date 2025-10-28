# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Integration tests for benchmark commands against echo server.

These tests verify end-to-end benchmark execution with real HTTP server:
- Offline mode (max throughput)
- Online mode (Poisson distribution)
- Dataset loading and processing
- Results collection and reporting
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark import run_benchmark_command


class TestBenchmarkCommandIntegration:
    """Integration tests for benchmark commands with echo server."""

    @pytest.mark.asyncio
    async def test_offline_benchmark_with_echo_server(
        self, mock_http_echo_server, ds_pickle_dataset_path, caplog
    ):
        """Test offline benchmark completes successfully."""
        args = MagicMock()
        args.benchmark_mode = "offline"
        args.config = None
        args.endpoint = mock_http_echo_server.url
        args.dataset = Path(ds_pickle_dataset_path)
        args.api_key = None
        args.qps = None  # Use default
        args.concurrency = None
        args.workers = 1
        args.duration = None  # Use default
        args.min_tokens = None
        args.max_tokens = None
        args.mode = None
        args.output = None
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_benchmark_command(args)

        log_text = caplog.text
        # Verify completion
        # TODO: this might be changed later to the actual output of the benchmark.
        assert "Completed in" in log_text
        assert "successful" in log_text
        assert "QPS:" in log_text
        # Verify scheduler used
        assert "MaxThroughputScheduler" in log_text

    @pytest.mark.asyncio
    async def test_online_benchmark_with_echo_server(
        self, mock_http_echo_server, ds_pickle_dataset_path, caplog
    ):
        """Test online benchmark with Poisson distribution."""
        args = MagicMock()
        args.benchmark_mode = "online"
        args.config = None
        args.endpoint = mock_http_echo_server.url
        args.dataset = Path(ds_pickle_dataset_path)
        args.api_key = None
        args.qps = 50  # Override default
        args.concurrency = None
        args.workers = 1
        args.duration = 2  # Short test
        args.min_tokens = None
        args.max_tokens = None
        args.mode = None
        args.output = None
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_benchmark_command(args)

        log_text = caplog.text
        # Verify completion
        # TODO: this might be changed later to the actual output of the benchmark.
        assert "Completed in" in log_text
        assert "successful" in log_text
        # Verify Poisson scheduler used
        assert "PoissonDistributionScheduler" in log_text
        assert "50" in log_text  # QPS target

    @pytest.mark.asyncio
    async def test_benchmark_with_output_file(
        self, mock_http_echo_server, ds_pickle_dataset_path, tmp_path
    ):
        """Test benchmark saves results to JSON file."""
        output_file = tmp_path / "benchmark_results.json"

        args = MagicMock()
        args.benchmark_mode = "offline"
        args.config = None
        args.endpoint = mock_http_echo_server.url
        args.dataset = Path(ds_pickle_dataset_path)
        args.api_key = None
        args.qps = None
        args.concurrency = None
        args.workers = 1
        args.duration = None
        args.min_tokens = None
        args.max_tokens = None
        args.mode = None
        args.output = output_file
        args.verbose = 0

        await run_benchmark_command(args)

        # Verify file was created
        assert output_file.exists()

        # Verify JSON structure
        import json

        with open(output_file) as f:
            results = json.load(f)

        assert "config" in results
        assert "results" in results
        assert results["results"]["total"] > 0
        assert results["results"]["successful"] >= 0

    @pytest.mark.asyncio
    async def test_benchmark_mode_logging(
        self, mock_http_echo_server, ds_pickle_dataset_path, caplog
    ):
        """Test that benchmark logs mode and scheduler information."""
        args = MagicMock()
        args.benchmark_mode = "online"
        args.config = None
        args.endpoint = mock_http_echo_server.url
        args.dataset = Path(ds_pickle_dataset_path)
        args.api_key = None
        args.qps = 20
        args.concurrency = None
        args.workers = 1
        args.duration = 2
        args.min_tokens = None
        args.max_tokens = None
        args.mode = "perf"
        args.output = None
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_benchmark_command(args)

        log_text = caplog.text
        # Should log mode and configuration
        # TODO: this might be changed later to the actual output of the benchmark.
        # Test mode is now TestMode enum, shown as TestMode.PERF
        assert "Mode:" in log_text and ("perf" in log_text or "PERF" in log_text)
        assert "QPS: 20" in log_text
        assert "Responses: False" in log_text  # perf mode
