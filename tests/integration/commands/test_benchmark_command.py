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

"""Integration tests for benchmark commands against echo server.

These tests verify end-to-end benchmark execution with real HTTP server:
- Offline mode (max throughput)
- Online mode (Poisson distribution)
- Dataset loading and processing
- Results collection and reporting
"""

import argparse

# Verify JSON structure
import json
from pathlib import Path

import pytest
from inference_endpoint.commands.benchmark import run_benchmark_command


class TestBenchmarkCommandIntegration:
    """Integration tests for benchmark commands with echo server."""

    @pytest.mark.asyncio
    async def test_offline_benchmark_with_echo_server(
        self, mock_http_echo_server, ds_pickle_dataset_path, caplog
    ):
        """Test offline benchmark completes successfully."""
        args = argparse.Namespace(
            benchmark_mode="offline",
            config=None,
            endpoints=mock_http_echo_server.url,
            dataset=Path(ds_pickle_dataset_path),
            api_key=None,
            target_qps=None,
            concurrency=None,
            workers=1,
            duration=None,
            num_samples=None,
            streaming="auto",
            min_output_tokens=None,
            max_output_tokens=None,
            mode=None,
            output=None,
            verbose=1,
            model="echo-server",
            timeout=None,
            warmup_connections=False,
        )

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

        args = argparse.Namespace(
            benchmark_mode="online",
            config=None,
            endpoints=mock_http_echo_server.url,
            dataset=Path(ds_pickle_dataset_path),
            api_key=None,
            target_qps=50,
            concurrency=None,
            workers=1,
            duration=2,
            num_samples=None,
            streaming="auto",
            min_output_tokens=None,
            max_output_tokens=None,
            mode=None,
            output=None,
            verbose=1,
            model="echo-server",
            timeout=None,
            warmup_connections=False,
        )
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
        # The benchmark command writes results to `results.json` inside the
        # configured `report_dir`. Pass `report_dir=tmp_path` so the command
        # will write output into this temporary directory and we can assert on
        # the produced file.
        report_dir = tmp_path

        args = argparse.Namespace(
            benchmark_mode="offline",
            config=None,
            endpoints=mock_http_echo_server.url,
            dataset=Path(ds_pickle_dataset_path),
            api_key=None,
            target_qps=None,
            concurrency=None,
            workers=1,
            duration=None,
            num_samples=None,
            streaming="auto",
            min_output_tokens=None,
            max_output_tokens=None,
            mode=None,
            report_dir=report_dir,
            verbose=0,
            model="echo-server",
            timeout=None,
            warmup_connections=False,
        )

        await run_benchmark_command(args)

        # Verify file was created at <report_dir>/results.json
        results_path = report_dir / "results.json"
        assert results_path.exists()

        with open(results_path) as f:
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

        args = argparse.Namespace(
            benchmark_mode="online",
            config=None,
            endpoints=mock_http_echo_server.url,
            dataset=Path(ds_pickle_dataset_path),
            api_key=None,
            target_qps=20,
            concurrency=None,
            workers=1,
            duration=2,
            num_samples=None,
            streaming="auto",
            min_output_tokens=None,
            max_output_tokens=None,
            mode="perf",
            output=None,
            verbose=1,
            model="echo-server",
            timeout=None,
            warmup_connections=False,
        )
        with caplog.at_level("INFO"):
            await run_benchmark_command(args)

        log_text = caplog.text
        # Should log mode and configuration
        # TODO: this might be changed later to the actual output of the benchmark.
        # Test mode is now TestMode enum, shown as TestMode.PERF
        assert "Mode:" in log_text and ("perf" in log_text or "PERF" in log_text)
        assert "QPS: 20" in log_text
        assert "Responses: False" in log_text  # perf mode

    @pytest.mark.asyncio
    async def test_cli_argument_names_match_benchmark_implementation(
        self, mock_http_echo_server, ds_pickle_dataset_path, caplog
    ):
        """Test that CLI argument names match what benchmark command expects.

        This test uses the actual CLI parser to parse arguments, ensuring that
        the argument names defined in the parser (e.g., --endpoint) match what
        _build_config_from_cli() expects (e.g., args.endpoint).

        Without this test, a mismatch like defining --endpoint in CLI but
        accessing args.endpoints in the code would pass unit tests but fail
        in actual execution.
        """
        from inference_endpoint.cli import create_parser

        # Build CLI command as would be used in actual execution
        cli_args = [
            "benchmark",
            "offline",
            "--endpoint",
            mock_http_echo_server.url,
            "--model",
            "echo-server",
            "--dataset",
            str(ds_pickle_dataset_path),
            "--workers",
            "1",
        ]

        # Parse using the actual CLI parser
        parser = create_parser()
        args = parser.parse_args(cli_args)

        # Verify essential attributes exist with expected names
        # These assertions catch mismatches like 'endpoint' vs 'endpoints'
        assert hasattr(args, "endpoint"), "CLI should define 'endpoint' attribute"
        assert hasattr(args, "model"), "CLI should define 'model' attribute"
        assert hasattr(args, "dataset"), "CLI should define 'dataset' attribute"
        assert hasattr(
            args, "benchmark_mode"
        ), "CLI should define 'benchmark_mode' attribute"

        # Run the benchmark with parsed args to ensure end-to-end compatibility
        with caplog.at_level("INFO"):
            await run_benchmark_command(args)

        log_text = caplog.text
        # Verify the benchmark actually ran successfully
        assert "Completed in" in log_text
        assert "successful" in log_text
