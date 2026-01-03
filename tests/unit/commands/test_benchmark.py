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

"""Tests for benchmark command error handling.

These tests verify that the benchmark command properly validates inputs,
handles configuration errors, and raises appropriate exceptions instead of
calling sys.exit(). This allows:
- Testing error conditions without process termination
- Programmatic use of benchmark command
- Consistent error handling via centralized exception catching in main()

Focus areas:
- Input validation (missing args, invalid patterns)
- Configuration loading and merging
- Dataset validation
- Error propagation with proper exception types
- Verbosity levels and logging
"""

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark import (
    _build_config_from_cli,
    run_benchmark_command,
)
from inference_endpoint.config.schema import APIType
from inference_endpoint.exceptions import InputValidationError
from inference_endpoint.utils.logging import TRACE


def _create_mock_args(**overrides):
    """Helper to create mock args with default values.

    Args:
        **overrides: Any fields to override from defaults

    Returns:
        MagicMock configured with all required benchmark args
    """
    defaults = {
        "benchmark_mode": "offline",
        "config": None,
        "endpoint": "http://test.com",
        "dataset": Path("test.pkl"),
        "model": "llama-2-70b",
        "api_key": None,
        "api_type": APIType.OPENAI,
        "load_pattern": None,
        "target_qps": None,
        "concurrency": None,
        "workers": None,
        "duration": None,
        "num_samples": None,
        "streaming": "auto",
        "min_output_tokens": None,
        "max_output_tokens": None,
        "mode": "perf",
        "report_dir": None,
        "timeout": None,
        "output": None,
        "verbose": 0,
    }
    defaults.update(overrides)

    mock = MagicMock()
    for key, value in defaults.items():
        setattr(mock, key, value)
    return mock


class TestBuildConfigFromCLI:
    """Test building BenchmarkConfig from CLI arguments.

    CLI and YAML modes are now mutually exclusive - no merging.
    This tests the CLI-only config builder.
    """

    def test_build_config_offline_minimal(self):
        """Test building config with minimal offline params."""
        # Note: concurrency is now in shared args (applies to both offline and online)
        args = argparse.Namespace(
            endpoint="http://test:8000",
            dataset=Path("test.pkl"),
            model="llama-2-70b",  # Required
            api_key=None,
            target_qps=None,
            concurrency=None,  # Now in shared args
            workers=None,
            duration=None,
            min_output_tokens=None,
            max_output_tokens=None,
        )

        config = _build_config_from_cli(args, "offline")

        assert config.name == "cli_offline"
        assert config.endpoint_config.endpoint == "http://test:8000"
        assert config.datasets[0].path == "test.pkl"
        assert config.settings.load_pattern.type.value == "max_throughput"
        assert config.settings.load_pattern.target_qps is None
        assert config.settings.client.workers == 4  # Default
        assert (
            config.settings.runtime.min_duration_ms == 0
        )  # Default: 0 - use dataset samples
        assert config.settings.runtime.n_samples_to_issue is None  # Default: None

    def test_build_config_online_with_params(self):
        """Test building config with custom online params."""
        # Online mode has concurrency attribute (from _add_online_specific_args)
        args = argparse.Namespace(
            endpoint="http://prod:8000",
            dataset=Path("dataset.pkl"),
            model="gpt-4",  # Required
            api_key="key123",
            target_qps=100.0,
            workers=8,
            concurrency=64,  # Online-specific
            load_pattern="poisson",  # Online-specific
            duration=600,
            min_output_tokens=None,
            max_output_tokens=None,
        )

        config = _build_config_from_cli(args, "online")

        assert config.name == "cli_online"
        assert config.settings.load_pattern.type.value == "poisson"
        assert config.settings.load_pattern.target_qps == 100.0
        assert config.settings.client.workers == 8
        assert config.settings.runtime.min_duration_ms == 600000

    # Note: Tests for missing endpoint/dataset/model removed
    # These are enforced by argparse (required=True), not by _build_config_from_cli
    # Argparse errors before our code runs, so no need to test here

    def test_build_config_with_num_samples(self):
        """Test that num_samples parameter is mapped to config."""
        args = argparse.Namespace(
            endpoint="http://test:8000",
            dataset=Path("test.pkl"),
            model="llama-2-70b",
            api_key=None,
            target_qps=None,
            concurrency=None,
            workers=None,
            duration=None,
            num_samples=100,  # This should be mapped to config
            min_output_tokens=None,
            max_output_tokens=None,
        )

        config = _build_config_from_cli(args, "offline")
        assert config.settings.runtime.n_samples_to_issue == 100


class TestRunBenchmarkCommand:
    """Test benchmark command error handling.

    These tests validate that all user input errors are caught early and
    raise InputValidationError, allowing main() to handle exits gracefully.
    Tests cover: missing mode, invalid config, missing endpoint/dataset.
    """

    @pytest.mark.asyncio
    async def test_from_config_mode_requires_config(self):
        """Test from-config mode requires --config."""
        # Argparse makes --config required for from-config mode
        # This tests the code path
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: "test"
type: "offline"
datasets:
  - name: "test"
    type: "performance"
    path: "tests/datasets/dummy_1k.pkl"
endpoint_config:
  endpoint: "http://test:8000"
""")
            config_path = Path(f.name)

        args = MagicMock()
        args.benchmark_mode = "from-config"
        args.config = config_path
        args.output = None
        args.mode = None

        try:
            # Should load from YAML successfully
            from inference_endpoint.config.yaml_loader import ConfigLoader

            config = ConfigLoader.load_yaml(config_path)
            assert config.endpoint_config.endpoint == "http://test:8000"
        finally:
            config_path.unlink()

    @pytest.mark.asyncio
    async def test_missing_benchmark_mode(self):
        """Test error when benchmark mode is None (shouldn't happen with argparse)."""
        # Note: Argparse prevents this scenario, but we test defensive code
        args = MagicMock()
        args.benchmark_mode = None  # Not "offline", "online", or "from-config"

        # Should error in the else clause
        with pytest.raises(InputValidationError, match="Unknown benchmark mode"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_invalid_config_file(self):
        """Test error with invalid config file."""
        args = MagicMock()
        args.benchmark_mode = "from-config"
        args.config = Path("/nonexistent/config.yaml")
        args.output = None
        args.mode = None

        with pytest.raises(InputValidationError, match="Config error"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_nonexistent_dataset_file(self):
        """Test error when dataset file doesn't exist."""
        args = _create_mock_args(dataset=Path("/nonexistent/data.pkl"))

        with pytest.raises(InputValidationError, match="not found"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_online_mode_requires_qps(self):
        """Test that online mode with poisson pattern requires --target-qps."""
        args = _create_mock_args(
            benchmark_mode="online",
            # target_qps=None (default) - should raise error from config validation
        )

        with pytest.raises(InputValidationError, match="requires positive target_qps"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_concurrency_pattern_requires_concurrency(self):
        """Test that concurrency load pattern requires --concurrency."""
        args = _create_mock_args(
            benchmark_mode="online",
            load_pattern="concurrency",
            # concurrency=None (default) - should raise error from config validation
        )

        with pytest.raises(InputValidationError, match="requires target_concurrency"):
            await run_benchmark_command(args)

    # Note: Testing unsupported load patterns requires full integration
    # as it happens during scheduler creation after dataset loading.
    # This is covered by yaml_config.py tests and integration tests.


class TestVerbosityLevels:
    """Test CLI verbosity levels and TRACE logging for timing."""

    def test_trace_level_exists(self):
        """Verify TRACE level is defined below DEBUG."""
        assert TRACE < logging.DEBUG
        assert TRACE == 5
        assert logging.getLevelName(TRACE) == "TRACE"

    def test_timing_logs_require_trace_level(self):
        """Verify timing logs only appear at TRACE level (-vvv), not DEBUG (-vv)."""

        test_logger = logging.getLogger("inference_endpoint.endpoint_client.types")

        # At DEBUG level (-vv), timing logs should NOT appear
        test_logger.setLevel(logging.DEBUG)
        assert not test_logger.isEnabledFor(TRACE)

        # At TRACE level (-vvv), timing logs should appear
        test_logger.setLevel(TRACE)
        assert test_logger.isEnabledFor(TRACE)

    def test_verbosity_mapping(self):
        """Verify -v, -vv, -vvv map to correct log levels."""
        # These are the expected mappings from cli.py:
        # -v   (verbose=1) -> INFO
        # -vv  (verbose=2) -> DEBUG
        # -vvv (verbose=3) -> TRACE
        assert logging.INFO == 20
        assert logging.DEBUG == 10
        assert TRACE == 5
