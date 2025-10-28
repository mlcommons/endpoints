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
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark import (
    _extract_cli_overrides,
    run_benchmark_command,
)
from inference_endpoint.exceptions import InputValidationError


class TestExtractCLIOverrides:
    """Test CLI override extraction.

    Validates that CLI arguments are correctly extracted and only non-None
    values are included in the override dictionary. This is critical for
    the config merging logic where CLI args override YAML values.
    """

    def test_extract_all_overrides(self):
        """Test extracting all possible CLI overrides."""
        args = MagicMock()
        args.endpoint = "http://test.com"
        args.api_key = "test-key"
        args.qps = 100
        args.concurrency = 50
        args.workers = 8
        args.duration = 60
        args.min_tokens = 100
        args.max_tokens = 2000

        overrides = _extract_cli_overrides(args)

        assert overrides["endpoint"] == "http://test.com"
        assert overrides["api_key"] == "test-key"
        assert overrides["qps"] == 100
        assert overrides["concurrency"] == 50
        assert overrides["workers"] == 8
        assert overrides["duration"] == 60
        assert overrides["min_tokens"] == 100
        assert overrides["max_tokens"] == 2000

    def test_extract_none_values_excluded(self):
        """Test that None values are not included."""
        args = MagicMock()
        args.endpoint = "http://test.com"
        args.api_key = None
        args.model = None
        args.qps = None
        args.concurrency = None
        args.workers = None
        args.duration = None
        args.min_tokens = None
        args.max_tokens = None

        overrides = _extract_cli_overrides(args)

        assert overrides == {"endpoint": "http://test.com"}
        assert "api_key" not in overrides
        assert "model" not in overrides
        assert "qps" not in overrides

    def test_extract_missing_attributes(self):
        """Test with missing attributes."""
        args = MagicMock(spec=["endpoint"])  # Only has endpoint attribute
        args.endpoint = "http://test.com"

        overrides = _extract_cli_overrides(args)

        assert overrides == {"endpoint": "http://test.com"}


class TestRunBenchmarkCommand:
    """Test benchmark command error handling.

    These tests validate that all user input errors are caught early and
    raise InputValidationError, allowing main() to handle exits gracefully.
    Tests cover: missing mode, invalid config, missing endpoint/dataset.
    """

    @pytest.mark.asyncio
    async def test_missing_benchmark_mode_and_config(self):
        """Test error when neither mode nor config specified."""
        args = MagicMock()
        args.benchmark_mode = None
        args.config = None

        with pytest.raises(InputValidationError, match="Benchmark mode required"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_invalid_config_file(self):
        """Test error with invalid config file."""
        args = MagicMock()
        args.benchmark_mode = None
        args.config = Path("/nonexistent/config.yaml")
        args.endpoint = "http://test.com"

        with pytest.raises(InputValidationError, match="Config error"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_missing_endpoint_offline_mode(self):
        """Test error when endpoint not specified in offline mode."""
        args = MagicMock()
        args.benchmark_mode = "offline"
        args.config = None
        args.endpoint = None
        args.api_key = None
        args.dataset = None
        args.qps = None
        args.concurrency = None
        args.workers = None
        args.duration = None
        args.min_tokens = None
        args.max_tokens = None
        args.mode = None  # TestMode

        with pytest.raises(InputValidationError, match="Endpoint required"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_missing_dataset(self):
        """Test error when dataset not specified."""
        args = MagicMock()
        args.benchmark_mode = "offline"
        args.config = None
        args.endpoint = "http://test.com"
        args.dataset = None
        args.api_key = None
        args.target_qps = None
        args.concurrency = None
        args.workers = None
        args.duration = None
        args.min_tokens = None
        args.max_tokens = None
        args.mode = "perf"
        args.verbose = 0

        with pytest.raises(InputValidationError, match="Dataset required"):
            await run_benchmark_command(args)

    @pytest.mark.asyncio
    async def test_nonexistent_dataset_file(self):
        """Test error when dataset file doesn't exist."""
        args = MagicMock()
        args.benchmark_mode = "offline"
        args.config = None
        args.endpoint = "http://test.com"
        args.dataset = Path("/nonexistent/data.pkl")
        args.api_key = None
        args.target_qps = None
        args.concurrency = None
        args.workers = None
        args.duration = None
        args.min_tokens = None
        args.max_tokens = None
        args.mode = "perf"
        args.verbose = 0

        with pytest.raises(InputValidationError, match="Dataset not found"):
            await run_benchmark_command(args)

    # Note: Testing unsupported load patterns requires full integration
    # as it happens during scheduler creation after dataset loading.
    # This is covered by yaml_config.py tests and integration tests.
