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

"""Tests for utility commands.

These tests verify the utility commands (info, validate, init) which are
essential for:
- User onboarding (init generates templates)
- Config validation before running benchmarks
- Version and system information

Testing these commands ensures users have a smooth experience when setting
up and validating their benchmark configurations.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint import __version__
from inference_endpoint.commands.utils import (
    monotime_to_datetime,
    run_info_command,
    run_init_command,
    run_validate_command,
)
from inference_endpoint.exceptions import InputValidationError


class TestRunInfoCommand:
    """Test info command.

    Validates that version and system information are displayed correctly
    with appropriate detail based on verbosity level.
    """

    @pytest.mark.asyncio
    async def test_info_basic(self, caplog):
        """Test basic info command shows version and system information."""
        args = MagicMock()
        args.verbose = 0

        with caplog.at_level("INFO"):
            await run_info_command(args)

        log_text = caplog.text
        # Use single source of truth for version
        assert __version__ in log_text
        assert "System Information" in log_text
        assert "Python Environment" in log_text

    @pytest.mark.asyncio
    async def test_info_verbose(self, caplog):
        """Test info command shows detailed system information."""
        args = MagicMock()
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_info_command(args)

        log_text = caplog.text
        # Should show version
        assert __version__ in log_text
        # Should show system details
        assert "Operating System:" in log_text
        assert "CPU:" in log_text
        assert "Memory:" in log_text


class TestRunValidateCommand:
    """Test validate command.

    Validates YAML config validation logic, ensuring invalid configs are
    caught before benchmark execution. Critical for preventing runtime
    errors due to misconfiguration.
    """

    @pytest.mark.asyncio
    async def test_validate_missing_config(self):
        """Test validate without config file."""
        args = MagicMock()
        args.config = None

        with pytest.raises(InputValidationError, match="Config file required"):
            await run_validate_command(args)

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self):
        """Test validate with non-existent file."""
        args = MagicMock()
        args.config = Path("/nonexistent/file.yaml")
        args.verbose = 0

        with pytest.raises(InputValidationError, match="not found"):
            await run_validate_command(args)

    @pytest.mark.asyncio
    async def test_validate_success(self, tmp_path):
        """Test successful validation."""
        # Create valid config
        config_content = """
name: "test"
type: "offline"

datasets:
  - name: "test"
    type: "performance"
    path: "test.pkl"
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        args = MagicMock()
        args.config = config_file
        args.verbose = 0

        # Should not raise
        await run_validate_command(args)


class TestRunInitCommand:
    """Test init command.

    Validates template generation which is key for user onboarding.
    Users should be able to quickly generate valid YAML configs as starting
    points for their benchmarks.
    """

    @pytest.mark.asyncio
    async def test_init_unknown_template(self):
        """Test init with unknown template type."""
        args = MagicMock()
        args.template = "unknown"

        with pytest.raises(InputValidationError, match="Unknown template"):
            await run_init_command(args)

    @pytest.mark.asyncio
    async def test_init_success(self):
        """Test successful template generation."""

        args = MagicMock()
        args.template = "offline"

        output_file = Path(f"{args.template}_template.yaml")

        try:
            await run_init_command(args)

            assert output_file.exists()
            content = output_file.read_text()
            assert "offline-benchmark" in content
            assert "max_throughput" in content
        finally:
            output_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_init_warns_on_overwrite(self, caplog):
        """Test warning when file already exists."""

        args = MagicMock()
        args.template = "online"

        output_file = Path(f"{args.template}_template.yaml")
        output_file.write_text("existing content")

        try:
            await run_init_command(args)

            assert "will be overwritten" in caplog.text
            # File should be replaced
            assert "online-benchmark" in output_file.read_text()
        finally:
            output_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_init_all_templates(self):
        """Test generating all template types."""
        templates = ["offline", "online", "eval", "submission"]

        for template_type in templates:
            output_file = Path(f"{template_type}_template.yaml")
            args = MagicMock()
            args.template = template_type

            try:
                await run_init_command(args)

                assert output_file.exists()
                assert output_file.stat().st_size > 0
            finally:
                output_file.unlink(missing_ok=True)


class TestMonotimeToDatetime:
    """Test monotime_to_datetime conversion for past and current monotonic times."""

    def test_monotime_to_datetime_backward_past_time(self):
        """Past monotonic time converts to a datetime in the past relative to now."""
        # Monotonic time 2 seconds ago
        mono_now_ns = time.monotonic_ns()
        past_mono_ns = mono_now_ns - 2 * 10**9  # 2 seconds in nanoseconds

        result = monotime_to_datetime(past_mono_ns)
        now = datetime.now()
        delta = now - result

        # Result should be roughly 2 seconds before now (allow 0.5s tolerance)
        assert timedelta(seconds=1.5) <= delta <= timedelta(seconds=2.5)

    def test_monotime_to_datetime_forward_current_time(self):
        """Current monotonic time converts to a datetime close to now."""
        mono_now_ns = time.monotonic_ns()

        result = monotime_to_datetime(mono_now_ns)
        now = datetime.now()
        delta = abs((now - result).total_seconds())

        # Result should be within 5 second of now (conversion and execution delay)
        assert delta <= 5.0
