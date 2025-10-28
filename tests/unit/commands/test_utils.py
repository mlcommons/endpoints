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

"""Tests for utility commands.

These tests verify the utility commands (info, validate, init) which are
essential for:
- User onboarding (init generates templates)
- Config validation before running benchmarks
- Version and system information

Testing these commands ensures users have a smooth experience when setting
up and validating their benchmark configurations.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint import __version__
from inference_endpoint.commands.utils import (
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
        """Test basic info command shows version and status."""
        args = MagicMock()
        args.verbose = 0

        with caplog.at_level("INFO"):
            await run_info_command(args)

        log_text = caplog.text
        # Use single source of truth for version
        assert __version__ in log_text
        assert "Operational" in log_text

    @pytest.mark.asyncio
    async def test_info_verbose(self, caplog):
        """Test info command with verbose mode shows additional details."""
        args = MagicMock()
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_info_command(args)

        log_text = caplog.text
        # Should show version
        assert __version__ in log_text
        # Should show architecture/capabilities in verbose mode
        assert "Architecture:" in log_text or "Capabilities:" in log_text


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
        args.output = None

        with pytest.raises(InputValidationError, match="Unknown template"):
            await run_init_command(args)

    @pytest.mark.asyncio
    async def test_init_success(self, tmp_path):
        """Test successful template generation."""
        output_file = tmp_path / "test_template.yaml"

        args = MagicMock()
        args.template = "offline"
        args.output = str(output_file)

        await run_init_command(args)

        assert output_file.exists()
        content = output_file.read_text()
        assert "offline-benchmark" in content
        assert "max_throughput" in content

    @pytest.mark.asyncio
    async def test_init_warns_on_overwrite(self, tmp_path, caplog):
        """Test warning when file already exists."""
        output_file = tmp_path / "existing.yaml"
        output_file.write_text("existing content")

        args = MagicMock()
        args.template = "online"
        args.output = str(output_file)

        await run_init_command(args)

        assert "will be overwritten" in caplog.text
        # File should be replaced
        assert "online-benchmark" in output_file.read_text()

    @pytest.mark.asyncio
    async def test_init_all_templates(self, tmp_path):
        """Test generating all template types."""
        templates = ["offline", "online", "eval", "submission"]

        for template_type in templates:
            output_file = tmp_path / f"{template_type}_test.yaml"

            args = MagicMock()
            args.template = template_type
            args.output = str(output_file)

            await run_init_command(args)

            assert output_file.exists()
            assert output_file.stat().st_size > 0
