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

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint import __version__
from inference_endpoint.commands.utils import (
    generate_user_conf_submission_checker,
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


class TestGenerateUserConfSubmissionChecker:
    """Test user.conf generation for submission checker.

    Validates that the user.conf file is generated correctly with proper
    key mapping from endpoints runtime settings to MLPerf loadgen format.
    This is critical for submission checker compatibility.
    """

    @pytest.fixture
    def sample_runtime_settings(self):
        """Sample runtime settings data for testing."""
        return {
            "n_samples_from_dataset": 1000,
            "n_samples_to_issue": 500,
            "total_samples_to_issue": 500,
            "max_duration_ms": 60000,
            "min_duration_ms": 30000,
            "min_sample_count": 100,
            "scheduler_random_seed": 42,
            "dataloader_random_seed": 123,
        }

    @pytest.fixture
    def report_dir_with_settings(self, tmp_path, sample_runtime_settings):
        """Create a report directory with runtime_settings.json."""
        report_dir = tmp_path / "test_report"
        report_dir.mkdir()

        runtime_settings_file = report_dir / "runtime_settings.json"
        with open(runtime_settings_file, "w") as f:
            json.dump(sample_runtime_settings, f)

        return report_dir

    def test_generate_user_conf_success(self, report_dir_with_settings):
        """Test successful user.conf generation."""
        # Generate user.conf
        generate_user_conf_submission_checker(report_dir_with_settings)

        # Check if user.conf exists
        user_conf_path = report_dir_with_settings / "user.conf"
        assert user_conf_path.exists(), "user.conf file should be created"

        # Read and verify contents
        content = user_conf_path.read_text()
        lines = content.strip().split("\n")

        # Verify file is not empty
        assert (
            len(lines) > 0
        ), "user.conf should not be empty when runtime_settings exists with data"

        # Verify format: each line should be in format "<text>.<text>.<text>=<value>"
        for line in lines:
            assert "=" in line, f"Line should contain '=' but got: {line}"
            key_part, value_part = line.split("=")
            # Should have at least 3 parts separated by dots
            parts = key_part.split(".")
            assert (
                len(parts) == 3
            ), f"Key should have format '<text>.<text>.<text>' but got: {key_part}"
            # Each part should be non-empty
            for part in parts:
                assert (
                    len(part) > 0
                ), f"Each part in key should be non-empty but got: {key_part}"
            # Value should not be empty
            assert len(value_part) > 0, f"Value should not be empty: {line}"

    def test_missing_runtime_settings_file(self, tmp_path):
        """Test error handling when runtime_settings.json is missing."""
        report_dir = tmp_path / "empty_report"
        report_dir.mkdir()

        # Should raise FileNotFoundError
        with pytest.raises(
            FileNotFoundError, match=f"runtime_settings.json not found in {report_dir}"
        ):
            generate_user_conf_submission_checker(report_dir)

        # user.conf should not be created
        user_conf_path = report_dir / "user.conf"
        assert (
            not user_conf_path.exists()
        ), "user.conf should not be created when runtime_settings.json is missing"

    def test_empty_runtime_settings(self, tmp_path):
        """Test handling of empty runtime settings."""
        report_dir = tmp_path / "empty_settings_report"
        report_dir.mkdir()

        # Create empty runtime_settings.json
        runtime_settings_file = report_dir / "runtime_settings.json"
        with open(runtime_settings_file, "w") as f:
            json.dump({}, f)

        # Should succeed but create empty user.conf
        generate_user_conf_submission_checker(report_dir)

        user_conf_path = report_dir / "user.conf"
        assert (
            user_conf_path.exists()
        ), "user.conf should be created even with empty settings"

        content = user_conf_path.read_text()
        assert (
            content.strip() == ""
        ), "user.conf should be empty when runtime_settings is empty"

    def test_user_conf_with_unmapped_keys(self, tmp_path):
        """Test that unmapped keys are included with their original names."""
        report_dir = tmp_path / "unmapped_report"
        report_dir.mkdir()

        # Create runtime_settings with both mapped and unmapped keys
        runtime_settings = {
            "n_samples_from_dataset": 1000,  # This will be mapped to qsl_reported_performance_count
            "custom_key": "custom_value",  # This should remain as-is
            "another_setting": 42,  # This should remain as-is
        }

        runtime_settings_file = report_dir / "runtime_settings.json"
        with open(runtime_settings_file, "w") as f:
            json.dump(runtime_settings, f)

        generate_user_conf_submission_checker(report_dir)

        user_conf_path = report_dir / "user.conf"
        content = user_conf_path.read_text()

        # Check mapped key
        assert "*.*.qsl_reported_performance_count=1000" in content

        # Check unmapped keys (should use original names)
        assert "*.*.custom_key=custom_value" in content
        assert "*.*.another_setting=42" in content

    def test_user_conf_overwrites_existing(self, report_dir_with_settings):
        """Test that generating user.conf overwrites existing file."""
        user_conf_path = report_dir_with_settings / "user.conf"

        # Create existing user.conf with different content
        user_conf_path.write_text("*.*.old_key=old_value\n")

        # Generate new user.conf
        generate_user_conf_submission_checker(report_dir_with_settings)

        # Read new content
        content = user_conf_path.read_text()

        # Should not contain old content
        assert "old_key" not in content
        assert "old_value" not in content

        # Should contain new content
        assert "*.*.qsl_reported_performance_count=1000" in content
