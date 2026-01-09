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

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inference_endpoint import __version__
from inference_endpoint.commands.utils import (
    generate_mlperf_log_details_submission_checker,
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
        """Test successful user.conf generation with correct key mapping.

        Verifies that:
        1. user.conf file is created
        2. All entries have correct format (*.*.loadgen_key=value)
        3. All keys in sample_runtime_settings are correctly transformed
        4. Mapped keys use their loadgen equivalents per ENDPOINTS_TO_LOADGEN_KEY_MAPPING
        5. Unmapped keys use their original names
        """
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

        # Verify format and check key mappings

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

        # Parse content into a dictionary for easier verification
        content_dict = {}
        for line in lines:
            key_part, value_part = line.split("=")
            # Extract the actual key (after the first two dots which are wildcards)
            actual_key = ".".join(key_part.split(".")[2:])
            content_dict[actual_key] = value_part

        # Verify all mappings from sample_runtime_settings
        # Expected mappings based on ENDPOINTS_TO_LOADGEN_KEY_MAPPING:
        expected_mappings = {
            # Mapped keys (will use loadgen names)
            "qsl_reported_performance_count": "1000",  # from n_samples_from_dataset
            "effective_max_duration_ms": "60000",  # from max_duration_ms
            "effective_min_duration_ms": "30000",  # from min_duration_ms
            "effective_min_query_count": "100",  # from min_sample_count
            # Unmapped keys (will use original names)
            "n_samples_to_issue": "500",
            "total_samples_to_issue": "500",
            "scheduler_random_seed": "42",
            "dataloader_random_seed": "123",
        }

        # Verify each expected mapping
        for expected_key, expected_value in expected_mappings.items():
            assert (
                expected_key in content_dict
            ), f"Expected key '{expected_key}' not found in user.conf. Available keys: {list(content_dict.keys())}"
            assert (
                content_dict[expected_key] == expected_value
            ), f"Key '{expected_key}' should have value '{expected_value}' but got '{content_dict[expected_key]}'"

        # Verify that the correct number of keys are present
        assert (
            len(content_dict) == len(expected_mappings)
        ), f"Expected {len(expected_mappings)} keys but got {len(content_dict)}: {list(content_dict.keys())}"

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


class TestGenerateMlperfLogDetailsSubmissionChecker:
    """Test mlperf_log_details.txt generation for submission checker.

    Validates that the mlperf_log_details.txt file is generated correctly with
    proper key mapping from endpoints summary to MLPerf loadgen format.
    This is critical for submission checker compatibility.
    """

    @pytest.fixture
    def sample_summary_data(self):
        """Sample summary data for testing (ENDPTS format)."""
        return [
            {
                "key": "endpoints_version",
                "value": "5.0.25",
                "time_ms": 0.009344,
                "namespace": "mlperf::logging",
                "event_type": "POINT_IN_TIME",
                "metadata": {"is_error": False, "is_warning": False},
            },
            {
                "key": "n_samples_from_dataset",
                "value": 1000,
                "time_ms": 0.021440,
                "namespace": "mlperf::logging",
                "event_type": "POINT_IN_TIME",
                "metadata": {"is_error": False, "is_warning": False},
            },
            {
                "key": "effective_scenario",
                "value": "Offline",
                "time_ms": 0.032160,
                "namespace": "mlperf::logging",
                "event_type": "POINT_IN_TIME",
                "metadata": {"is_error": False, "is_warning": False},
            },
            {
                "key": "custom_key",
                "value": "custom_value",
                "time_ms": 0.050000,
                "namespace": "mlperf::logging",
                "event_type": "POINT_IN_TIME",
                "metadata": {"is_error": False, "is_warning": False},
            },
        ]

    @pytest.fixture
    def report_dir_with_summary(self, tmp_path, sample_summary_data):
        """Create a report directory with summary.json in ENDPTS format."""
        report_dir = tmp_path / "test_report"
        report_dir.mkdir()

        summary_file = report_dir / "summary.json"
        marker = ":::ENDPTS"
        with open(summary_file, "w") as f:
            for record in sample_summary_data:
                f.write(f"{marker} {json.dumps(record)}\n")

        return report_dir

    def test_generate_mlperf_log_details_success(
        self, report_dir_with_summary, sample_summary_data
    ):
        """Test successful mlperf_log_details.txt generation with correct key mapping.

        Verifies that:
        1. mlperf_log_details.txt file is created
        2. All lines start with :::ENDPTS marker
        3. All records are valid JSON
        4. Mapped keys use their loadgen equivalents
        5. Unmapped keys use their original names
        6. Record structure is preserved
        """
        # Generate mlperf_log_details.txt
        generate_mlperf_log_details_submission_checker(
            report_dir_with_summary, strict=True
        )

        # Check if mlperf_log_details.txt exists
        log_details_path = report_dir_with_summary / "mlperf_log_details.txt"
        assert (
            log_details_path.exists()
        ), "mlperf_log_details.txt file should be created"

        # Read and verify contents
        content = log_details_path.read_text()
        lines = content.strip().split("\n")

        # Verify file is not empty
        assert (
            len(lines) > 0
        ), "mlperf_log_details.txt should not be empty when summary exists with data"

        marker = ":::ENDPTS"
        records = []

        # Verify format and parse records
        for line in lines:
            assert line.startswith(
                marker
            ), f"Line should start with '{marker}' but got: {line}"

            # Extract JSON part
            json_str = line[len(marker) :].strip()
            try:
                record = json.loads(json_str)
                records.append(record)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in line: {line}. Error: {e}")

            # Verify record structure
            assert "key" in record, f"Record should have 'key' field: {record}"
            assert "value" in record, f"Record should have 'value' field: {record}"

        # Verify correct number of records
        assert len(records) == len(
            sample_summary_data
        ), f"Expected {len(sample_summary_data)} records but got {len(records)}"

        # Verify key mappings
        # endpoints_version should map to loadgen_version
        version_record = next(
            (r for r in records if r["key"] == "loadgen_version"), None
        )
        assert (
            version_record is not None
        ), "endpoints_version should be mapped to loadgen_version"
        assert version_record["value"] == "5.0.25"

        # n_samples_from_dataset should map to qsl_reported_performance_count
        samples_record = next(
            (r for r in records if r["key"] == "qsl_reported_performance_count"), None
        )
        assert (
            samples_record is not None
        ), "n_samples_from_dataset should be mapped to qsl_reported_performance_count"
        assert samples_record["value"] == 1000

        # effective_scenario should remain as-is (not in mapping)
        scenario_record = next(
            (r for r in records if r["key"] == "effective_scenario"), None
        )
        assert scenario_record is not None, "effective_scenario should remain unmapped"
        assert scenario_record["value"] == "Offline"

        # custom_key should remain as-is (not in mapping)
        custom_record = next((r for r in records if r["key"] == "custom_key"), None)
        assert custom_record is not None, "custom_key should remain unmapped"
        assert custom_record["value"] == "custom_value"

    def test_missing_summary_file(self, tmp_path):
        """Test error handling when summary.json is missing."""
        report_dir = tmp_path / "empty_report"
        report_dir.mkdir()

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="summary.json not found in"):
            generate_mlperf_log_details_submission_checker(report_dir, strict=True)

        # mlperf_log_details.txt should not be created
        log_details_path = report_dir / "mlperf_log_details.txt"
        assert (
            not log_details_path.exists()
        ), "mlperf_log_details.txt should not be created when summary.json is missing"

    def test_empty_summary_file(self, tmp_path):
        """Test handling of empty summary file."""
        report_dir = tmp_path / "empty_summary_report"
        report_dir.mkdir()

        # Create empty summary.json
        summary_file = report_dir / "summary.json"
        summary_file.write_text("")

        # Should succeed but create empty mlperf_log_details.txt
        generate_mlperf_log_details_submission_checker(report_dir, strict=True)

        log_details_path = report_dir / "mlperf_log_details.txt"
        assert (
            log_details_path.exists()
        ), "mlperf_log_details.txt should be created even with empty summary"

        content = log_details_path.read_text()
        assert (
            content.strip() == ""
        ), "mlperf_log_details.txt should be empty when summary is empty"

    def test_strict_mode_invalid_json(self, tmp_path):
        """Test strict mode raises error on invalid JSON."""
        report_dir = tmp_path / "invalid_json_report"
        report_dir.mkdir()

        # Create summary with invalid JSON
        summary_file = report_dir / "summary.json"
        marker = ":::ENDPTS"
        with open(summary_file, "w") as f:
            f.write(f"{marker} {{invalid json\n")

        # Should raise json.JSONDecodeError in strict mode
        with pytest.raises(json.JSONDecodeError):
            generate_mlperf_log_details_submission_checker(report_dir, strict=True)

    def test_non_strict_mode_invalid_json(self, tmp_path, caplog):
        """Test non-strict mode skips invalid JSON lines with warning."""
        report_dir = tmp_path / "invalid_json_report"
        report_dir.mkdir()

        # Create summary with mix of valid and invalid JSON
        summary_file = report_dir / "summary.json"
        marker = ":::ENDPTS"
        valid_record = {"key": "test_key", "value": "test_value"}
        with open(summary_file, "w") as f:
            f.write(f"{marker} {json.dumps(valid_record)}\n")
            f.write(f"{marker} invalid json\n")
            f.write(f"{marker} {json.dumps(valid_record)}\n")

        # Should succeed in non-strict mode
        with caplog.at_level("WARNING"):
            generate_mlperf_log_details_submission_checker(report_dir, strict=False)

        # Should have warning about invalid line
        assert any(
            "Skipping invalid line" in record.message for record in caplog.records
        ), "Should have warning about skipping invalid lines"

        # Check output file only has valid records
        log_details_path = report_dir / "mlperf_log_details.txt"
        content = log_details_path.read_text()
        print(content)
        lines = [line for line in content.strip().split("\n") if line]
        assert (
            len(lines) == 2
        ), f"{content}\nShould only have valid records in output (invalid line skipped)"

    def test_lines_without_marker(self, tmp_path):
        """Test that lines without marker are ignored."""
        report_dir = tmp_path / "marker_report"
        report_dir.mkdir()

        # Create summary with lines both with and without marker
        summary_file = report_dir / "summary.json"
        marker = ":::ENDPTS"
        valid_record = {"key": "test_key", "value": "test_value"}
        with open(summary_file, "w") as f:
            f.write(f"{marker} {json.dumps(valid_record)}\n")
            f.write("This is a comment without marker\n")
            f.write(f"{marker} {json.dumps(valid_record)}\n")

        generate_mlperf_log_details_submission_checker(report_dir, strict=True)

        log_details_path = report_dir / "mlperf_log_details.txt"
        content = log_details_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) == 2, "Should only include lines with marker in output"

    def test_mlperf_log_details_overwrites_existing(self, report_dir_with_summary):
        """Test that generating mlperf_log_details.txt overwrites existing file."""
        log_details_path = report_dir_with_summary / "mlperf_log_details.txt"

        # Create existing mlperf_log_details.txt with different content
        log_details_path.write_text(':::ENDPTS {"key":"old_key","value":"old_value"}\n')

        # Generate new mlperf_log_details.txt
        generate_mlperf_log_details_submission_checker(
            report_dir_with_summary, strict=True
        )

        # Read new content
        content = log_details_path.read_text()

        # Should not contain old content
        assert "old_key" not in content
        assert "old_value" not in content

        # Should contain new content
        assert "endpoints_version" in content or "loadgen_version" in content

    def test_unmapped_keys_preserved(self, tmp_path):
        """Test that unmapped keys are preserved with original names."""
        report_dir = tmp_path / "unmapped_report"
        report_dir.mkdir()

        # Create summary with unmapped keys
        summary_file = report_dir / "summary.json"
        marker = ":::ENDPTS"
        records = [
            {"key": "unmapped_key_1", "value": "value1"},
            {"key": "custom_metric", "value": 42},
            {"key": "another_custom", "value": "data"},
        ]
        with open(summary_file, "w") as f:
            for record in records:
                f.write(f"{marker} {json.dumps(record)}\n")

        generate_mlperf_log_details_submission_checker(report_dir, strict=True)

        log_details_path = report_dir / "mlperf_log_details.txt"
        content = log_details_path.read_text()

        # Check that unmapped keys are preserved
        assert "unmapped_key_1" in content
        assert "custom_metric" in content
        assert "another_custom" in content

    def test_json_output_format(self, report_dir_with_summary):
        """Test that output records are valid compact JSON format."""
        generate_mlperf_log_details_submission_checker(
            report_dir_with_summary, strict=True
        )

        log_details_path = report_dir_with_summary / "mlperf_log_details.txt"
        content = log_details_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        marker = ":::ENDPTS"
        for line in lines:
            # Extract JSON part
            json_str = line[len(marker) :].strip()

            # Verify JSON is parseable
            json.loads(json_str)

            # Verify no spaces after separators (compact format)
            # Should have format like {"key":"value","time_ms":123}
            assert (
                ", " not in json_str
            ), f"JSON should be compact without spaces: {json_str}"

    def test_metadata_preservation(self, tmp_path):
        """Test that metadata and other fields are preserved in output."""
        report_dir = tmp_path / "metadata_report"
        report_dir.mkdir()

        # Create summary with various fields
        summary_file = report_dir / "summary.json"
        marker = ":::ENDPTS"
        record = {
            "key": "test_key",
            "value": "test_value",
            "time_ms": 1234.567,
            "namespace": "mlperf::logging",
            "event_type": "POINT_IN_TIME",
            "metadata": {
                "is_error": False,
                "is_warning": True,
                "file": "test.cc",
                "line_no": 42,
            },
        }
        with open(summary_file, "w") as f:
            f.write(f"{marker} {json.dumps(record)}\n")

        generate_mlperf_log_details_submission_checker(report_dir, strict=True)

        log_details_path = report_dir / "mlperf_log_details.txt"
        content = log_details_path.read_text().strip()

        # Extract and verify record
        json_str = content[len(marker) :].strip()
        output_record = json.loads(json_str)

        # All fields except 'key' should be preserved
        assert output_record["value"] == "test_value"
        assert output_record["time_ms"] == 1234.567
        assert output_record["namespace"] == "mlperf::logging"
        assert output_record["event_type"] == "POINT_IN_TIME"
        assert output_record["metadata"]["is_warning"] is True
        assert output_record["metadata"]["line_no"] == 42

    def test_multiple_mapped_keys(self, tmp_path):
        """Test multiple keys with different mapping scenarios."""
        report_dir = tmp_path / "multiple_keys_report"
        report_dir.mkdir()

        # Create summary with multiple mapped and unmapped keys
        summary_file = report_dir / "summary.json"
        marker = ":::ENDPTS"
        records = [
            {"key": "endpoints_version", "value": "5.0.25"},
            {"key": "n_samples_from_dataset", "value": 2000},
            {"key": "effective_scenario", "value": "Offline"},
            {"key": "qps", "value": 100},
            {"key": "latency.min", "value": 10},
            {"key": "latency.max", "value": 100},
            {"key": "custom_user_metric", "value": "user_data"},
        ]
        with open(summary_file, "w") as f:
            for record in records:
                f.write(f"{marker} {json.dumps(record)}\n")

        generate_mlperf_log_details_submission_checker(report_dir, strict=True)

        log_details_path = report_dir / "mlperf_log_details.txt"
        content = log_details_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        # Parse all records
        parsed_records = []
        for line in lines:
            json_str = line[len(marker) :].strip()
            parsed_records.append(json.loads(json_str))

        # Verify mappings
        keys_in_output = {r["key"] for r in parsed_records}

        # Mapped keys should use loadgen names
        assert "loadgen_version" in keys_in_output
        assert "qsl_reported_performance_count" in keys_in_output
        assert "result_completed_samples_per_sec" in keys_in_output
        assert "result_min_latency_ns" in keys_in_output
        assert "result_max_latency_ns" in keys_in_output

        # Unmapped keys should use original names
        assert "effective_scenario" in keys_in_output
        assert "custom_user_metric" in keys_in_output

        # Original mapped names should not be present
        assert "endpoints_version" not in keys_in_output
        assert "n_samples_from_dataset" not in keys_in_output
        assert "qps" not in keys_in_output
        assert "latency.min" not in keys_in_output
        assert "latency.max" not in keys_in_output
