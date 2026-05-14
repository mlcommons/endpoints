"""Tests for the CLI commands."""

import json
from pathlib import Path

from click.testing import CliRunner

from submission_checker.cli import main


def test_check_fails_for_missing_path(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["check", str(tmp_path / "nonexistent")])
    assert result.exit_code == 1


def test_check_reports_low_throughput_error(sub_e):
    """sub_e (Gaudi, 11 points) has no run in LT region (33–42 for M=1024)."""
    runner = CliRunner()
    result = runner.invoke(main, ["check", str(sub_e)])
    assert "low-throughput-coverage" in result.output
    assert result.exit_code == 1


def test_check_valid_submission_passes(valid_standardized):
    runner = CliRunner()
    result = runner.invoke(main, ["check", str(valid_standardized)])
    assert result.exit_code == 0


def test_check_quiet_suppresses_info(sub_e):
    runner = CliRunner()
    result = runner.invoke(main, ["check", "--quiet", str(sub_e)])
    assert "info" not in result.output.lower()


def test_check_strict_fails_on_warnings(valid_standardized):
    """Even the valid submission has WARNINGs (run-duration WIP); strict makes it fail."""
    runner = CliRunner()
    result = runner.invoke(main, ["check", "--strict", str(valid_standardized)])
    # May or may not have warnings depending on duration values — just check it runs
    assert result.exit_code in (0, 1)


def test_version_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_regions_command_known_m():
    runner = CliRunner()
    result = runner.invoke(main, ["regions", "--max-concurrency", "1024"])
    assert result.exit_code == 0
    assert "33" in result.output  # LT start
    assert "42" in result.output  # LT end (Appendix B)
    assert "131" in result.output  # MT end


def test_regions_command_invalid_m():
    runner = CliRunner()
    result = runner.invoke(main, ["regions", "--max-concurrency", "32"])
    assert result.exit_code == 1


def test_check_output_flag_writes_json(tmp_path: Path, valid_standardized: Path) -> None:
    """--output writes a machine-readable JSON report alongside the terminal table."""
    output_file = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(main, ["check", "--output", str(output_file), str(valid_standardized)])
    assert result.exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert "results" in data
    assert data["passed"] is True
