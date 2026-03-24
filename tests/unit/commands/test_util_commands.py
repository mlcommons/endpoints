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

"""Tests for utility commands (info, validate, init, probe) and main.py dispatch."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from inference_endpoint import __version__
from inference_endpoint.commands.info import execute_info
from inference_endpoint.commands.init import execute_init
from inference_endpoint.commands.probe import ProbeConfig, execute_probe
from inference_endpoint.commands.validate import execute_validate
from inference_endpoint.config.schema import APIType
from inference_endpoint.exceptions import (
    CLIError,
    ExecutionError,
    InputValidationError,
    SetupError,
)


class TestInfoCommand:
    @pytest.mark.unit
    def test_shows_version_and_system_info(self, caplog):
        with caplog.at_level("INFO"):
            execute_info()
        assert __version__ in caplog.text
        assert "System Information" in caplog.text
        assert "Operating System:" in caplog.text
        assert "CPU:" in caplog.text
        assert "Memory:" in caplog.text


class TestValidateCommand:
    @pytest.mark.unit
    def test_nonexistent_file(self):
        with pytest.raises(InputValidationError, match="not found"):
            execute_validate(Path("/nonexistent/file.yaml"))

    @pytest.mark.unit
    def test_valid_config(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
type: "offline"
model_params:
  name: "test-model"
endpoint_config:
  endpoints: ["http://localhost:8000"]
datasets:
  - path: "test.pkl"
""")
        execute_validate(config_file)

    @pytest.mark.unit
    def test_submission_ref_logging(self, tmp_path, caplog):
        config_file = tmp_path / "sub.yaml"
        config_file.write_text("""
type: "submission"
benchmark_mode: "offline"
model_params:
  name: "llama"
endpoint_config:
  endpoints: ["http://localhost:8000"]
datasets:
  - path: "test.pkl"
submission_ref:
  model: "llama"
  ruleset: "mlperf-v6"
""")
        with caplog.at_level("INFO"):
            execute_validate(config_file)
        assert "Submission:" in caplog.text
        assert "mlperf-v6" in caplog.text

    @pytest.mark.unit
    def test_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml")
        with pytest.raises(InputValidationError, match="validation failed"):
            execute_validate(config_file)


class TestInitCommand:
    @pytest.mark.unit
    def test_unknown_template(self):
        with pytest.raises(InputValidationError, match="Unknown template"):
            execute_init("unknown")

    @pytest.mark.unit
    @pytest.mark.parametrize("template", ["offline", "online", "eval", "submission"])
    def test_generates_template(self, template):
        output_file = Path(f"{template}_template.yaml")
        try:
            execute_init(template)
            assert output_file.exists()
            assert output_file.stat().st_size > 0
        finally:
            output_file.unlink(missing_ok=True)

    @pytest.mark.unit
    def test_warns_on_overwrite(self, caplog):
        output_file = Path("online_template.yaml")
        output_file.write_text("existing")
        try:
            execute_init("online")
            assert "will be overwritten" in caplog.text
        finally:
            output_file.unlink(missing_ok=True)

    @pytest.mark.unit
    def test_fallback_when_template_missing(self, tmp_path, monkeypatch):
        """When template file doesn't exist, falls back to create_default_config."""
        monkeypatch.setattr(
            "inference_endpoint.commands.init.TEMPLATES_DIR",
            tmp_path / "nonexistent",
        )
        output_file = Path("offline_template.yaml")
        try:
            execute_init("offline")
            assert output_file.exists()
        finally:
            output_file.unlink(missing_ok=True)

    @pytest.mark.unit
    def test_os_error_raises_setup_error(self, monkeypatch):
        monkeypatch.setattr(
            "inference_endpoint.commands.init.TEMPLATES_DIR",
            Path("/nonexistent"),
        )
        monkeypatch.setattr(
            "inference_endpoint.commands.init.BenchmarkConfig.create_default_config",
            MagicMock(side_effect=OSError("permission denied")),
        )
        with pytest.raises(SetupError, match="Failed to create"):
            execute_init("offline")


class TestProbeConfig:
    @pytest.mark.unit
    def test_defaults(self):
        config = ProbeConfig(endpoints="http://localhost:8000", model="test")
        assert config.api_type == APIType.OPENAI
        assert config.requests == 10

    @pytest.mark.unit
    def test_sglang_api_type(self):
        config = ProbeConfig(
            endpoints="http://localhost:8000", model="test", api_type=APIType.SGLANG
        )
        assert config.api_type == APIType.SGLANG


class TestProbeExecution:
    @pytest.mark.unit
    @patch("inference_endpoint.commands.probe.run_async")
    def test_execute_probe_calls_async(self, mock_run_async):
        config = ProbeConfig(endpoints="http://localhost:8000", model="test")
        execute_probe(config)
        mock_run_async.assert_called_once()

    @pytest.mark.unit
    def test_empty_model_raises(self):
        config = ProbeConfig(endpoints="http://localhost:8000", model="")
        with pytest.raises(InputValidationError, match="Model required"):
            import asyncio

            from inference_endpoint.commands.probe import _probe_async

            asyncio.run(_probe_async(config))

    @pytest.mark.unit
    @patch("inference_endpoint.commands.probe.HTTPEndpointClient")
    def test_setup_failure_raises(self, mock_client_cls):
        mock_client_cls.side_effect = ConnectionError("refused")

        config = ProbeConfig(endpoints="http://localhost:8000", model="test")
        with pytest.raises(SetupError, match="Probe setup failed"):
            import asyncio

            from inference_endpoint.commands.probe import _probe_async

            asyncio.run(_probe_async(config))

    @pytest.mark.unit
    @patch("inference_endpoint.commands.probe.HTTPEndpointClient")
    def test_all_issues_fail_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.issue.side_effect = RuntimeError("send failed")
        mock_client_cls.return_value = mock_client

        config = ProbeConfig(
            endpoints="http://localhost:8000", model="test", requests=2
        )
        with pytest.raises(ExecutionError, match="no queries could be issued"):
            import asyncio

            from inference_endpoint.commands.probe import _probe_async

            asyncio.run(_probe_async(config))


class TestMainRunExceptionHandling:
    """Test that main.run() maps exceptions to correct exit codes."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "exc, code",
        [
            (InputValidationError("bad input"), 2),
            (SetupError("setup failed"), 3),
            (ExecutionError("exec failed"), 4),
            (CLIError("cli error"), 1),
            (NotImplementedError("not impl"), 1),
            (RuntimeError("unexpected"), 1),
        ],
    )
    def test_exception_exit_codes(self, exc, code):
        from inference_endpoint.main import run

        with patch("inference_endpoint.main.app") as mock_app:
            mock_app.meta.side_effect = exc
            with pytest.raises(SystemExit) as exc_info:
                run()
            assert exc_info.value.code == code
