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

"""Integration tests for probe command against echo server.

These tests verify the probe command works end-to-end with a real HTTP server,
validating:
- Successful probes with various request counts
- Response collection and display
- Latency measurement
- Error handling with failing endpoints
"""

import pytest
from inference_endpoint.commands.probe import ProbeConfig, execute_probe
from inference_endpoint.config.schema import APIType
from inference_endpoint.exceptions import ExecutionError, InputValidationError


class TestProbeCommandIntegration:
    """Integration tests for probe command with echo server."""

    @pytest.mark.integration
    def test_probe_with_echo_server(self, mock_http_echo_server, caplog):
        """Test successful probe against echo server."""
        config = ProbeConfig(
            endpoints=mock_http_echo_server.url,
            model="gpt-3.5-turbo",
            requests=5,
            prompt="Test probe message",
        )

        with caplog.at_level("INFO"):
            execute_probe(config)

            log_text = caplog.text
            assert "Completed: 5/5 successful" in log_text
            assert "Avg latency:" in log_text
            assert "Sample responses" in log_text
            assert "Test probe message" in log_text
            assert "Probe successful" in log_text

    @pytest.mark.integration
    def test_probe_with_default_prompt(self, mock_http_echo_server, caplog):
        """Test probe with default prompt."""
        config = ProbeConfig(
            endpoints=mock_http_echo_server.url,
            model="gpt-3.5-turbo",
            requests=3,
        )

        with caplog.at_level("INFO"):
            execute_probe(config)

            log_text = caplog.text
            assert "Completed: 3/3 successful" in log_text
            assert "joke in 30 words" in log_text

    @pytest.mark.integration
    def test_probe_shows_multiple_responses(self, mock_http_echo_server, caplog):
        """Test that probe shows sample responses."""
        config = ProbeConfig(
            endpoints=mock_http_echo_server.url,
            model="gpt-3.5-turbo",
            requests=15,
            prompt="Sample response text",
        )

        with caplog.at_level("INFO"):
            execute_probe(config)

            assert "Sample responses (15 collected)" in caplog.text
            assert "[probe-0]" in caplog.text
            assert "Sample response text" in caplog.text

    @pytest.mark.integration
    def test_probe_rejects_videogen_api_type(self):
        """Probe assumes second-scale latencies; videogen requests run for minutes."""
        config = ProbeConfig(
            endpoints="http://localhost:8000",
            model="wan22",
            api_type=APIType.VIDEOGEN,
            requests=1,
            prompt="a cat",
        )
        with pytest.raises(InputValidationError, match="videogen"):
            execute_probe(config)

    @pytest.mark.integration
    def test_probe_with_invalid_endpoint(self):
        """Test probe fails gracefully with invalid endpoint."""
        config = ProbeConfig(
            endpoints="http://invalid-host-does-not-exist:9999",
            model="gpt-3.5-turbo",
            requests=3,
            prompt="Test",
        )

        # With lazy connection pooling, client creation succeeds but requests fail
        # during execution when the worker can't resolve the hostname
        with pytest.raises(ExecutionError, match="Probe failed"):
            execute_probe(config)

    @pytest.mark.integration
    def test_probe_with_custom_prompt(self, mock_http_echo_server, caplog):
        """Test probe with custom prompt text."""
        custom_prompt = "This is my custom probe message with special chars: @#$%"

        config = ProbeConfig(
            endpoints=mock_http_echo_server.url,
            model="gpt-3.5-turbo",
            requests=2,
            prompt=custom_prompt,
        )

        with caplog.at_level("INFO"):
            execute_probe(config)

            log_text = caplog.text
            # Echo server should return the prompt
            assert custom_prompt in log_text
            assert "Probe successful" in log_text
