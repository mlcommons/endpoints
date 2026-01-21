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

from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.probe import run_probe_command
from inference_endpoint.exceptions import ExecutionError


class TestProbeCommandIntegration:
    """Integration tests for probe command with echo server."""

    @pytest.mark.asyncio
    async def test_probe_with_echo_server(self, mock_http_echo_server, caplog):
        """Test successful probe against echo server."""
        args = MagicMock()
        args.endpoint = mock_http_echo_server.url
        args.api_type = "openai"
        args.requests = 5
        args.prompt = "Test probe message"
        args.model = "gpt-3.5-turbo"  # Required
        args.verbose = 1

        with caplog.at_level("INFO"):
            # Should complete successfully
            await run_probe_command(args)

            log_text = caplog.text
            # Verify success indicators
            assert "✓ Completed: 5/5 successful" in log_text
            assert "✓ Avg latency:" in log_text
            assert "✓ Sample responses" in log_text
            assert "Test probe message" in log_text
            assert "✓ Probe successful" in log_text

    @pytest.mark.asyncio
    async def test_probe_with_default_prompt(self, mock_http_echo_server, caplog):
        """Test probe with default prompt."""
        args = MagicMock()
        args.endpoint = mock_http_echo_server.url
        args.api_type = "openai"
        args.requests = 3
        args.prompt = "Please write me a joke in 30 words."  # Default
        args.model = "gpt-3.5-turbo"  # Required
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_probe_command(args)

            log_text = caplog.text
            assert "✓ Completed: 3/3 successful" in log_text
            assert "joke in 30 words" in log_text

    @pytest.mark.asyncio
    async def test_probe_shows_multiple_responses(self, mock_http_echo_server, caplog):
        """Test that probe shows sample responses."""
        args = MagicMock()
        args.endpoint = mock_http_echo_server.url
        args.api_type = "openai"
        args.requests = 15  # More than 10 to test truncation
        args.prompt = "Sample response text"
        args.model = "gpt-3.5-turbo"  # Required
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_probe_command(args)

            # Should show up to 10 responses
            assert "Sample responses (15 collected)" in caplog.text
            assert "[probe-0]" in caplog.text
            assert "Sample response text" in caplog.text

    @pytest.mark.asyncio
    async def test_probe_with_invalid_endpoint(self, caplog):
        """Test probe fails gracefully with invalid endpoint."""
        args = MagicMock()
        args.endpoint = "http://invalid-host-does-not-exist:9999"
        args.api_type = "openai"
        args.requests = 3
        args.prompt = "Test"
        args.model = "gpt-3.5-turbo"  # Required
        args.verbose = 0

        # With lazy connection pooling, client creation succeeds but requests fail
        # during execution when the worker can't resolve the hostname
        with pytest.raises(ExecutionError, match="Probe failed"):
            await run_probe_command(args)

    @pytest.mark.asyncio
    async def test_probe_with_custom_prompt(self, mock_http_echo_server, caplog):
        """Test probe with custom prompt text."""
        custom_prompt = "This is my custom probe message with special chars: @#$%"

        args = MagicMock()
        args.endpoint = mock_http_echo_server.url
        args.api_type = "openai"
        args.requests = 2
        args.prompt = custom_prompt
        args.model = "gpt-3.5-turbo"  # Required
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_probe_command(args)

            log_text = caplog.text
            # Echo server should return the prompt
            assert custom_prompt in log_text
            assert "✓ Probe successful" in log_text
