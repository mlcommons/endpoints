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
        args.requests = 15  # More than 10 to test truncation
        args.prompt = "Sample response text"
        args.model = "gpt-3.5-turbo"  # Required
        args.verbose = 1

        with caplog.at_level("INFO"):
            await run_probe_command(args)

        log_text = caplog.text
        # Should show up to 10 responses
        assert "Sample responses (15 collected)" in log_text
        assert "[probe-0]" in log_text
        assert "Sample response text" in log_text

    @pytest.mark.asyncio
    async def test_probe_with_invalid_endpoint(self, caplog):
        """Test probe fails gracefully with invalid endpoint."""
        args = MagicMock()
        args.endpoint = "http://invalid-host-does-not-exist:9999"
        args.requests = 3
        args.prompt = "Test"
        args.model = "gpt-3.5-turbo"  # Required
        args.verbose = 0

        # Should raise ExecutionError when all requests fail
        with pytest.raises(ExecutionError, match="Probe failed"):
            await run_probe_command(args)

    @pytest.mark.asyncio
    async def test_probe_with_custom_prompt(self, mock_http_echo_server, caplog):
        """Test probe with custom prompt text."""
        custom_prompt = "This is my custom probe message with special chars: @#$%"

        args = MagicMock()
        args.endpoint = mock_http_echo_server.url
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
