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

"""Unit tests for logging module."""

import logging

import pytest
from colorama import Fore, Style
from inference_endpoint.utils.logging import ColoredFormatter, setup_logging


class TestColoredFormatter:
    """Tests for the ColoredFormatter class."""

    def test_colored_formatter_with_color_enabled(self):
        """Test that ColoredFormatter applies colors when use_color=True."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s", use_color=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Check that color code is present in output
        assert Fore.GREEN in output
        assert Style.RESET_ALL in output
        assert "test message" in output

    def test_colored_formatter_with_color_disabled(self):
        """Test that ColoredFormatter does not apply colors when use_color=False."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s", use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Check that no color codes are present
        assert Fore.GREEN not in output
        assert Style.RESET_ALL not in output
        assert "INFO" in output
        assert "test message" in output

    def test_colored_formatter_levelname_restoration(self):
        """Test that ColoredFormatter restores original levelname after formatting."""
        formatter = ColoredFormatter(fmt="%(levelname)s", use_color=True)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        original_levelname = record.levelname
        formatter.format(record)

        # Verify levelname is restored
        assert record.levelname == original_levelname

    @pytest.mark.parametrize(
        "level, expected_color",
        [
            (logging.INFO, Fore.GREEN),
            (logging.WARNING, Fore.YELLOW),
            (logging.ERROR, Fore.RED),
            (logging.CRITICAL, Fore.RED),
        ],
    )
    def test_colored_formatter_level_colors(self, level, expected_color):
        """Test that correct colors are applied for each log level."""
        formatter = ColoredFormatter(fmt="%(levelname)s", use_color=True)
        record = logging.LogRecord(
            name="test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert expected_color in output
        assert Style.RESET_ALL in output

    def test_colored_formatter_debug_level_no_color(self):
        """Test that DEBUG level (unmapped) does not crash or apply colors."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s", use_color=True)
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="debug message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # DEBUG is not in the color map, so no color should be applied
        assert "DEBUG" in output
        assert "debug message" in output

        # Verify NO colors were applied (since DEBUG is unmapped)
        assert Fore.GREEN not in output
        assert Fore.YELLOW not in output
        assert Fore.RED not in output
        assert Style.RESET_ALL not in output


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging_configures_root_logger(self, monkeypatch):
        """Test that setup_logging configures the root logger with a handler."""
        # Remove NO_COLOR to ensure colors are enabled
        monkeypatch.delenv("NO_COLOR", raising=False)

        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging()

        # Verify root logger has at least one handler
        assert len(root_logger.handlers) > 0

    def test_setup_logging_colors_disabled_with_no_color_env(self, monkeypatch):
        """Test that NO_COLOR env var disables colors by testing output directly."""
        monkeypatch.setenv("NO_COLOR", "1")

        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging()

        # Create a log record and format it
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        # Get the first handler's formatter
        handler = root_logger.handlers[0]
        formatted = handler.formatter.format(record)

        # Verify no color codes in output when NO_COLOR is set
        assert Fore.GREEN not in formatted

    def test_setup_logging_colors_enabled_by_default(self, monkeypatch):
        """Test that colors are enabled by default when NO_COLOR is not set."""
        monkeypatch.delenv("NO_COLOR", raising=False)

        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging()

        # Create a log record and format it
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        # Get the first handler's formatter
        handler = root_logger.handlers[0]
        formatted = handler.formatter.format(record)

        # Verify color codes are present in output by default
        assert Fore.GREEN in formatted

    def test_setup_logging_asyncio_logger_level(self, monkeypatch):
        """Test that asyncio logger is set to WARNING level after calling setup_logging()."""
        monkeypatch.delenv("NO_COLOR", raising=False)

        setup_logging()

        asyncio_logger = logging.getLogger("asyncio")
        assert asyncio_logger.level == logging.WARNING

    def test_setup_logging_urllib3_logger_level(self, monkeypatch):
        """Test that urllib3 logger is set to WARNING level after calling setup_logging()."""
        monkeypatch.delenv("NO_COLOR", raising=False)

        setup_logging()

        urllib3_logger = logging.getLogger("urllib3")
        assert urllib3_logger.level == logging.WARNING
