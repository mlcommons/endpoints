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

"""
Logging configuration for the MLPerf Inference Endpoint Benchmarking System.

This module provides centralized logging setup and configuration.
"""

import logging
import os
import sys

from colorama import Fore, Style
from colorama import init as _colorama_init

# Initialize colorama
_colorama_init(autoreset=True)

# Custom TRACE level (below DEBUG) for deep instrumentation
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def verbosity_to_log_level(verbosity: int, as_int: bool = False) -> str | int:
    """Convert CLI verbosity count to log level.

    Args:
        verbosity: Count of -v flags (0, 1, 2, 3+)
        as_int: If True, return integer level; if False, return string name

    Returns:
        Log level as string ("INFO", "DEBUG", "TRACE") or int (20, 10, 5)
    """
    if verbosity >= 3:
        return TRACE if as_int else "TRACE"
    elif verbosity >= 2:
        return logging.DEBUG if as_int else "DEBUG"
    return logging.INFO if as_int else "INFO"


# Map levelname -> color
_LEVEL_COLORS = {
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED,
}


class ColoredFormatter(logging.Formatter):
    """Formatter that applies colors to log level names.

    Applies colorama colors (green for INFO, yellow for WARNING, red for ERROR/CRITICAL)
    to the levelname field only, leaving the rest of the log message unmodified.
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
        use_color: bool = False,
    ):
        """Initialize the formatter.

        Args:
            fmt: Log format string.
            datefmt: Date format string.
            style: Format style (% or {).
            use_color: Whether to apply colors to levelname. Defaults to False.
                      Enable by setting FORCE_COLOR_LOGGING environment variable.
        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with optional colors.

        Args:
            record: The log record to format.

        Returns:
            Formatted log message with colors applied if enabled.
        """
        # If coloring disabled, delegate
        if not self.use_color:
            return super().format(record)

        orig = record.levelname
        color = _LEVEL_COLORS.get(orig)
        if color:
            try:
                record.levelname = f"{color}{orig}{Style.RESET_ALL}"
                return super().format(record)
            finally:
                record.levelname = orig

        return super().format(record)


def setup_logging(level: str | None = None, format_string: str | None = None) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    # Default logging level
    if level is None:
        level = "INFO"

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # Disable colors by default to avoid potential formatting overhead during benchmarking.
    # Colors can be explicitly enabled via FORCE_COLOR_LOGGING environment variable.
    use_color = os.getenv("FORCE_COLOR_LOGGING") is not None

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(fmt=format_string, use_color=use_color))

    # Handle custom TRACE level
    if level.upper() == "TRACE":
        log_level = TRACE
    else:
        log_level = getattr(logging, level.upper())

    logging.basicConfig(level=log_level, handlers=[handler], force=True)

    # Set specific logger levels
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.debug(
        f"Logging configured with level: {level}, colors={'on' if use_color else 'off'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
