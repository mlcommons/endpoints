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

"""
Line-by-line profiling using the line_profiler library.

This module provides a clean singleton API for profiling:
- Controlled via ENABLE_LINE_PROFILER environment variable
- No-op decorators when disabled (zero overhead)
- Support for both sync and async functions
- Automatic cleanup on process exit
"""

import atexit
import contextlib
import io
import os
import sys

try:
    from line_profiler import LineProfiler
except ImportError:
    LineProfiler = None
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Environment variable names
ENV_VAR_ENABLE_LINE_PROFILER = "ENABLE_LINE_PROFILER"
ENV_VAR_LINE_PROFILER_LOGFILE = "LINE_PROFILER_LOGFILE"


class ProfilerState:
    """
    Singleton class that manages the line profiler state and lifecycle.

    This encapsulates all profiling functionality and avoids multiple globals.
    """

    _instance: Optional["ProfilerState"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        enable_profiler = os.environ.get(ENV_VAR_ENABLE_LINE_PROFILER, "0")
        self.enabled = enable_profiler == "1"
        self.profiler = None
        self._stats_printed = False
        logfile = os.environ.get(ENV_VAR_LINE_PROFILER_LOGFILE, None)
        self.output_file = Path(logfile) if logfile else None
        self._atexit_registered = False

        if self.enabled:
            if LineProfiler is None:
                raise ImportError(
                    f"line_profiler not installed but {ENV_VAR_ENABLE_LINE_PROFILER}={enable_profiler} is set. "
                    f"Install with: pip install line_profiler"
                )
            self.profiler = LineProfiler()
            self.profiler.enable()
            atexit.register(self._safe_cleanup)
            self._atexit_registered = True

    def _safe_cleanup(self):
        """Safe cleanup wrapper that suppresses all errors during atexit."""
        if not self._atexit_registered:
            return

        try:
            self._cleanup()
        except:  # noqa: E722
            pass  # Suppress all errors during shutdown

    def _cleanup(self):
        """Cleanup function called at interpreter exit or explicit shutdown.

        Prints stats (if any) and then completely tears down the profiler
        to prevent shutdown errors.
        """
        if not self.profiler or self._stats_printed or not self.profiler.functions:
            self._teardown_profiler()
            return

        with contextlib.suppress(Exception):
            self.pause()
            self._print_stats_to_destination()
            self._stats_printed = True
            self._teardown_profiler()

    def _print_stats_to_destination(self):
        """Print stats to configured output destination."""
        pid = os.getpid()

        if self.output_file:
            output_path = self.output_file.with_name(f"{self.output_file.name}.{pid}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open(mode="w") as stream:
                self.print_stats(stream=stream, prefix=f"PID {pid}")
        else:
            self.print_stats(stream=sys.stderr, prefix=f"PID {pid}")

    def _teardown_profiler(self):
        """Teardown profiler to prevent shutdown errors."""
        if not self.profiler:
            return

        self.profiler.disable()
        self.profiler.functions.clear()
        self.profiler.enable_count = 0
        self.profiler = None

    def profile(self, func: F) -> F:
        """Profile decorator for functions."""
        if not self.profiler:
            return func
        return self.profiler(func)

    def print_stats(self, stream=None, prefix=None):
        """
        Print profiling statistics.

        Args:
            stream: Output stream (defaults to sys.stdout)
            prefix: Optional prefix for the output (e.g., "PID 1234")
        """
        if not self.profiler or not self.profiler.functions:
            return

        out = stream or sys.stdout

        if prefix:
            print(f"\n{'=' * 80}", file=out)
            print(f"{prefix} - LINE PROFILER RESULTS", file=out)
            print(f"{'=' * 80}", file=out)

        self.profiler.print_stats(stream=out, output_unit=1e-6, stripzeros=True)
        out.flush()
        self._stats_printed = True

    def get_stats(self) -> str:
        """Get profiling statistics as a string."""
        if not self.profiler or not self.profiler.functions:
            return ""

        output = io.StringIO()
        self.profiler.print_stats(stream=output, output_unit=1e-6, stripzeros=True)
        return output.getvalue()

    def resume(self):
        """Resume profiling data collection."""
        if self.profiler:
            self.profiler.enable()

    def pause(self):
        """Pause profiling data collection."""
        if self.profiler:
            try:
                self.profiler.disable()
            except (AttributeError, TypeError):
                pass  # Already torn down

    def shutdown(self):
        """Explicit shutdown for worker processes. Safe to call multiple times."""
        if self._stats_printed:
            return

        self._atexit_registered = False  # Prevent double-printing via atexit
        self._cleanup()

    def is_enabled(self) -> bool:
        """Check if profiling is currently enabled."""
        return self.enabled


# Create singleton instance
_profiler_state = ProfilerState()

# Public API
profile = _profiler_state.profile
print_stats = _profiler_state.print_stats
get_stats = _profiler_state.get_stats
resume = _profiler_state.resume
pause = _profiler_state.pause
shutdown = _profiler_state.shutdown
is_enabled = _profiler_state.is_enabled
