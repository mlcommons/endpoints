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

"""Line-by-line profiling using the line_profiler library."""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from . import ENV_VAR_ENABLE_LINE_PROFILER
from .profiler_utils import (
    get_output_path,
    get_run_output_dir,
    is_pytest_mode,
    log_profiler_start,
    print_all_profiles,
)

F = TypeVar("F", bound=Callable[..., Any])


class LineProfiler:
    """Line profiler implementation."""

    def __init__(self):
        # Configuration
        self.enabled = os.environ.get(ENV_VAR_ENABLE_LINE_PROFILER, "0") == "1"
        self._output_dir = None  # Lazy initialization

        # Track original PID for fork detection
        self._original_pid = os.getpid()

        # State
        self.profiler = None
        self._session_active = False
        self._stats_written = False

        # Initialize if enabled
        if self.enabled:
            self._initialize_profiler()
            self.start()

    @property
    def output_dir(self) -> Path:
        """Lazy initialization of output directory."""
        if self._output_dir is None:
            if self.enabled:
                self._output_dir = get_run_output_dir() / "line_profiler"
            else:
                self._output_dir = Path("/tmp/mlperf_client_profiles")
        return self._output_dir

    def _initialize_profiler(self):
        """Initialize line profiler."""
        try:
            from line_profiler import LineProfiler as LP

            self.profiler = LP()
        except ImportError as e:
            raise ImportError(
                f"line_profiler not installed but {ENV_VAR_ENABLE_LINE_PROFILER}=1 is set. "
                f"Install with: pip install line_profiler"
            ) from e

    def check_subprocess_spawn(self):
        """Check if we're in a subprocess and reinitialize if needed."""
        if not self.enabled:
            return

        current_pid = os.getpid()

        # Fork mode: PID changed from parent
        if current_pid != self._original_pid:
            self._original_pid = current_pid
            self._stats_written = False
            self._session_active = False

            # Clear parent's profiler completely
            if self.profiler:
                try:
                    self.profiler.disable()
                except Exception:
                    # Ignore errors - profiler may be in invalid state
                    pass
                try:
                    self.profiler.functions.clear()
                except Exception:
                    # Ignore errors - functions may not exist
                    pass

            # Create fresh profiler for child
            self.profiler = None
            self._initialize_profiler()
            self.start()
        # Spawn mode: ensure started
        elif not self._session_active:
            self.start()

    def start(self):
        """Start profiling session."""
        if not self.enabled or self._session_active:
            return

        if self.profiler is None:
            self._initialize_profiler()

        self.profiler.enable()
        self._session_active = True
        log_profiler_start("line_profiler")

    def stop(self):
        """Stop profiling session."""
        if not self.enabled or not self._session_active:
            return

        if self.profiler:
            try:
                self.profiler.disable()
                self._session_active = False
            except (AttributeError, TypeError, RuntimeError):
                # Ignore errors during disable - profiler may be in invalid state
                self._session_active = False

    def shutdown(self):
        """Shutdown profiler and write output."""
        if not self.enabled or self._stats_written:
            return

        # Stop if active
        if self._session_active:
            self.stop()

        # Write output
        try:
            self._write_output()
            self._stats_written = True
        except Exception as e:
            print(
                f"[LineProfiler] Error writing profile for PID {os.getpid()}: {e}",
                file=sys.stderr,
            )

    def profile(self, func: F) -> F:
        """Decorator to mark a function for line-by-line profiling."""
        if not self.profiler:
            return func
        return self.profiler(func)

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled

    def pause(self):
        """Pause profiling (alias for stop)."""
        self.stop()

    def resume(self):
        """Resume profiling (alias for start)."""
        self.start()

    def print_stats(self, stream=None, prefix: str = ""):
        """Print profiler statistics to stream."""
        if not self.enabled or not self.profiler or not self.profiler.functions:
            return

        import sys

        if stream is None:
            stream = sys.stdout

        if prefix:
            print(f"\n{prefix} - LINE PROFILER RESULTS", file=stream)
            print(f"{'=' * 80}", file=stream)

        self.profiler.print_stats(stream=stream, output_unit=1e-6, stripzeros=True)

        if prefix:
            print(f"{'=' * 80}\n", file=stream)

    def get_stats(self) -> str:
        """Get profiler statistics as a string."""
        if not self.enabled or not self.profiler or not self.profiler.functions:
            return ""

        import io

        buffer = io.StringIO()
        self.profiler.print_stats(stream=buffer, output_unit=1e-6, stripzeros=True)
        return buffer.getvalue()

    def _write_output(self):
        """Write profiler output to file(s)."""
        if not self.profiler or not self.profiler.functions:
            return

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write to file
        output_path = get_output_path(self.output_dir, "line_profiler", "txt")
        with open(output_path, "w") as f:
            self.profiler.print_stats(stream=f, output_unit=1e-6, stripzeros=True)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # In standalone mode (not pytest), also print inline
        if not is_pytest_mode():
            self._print_inline(output_path)

    def _print_inline(self, output_path: Path):
        """Print line profiler output inline to stderr."""
        pid = os.getpid()

        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"LINE PROFILER RESULTS - PID {pid}", file=sys.stderr)
        print(f"Saved to: {output_path}", file=sys.stderr)
        print(f"{'=' * 80}", file=sys.stderr)

        # Print file content
        content = output_path.read_text()
        print(content, file=sys.stderr)

        print(f"{'=' * 80}\n", file=sys.stderr)
        sys.stderr.flush()


# Create singleton instance
_profiler = LineProfiler()

# Public API
profile = _profiler.profile
shutdown = _profiler.shutdown
is_enabled = _profiler.is_enabled
check_subprocess_spawn = _profiler.check_subprocess_spawn
pause = _profiler.pause
resume = _profiler.resume
print_stats = _profiler.print_stats
get_stats = _profiler.get_stats


def pytest_configure(config):
    """Configure line_profiler for pytest session."""
    if not _profiler.enabled:
        return

    print(
        f"\n[Profiling] {ENV_VAR_ENABLE_LINE_PROFILER}=1 detected, profiling enabled",
        file=sys.stderr,
    )
    print(f"[Profiling] Output directory: {_profiler.output_dir}", file=sys.stderr)


def pytest_sessionfinish(session, exitstatus):
    """Collect and display line_profiler results after pytest session."""
    if not _profiler.enabled:
        return

    # Shutdown main process profiler (writes its own profile)
    try:
        shutdown()

        # Completely disable the profiler to prevent shutdown errors
        if _profiler.profiler:
            try:
                _profiler.profiler.disable()
                # Clear functions to prevent monitoring errors during cleanup
                _profiler.profiler.functions.clear()
                _profiler.profiler = None
            except Exception:
                pass

    except Exception as e:
        # Ignore errors during shutdown - line_profiler may have internal state issues
        print(f"[LineProfiler] Warning: Error during shutdown: {e}", file=sys.stderr)

    # Collect and display all profiles (main + workers)
    try:
        print_all_profiles(
            profiler_name="line_profiler",
            output_dir=_profiler.output_dir,
        )
    except Exception as e:
        print(f"[LineProfiler] Warning: Error printing profiles: {e}", file=sys.stderr)
