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

"""Pyinstrument profiling module."""

import atexit
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from . import ENV_VAR_ENABLE_PYINSTRUMENT
from .profiler_utils import (
    get_output_path,
    get_run_output_dir,
    is_pytest_mode,
    log_profiler_start,
    print_all_profiles,
)

F = TypeVar("F", bound=Callable[..., Any])


class PyinstrumentProfiler:
    """Pyinstrument profiler implementation."""

    def __init__(self):
        # Configuration
        self.enabled = os.environ.get(ENV_VAR_ENABLE_PYINSTRUMENT, "0") == "1"
        self._output_dir = None  # Lazy initialization

        # Track original PID for fork detection
        self._original_pid = os.getpid()

        # State
        self.profiler = None
        self._session_active = False
        self._stats_written = False
        self._atexit_registered = False

        # Auto-start if enabled
        if self.enabled:
            self.start()
            # Register atexit handler to ensure worker profiles are written
            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True

    @property
    def output_dir(self) -> Path:
        """Lazy initialization of output directory."""
        if self._output_dir is None:
            if self.enabled:
                self._output_dir = get_run_output_dir() / "pyinstrument"
            else:
                self._output_dir = Path("/tmp/pyinstrument_profiles")
        return self._output_dir

    def _create_profiler(self):
        """Create a new profiler instance."""
        try:
            from pyinstrument import Profiler

            self.profiler = Profiler(
                interval=0.0005,
                async_mode="enabled",
            )
        except ImportError as e:
            raise ImportError(
                f"pyinstrument not installed but {ENV_VAR_ENABLE_PYINSTRUMENT}=1 is set. "
                f"Install with: pip install pyinstrument"
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
            self.profiler = None
            self._atexit_registered = False
            self.start()
            # Register atexit handler for worker process
            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True
        # Spawn mode: ensure started
        elif not self._session_active:
            self.start()
            # Register atexit handler if not already done
            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True

    def start(self):
        """Start profiling session."""
        if not self.enabled or self._session_active:
            return

        if self.profiler is None:
            self._create_profiler()

        try:
            self.profiler.start()
            log_profiler_start("pyinstrument")
        except RuntimeError as e:
            if "already a profiler running" in str(e):
                # Profiler already running, which is fine
                pass
            else:
                raise

        # Set active regardless of whether we started it or it was already running
        self._session_active = True

    def stop(self):
        """Stop profiling session."""
        if not self.enabled or not self._session_active:
            return

        if self.profiler:
            try:
                self.profiler.stop()
            except Exception:
                pass

        # Set inactive regardless of whether stop succeeded
        self._session_active = False

    def shutdown(self):
        """Shutdown profiler and write output. Safe to call multiple times."""
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
                f"[Pyinstrument] Error writing profile for PID {os.getpid()}: {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc()

    def profile(self, func: F) -> F:
        """Decorator to mark a function for profiling."""
        if not self.enabled:
            return func

        func._pyinstrument_profiled = True
        return func

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled

    def _write_output(self):
        """Write profiler output to file(s)."""
        if not self.profiler:
            return

        # Check if we have data
        try:
            session = self.profiler.last_session
            if not session:
                return

            root_frame = session.root_frame()
            if not root_frame:
                return
        except Exception:
            return

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write to file
        output_path = get_output_path(self.output_dir, "pyinstrument", "txt")

        # Generate profile text (simplified for speed)
        profile_text = self.profiler.output_text(
            unicode=True,
            color=True,
            show_all=False,
            timeline=False,
            time="percent_of_total",
        )

        with open(output_path, "w") as f:
            f.write(profile_text)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # In standalone mode (not pytest), print inline
        if not is_pytest_mode():
            pid = os.getpid()
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"PYINSTRUMENT PROFILE - PID {pid}", file=sys.stderr)
            print(f"Saved to: {output_path}", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            print(profile_text, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            sys.stderr.flush()


# Create singleton instance
_profiler = PyinstrumentProfiler()

# Public API
profile = _profiler.profile
start = _profiler.start
stop = _profiler.stop
shutdown = _profiler.shutdown
is_enabled = _profiler.is_enabled
check_subprocess_spawn = _profiler.check_subprocess_spawn


def pytest_configure(config):
    """Configure pyinstrument for pytest session."""
    if not _profiler.enabled:
        return

    print(
        f"\n[Profiling] {ENV_VAR_ENABLE_PYINSTRUMENT}=1 detected, profiling enabled",
        file=sys.stderr,
    )
    print(f"[Profiling] Output directory: {_profiler.output_dir}", file=sys.stderr)


def pytest_sessionfinish(session, exitstatus):
    """Collect and display pyinstrument results after pytest session."""
    if not _profiler.enabled:
        return

    # Shutdown main process profiler (writes its own profile)
    shutdown()

    # Collect and display all profiles (main + workers)
    print_all_profiles(
        profiler_name="pyinstrument",
        output_dir=_profiler.output_dir,
    )
