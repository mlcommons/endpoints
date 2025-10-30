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
Profiling module for inference endpoint.

Provides unified API for profiling backends (only one can be enabled at a time):
- line_profiler: Line-by-line profiling (ENABLE_LINE_PROFILER=1)
- pyinstrument: Statistical sampling profiler (ENABLE_PYINSTRUMENT=1)
- yappi: Multi-threaded deterministic profiler (ENABLE_YAPPI=1)

## Automatic Profiling Architecture

Profiling is completely automatic when enabled via environment variable:
1. Set ENABLE_YAPPI=1 (or ENABLE_PYINSTRUMENT=1, ENABLE_LINE_PROFILER=1)
2. Profiler auto-starts when the module is imported
3. Fork/spawn detection automatically reinitializes profiler in child processes
4. Profiler auto-shuts down on process exit and writes profile data

No manual profiler_start() calls needed! Just:
- Use @profile decorator on functions you want to profile
- Call profiler_shutdown() explicitly if you need to write profiles before exit

## Usage

    from inference_endpoint.profiling import profile

    @profile
    def my_function():
        # This function will be profiled automatically
        pass
"""

import os
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)

# Environment variable constants - centralized for all profilers
ENV_VAR_ENABLE_LINE_PROFILER = "ENABLE_LINE_PROFILER"
ENV_VAR_ENABLE_PYINSTRUMENT = "ENABLE_PYINSTRUMENT"
ENV_VAR_ENABLE_YAPPI = "ENABLE_YAPPI"
ENV_VAR_ENABLE_LOOP_STATS = "ENABLE_LOOP_STATS"

# Shared output directory for all profilers
ENV_VAR_PROFILER_OUT_DIR = "PROFILER_OUT_DIR"

# Profiler-specific settings
ENV_VAR_YAPPI_FILTER_PROFILED = "YAPPI_FILTER_PROFILED"

__all__ = [
    # API functions
    "profile",
    "profiler_start",
    "profiler_shutdown",
    "profiler_is_enabled",
    "profiler_check_subprocess",
    "profiler_prevent_init",
    "profiler_pytest_configure",
    "profiler_pytest_sessionfinish",
    # Environment variable constants
    "ENV_VAR_ENABLE_LINE_PROFILER",
    "ENV_VAR_ENABLE_PYINSTRUMENT",
    "ENV_VAR_ENABLE_YAPPI",
    "ENV_VAR_ENABLE_LOOP_STATS",
    "ENV_VAR_PROFILER_OUT_DIR",
    "ENV_VAR_YAPPI_FILTER_PROFILED",
]

# Determine which profiler to use (only one at a time)
_active_profiler = None

if os.environ.get(ENV_VAR_ENABLE_LINE_PROFILER) == "1":
    from . import line_profiler as _active_profiler
elif os.environ.get(ENV_VAR_ENABLE_PYINSTRUMENT) == "1":
    from . import pyinstrument_profiler as _active_profiler
elif os.environ.get(ENV_VAR_ENABLE_YAPPI) == "1":
    from . import yappi_profiler as _active_profiler

# Event loop stats is handled separately (not a traditional profiler)
# It sets up event loop policy when check_subprocess_spawn() is called
_loop_stats_enabled = os.environ.get(ENV_VAR_ENABLE_LOOP_STATS) == "1"


def profile(func: F) -> F:
    """
    Unified profiling decorator.

    Usage:
        @profile
        def my_function():
            pass

    This automatically uses whichever profiler is enabled via environment variable.
    No-op if no profiler is enabled.
    """
    if _active_profiler:
        return _active_profiler.profile(func)
    return func


def profiler_start():
    """
    Explicitly start the profiler (rarely needed).

    NOTE: Profiling starts automatically when enabled! This function is only
    needed for advanced use cases like restarting profiling after stop().

    No-op if no profiler is enabled or if already started.
    """
    if _active_profiler and hasattr(_active_profiler, "start"):
        _active_profiler.start()


def profiler_shutdown():
    """
    Shutdown the enabled profiler and print statistics.

    Called automatically at process exit, but can be called explicitly
    for worker processes to ensure stats are written.
    """
    if _active_profiler:
        _active_profiler.shutdown()


def profiler_is_enabled() -> bool:
    """
    Check if a profiler is enabled.

    Returns:
        True if a profiler is enabled, False otherwise.
    """
    return _active_profiler is not None


def profiler_check_subprocess():
    """
    Check if we're in a subprocess and reinitialize the profiler if needed.

    This should be called at the start of worker processes to ensure the profiler
    is properly initialized. Handles both fork and spawn multiprocessing modes:

    - Fork mode: Detects PID change and reinitializes profiler state
    - Spawn mode: Ensures profiling is started in the fresh interpreter

    Automatically starts profiling in the subprocess if profiling is enabled.

    No-op if no profiler is enabled or if the profiler doesn't support subprocess detection.
    """
    if _active_profiler and hasattr(_active_profiler, "check_subprocess_spawn"):
        _active_profiler.check_subprocess_spawn()

    # Also setup event loop stats if enabled (independent of main profiler)
    if _loop_stats_enabled:
        from inference_endpoint.profiling.event_loop_profiler import (
            setup_event_loop_policy,
        )

        setup_event_loop_policy()


def profiler_prevent_init():
    """
    Prevent profiler from initializing in the current process.

    This should be called EARLY in subprocess initialization (before imports)
    to prevent profilers from auto-starting. Useful for test infrastructure
    processes that should not be profiled (e.g., echo servers, mock services).

    Clears profiling environment variables and prevents any profiler state
    from being initialized.
    """
    # Clear environment variables to prevent auto-initialization
    os.environ.pop(ENV_VAR_ENABLE_PYINSTRUMENT, None)
    os.environ.pop(ENV_VAR_ENABLE_LINE_PROFILER, None)
    os.environ.pop(ENV_VAR_ENABLE_YAPPI, None)
    os.environ.pop(ENV_VAR_ENABLE_LOOP_STATS, None)

    # If a profiler is already loaded, disable it
    global _active_profiler, _loop_stats_enabled
    if _active_profiler and hasattr(_active_profiler, "shutdown"):
        try:
            _active_profiler.shutdown()
        except Exception:
            pass
    _active_profiler = None
    _loop_stats_enabled = False


def profiler_pytest_configure(config):
    """
    Configure profiling for pytest session.

    Delegates to each profiler module's pytest_configure() if it exists.
    This keeps profiler-specific logic in their respective modules.

    Args:
        config: pytest config object
    """
    # List of profiler modules that might have pytest integration
    profiler_modules = []

    # Add active profiler if it has pytest integration
    if _active_profiler and hasattr(_active_profiler, "pytest_configure"):
        profiler_modules.append(_active_profiler)

    # Add event loop profiler if enabled
    if _loop_stats_enabled:
        from inference_endpoint.profiling import event_loop_profiler

        profiler_modules.append(event_loop_profiler)

    # Call each profiler's pytest_configure
    for profiler_module in profiler_modules:
        if hasattr(profiler_module, "pytest_configure"):
            profiler_module.pytest_configure(config)


def profiler_pytest_sessionfinish(session, exitstatus):
    """
    Collect and display profiling results after pytest session completes.

    Delegates to each profiler module's pytest_sessionfinish() if it exists.
    This keeps profiler-specific logic in their respective modules.

    Args:
        session: pytest session object
        exitstatus: pytest exit status
    """
    # List of profiler modules that might have pytest integration
    profiler_modules = []

    # Add active profiler if it has pytest integration
    if _active_profiler and hasattr(_active_profiler, "pytest_sessionfinish"):
        profiler_modules.append(_active_profiler)

    # Add event loop profiler if enabled
    if _loop_stats_enabled:
        from inference_endpoint.profiling import event_loop_profiler

        profiler_modules.append(event_loop_profiler)

    # Call each profiler's pytest_sessionfinish
    for profiler_module in profiler_modules:
        if hasattr(profiler_module, "pytest_sessionfinish"):
            profiler_module.pytest_sessionfinish(session, exitstatus)
