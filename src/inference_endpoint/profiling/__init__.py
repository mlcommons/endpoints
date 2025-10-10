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

__all__ = [
    "profile",
    "profiler_start",
    "profiler_shutdown",
    "profiler_is_enabled",
    "profiler_check_subprocess",
]

# Determine which profiler to use (only one at a time)
_active_profiler = None

if os.environ.get("ENABLE_LINE_PROFILER") == "1":
    from . import line_profiler as _active_profiler
elif os.environ.get("ENABLE_PYINSTRUMENT") == "1":
    from . import pyinstrument_profiler as _active_profiler
elif os.environ.get("ENABLE_YAPPI") == "1":
    from . import yappi_profiler as _active_profiler


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
