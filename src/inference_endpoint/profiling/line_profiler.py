"""
Line-by-line profiling using the line_profiler library.

This module provides a clean singleton API for profiling:
- Controlled via ENABLE_LINE_PROFILER environment variable
- No-op decorators when disabled (zero overhead)
- Support for both sync and async functions
- Automatic cleanup on process exit
"""

import atexit
import io
import os
import sys
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
            try:
                from line_profiler import LineProfiler

                self.profiler = LineProfiler()
                self.profiler.enable()
                atexit.register(self._safe_cleanup)
                self._atexit_registered = True
            except ImportError as e:
                raise ImportError(
                    f"line_profiler not installed but {ENV_VAR_ENABLE_LINE_PROFILER}={enable_profiler} is set. "
                    f"Install with: pip install line_profiler"
                ) from e

    def _safe_cleanup(self):
        """Safe cleanup wrapper that suppresses all errors during atexit."""
        if not self._atexit_registered:
            return

        try:
            self._cleanup()
        except:  # noqa: E722
            pass  # Suppress all errors during shutdown

    def _cleanup(self):
        """Cleanup function called at interpreter exit or explicit shutdown."""
        if not self.profiler or self._stats_printed:
            return

        # Check if profiler has any tracked functions
        try:
            if not self.profiler.functions:
                return
        except (AttributeError, TypeError):
            return  # Profiler partially torn down

        try:
            self.pause()
            pid = os.getpid()

            # Determine output destination
            if self.output_file:
                if self.output_file.parent != Path("."):
                    self.output_file.parent.mkdir(parents=True, exist_ok=True)
                stream = self.output_file.with_name(
                    f"{self.output_file.name}.{pid}"
                ).open(mode="w")
                should_close = True
            else:
                stream = sys.stderr
                should_close = False

            try:
                self.print_stats(stream=stream, prefix=f"PID {pid}")
            finally:
                if should_close:
                    stream.close()

            self._stats_printed = True
        except Exception:
            pass  # Silently fail during cleanup

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
