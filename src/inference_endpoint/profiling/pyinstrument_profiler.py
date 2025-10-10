"""
Pyinstrument profiling module.
"""

import atexit
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, TypeVar

# Import environment variable constants from parent module
from . import ENV_VAR_ENABLE_PYINSTRUMENT, ENV_VAR_PYINSTRUMENT_OUTPUT_DIR

F = TypeVar("F", bound=Callable[..., Any])


class PyinstrumentProfilerState:
    """
    Singleton class that manages the pyinstrument profiler state and lifecycle.

    This handles:
    - Profiler initialization and lifecycle
    - Multiprocessing/fork detection and reinitialization
    - Output formatting and file writing
    - Decorator implementation for profiling functions

    NOTE: Pyinstrument only allows ONE profiler per thread/process.
    The @profile decorator is just a marker - actual profiling happens
    at the process level between start() and stop() calls.
    """

    _instance: Optional["PyinstrumentProfilerState"] = None
    _original_pid: int | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._original_pid = os.getpid()

        # Check if pyinstrument is enabled
        enable_profiler = os.environ.get(ENV_VAR_ENABLE_PYINSTRUMENT, "0")
        self.enabled = enable_profiler == "1"

        # Configuration
        self.output_dir = Path(
            os.environ.get(
                ENV_VAR_PYINSTRUMENT_OUTPUT_DIR, "/tmp/pyinstrument_profiles"
            )
        )

        # Profiler instance
        self.profiler = None
        self._session_active = False
        self._stats_written = False
        self._atexit_registered = False

        # Auto-start profiler if enabled
        if self.enabled:
            self.start()

    def _create_profiler(self):
        """Create a new profiler instance."""
        try:
            from pyinstrument import Profiler

            # Create profiler with appropriate settings
            self.profiler = Profiler(
                # Use wall clock time (not CPU time) for async code
                interval=0.0005,
                async_mode="enabled",  # Enable async profiling
            )

        except ImportError as e:
            raise ImportError(
                f"pyinstrument not installed but {ENV_VAR_ENABLE_PYINSTRUMENT}=1 is set. "
                f"Install with: pip install pyinstrument"
            ) from e

    def check_subprocess_spawn(self):
        """
        Check if we're in a subprocess and reinitialize if needed.

        This handles both multiprocessing modes:

        1. Fork mode: Child process inherits parent's memory (copy-on-write).
           Singleton state is copied from parent, so we detect PID change
           and reinitialize the profiler with fresh state.

        2. Spawn mode: Child starts with fresh Python interpreter.
           Singleton is created fresh with child's PID, so PID check doesn't
           detect anything. We just ensure profiling is started.

        This function handles both cases gracefully.
        """
        current_pid = os.getpid()

        # Case 1: Fork mode - PID changed from parent
        if current_pid != self._original_pid:
            self._original_pid = current_pid
            self._stats_written = False
            self._session_active = False
            self._atexit_registered = False

            # Reset profiler and auto-start in the new process
            self.profiler = None
            if self.enabled:
                self.start()

        # Case 2: Spawn mode - already fresh, just ensure started
        # This is also safe for fork mode after reinitialization above
        elif self.enabled and not self._session_active:
            self.start()

    def start(self):
        """
        Start a profiling session.

        Called automatically when profiler is enabled. Can also be called
        explicitly to restart profiling after stop().
        """
        if not self.enabled:
            return

        # Don't start if already active
        if self._session_active:
            return

        # Create profiler if needed
        if self.profiler is None:
            self._create_profiler()

        # Register atexit handler if not already done
        if not self._atexit_registered:
            atexit.register(self._safe_cleanup)
            self._atexit_registered = True

        # Start profiling (catch error if profiler already running in this thread)
        try:
            self.profiler.start()
            self._session_active = True
            # Log that profiling started
            print(
                f"[Pyinstrument] Auto-started profiling for PID {os.getpid()}",
                file=sys.stderr,
            )
        except RuntimeError as e:
            if "already a profiler running" in str(e):
                # Another profiler is already active in this thread - this can happen
                # during pytest initialization when modules are imported multiple times
                # Just mark as active and continue
                self._session_active = True
            else:
                raise

    def stop(self):
        """Stop the current profiling session and write output immediately."""
        if not self.enabled or not self._session_active:
            return

        if self.profiler:
            try:
                self.profiler.stop()
                self._session_active = False

                # Write output immediately on stop (don't wait for atexit)
                # This ensures worker processes that get killed write their stats
                if not self._stats_written:
                    try:
                        self._write_final_output()
                        self._stats_written = True
                    except Exception:
                        pass
            except Exception:
                # Handle case where profiler was already stopped
                self._session_active = False

    def profile(self, func: F) -> F:
        """
        Decorator to mark a function for profiling.

        NOTE: This is just a marker decorator when pyinstrument is enabled.
        Actual profiling happens at the process level between start() and stop().

        When profiling is disabled, this is a complete no-op.
        """
        if not self.enabled:
            return func

        # Just mark the function with an attribute
        # This allows us to identify profiled functions if needed
        func._pyinstrument_profiled = True
        return func

    def _safe_cleanup(self):
        """Safe cleanup wrapper for atexit."""
        try:
            self.shutdown()
        except Exception:
            pass  # Suppress all errors during shutdown

    def shutdown(self):
        """
        Shutdown profiler and write final output.

        This is called automatically at exit but can also be called
        explicitly by worker processes.
        """
        if self._stats_written or not self.enabled:
            return

        # Stop any active session (this will write output if not already written)
        if self._session_active:
            self.stop()

        # Mark as complete (stop() already wrote output if needed)
        self._stats_written = True

    def _write_final_output(self):
        """Write the final profiler output."""
        if not self.profiler:
            return

        # Check if we have any data
        try:
            session = self.profiler.last_session
            if not session or not session.root_frame():
                return
        except Exception:
            return

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        pid = os.getpid()
        timestamp = int(time.time() * 1000)
        filename = f"profile_pid{pid}_{timestamp}.txt"
        output_path = self.output_dir / filename

        # Generate and write text output
        profile_text = self.profiler.output_text(
            unicode=True,
            color=True,
            show_all=False,
            timeline=False,
            time="percent_of_total",
        )

        with open(output_path, "w") as f:
            f.write(profile_text)

        # Print profile inline to stderr for immediate viewing
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"PYINSTRUMENT PROFILE - PID {pid}", file=sys.stderr)
        print(f"Saved to: {output_path}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(profile_text, file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        sys.stderr.flush()

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled


# Create singleton instance
_profiler_state = PyinstrumentProfilerState()

# Public API matching the expected interface
profile = _profiler_state.profile
start = _profiler_state.start
stop = _profiler_state.stop
shutdown = _profiler_state.shutdown
is_enabled = _profiler_state.is_enabled
check_subprocess_spawn = _profiler_state.check_subprocess_spawn
