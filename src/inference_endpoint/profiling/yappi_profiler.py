"""
Yappi profiling module.

Yappi is a multi-threaded profiler configured to measure CPU time.
"""

import atexit
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, TypeVar

# Import environment variable constants from parent module
from . import (
    ENV_VAR_ENABLE_YAPPI,
    ENV_VAR_YAPPI_FILTER_PROFILED,
    ENV_VAR_YAPPI_OUTPUT_DIR,
)

F = TypeVar("F", bound=Callable[..., Any])


class YappiProfilerState:
    """
    Singleton class that manages the yappi profiler state and lifecycle.

    This handles:
    - Profiler initialization and lifecycle
    - Multiprocessing/fork detection and reinitialization
    - Output formatting and file writing
    - Decorator implementation for filtering profiled functions

    NOTE: Yappi profiles everything once started. The @profile decorator
    is used as a filter to only show results for marked functions.
    """

    _instance: Optional["YappiProfilerState"] = None
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

        # Check if yappi is enabled
        enable_profiler = os.environ.get(ENV_VAR_ENABLE_YAPPI, "0")
        self.enabled = enable_profiler == "1"

        # Configuration
        self.output_dir = Path(
            os.environ.get(ENV_VAR_YAPPI_OUTPUT_DIR, "/tmp/yappi_profiles")
        )
        self.filter_profiled = os.environ.get(ENV_VAR_YAPPI_FILTER_PROFILED, "1") == "1"

        # Track functions marked with @profile decorator
        self._profiled_functions = set()

        # State tracking
        self._session_active = False
        self._stats_written = False
        self._atexit_registered = False

        # yappi module (lazy import)
        self._yappi = None

        # Auto-start profiler if enabled
        if self.enabled:
            self.start()

    def _import_yappi(self):
        """Lazy import of yappi module."""
        if self._yappi is None:
            try:
                import yappi

                self._yappi = yappi
            except ImportError as e:
                raise ImportError(
                    f"yappi not installed but {ENV_VAR_ENABLE_YAPPI}=1 is set. "
                    f"Install with: pip install yappi"
                ) from e
        return self._yappi

    def check_subprocess_spawn(self):
        """
        Check if we're in a subprocess and reinitialize if needed.

        This handles both multiprocessing modes:

        1. Fork mode: Child process inherits parent's memory (copy-on-write).
           Singleton state is copied from parent, so we detect PID change
           and reinitialize the profiler with fresh state.

        2. Spawn mode: Child starts with fresh Python interpreter.
           Singleton is created fresh with child's PID, so PID check doesn't
           detect anything. But decorators are re-applied during module import,
           so _profiled_functions is repopulated correctly. We just ensure
           profiling is started.

        This function handles both cases gracefully.
        """
        current_pid = os.getpid()

        # Case 1: Fork mode - PID changed from parent
        if current_pid != self._original_pid:
            # We're in a forked process - need full reinitialization
            self._original_pid = current_pid
            self._stats_written = False
            self._session_active = False
            self._atexit_registered = False

            # Yappi needs to be cleared/restarted in the new process
            if self.enabled:
                yappi = self._import_yappi()
                # Clear any stats from parent process
                if yappi.is_running():
                    yappi.stop()
                yappi.clear_stats()
                # Start fresh in child process
                self.start()

        # Case 2: Spawn mode - already fresh, just ensure started
        # This is also safe for fork mode after reinitialization above
        elif self.enabled and not self._session_active:
            # In spawn mode, singleton is fresh but profiling might not have started yet
            # This can happen if check_subprocess_spawn() is called before module imports finish
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

        # Import yappi
        yappi = self._import_yappi()

        # Register atexit handler if not already done
        if not self._atexit_registered:
            atexit.register(self._safe_cleanup)
            self._atexit_registered = True

        # Set clock type to CPU time
        yappi.set_clock_type("cpu")

        # Start profiling
        if not yappi.is_running():
            yappi.start(builtins=False, profile_threads=True)
            self._session_active = True
            print(
                f"[Yappi] Auto-started profiling for PID {os.getpid()} (clock=cpu)",
                file=sys.stderr,
            )

    def stop(self):
        """Stop the current profiling session."""
        if not self.enabled or not self._session_active:
            return

        yappi = self._import_yappi()
        if yappi.is_running():
            yappi.stop()
            self._session_active = False

    def profile(self, func: F) -> F:
        """
        Decorator to mark a function for profiling.

        When filter_profiled is enabled, only functions marked with this
        decorator will appear in the final output.

        When profiling is disabled, this is a complete no-op.
        """
        if not self.enabled:
            return func

        # Track this function for filtering
        # We store the full name: module.qualname
        full_name = f"{func.__module__}.{func.__qualname__}"
        self._profiled_functions.add(full_name)

        # Mark the function with an attribute
        func._yappi_profiled = True
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

        # Stop any active session
        if self._session_active:
            self.stop()

        # Write output
        try:
            self._write_final_output()
        except Exception as e:
            print(
                f"[PROFILER] Error writing yappi profile for PID {os.getpid()}: {e}",
                file=sys.stderr,
            )

        self._stats_written = True

    def _write_final_output(self):
        """Write the final profiler output."""
        yappi = self._import_yappi()

        # Get all function stats (before filtering)
        all_func_stats = yappi.get_func_stats()

        # Check if we have any data
        if all_func_stats.empty():
            return

        # Filter to only @profile decorated functions if requested
        func_stats = all_func_stats
        if self.filter_profiled and self._profiled_functions:
            # Yappi's full_name format: "<file_path>:<line_number> <class>.<method>"
            # We store as: "<module>.<qualname>"
            # Need to match by extracting the function name from yappi's format

            def matches_profiled_function(stat):
                """Check if a yappi stat matches any of our @profile decorated functions."""
                # Extract the function name part after the line number
                # Format: "/path/to/file.py:123 ClassName.method_name"
                if " " in stat.full_name:
                    func_part = stat.full_name.split(" ", 1)[
                        1
                    ]  # "ClassName.method_name"
                else:
                    func_part = stat.name  # fallback to just the name

                # Check if any of our tracked function names ends with this pattern
                for tracked_func in self._profiled_functions:
                    # Match if tracked name ends with the function part
                    # e.g., "inference_endpoint.endpoint_client.worker.Worker._main_loop"
                    #       should match "Worker._main_loop"
                    if tracked_func.endswith(func_part):
                        return True
                return False

            func_stats = yappi.get_func_stats(filter_callback=matches_profiled_function)

            # If no profiled functions have data, return
            if func_stats.empty():
                print(
                    f"[Yappi] No data for @profile decorated functions in PID {os.getpid()}",
                    file=sys.stderr,
                )
                return

        # Calculate overhead metrics
        # Total time across ALL functions (including libraries, system, etc)
        total_all_time = sum(stat.tsub for stat in all_func_stats)

        # Time in profiled functions (your code being measured)
        total_profiled_time = sum(stat.tsub for stat in func_stats)

        # External overhead = everything else (libraries, framework, system calls, etc.)
        external_time = total_all_time - total_profiled_time

        # Calculate percentages
        profiled_pct = (
            (total_profiled_time / total_all_time * 100) if total_all_time > 0 else 0
        )
        external_overhead_pct = (
            (external_time / total_all_time * 100) if total_all_time > 0 else 0
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        pid = os.getpid()
        timestamp = int(time.time() * 1000)

        try:
            # Save to text file
            filename = f"profile_pid{pid}_{timestamp}.txt"
            output_path = self.output_dir / filename

            # Write to file
            with open(output_path, "w") as f:
                # Write header with overhead breakdown
                f.write(f"Yappi Profile - PID {pid}\n")
                f.write("Clock type: cpu\n")
                f.write(f"Filter @profile: {self.filter_profiled}\n")
                f.write("\n")
                f.write("Time Breakdown:\n")
                f.write(f"  Total measured time:    {total_all_time:.6f}s (100.00%)\n")
                f.write(
                    f"  @profile functions:     {total_profiled_time:.6f}s ({profiled_pct:>6.2f}%) - your code\n"
                )
                f.write(
                    f"  External overhead:      {external_time:.6f}s ({external_overhead_pct:>6.2f}%) - libraries, frameworks, system\n"
                )
                f.write("=" * 150 + "\n\n")

                # Write stats with wider columns for full function names + percentage
                func_stats.sort("tsub", "desc")

                # Get thread stats to map thread IDs to names
                thread_stats = yappi.get_thread_stats()
                thread_id_to_name = {}
                thread_name_counts = {}  # Track duplicate names

                for tstat in thread_stats:
                    base_name = tstat.name
                    # If name is generic (Thread, Thread-1, etc), add ID suffix for clarity
                    if base_name.startswith("Thread") or base_name == "_MainThread":
                        # Track how many threads have this base name
                        thread_name_counts[base_name] = (
                            thread_name_counts.get(base_name, 0) + 1
                        )
                        # Add ID suffix to distinguish threads
                        thread_id_to_name[tstat.id] = f"{base_name}#{tstat.id}"
                    else:
                        thread_id_to_name[tstat.id] = base_name

                # Write custom formatted output with percentage and thread info
                # yappi: tsub = exclusive time (just this function), ttot = inclusive time (with children)
                # tsub-e2e% = tsub (exclusive time) / total_all_time (e2e time)
                # ttot-e2e% = ttot (inclusive time) / total_all_time (e2e time)
                f.write("=" * 160 + "\n")
                f.write(
                    "PROFILED FUNCTIONS - TOP 20 (sorted by total time in subtree)\n"
                )
                f.write("=" * 160 + "\n\n")
                f.write(
                    f"{'name':<90} {'thread':<20} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}\n"
                )
                for i, stat in enumerate(func_stats):
                    if i >= 20:  # Limit to top 20
                        break
                    tsub_pct_e2e = (
                        (stat.tsub / total_all_time * 100) if total_all_time > 0 else 0
                    )
                    ttot_pct_e2e = (
                        (stat.ttot / total_all_time * 100) if total_all_time > 0 else 0
                    )
                    # Get thread name from context id
                    thread_name = thread_id_to_name.get(
                        stat.ctx_id, f"tid:{stat.ctx_id}"
                    )
                    # Truncate function name to fit with thread column
                    func_name = (
                        stat.full_name
                        if len(stat.full_name) <= 90
                        else "..." + stat.full_name[-87:]
                    )
                    # Truncate thread name if too long
                    thread_display = (
                        thread_name[:20]
                        if len(thread_name) <= 20
                        else thread_name[:17] + "..."
                    )
                    f.write(
                        f"{func_name:<90} {thread_display:<20} {stat.ncall:>8} {stat.tsub:>10.6f} {tsub_pct_e2e:>10.2f} {stat.ttot:>10.6f} {ttot_pct_e2e:>10.2f} {stat.tavg:>10.6f}\n"
                    )

                # Write overall top 20 functions (all functions, not just @profiled)
                f.write("\n\n" + "=" * 160 + "\n")
                f.write("ALL FUNCTIONS - TOP 20 (sorted by total time in subtree)\n")
                f.write("=" * 160 + "\n\n")
                all_func_stats.sort("ttot", "desc")
                f.write(
                    f"{'name':<90} {'thread':<20} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}\n"
                )
                for i, stat in enumerate(all_func_stats):
                    if i >= 20:  # Limit to top 20
                        break
                    tsub_pct_e2e = (
                        (stat.tsub / total_all_time * 100) if total_all_time > 0 else 0
                    )
                    ttot_pct_e2e = (
                        (stat.ttot / total_all_time * 100) if total_all_time > 0 else 0
                    )
                    # Get thread name from context id
                    thread_name = thread_id_to_name.get(
                        stat.ctx_id, f"tid:{stat.ctx_id}"
                    )
                    # Truncate function name to fit with thread column
                    func_name = (
                        stat.full_name
                        if len(stat.full_name) <= 90
                        else "..." + stat.full_name[-87:]
                    )
                    # Truncate thread name if too long
                    thread_display = (
                        thread_name[:20]
                        if len(thread_name) <= 20
                        else thread_name[:17] + "..."
                    )
                    f.write(
                        f"{func_name:<90} {thread_display:<20} {stat.ncall:>8} {stat.tsub:>10.6f} {tsub_pct_e2e:>10.2f} {stat.ttot:>10.6f} {ttot_pct_e2e:>10.2f} {stat.tavg:>10.6f}\n"
                    )

                # Write subtree information for top 20 profiled functions
                f.write("\n\n" + "=" * 150 + "\n")
                f.write(
                    "SUBTREE DETAILS - TOP 20 (child functions called by each profiled function)\n"
                )
                f.write("=" * 150 + "\n\n")

                for i, stat in enumerate(func_stats):
                    if i >= 20:  # Limit to top 20 functions
                        break
                    children = stat.children
                    if children:
                        parent_tsub_pct_e2e = (
                            (stat.tsub / total_all_time * 100)
                            if total_all_time > 0
                            else 0
                        )
                        parent_ttot_pct_e2e = (
                            (stat.ttot / total_all_time * 100)
                            if total_all_time > 0
                            else 0
                        )
                        f.write(f"\n{stat.full_name}\n")
                        f.write(
                            f"  Total calls: {stat.ncall}, Exclusive time: {stat.tsub:.6f}s ({parent_tsub_pct_e2e:.2f}%), Total time: {stat.ttot:.6f}s ({parent_ttot_pct_e2e:.2f}%)\n"
                        )
                        f.write("-" * 150 + "\n")
                        children.sort("tsub", "desc")
                        # Write header with percentage columns (% of e2e time)
                        f.write(
                            f"{'name':<90} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}\n"
                        )
                        # Write all children
                        for child in children:
                            # Calculate percentages relative to e2e time
                            child_tsub_pct_e2e = (
                                (child.tsub / total_all_time * 100)
                                if total_all_time > 0
                                else 0
                            )
                            child_ttot_pct_e2e = (
                                (child.ttot / total_all_time * 100)
                                if total_all_time > 0
                                else 0
                            )
                            # Truncate long names to fit
                            name = (
                                child.full_name
                                if len(child.full_name) <= 90
                                else "..." + child.full_name[-87:]
                            )
                            f.write(
                                f"{name:<90} {child.ncall:>8} {child.tsub:>10.6f} {child_tsub_pct_e2e:>10.2f} {child.ttot:>10.6f} {child_ttot_pct_e2e:>10.2f} {child.tavg:>10.6f}\n"
                            )

            # Write thread stats to file with enhanced formatting
            with open(output_path, "a") as f:
                f.write("\n\n" + "=" * 150 + "\n")
                f.write("THREAD STATISTICS\n")
                f.write("=" * 150 + "\n\n")

            # Get thread stats and append to file with custom formatting
            if not thread_stats.empty():
                with open(output_path, "a") as f:
                    # Write custom header to match function output format
                    f.write(
                        f"{'thread_name':<25} {'id':>6} {'tid':<20} {'ttot':>12} {'scnt':>10}\n"
                    )
                    f.write(f"{'-'*75}\n")
                    for tstat in thread_stats:
                        # Format thread name with ID suffix to match function output
                        base_name = tstat.name
                        if base_name.startswith("Thread") or base_name == "_MainThread":
                            thread_display = f"{base_name}#{tstat.id}"
                        else:
                            thread_display = base_name
                        f.write(
                            f"{thread_display:<25} {tstat.id:>6} {tstat.tid:<20} {tstat.ttot:>12.6f} {tstat.sched_count:>10}\n"
                        )

            # Also save as callgrind format for visualization
            callgrind_filename = f"profile_pid{pid}_{timestamp}.callgrind"
            callgrind_path = self.output_dir / callgrind_filename
            func_stats.save(str(callgrind_path), type="callgrind")

            # Print profile inline to stderr for immediate viewing
            print(f"\n{'='*150}", file=sys.stderr)
            print(f"YAPPI PROFILE - PID {pid}", file=sys.stderr)
            print("Clock type: cpu", file=sys.stderr)
            print(f"Filter @profile: {self.filter_profiled}", file=sys.stderr)
            print("", file=sys.stderr)
            print(f"Saved to: {output_path}", file=sys.stderr)
            print(f"Callgrind: {callgrind_path}", file=sys.stderr)
            print(f"{'='*150}", file=sys.stderr)

            # Print stats to stderr with wider columns, percentage, and thread info
            # yappi: tsub = exclusive time (just this function), ttot = inclusive time (with children)
            # tsub-e2e% = tsub (exclusive time) / total_all_time (e2e time)
            # ttot-e2e% = ttot (inclusive time) / total_all_time (e2e time)
            print(
                "\nPROFILED FUNCTIONS - TOP 20 (sorted by total time in subtree)",
                file=sys.stderr,
            )
            print(f"{'-'*168}", file=sys.stderr)
            print(
                f"{'name':<90} {'thread':<20} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}",
                file=sys.stderr,
            )
            for i, stat in enumerate(func_stats):
                if i >= 20:  # Limit to top 20
                    break
                tsub_pct_e2e = (
                    (stat.tsub / total_all_time * 100) if total_all_time > 0 else 0
                )
                ttot_pct_e2e = (
                    (stat.ttot / total_all_time * 100) if total_all_time > 0 else 0
                )
                # Get thread name
                thread_name = thread_id_to_name.get(stat.ctx_id, f"tid:{stat.ctx_id}")
                # Truncate long names
                name = (
                    stat.full_name
                    if len(stat.full_name) <= 90
                    else "..." + stat.full_name[-87:]
                )
                # Truncate thread name if too long
                thread_display = (
                    thread_name[:20]
                    if len(thread_name) <= 20
                    else thread_name[:17] + "..."
                )
                print(
                    f"{name:<90} {thread_display:<20} {stat.ncall:>8} {stat.tsub:>10.6f} {tsub_pct_e2e:>10.2f} {stat.ttot:>10.6f} {ttot_pct_e2e:>10.2f} {stat.tavg:>10.6f}",
                    file=sys.stderr,
                )

            # Print overall top 20 functions (all functions, not just @profiled)
            print(
                "\n\nALL FUNCTIONS - TOP 20 (sorted by total time in subtree)",
                file=sys.stderr,
            )
            print(f"{'-'*168}", file=sys.stderr)
            print(
                f"{'name':<90} {'thread':<20} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}",
                file=sys.stderr,
            )
            for i, stat in enumerate(all_func_stats):
                if i >= 20:  # Limit to top 20
                    break
                tsub_pct_e2e = (
                    (stat.tsub / total_all_time * 100) if total_all_time > 0 else 0
                )
                ttot_pct_e2e = (
                    (stat.ttot / total_all_time * 100) if total_all_time > 0 else 0
                )
                # Get thread name
                thread_name = thread_id_to_name.get(stat.ctx_id, f"tid:{stat.ctx_id}")
                # Truncate long names
                name = (
                    stat.full_name
                    if len(stat.full_name) <= 90
                    else "..." + stat.full_name[-87:]
                )
                # Truncate thread name if too long
                thread_display = (
                    thread_name[:20]
                    if len(thread_name) <= 20
                    else thread_name[:17] + "..."
                )
                print(
                    f"{name:<90} {thread_display:<20} {stat.ncall:>8} {stat.tsub:>10.6f} {tsub_pct_e2e:>10.2f} {stat.ttot:>10.6f} {ttot_pct_e2e:>10.2f} {stat.tavg:>10.6f}",
                    file=sys.stderr,
                )

            # Print top children for first 5 profiled functions
            print(
                "\n\nTOP SUBTREES (child functions for slowest profiled functions)",
                file=sys.stderr,
            )
            print(f"{'='*150}", file=sys.stderr)
            for i, stat in enumerate(func_stats[:5]):
                children = stat.children
                if children:
                    # Calculate percentages relative to e2e time for parent
                    parent_tsub_pct_e2e = (
                        (stat.tsub / total_all_time * 100) if total_all_time > 0 else 0
                    )
                    parent_ttot_pct_e2e = (
                        (stat.ttot / total_all_time * 100) if total_all_time > 0 else 0
                    )
                    print(f"\n{i+1}. {stat.full_name}", file=sys.stderr)
                    print(
                        f"   Calls: {stat.ncall}, Exclusive time: {stat.tsub:.6f}s ({parent_tsub_pct_e2e:.2f}%), Total time: {stat.ttot:.6f}s ({parent_ttot_pct_e2e:.2f}%)",
                        file=sys.stderr,
                    )
                    print(f"   {'-'*146}", file=sys.stderr)
                    children.sort("tsub", "desc")
                    # Show top 10 children - can't slice yappi stats, so print manually
                    # % shows child's exclusive and total time as % of e2e time
                    print(
                        f"   {'name':<90} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}",
                        file=sys.stderr,
                    )
                    for j, child in enumerate(children):
                        if j >= 10:  # Only show top 10
                            break
                        # Calculate percentages relative to e2e time
                        child_tsub_pct_e2e = (
                            (child.tsub / total_all_time * 100)
                            if total_all_time > 0
                            else 0
                        )
                        child_ttot_pct_e2e = (
                            (child.ttot / total_all_time * 100)
                            if total_all_time > 0
                            else 0
                        )
                        # Truncate long names to fit
                        name = (
                            child.full_name
                            if len(child.full_name) <= 90
                            else "..." + child.full_name[-87:]
                        )
                        print(
                            f"   {name:<90} {child.ncall:>8} {child.tsub:>10.6f} {child_tsub_pct_e2e:>10.2f} {child.ttot:>10.6f} {child_ttot_pct_e2e:>10.2f} {child.tavg:>10.6f}",
                            file=sys.stderr,
                        )

            # Print thread stats to stderr with enhanced formatting
            print(f"\n\n{'='*150}", file=sys.stderr)
            print("THREAD STATISTICS", file=sys.stderr)
            print(f"{'='*150}", file=sys.stderr)
            if not thread_stats.empty():
                # Print custom header to match function output format
                print(
                    f"{'thread_name':<25} {'id':>6} {'tid':<20} {'ttot':>12} {'scnt':>10}",
                    file=sys.stderr,
                )
                print(f"{'-'*75}", file=sys.stderr)
                for tstat in thread_stats:
                    # Format thread name with ID suffix to match function output
                    base_name = tstat.name
                    if base_name.startswith("Thread") or base_name == "_MainThread":
                        thread_display = f"{base_name}#{tstat.id}"
                    else:
                        thread_display = base_name
                    print(
                        f"{thread_display:<25} {tstat.id:>6} {tstat.tid:<20} {tstat.ttot:>12.6f} {tstat.sched_count:>10}",
                        file=sys.stderr,
                    )

            print(f"\n{'='*150}\n", file=sys.stderr)
            sys.stderr.flush()

        except Exception as e:
            print(
                f"[PROFILER] Error writing profile for PID {pid}: {e}", file=sys.stderr
            )

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled


# Create singleton instance
_profiler_state = YappiProfilerState()

# Public API matching the expected interface
profile = _profiler_state.profile
start = _profiler_state.start
stop = _profiler_state.stop
shutdown = _profiler_state.shutdown
is_enabled = _profiler_state.is_enabled
check_subprocess_spawn = _profiler_state.check_subprocess_spawn
