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

"""Yappi profiling module.

Yappi is a multi-threaded profiler configured to measure CPU time.
"""

import atexit
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from . import (
    ENV_VAR_ENABLE_YAPPI,
    ENV_VAR_YAPPI_FILTER_PROFILED,
)
from .profiler_utils import (
    get_output_path,
    get_run_output_dir,
    is_pytest_mode,
    log_profiler_start,
    print_all_profiles,
)

F = TypeVar("F", bound=Callable[..., Any])


class YappiProfiler:
    """Yappi profiler implementation."""

    def __init__(self):
        # Configuration
        self.enabled = os.environ.get(ENV_VAR_ENABLE_YAPPI, "0") == "1"
        self._output_dir = None  # Lazy initialization
        self.filter_profiled = os.environ.get(ENV_VAR_YAPPI_FILTER_PROFILED, "1") == "1"

        # Track original PID for fork detection
        self._original_pid = os.getpid()

        # Track functions marked with @profile decorator
        self._profiled_functions = set()

        # State
        self._yappi = None
        self._session_active = False
        self._stats_written = False
        self._atexit_registered = False

        # Auto-start if enabled
        if self.enabled:
            self.start()

    @property
    def output_dir(self) -> Path:
        """Lazy initialization of output directory."""
        if self._output_dir is None:
            if self.enabled:
                self._output_dir = get_run_output_dir() / "yappi"
            else:
                self._output_dir = Path("/tmp/yappi_profiles")
        return self._output_dir

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
        """Check if we're in a subprocess and reinitialize if needed."""
        if not self.enabled:
            return

        current_pid = os.getpid()

        # Fork mode: PID changed from parent
        if current_pid != self._original_pid:
            self._original_pid = current_pid
            self._stats_written = False
            self._session_active = False
            self._atexit_registered = False

            # Clear yappi stats from parent
            yappi = self._import_yappi()
            if yappi.is_running():
                yappi.stop()
            yappi.clear_stats()

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

        yappi = self._import_yappi()
        yappi.set_clock_type("cpu")

        if not yappi.is_running():
            yappi.start(builtins=False, profile_threads=True)
            log_profiler_start("yappi")

        # Set active regardless of whether we started it or it was already running
        self._session_active = True

    def stop(self):
        """Stop profiling session."""
        if not self.enabled or not self._session_active:
            return

        yappi = self._import_yappi()
        if yappi.is_running():
            yappi.stop()

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
                f"[Yappi] Error writing profile for PID {os.getpid()}: {e}",
                file=sys.stderr,
            )

    def profile(self, func: F) -> F:
        """Mark a function for profiling."""
        if not self.enabled:
            return func

        # Track this function for filtering
        full_name = f"{func.__module__}.{func.__qualname__}"
        self._profiled_functions.add(full_name)
        func._yappi_profiled = True
        return func

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled

    def _write_output(self):
        """Write profiler output to file(s)."""
        yappi = self._import_yappi()

        # Get all function stats
        all_func_stats = yappi.get_func_stats()
        if all_func_stats.empty():
            return

        # Filter to @profile decorated functions if requested
        func_stats = all_func_stats
        if self.filter_profiled and self._profiled_functions:

            def matches_profiled_function(stat):
                if " " in stat.full_name:
                    func_part = stat.full_name.split(" ", 1)[1]
                else:
                    func_part = stat.name

                for tracked_func in self._profiled_functions:
                    if tracked_func.endswith(func_part):
                        return True
                return False

            func_stats = yappi.get_func_stats(filter_callback=matches_profiled_function)
            if func_stats.empty():
                func_stats = all_func_stats

        # Calculate metrics
        total_all_time = sum(stat.tsub for stat in all_func_stats)
        total_profiled_time = sum(stat.tsub for stat in func_stats)
        external_time = total_all_time - total_profiled_time
        profiled_pct = (
            (total_profiled_time / total_all_time * 100) if total_all_time > 0 else 0
        )
        external_overhead_pct = (
            (external_time / total_all_time * 100) if total_all_time > 0 else 0
        )

        # Get thread stats
        thread_stats = yappi.get_thread_stats()
        thread_id_to_name = {}
        for tstat in thread_stats:
            base_name = tstat.name
            if base_name.startswith("Thread") or base_name == "_MainThread":
                thread_id_to_name[tstat.id] = f"{base_name}#{tstat.id}"
            else:
                thread_id_to_name[tstat.id] = base_name

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write to file
        output_path = get_output_path(self.output_dir, "yappi", "txt")
        with open(output_path, "w") as f:
            self._write_profile_content(
                f,
                func_stats,
                all_func_stats,
                thread_stats,
                thread_id_to_name,
                total_all_time,
                total_profiled_time,
                external_time,
                profiled_pct,
                external_overhead_pct,
            )
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # Also save callgrind
        callgrind_path = output_path.with_suffix(".callgrind")
        func_stats.save(str(callgrind_path), type="callgrind")

        # In standalone mode (not pytest), also print inline
        if not is_pytest_mode():
            self._print_inline(output_path, callgrind_path)

    def _print_inline(self, output_path: Path, callgrind_path: Path):
        """Print yappi profile inline to stderr."""
        pid = os.getpid()

        print(f"\n{'='*150}", file=sys.stderr)
        print(f"YAPPI PROFILE - PID {pid}", file=sys.stderr)
        print("Clock type: cpu", file=sys.stderr)
        print(f"Filter @profile: {self.filter_profiled}", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"Saved to: {output_path}", file=sys.stderr)
        print(f"Callgrind: {callgrind_path}", file=sys.stderr)
        print(f"{'='*150}\n", file=sys.stderr)

        # Print file content
        content = output_path.read_text()
        print(content, file=sys.stderr)
        print(f"\n{'='*150}\n", file=sys.stderr)
        sys.stderr.flush()

    def _write_profile_content(
        self,
        f,
        func_stats,
        all_func_stats,
        thread_stats,
        thread_id_to_name,
        total_all_time,
        total_profiled_time,
        external_time,
        profiled_pct,
        external_overhead_pct,
    ):
        """Write formatted profile content to file."""
        pid = os.getpid()

        # Header
        f.write(f"Yappi Profile - PID {pid}\n")
        f.write("Clock type: cpu\n")
        f.write(f"Filter @profile: {self.filter_profiled}\n")
        f.write("=" * 150 + "\n\n")

        # Profiled functions - top 20
        func_stats.sort("ttot", "desc")
        f.write("PROFILED FUNCTIONS - TOP 20 (sorted by total time in subtree)\n")
        f.write("=" * 160 + "\n")
        f.write(
            f"{'name':<90} {'thread':<20} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}\n"
        )

        for i, stat in enumerate(func_stats):
            if i >= 20:
                break
            self._write_stat_line(f, stat, thread_id_to_name, total_all_time)

        # All functions - top 20
        f.write("\n\nALL FUNCTIONS - TOP 20 (sorted by total time in subtree)\n")
        f.write("=" * 160 + "\n")
        all_func_stats.sort("ttot", "desc")
        f.write(
            f"{'name':<90} {'thread':<20} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}\n"
        )

        for i, stat in enumerate(all_func_stats):
            if i >= 20:
                break
            self._write_stat_line(f, stat, thread_id_to_name, total_all_time)

        # Subtree details - top 20 profiled functions
        f.write(
            "\n\nSUBTREE DETAILS - TOP 20 (child functions called by each profiled function)\n"
        )
        f.write("=" * 150 + "\n")

        for i, stat in enumerate(func_stats):
            if i >= 20:
                break
            children = stat.children
            if children:
                self._write_subtree(f, stat, children, total_all_time)

        # Thread statistics
        f.write("\n\n" + "=" * 150 + "\n")
        f.write("THREAD STATISTICS\n")
        f.write("=" * 150 + "\n")
        f.write(
            f"{'thread_name':<25} {'id':>6} {'tid':<20} {'ttot':>12} {'scnt':>10}\n"
        )
        f.write(f"{'-'*75}\n")

        for tstat in thread_stats:
            base_name = tstat.name
            if base_name.startswith("Thread") or base_name == "_MainThread":
                thread_display = f"{base_name}#{tstat.id}"
            else:
                thread_display = base_name
            f.write(
                f"{thread_display:<25} {tstat.id:>6} {tstat.tid:<20} {tstat.ttot:>12.6f} {tstat.sched_count:>10}\n"
            )

    def _write_stat_line(self, f, stat, thread_id_to_name, total_all_time):
        """Write a single stat line."""
        tsub_pct_e2e = (stat.tsub / total_all_time * 100) if total_all_time > 0 else 0
        ttot_pct_e2e = (stat.ttot / total_all_time * 100) if total_all_time > 0 else 0
        thread_name = thread_id_to_name.get(stat.ctx_id, f"tid:{stat.ctx_id}")

        func_name = (
            stat.full_name
            if len(stat.full_name) <= 90
            else "..." + stat.full_name[-87:]
        )
        thread_display = (
            thread_name[:20] if len(thread_name) <= 20 else thread_name[:17] + "..."
        )

        f.write(
            f"{func_name:<90} {thread_display:<20} {stat.ncall:>8} {stat.tsub:>10.6f} {tsub_pct_e2e:>10.2f} {stat.ttot:>10.6f} {ttot_pct_e2e:>10.2f} {stat.tavg:>10.6f}\n"
        )

    def _write_subtree(self, f, parent_stat, children, total_all_time):
        """Write subtree information for a function."""
        parent_tsub_pct_e2e = (
            (parent_stat.tsub / total_all_time * 100) if total_all_time > 0 else 0
        )
        parent_ttot_pct_e2e = (
            (parent_stat.ttot / total_all_time * 100) if total_all_time > 0 else 0
        )

        f.write(f"\n{parent_stat.full_name}\n")
        f.write(
            f"  Total calls: {parent_stat.ncall}, Exclusive time: {parent_stat.tsub:.6f}s ({parent_tsub_pct_e2e:.2f}%), Total time: {parent_stat.ttot:.6f}s ({parent_ttot_pct_e2e:.2f}%)\n"
        )
        f.write("-" * 150 + "\n")

        children.sort("tsub", "desc")
        f.write(
            f"{'name':<90} {'ncall':>8} {'tsub':>10} {'tsub-e2e%':>10} {'ttot':>10} {'ttot-e2e%':>10} {'tavg':>10}\n"
        )

        for child in children:
            child_tsub_pct_e2e = (
                (child.tsub / total_all_time * 100) if total_all_time > 0 else 0
            )
            child_ttot_pct_e2e = (
                (child.ttot / total_all_time * 100) if total_all_time > 0 else 0
            )
            name = (
                child.full_name
                if len(child.full_name) <= 90
                else "..." + child.full_name[-87:]
            )
            f.write(
                f"{name:<90} {child.ncall:>8} {child.tsub:>10.6f} {child_tsub_pct_e2e:>10.2f} {child.ttot:>10.6f} {child_ttot_pct_e2e:>10.2f} {child.tavg:>10.6f}\n"
            )


# Create singleton instance
_profiler = YappiProfiler()

# Public API
profile = _profiler.profile
start = _profiler.start
stop = _profiler.stop
shutdown = _profiler.shutdown
is_enabled = _profiler.is_enabled
check_subprocess_spawn = _profiler.check_subprocess_spawn


def pytest_configure(config):
    """Configure yappi for pytest session."""
    if not _profiler.enabled:
        return

    print(
        f"\n[Profiling] {ENV_VAR_ENABLE_YAPPI}=1 detected, profiling enabled",
        file=sys.stderr,
    )
    print(f"[Profiling] Output directory: {_profiler.output_dir}", file=sys.stderr)


def pytest_sessionfinish(session, exitstatus):
    """Collect and display yappi results after pytest session."""
    if not _profiler.enabled:
        return

    # Shutdown main process profiler (writes its own profile)
    shutdown()

    # Collect and display all profiles (main + workers)
    print_all_profiles(
        profiler_name="yappi",
        output_dir=_profiler.output_dir,
    )
