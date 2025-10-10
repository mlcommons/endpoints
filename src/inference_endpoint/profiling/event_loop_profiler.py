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

"""Event loop profiling with I/O wait attribution and callback tracking.

This module provides accurate event-loop-specific profiling:
- Time in select() (I/O wait) vs time executing callbacks (CPU busy)
- Which file descriptors are causing I/O wait
- FD type detection (how many TCP sockets, Unix sockets, etc.)
- Per-callback execution statistics
- Per-task I/O wait attribution

Example:
    >>> import os
    >>> os.environ["ENABLE_LOOP_STATS"] = "1"
    >>> setup_event_loop_policy()
    >>> asyncio.run(main())  # Stats displayed at end
"""

import asyncio
import logging
import os
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from selectors import EVENT_READ, EVENT_WRITE, BaseSelector
from typing import Any

import uvloop

logger = logging.getLogger(__name__)


def get_fd_type(fd: int) -> str:
    """Determine file descriptor type."""
    try:
        sock = socket.socket(fileno=fd)
        try:
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                return "tcp" if sock.type == socket.SOCK_STREAM else "udp"
            elif sock.family == socket.AF_UNIX:
                return "unix"
            return "socket"
        finally:
            sock.detach()
    except (OSError, AttributeError):
        try:
            import stat

            mode = os.fstat(fd).st_mode
            if stat.S_ISFIFO(mode):
                return "pipe"
            if stat.S_ISREG(mode):
                return "file"
            return "other"
        except Exception:
            return "unknown"


def get_suspension_point(coro) -> str | None:
    """
    Extract ALL suspension points in the coroutine chain as a call stack.

    Returns a string showing the full await chain from top to bottom.
    """
    try:
        stack_parts = []
        current_coro = coro
        max_depth = 10  # Prevent infinite loops
        depth = 0

        while depth < max_depth:
            if not hasattr(current_coro, "cr_frame") or current_coro.cr_frame is None:
                break

            frame = current_coro.cr_frame
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            funcname = frame.f_code.co_name

            # Simplify path - only show relative path from workspace
            if "workspace" in filename:
                filename = filename.split("workspace/")[-1]
            elif "site-packages" in filename:
                # For library code, just show package name
                parts = filename.split("site-packages/")[-1].split("/")
                filename = (
                    f"{parts[0]}/.../{parts[-1]}" if len(parts) > 2 else "/".join(parts)
                )

            # Add this level to the stack
            location = f"{filename}:{lineno}:{funcname}"
            stack_parts.append(location)

            # Check if this coroutine is awaiting another coroutine
            # If cr_await exists and points to another coroutine, go deeper
            if hasattr(current_coro, "cr_await") and current_coro.cr_await is not None:
                next_coro = current_coro.cr_await
                # Only go deeper if next_coro is actually a coroutine
                if hasattr(next_coro, "cr_frame"):
                    current_coro = next_coro
                    depth += 1
                    continue

            # Reached the leaf
            break

        # Return the FULL coroutine stack to show the complete await chain
        if stack_parts:
            return " → ".join(stack_parts)
        return None

    except Exception as e:
        # Debug: show what failed
        import sys

        print(f"[DEBUG] get_suspension_point failed: {e}", file=sys.stderr, flush=True)
    return None


def get_callback_name(handle: Any) -> tuple[str, bool]:
    """Extract callback name from Handle, decoding TaskStepMethWrapper to actual coroutine.

    Returns:
        (callback_name, is_task_execution): Name and whether this is a task execution callback
    """
    try:
        cb = handle._callback if hasattr(handle, "_callback") else handle

        # Decode TaskStepMethWrapper to get the actual task/coroutine
        # TaskStepMethWrapper.__self__ points to the Task object
        if type(cb).__name__ == "TaskStepMethWrapper":
            if hasattr(cb, "__self__"):
                task = cb.__self__
                # Try to get the coroutine from the task
                if hasattr(task, "_coro"):
                    coro = task._coro
                    # Extract coroutine name
                    coro_name = getattr(coro, "__qualname__", None) or getattr(
                        coro, "__name__", None
                    )
                    if coro_name:
                        return f"Task({coro_name})", True
                elif hasattr(task, "get_coro"):
                    coro = task.get_coro()
                    coro_name = getattr(coro, "__qualname__", None) or getattr(
                        coro, "__name__", None
                    )
                    if coro_name:
                        return f"Task({coro_name})", True
                # Fallback: use task name if available
                if hasattr(task, "get_name"):
                    task_name = task.get_name()
                    if task_name and not task_name.startswith("Task-"):
                        return f"Task({task_name})", True

        # Decode Task.task_wakeup - merge with task execution stats
        # These are logically the same operation (wakeup schedules the task step)
        name = getattr(cb, "__qualname__", None) or getattr(cb, "__name__", None)
        if name == "Task.task_wakeup":
            # cb is a bound method, cb.__self__ is the task
            if hasattr(cb, "__self__"):
                task = cb.__self__
                if hasattr(task, "_coro"):
                    coro = task._coro
                    coro_name = getattr(coro, "__qualname__", None) or getattr(
                        coro, "__name__", None
                    )
                    if coro_name:
                        # Use same name as TaskStepMethWrapper to merge stats
                        return f"Task({coro_name})", False

        # Standard callback name extraction
        return (name or str(cb)[:50]), False
    except Exception:
        return "unknown", False


@dataclass
class FDStats:
    """Statistics for a file descriptor."""

    fd: int
    fd_type: str
    wait_time: float = 0.0
    wait_count: int = 0
    events: set[int] = field(default_factory=set)
    callbacks: set[str] = field(default_factory=set)


@dataclass
class CallbackStats:
    """Per-callback execution statistics."""

    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0


@dataclass
class TaskSuspensionPoint:
    """Track where tasks suspend/resume."""

    location: str  # "filename:lineno"
    count: int = 0
    total_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0


@dataclass
class EventLoopStats:
    """Event loop statistics."""

    # Time breakdown
    total_time: float = 0.0
    select_time: float = 0.0  # I/O wait
    callback_time: float = 0.0  # CPU busy
    iterations: int = 0

    # Concurrency
    max_fds: int = 0
    total_fds: int = 0
    max_tasks: int = 0
    total_tasks: int = 0
    max_ready: int = 0
    total_ready: int = 0
    samples: int = 0

    # FD tracking
    fd_stats: dict[int, FDStats] = field(default_factory=dict)
    fd_type_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fd_seen: set[int] = field(default_factory=set)

    # Callback tracking
    callback_stats: dict[str, CallbackStats] = field(default_factory=dict)
    total_callbacks: int = 0

    # Task I/O wait
    task_io_wait: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # Task scheduling latency (time from create to first run)
    task_schedule_latencies: list[float] = field(default_factory=list)

    # Task lifecycle metrics
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_cancelled: int = 0

    # Task suspension points (where tasks await/suspend) with I/O wait attribution
    task_suspension_points: dict[str, TaskSuspensionPoint] = field(default_factory=dict)

    @property
    def load_percent(self) -> float:
        """Event loop CPU utilization: CPU time / wall time * 100."""
        return (
            (self.callback_time / self.total_time * 100) if self.total_time > 0 else 0.0
        )

    @property
    def avg_fds(self) -> float:
        return self.total_fds / self.samples if self.samples > 0 else 0.0

    @property
    def avg_tasks(self) -> float:
        return self.total_tasks / self.samples if self.samples > 0 else 0.0

    @property
    def avg_ready(self) -> float:
        return self.total_ready / self.samples if self.samples > 0 else 0.0

    def format_stats(self) -> str:
        """Format statistics."""
        if self.total_time == 0:
            return "No activity"

        lines = [
            "=" * 80,
            "EVENT LOOP PROFILING",
            "=" * 80,
            "",
            "Time Breakdown:",
            f"  Wall time: {self.total_time:.3f}s (elapsed)",
            f"  CPU time:  {self.callback_time:.3f}s ({self.load_percent:.1f}% utilization)",
            f"  I/O wait:  {self.select_time:.3f}s ({100-self.load_percent:.1f}% idle)",
            "",
            "Iterations:",
            f"  Count:     {self.iterations:,} ({self.iterations/self.total_time:.1f}/s)",
            f"  Avg time:  {self.total_time/self.iterations*1000:.3f}ms",
            f"  Callbacks: {self.total_callbacks:,} ({self.total_callbacks/self.iterations:.1f} avg)",
            "",
            "Concurrency:",
            f"  FDs:       avg={self.avg_fds:.1f}, max={self.max_fds}",
            f"  Tasks:     avg={self.avg_tasks:.1f}, max={self.max_tasks}",
            f"  Ready:     avg={self.avg_ready:.1f}, max={self.max_ready}",
            "",
            "Task Lifecycle:",
            f"  Created:   {self.tasks_created:,}",
            f"  Completed: {self.tasks_completed:,}",
            f"  Cancelled: {self.tasks_cancelled:,}",
            f"  Active:    {self.tasks_created - self.tasks_completed - self.tasks_cancelled:,}",
            "",
        ]

        # Task scheduling latency
        if self.task_schedule_latencies:
            latencies_ms = sorted(
                [latency * 1000 for latency in self.task_schedule_latencies]
            )
            count = len(latencies_ms)
            lines.extend(
                [
                    "Task Scheduling Latency (create → run):",
                    f"  Count:     {count:,}",
                    f"  Min:       {latencies_ms[0]:.3f}ms",
                    f"  Avg:       {sum(latencies_ms)/count:.3f}ms",
                    f"  P50:       {latencies_ms[count//2]:.3f}ms",
                    f"  P95:       {latencies_ms[int(count*0.95)]:.3f}ms",
                    f"  P99:       {latencies_ms[int(count*0.99)]:.3f}ms",
                    f"  Max:       {latencies_ms[-1]:.3f}ms",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "Task Scheduling Latency:",
                    "  No data collected",
                    "",
                ]
            )

        # FD types
        if self.fd_type_counts:
            lines.append("File Descriptor Types:")
            total = sum(self.fd_type_counts.values())
            for fd_type, count in sorted(
                self.fd_type_counts.items(), key=lambda x: -x[1]
            ):
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"  {fd_type:<10} {count:>4} ({pct:>5.1f}%)")
            lines.append("")

        # Top FDs by wait time
        if self.fd_stats:
            lines.append("Top FDs by I/O Wait:")
            sorted_fds = sorted(
                self.fd_stats.items(), key=lambda x: x[1].wait_time, reverse=True
            )[:50]
            for fd, stats in sorted_fds:
                pct = (
                    stats.wait_time / self.select_time * 100
                    if self.select_time > 0
                    else 0
                )
                ev = ("R" if EVENT_READ in stats.events else "") + (
                    "W" if EVENT_WRITE in stats.events else ""
                )
                cbs = ", ".join(list(stats.callbacks)[:5])
                lines.append(
                    f"  FD{fd:>4} ({stats.fd_type:<6}) {ev:>2} | {stats.wait_time*1000:>7.1f}ms ({pct:>5.1f}%) | {cbs}"
                )
            lines.append("")

        # Callback summary by category
        if self.callback_stats:
            # Group callbacks by category
            categories = {
                "Tasks": defaultdict(lambda: {"time": 0.0, "count": 0}),
                "I/O": defaultdict(lambda: {"time": 0.0, "count": 0}),
                "ZMQ": defaultdict(lambda: {"time": 0.0, "count": 0}),
                "Other": defaultdict(lambda: {"time": 0.0, "count": 0}),
            }

            for cb in self.callback_stats.values():
                # Check ZMQ first (more specific) before I/O (more general)
                if "Zmq" in cb.name or "ZMQ" in cb.name:
                    cat = "ZMQ"
                elif "Task" in cb.name or "task_wakeup" in cb.name:
                    cat = "Tasks"
                elif (
                    "Transport" in cb.name
                    or "_read_ready" in cb.name
                    or "_write_ready" in cb.name
                ):
                    cat = "I/O"
                else:
                    cat = "Other"

                categories[cat][cb.name]["time"] += cb.total_time
                categories[cat][cb.name]["count"] += cb.count

            # Display category summaries
            lines.append("Callback Summary by Category:")
            for cat_name in ["Tasks", "I/O", "ZMQ", "Other"]:
                cat_data = categories[cat_name]
                if not cat_data:
                    continue
                total_time = sum(v["time"] for v in cat_data.values())
                total_count = sum(v["count"] for v in cat_data.values())
                if total_time > 0.0001:
                    pct = (
                        total_time / self.callback_time * 100
                        if self.callback_time > 0
                        else 0
                    )
                    lines.append(
                        f"  {cat_name:<10} {total_time*1000:>8.1f}ms ({total_count:>8,} calls, {pct:>5.1f}%)"
                    )
            lines.append("")

            # Top individual callbacks
            lines.append("Top Callbacks by Time:")
            sorted_cbs = sorted(
                self.callback_stats.values(), key=lambda x: x.total_time, reverse=True
            )[:50]
            for cb in sorted_cbs:
                if cb.total_time < 0.0001:
                    continue
                lines.append(
                    f"  {cb.total_time*1000:>8.1f}ms ({cb.count:>6} calls, {cb.avg_time*1000:>6.3f}ms avg) | {cb.name}"
                )
            lines.append("")

        # Top tasks by I/O wait
        if self.task_io_wait:
            lines.append("Top Tasks by I/O Wait:")
            sorted_tasks = sorted(
                self.task_io_wait.items(), key=lambda x: x[1], reverse=True
            )[:50]
            for task_name, wait_time in sorted_tasks:
                if wait_time < 0.001:
                    continue
                pct = wait_time / self.select_time * 100 if self.select_time > 0 else 0
                lines.append(f"  {wait_time*1000:>7.1f}ms ({pct:>5.1f}%) | {task_name}")
            lines.append("")

        # ============================================================
        # I/O WAIT ATTRIBUTION
        # ============================================================
        lines.append("")
        lines.append("=" * 80)
        lines.append("I/O WAIT ATTRIBUTION BY CODE LOCATION")
        lines.append(
            "(Cumulative task suspension time - can exceed wall-clock I/O due to parallelism)"
        )
        lines.append("=" * 80)
        if self.task_suspension_points:
            sorted_points = sorted(
                self.task_suspension_points.values(),
                key=lambda x: x.total_time,
                reverse=True,
            )[:30]
            total_suspension = sum(p.total_time for p in sorted_points)
            for point in sorted_points:
                if point.total_time > 0.001:
                    # Percentage of total suspension time across all tasks
                    pct = (
                        point.total_time / total_suspension * 100
                        if total_suspension > 0
                        else 0
                    )
                    lines.append(
                        f"{point.total_time*1000:>9.1f}ms ({pct:>5.1f}%) | "
                        f"{point.count:>5} suspends @ {point.avg_time*1000:>6.3f}ms avg"
                    )
                    lines.append(f"  {point.location}")
                    lines.append("")
        else:
            lines.append("No I/O suspension points captured")

        # ============================================================
        # CPU TIME ATTRIBUTION
        # ============================================================
        lines.append("=" * 80)
        lines.append("CPU TIME ATTRIBUTION BY CALLBACK")
        lines.append("=" * 80)
        if self.callback_stats:
            # Filter to show only Task callbacks (your code), not internals
            task_callbacks = {
                k: v for k, v in self.callback_stats.items() if "Task(" in k
            }
            if task_callbacks:
                sorted_cbs = sorted(
                    task_callbacks.values(), key=lambda x: x.total_time, reverse=True
                )[:50]
                for cb in sorted_cbs:
                    if cb.total_time > 0.0001:
                        pct = (
                            cb.total_time / self.callback_time * 100
                            if self.callback_time > 0
                            else 0
                        )
                        lines.append(
                            f"  {cb.total_time*1000:>8.1f}ms ({pct:>5.1f}% of CPU) | "
                            f"{cb.count:>6} calls @ {cb.avg_time*1000:>6.3f}ms avg | {cb.name}"
                        )
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


class InstrumentedSelector:
    """Selector wrapper that tracks I/O wait and attributes to FDs."""

    def __init__(self, selector: BaseSelector, stats: EventLoopStats):
        self._selector = selector
        self._stats = stats

    def select(self, timeout=None):
        """Track select() time and attribute to waiting FDs."""
        # Capture FDs before select
        waiting_fds = []
        try:
            for key in self._selector.get_map().values():
                fd = key.fileobj if isinstance(key.fileobj, int) else key.fd
                fd_type = get_fd_type(fd)
                callback_name, _ = (
                    get_callback_name(key.data)
                    if hasattr(key, "data")
                    else ("unknown", False)
                )
                waiting_fds.append((fd, fd_type, key.events, callback_name))
        except Exception:
            pass

        # Measure select time
        start = time.perf_counter()
        try:
            return self._selector.select(timeout)
        finally:
            elapsed = time.perf_counter() - start
            self._stats.select_time += elapsed

            # Attribute wait to FDs
            if waiting_fds and elapsed > 0:
                time_per_fd = elapsed / len(waiting_fds)
                for fd, fd_type, events, callback_name in waiting_fds:
                    if fd not in self._stats.fd_stats:
                        self._stats.fd_stats[fd] = FDStats(fd=fd, fd_type=fd_type)

                    fd_stats = self._stats.fd_stats[fd]
                    fd_stats.wait_time += time_per_fd
                    fd_stats.wait_count += 1
                    fd_stats.events.add(events)
                    fd_stats.callbacks.add(callback_name)

                    # Track unique FD types
                    if fd not in self._stats.fd_seen:
                        self._stats.fd_seen.add(fd)
                        self._stats.fd_type_counts[fd_type] += 1

    def __getattr__(self, name):
        return getattr(self._selector, name)


class MeasuredEventLoop(asyncio.SelectorEventLoop):
    """Event loop with comprehensive profiling."""

    def __init__(self, stats: EventLoopStats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats = stats
        self._start_time: float | None = None
        self._stats_file: str | None = None
        self._finalized = False  # Guard against double-finalize

        # Task scheduling latency tracking
        self._task_create_times: dict[int, float] = {}  # task_id -> creation_time
        self._task_first_run: set[int] = set()  # tasks that have run at least once

        # Task suspension tracking (where tasks await)
        self._task_suspend_times: dict[
            int, tuple[float, str]
        ] = {}  # task_id -> (suspend_time, location)

        # Instrument selector
        if hasattr(self, "_selector") and self._selector is not None:
            self._selector = InstrumentedSelector(self._selector, self._stats)

    def run_until_complete(self, future):
        self._start_time = time.perf_counter()
        self._setup_stats_file()

        # Register atexit to ensure stats are written even on interrupt
        import atexit

        atexit.register(self._finalize)

        try:
            return super().run_until_complete(future)
        finally:
            self._finalize()

    def close(self):
        """Override close to finalize stats."""
        self._finalize()
        super().close()

    def create_task(self, coro, *, name=None, context=None):
        """Override create_task to track task creation time and lifecycle."""
        task = super().create_task(coro, name=name, context=context)

        # Track task creation
        self._stats.tasks_created += 1
        task_id = id(task)
        creation_time = time.perf_counter()
        self._task_create_times[task_id] = creation_time

        # Record initial suspension point (task starts suspended before first run)
        if hasattr(task, "_coro") and task._coro:
            suspend_location = get_suspension_point(task._coro)
            if suspend_location:
                self._task_suspend_times[task_id] = (creation_time, suspend_location)

        # Add done callback to track completion/cancellation
        def _task_done_callback(t):
            if t.cancelled():
                self._stats.tasks_cancelled += 1
            else:
                self._stats.tasks_completed += 1
            # Clean up tracking data
            tid = id(t)
            if tid in self._task_create_times:
                del self._task_create_times[tid]
            if tid in self._task_suspend_times:
                del self._task_suspend_times[tid]
            if tid in self._task_first_run:
                self._task_first_run.discard(tid)

        task.add_done_callback(_task_done_callback)
        return task

    def _run_once(self):
        """Track iteration and profile callbacks."""
        self._stats.iterations += 1

        # Count ready callbacks
        ready_count = len(self._ready) if hasattr(self, "_ready") else 0
        self._stats.total_callbacks += ready_count

        # Time callback execution using CPU time (not wall time)
        callback_cpu_start = time.thread_time()

        # Profile individual callbacks
        if hasattr(self, "_ready") and ready_count > 0:
            self._profile_callbacks()
        else:
            super()._run_once()

        # Calculate CPU time spent in callbacks
        callback_cpu_elapsed = time.thread_time() - callback_cpu_start
        if callback_cpu_elapsed > 0:
            self._stats.callback_time += callback_cpu_elapsed

        # Sample metrics
        self._sample_metrics()

        # Periodically write stats (every 5000 iterations) to survive hard kills
        if self._stats.iterations % 5000 == 0 and self._stats_file:
            self._write_snapshot()

    def _profile_callbacks(self):
        """Profile individual callback execution."""
        import asyncio.events

        original_run = asyncio.events.Handle._run
        stats = self._stats
        task_create_times = self._task_create_times
        task_first_run = self._task_first_run
        task_suspend_times = self._task_suspend_times

        def profiled_run(handle_self):
            name, is_task_execution = get_callback_name(handle_self)
            wall_start = time.perf_counter()  # Wall time for I/O tracking
            cpu_start = time.thread_time()  # CPU time for callback execution

            # Track task scheduling latency and suspension points
            # Only for actual task executions (TaskStepMethWrapper), not wakeup callbacks
            cb = (
                handle_self._callback
                if hasattr(handle_self, "_callback")
                else handle_self
            )
            task = None
            task_id = None

            if (
                is_task_execution
                and type(cb).__name__ == "TaskStepMethWrapper"
                and hasattr(cb, "__self__")
            ):
                task = cb.__self__
                task_id = id(task)

                # Track scheduling latency on first run (wall time)
                if task_id not in task_first_run and task_id in task_create_times:
                    create_time = task_create_times[task_id]
                    latency = wall_start - create_time
                    stats.task_schedule_latencies.append(latency)
                    task_first_run.add(task_id)
                    del task_create_times[task_id]

                # Track suspension time (task is resuming now) - wall time for I/O wait
                if task_id in task_suspend_times:
                    suspend_time, suspend_location = task_suspend_times[task_id]
                    suspension_duration = wall_start - suspend_time

                    # Record suspension point with I/O wait time
                    if suspend_location not in stats.task_suspension_points:
                        stats.task_suspension_points[suspend_location] = (
                            TaskSuspensionPoint(location=suspend_location)
                        )
                    point = stats.task_suspension_points[suspend_location]
                    point.count += 1
                    point.total_time += suspension_duration

                    # Clean up
                    del task_suspend_times[task_id]

            try:
                return original_run(handle_self)
            finally:
                # Measure CPU time spent in callback
                cpu_elapsed = time.thread_time() - cpu_start
                wall_end = time.perf_counter()

                # After callback runs, record new suspension point if task is still alive
                # Only for actual task executions, not wakeup callbacks
                if (
                    is_task_execution
                    and task is not None
                    and task_id is not None
                    and not task.done()
                ):
                    if hasattr(task, "_coro"):
                        suspend_location = get_suspension_point(task._coro)
                        if suspend_location:
                            task_suspend_times[task_id] = (wall_end, suspend_location)

                # Record CPU time for this callback
                if name not in stats.callback_stats:
                    stats.callback_stats[name] = CallbackStats(name=name)
                cb_stats = stats.callback_stats[name]
                cb_stats.count += 1
                cb_stats.total_time += cpu_elapsed  # Use CPU time, not wall time
                cb_stats.min_time = min(cb_stats.min_time, cpu_elapsed)
                cb_stats.max_time = max(cb_stats.max_time, cpu_elapsed)

        try:
            asyncio.events.Handle._run = profiled_run
            super()._run_once()
        finally:
            asyncio.events.Handle._run = original_run

    def _sample_metrics(self):
        """Sample concurrency metrics."""
        self._stats.samples += 1

        # FDs
        try:
            selector = (
                self._selector._selector
                if hasattr(self._selector, "_selector")
                else self._selector
            )
            fd_count = len(selector.get_map())
            self._stats.max_fds = max(self._stats.max_fds, fd_count)
            self._stats.total_fds += fd_count
        except (AttributeError, RuntimeError):
            pass

        # Tasks
        try:
            tasks = [t for t in asyncio.all_tasks(self) if not t.done()]
            count = len(tasks)
            self._stats.max_tasks = max(self._stats.max_tasks, count)
            self._stats.total_tasks += count

            # Sample task suspension points EVERY iteration for comprehensive coverage
            # This captures all the different places tasks are waiting
            for task in tasks:
                if hasattr(task, "_coro") and task._coro:
                    suspend_loc = get_suspension_point(task._coro)
                    if (
                        suspend_loc
                        and suspend_loc not in self._stats.task_suspension_points
                    ):
                        self._stats.task_suspension_points[suspend_loc] = (
                            TaskSuspensionPoint(location=suspend_loc)
                        )

        except RuntimeError:
            pass

        # Ready queue
        try:
            ready_count = len(self._ready)
            self._stats.max_ready = max(self._stats.max_ready, ready_count)
            self._stats.total_ready += ready_count
        except (AttributeError, TypeError):
            pass

    def _write_snapshot(self):
        """Write current stats snapshot (called periodically)."""
        if not self._stats_file or self._stats.iterations == 0:
            return

        # Calculate current stats
        current_time = time.perf_counter() - self._start_time if self._start_time else 0
        current_callback_time = max(0, current_time - self._stats.select_time)

        # Temporarily set for formatting
        old_total = self._stats.total_time
        old_cb = self._stats.callback_time
        self._stats.total_time = current_time
        self._stats.callback_time = current_callback_time

        try:
            output = f"\n[EventLoop] PID {os.getpid()}:\n{self._stats.format_stats()}\n"
            with open(self._stats_file, "w") as f:
                f.write(output)
                f.flush()
        except Exception:
            pass
        finally:
            # Restore
            self._stats.total_time = old_total
            self._stats.callback_time = old_cb

    def _finalize(self):
        """Write final stats."""
        if self._finalized or self._stats.iterations == 0:
            return

        if self._start_time is not None:
            self._stats.total_time = time.perf_counter() - self._start_time

        if self._stats.total_time == 0:
            return

        # Calculate CPU time
        self._stats.callback_time = max(
            0, self._stats.total_time - self._stats.select_time
        )

        # Format output
        output = f"\n[EventLoop] PID {os.getpid()}:\n{self._stats.format_stats()}\n"

        # Write to file
        if self._stats_file:
            try:
                os.makedirs(os.path.dirname(self._stats_file), exist_ok=True)
                with open(self._stats_file, "w") as f:
                    f.write(output)
                    f.flush()
                    os.fsync(f.fileno())
                self._finalized = True

                # In standalone mode (not pytest), also print inline
                if "pytest" not in sys.modules:
                    print(output, file=sys.stderr)
                    sys.stderr.flush()
            except Exception as e:
                logger.error(f"Failed to write event loop stats: {e}")

    def _setup_stats_file(self):
        """Setup stats file for all processes when profiling is enabled."""
        if self._stats_file is None:
            try:
                base = os.environ.get(
                    "EVENT_LOOP_STATS_LOGFILE", "/tmp/mlperf_event_loop_stats/stats"
                )
                os.makedirs(os.path.dirname(base), exist_ok=True)
                self._stats_file = f"{base}.{os.getpid()}"
            except Exception:
                pass


class MeasuredEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Policy that creates measured event loops."""

    def __init__(self):
        super().__init__()
        self.stats = EventLoopStats()

    def new_event_loop(self):
        return MeasuredEventLoop(self.stats)


def setup_event_loop_policy() -> None:
    """Setup event loop policy based on ENABLE_LOOP_STATS env var."""
    if os.environ.get("ENABLE_LOOP_STATS", "0") == "1":
        asyncio.set_event_loop_policy(MeasuredEventLoopPolicy())
    else:
        uvloop.install()


def new_event_loop() -> asyncio.AbstractEventLoop:
    """Create new event loop respecting ENABLE_LOOP_STATS."""
    if os.environ.get("ENABLE_LOOP_STATS", "0") == "1":
        policy = asyncio.get_event_loop_policy()
        if isinstance(policy, MeasuredEventLoopPolicy):
            return policy.new_event_loop()
        else:
            return MeasuredEventLoop(EventLoopStats())
    else:
        return uvloop.new_event_loop()


def pytest_configure(config):
    """Configure pytest."""
    if os.environ.get("ENABLE_LOOP_STATS", "0") != "1":
        return

    print("\n[EventLoop] Profiling enabled", file=sys.stderr)

    if not os.environ.get("EVENT_LOOP_STATS_LOGFILE"):
        os.environ["EVENT_LOOP_STATS_LOGFILE"] = "/tmp/mlperf_event_loop_stats/stats"

    # Install the profiling event loop policy for this process and any workers
    setup_event_loop_policy()


def pytest_sessionfinish(session, exitstatus):
    """Display worker stats after pytest."""
    import glob
    import shutil

    if os.environ.get("ENABLE_LOOP_STATS", "0") != "1":
        return

    base = os.environ.get(
        "EVENT_LOOP_STATS_LOGFILE", "/tmp/mlperf_event_loop_stats/stats"
    )
    all_files = glob.glob(f"{base}.*")
    main_pid = os.getpid()
    worker_files = sorted([f for f in all_files if not f.endswith(f".{main_pid}")])

    if worker_files:
        print("\n" + "=" * 80, file=sys.stderr)
        print("EVENT LOOP PROFILING RESULTS", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

        for filepath in worker_files:
            try:
                with open(filepath) as f:
                    content = f.read()
                    if content.strip():
                        print(content, file=sys.stderr)
                        print("\n", file=sys.stderr)
            except Exception as e:
                print(f"Error reading {filepath}: {e}", file=sys.stderr)

        # Cleanup
        try:
            stats_dir = os.path.dirname(base)
            if stats_dir and os.path.exists(stats_dir):
                shutil.rmtree(stats_dir, ignore_errors=True)
        except Exception:
            pass
