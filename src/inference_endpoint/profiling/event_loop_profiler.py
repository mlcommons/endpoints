"""Event loop load monitoring for asyncio applications.

This module provides a simple way to measure event loop "busyness" by tracking
how much time the loop spends executing code (CPU/busy) vs waiting for I/O (idle).

Based on: https://stackoverflow.com/questions/54222794/how-do-i-monitor-how-busy-a-python-event-loop-is

The approach instruments the event loop's _run_once() and _process_events() methods:
- Time before select() = when event loop iteration started
- Time after select() = when I/O became ready
- Difference = time spent idle waiting for I/O
- Time executing callbacks = CPU busy time

Event loop load = CPU time / (CPU time + I/O wait time) * 100%

Additional metrics tracked:
- Storage I/O bytes from /proc/self/io (disk reads/writes, kernel-level accuracy)
- Network I/O bytes from /proc/net/dev (network send/recv, packets, throughput)
- CPU usage via resource.getrusage() (user vs system time)
- Context switches (voluntary = I/O wait, involuntary = preemption)
- File descriptor counts, task counts, ready queue depth
- Per-coroutine CPU and I/O wait attribution

Example:
        >>> import os
        >>> os.environ["ENABLE_LOOP_STATS"] = "1"
        >>>
        >>> setup_event_loop_policy()
        >>> asyncio.run(main()) # Stats are automatically logged when event loop completes
"""

import asyncio
import logging
import os
import resource
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import psutil
import uvloop

logger = logging.getLogger(__name__)


@dataclass
class EventLoopStats:
    """Event loop load statistics tracking CPU vs I/O wait time."""

    # Time breakdown
    total_time: float = 0.0  # Total time (CPU + I/O wait)
    select_time: float = 0.0  # Time spent waiting in select/poll (idle)
    cpu_time: float = 0.0  # Time spent executing callbacks (busy)
    callback_count: int = 0  # Number of event loop iterations

    # Socket/connection tracking
    max_fds: int = 0  # Maximum file descriptors (TCP sockets) open at once
    total_fds: int = 0  # Sum of FD counts (for averaging)
    fd_samples: int = 0  # Number of FD samples taken

    # Task/concurrency tracking
    max_tasks: int = 0  # Maximum async tasks running concurrently
    total_tasks: int = 0  # Sum of task counts (for averaging)
    task_samples: int = 0  # Number of task samples taken

    # Ready queue tracking (callbacks waiting to execute)
    max_ready: int = 0  # Maximum callbacks in ready queue
    total_ready: int = 0  # Sum of ready queue sizes
    ready_samples: int = 0  # Number of samples

    # Per-task I/O wait tracking (what tasks are suspended waiting for I/O)
    task_io_wait: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    task_io_samples: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Per-task CPU tracking (what tasks consume CPU time)
    task_cpu_time: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    task_cpu_samples: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Thread/GIL tracking
    thread_count_start: int = 0  # Thread count at start
    thread_count_end: int = 0  # Thread count at end
    max_threads: int = 0  # Maximum threads seen
    total_threads: int = 0  # Sum for averaging
    thread_samples: int = 0  # Number of samples

    # CPU & Context Switch Metrics (from resource.getrusage)
    rusage_start: dict[str, float] = field(default_factory=dict)  # rusage at start
    rusage_end: dict[str, float] = field(default_factory=dict)  # rusage at end
    user_cpu_time: float = 0.0  # User CPU time (seconds)
    system_cpu_time: float = 0.0  # System/kernel CPU time (seconds)
    voluntary_ctx_switches: int = 0  # Voluntary context switches (I/O wait)
    involuntary_ctx_switches: int = 0  # Involuntary context switches (preempted)
    max_rss_kb: int = 0  # Maximum resident set size (KB)

    # Scheduler/Timer Metrics
    max_scheduled_callbacks: int = 0  # Max pending scheduled callbacks
    total_scheduled_callbacks: int = 0  # Sum for averaging
    scheduled_callback_samples: int = 0  # Number of samples

    # I/O Bytes Metrics (from /proc/self/io - storage I/O only, not network)
    io_stats_start: dict[str, int] = field(default_factory=dict)  # I/O stats at start
    io_stats_end: dict[str, int] = field(default_factory=dict)  # I/O stats at end
    bytes_read: int = 0  # Actual bytes read from storage (disk)
    bytes_written: int = 0  # Actual bytes written to storage (disk)
    read_syscalls: int = 0  # Number of read syscalls
    write_syscalls: int = 0  # Number of write syscalls

    # Network I/O Metrics (from /proc/net/dev)
    net_stats_start: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # Network stats at start
    net_stats_end: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # Network stats at end
    net_bytes_recv: int = 0  # Network bytes received
    net_bytes_sent: int = 0  # Network bytes sent
    net_packets_recv: int = 0  # Network packets received
    net_packets_sent: int = 0  # Network packets sent

    _finalized: bool = False  # Track if we've already logged
    _process: psutil.Process | None = None  # psutil process handle

    def finalize(self) -> None:
        """Finalize stats after event loop completes.

        Calculates cpu_time from total_time and select_time.
        Following Stack Overflow reference: cpu_time = total_time - select_time

        Note: This may be called multiple times, but we only log once.
        total_time is set directly in run_forever()/run_until_complete().
        """
        # CPU time is derived: total time minus time spent waiting in select
        if self.total_time > 0:
            self.cpu_time = self.total_time - self.select_time

        # Only log once (on first finalize call with data)
        if self.total_time > 0 and not self._finalized:
            self._finalized = True
            # Print to stderr to ensure visibility (logger might not be at INFO level)
            import sys

            print(f"\n[EventLoop] Stats for PID {os.getpid()}:", file=sys.stderr)
            print(f"{self.format_stats()}", file=sys.stderr)
            sys.stderr.flush()

    @property
    def load_percent(self) -> float:
        """Event loop load as percentage (0-100%).

        Load = (CPU time / Total time) * 100

        Returns:
            0-100 percentage. 0% = fully idle, 100% = fully busy
        """
        if self.total_time == 0:
            return 0.0
        return (self.cpu_time / self.total_time) * 100

    @property
    def avg_iteration_time_ms(self) -> float:
        """Average time per event loop iteration in milliseconds."""
        if self.callback_count == 0:
            return 0.0
        return (self.total_time / self.callback_count) * 1000

    @property
    def avg_cpu_per_iteration_ms(self) -> float:
        """Average CPU time per iteration in milliseconds."""
        if self.callback_count == 0:
            return 0.0
        return (self.cpu_time / self.callback_count) * 1000

    @property
    def avg_io_wait_per_iteration_ms(self) -> float:
        """Average I/O wait time per iteration in milliseconds."""
        if self.callback_count == 0:
            return 0.0
        return (self.select_time / self.callback_count) * 1000

    @property
    def iterations_per_sec(self) -> float:
        """Event loop iterations per second (throughput)."""
        if self.total_time == 0:
            return 0.0
        return self.callback_count / self.total_time

    @property
    def avg_fds(self) -> float:
        """Average number of file descriptors (sockets) open."""
        if self.fd_samples == 0:
            return 0.0
        return self.total_fds / self.fd_samples

    @property
    def avg_tasks(self) -> float:
        """Average number of async tasks running concurrently."""
        if self.task_samples == 0:
            return 0.0
        return self.total_tasks / self.task_samples

    @property
    def avg_ready(self) -> float:
        """Average number of callbacks in ready queue."""
        if self.ready_samples == 0:
            return 0.0
        return self.total_ready / self.ready_samples

    @property
    def avg_threads(self) -> float:
        """Average number of threads running."""
        if self.thread_samples == 0:
            return 0.0
        return self.total_threads / self.thread_samples

    @property
    def total_cpu_time(self) -> float:
        """Total CPU time (user + system)."""
        return self.user_cpu_time + self.system_cpu_time

    @property
    def cpu_efficiency_percent(self) -> float:
        """CPU efficiency: what % of wall-clock time was actual CPU work."""
        if self.total_time == 0:
            return 0.0
        return (self.total_cpu_time / self.total_time) * 100

    @property
    def user_vs_system_ratio(self) -> float:
        """Ratio of user CPU to system CPU (higher is better)."""
        if self.system_cpu_time == 0:
            return 0.0
        return self.user_cpu_time / self.system_cpu_time

    @property
    def avg_scheduled_callbacks(self) -> float:
        """Average number of scheduled callbacks (timers) pending."""
        if self.scheduled_callback_samples == 0:
            return 0.0
        return self.total_scheduled_callbacks / self.scheduled_callback_samples

    @property
    def read_throughput_mbps(self) -> float:
        """Read throughput in MB/s."""
        if self.total_time == 0:
            return 0.0
        return (self.bytes_read / (1024 * 1024)) / self.total_time

    @property
    def write_throughput_mbps(self) -> float:
        """Write throughput in MB/s."""
        if self.total_time == 0:
            return 0.0
        return (self.bytes_written / (1024 * 1024)) / self.total_time

    @property
    def avg_read_size_kb(self) -> float:
        """Average read size in KB."""
        if self.read_syscalls == 0:
            return 0.0
        return (self.bytes_read / 1024) / self.read_syscalls

    @property
    def avg_write_size_kb(self) -> float:
        """Average write size in KB."""
        if self.write_syscalls == 0:
            return 0.0
        return (self.bytes_written / 1024) / self.write_syscalls

    @property
    def net_recv_throughput_mbps(self) -> float:
        """Network receive throughput in MB/s."""
        if self.total_time == 0:
            return 0.0
        return (self.net_bytes_recv / (1024 * 1024)) / self.total_time

    @property
    def net_send_throughput_mbps(self) -> float:
        """Network send throughput in MB/s."""
        if self.total_time == 0:
            return 0.0
        return (self.net_bytes_sent / (1024 * 1024)) / self.total_time

    @property
    def net_total_throughput_mbps(self) -> float:
        """Total network throughput (send + recv) in MB/s."""
        return self.net_recv_throughput_mbps + self.net_send_throughput_mbps

    def format_stats(self) -> str:
        """Format statistics as human-readable string."""
        if self.total_time == 0:
            return "No event loop activity recorded"

        io_pct = (
            (self.select_time / self.total_time * 100) if self.total_time > 0 else 0
        )

        lines = [
            "=" * 80,
            "EVENT LOOP LOAD ANALYSIS",
            "=" * 80,
            "",
            "Time Breakdown:",
            f"  Total elapsed:   {self.total_time:.3f}s",
            f"  CPU/busy time:   {self.cpu_time:.3f}s ({self.load_percent:.1f}%) ← Python executing code",
            f"  I/O wait time:   {self.select_time:.3f}s ({io_pct:.1f}%) ← Waiting for network/disk",
            "",
            "Event Loop Metrics:",
            f"  Iterations:          {self.callback_count:,} total",
            f"  Iterations/sec:      {self.iterations_per_sec:.1f}",
            f"  Avg iteration time:  {self.avg_iteration_time_ms:.3f}ms",
            f"    ├─ Avg CPU/busy:   {self.avg_cpu_per_iteration_ms:.3f}ms ({self.load_percent:.1f}%)",
            f"    └─ Avg I/O wait:   {self.avg_io_wait_per_iteration_ms:.3f}ms ({io_pct:.1f}%)",
            f"  Event loop load:     {self.load_percent:.1f}%",
            "",
        ]

        # Add concurrency stats if we have samples
        if (
            self.fd_samples > 0
            or self.task_samples > 0
            or self.ready_samples > 0
            or self.thread_samples > 0
        ):
            lines.append("Concurrency Metrics:")

            if self.fd_samples > 0:
                lines.append(
                    f"  Open file descriptors: avg={self.avg_fds:.1f}, max={self.max_fds}"
                )

            if self.task_samples > 0:
                lines.append(
                    f"  Active async tasks:    avg={self.avg_tasks:.1f}, max={self.max_tasks}"
                )

            if self.ready_samples > 0:
                lines.append(
                    f"  Ready callbacks queue: avg={self.avg_ready:.1f}, max={self.max_ready}"
                )

            if self.thread_samples > 0:
                lines.append(
                    f"  Thread count:          avg={self.avg_threads:.1f}, max={self.max_threads} "
                    f"(start={self.thread_count_start}, end={self.thread_count_end})"
                )

        lines.append("")

        # Add CPU & Context Switch stats
        if self.total_cpu_time > 0:
            lines.append("CPU & Context Switch Metrics:")
            lines.append(
                f"  User CPU:     {self.user_cpu_time:.3f}s ({self.user_cpu_time / self.total_time * 100 if self.total_time > 0 else 0:.1f}%)"
            )
            lines.append(
                f"  System CPU:   {self.system_cpu_time:.3f}s ({self.system_cpu_time / self.total_time * 100 if self.total_time > 0 else 0:.1f}%)"
            )
            lines.append(
                f"  Total CPU:    {self.total_cpu_time:.3f}s ({self.cpu_efficiency_percent:.1f}% of wall time)"
            )
            lines.append(
                f"  User/Sys:     {self.user_vs_system_ratio:.2f}x (higher = less kernel overhead)"
            )
            lines.append(f"  Vol ctx sw:   {self.voluntary_ctx_switches:,} (I/O waits)")
            lines.append(
                f"  Invol ctx sw: {self.involuntary_ctx_switches:,} (CPU preemptions)"
            )
            if self.max_rss_kb > 0:
                lines.append(f"  Max RSS:      {self.max_rss_kb / 1024:.1f} MB")
            lines.append("")

        # Add Storage I/O Bytes Metrics
        if self.bytes_read > 0 or self.bytes_written > 0:
            lines.append("Storage I/O Metrics (from /proc/self/io):")
            if self.bytes_read > 0:
                lines.append(
                    f"  Disk read:    {self.bytes_read / (1024 * 1024):.2f} MB "
                    f"({self.read_throughput_mbps:.2f} MB/s, {self.read_syscalls:,} syscalls, "
                    f"avg {self.avg_read_size_kb:.1f} KB/read)"
                )
            if self.bytes_written > 0:
                lines.append(
                    f"  Disk written: {self.bytes_written / (1024 * 1024):.2f} MB "
                    f"({self.write_throughput_mbps:.2f} MB/s, {self.write_syscalls:,} syscalls, "
                    f"avg {self.avg_write_size_kb:.1f} KB/write)"
                )
            if self.bytes_read > 0 and self.bytes_written > 0:
                total_io_mb = (self.bytes_read + self.bytes_written) / (1024 * 1024)
                total_throughput = (
                    self.read_throughput_mbps + self.write_throughput_mbps
                )
                lines.append(
                    f"  Total disk:   {total_io_mb:.2f} MB ({total_throughput:.2f} MB/s)"
                )
            lines.append("")

        # Add Network I/O Metrics
        if self.net_bytes_recv > 0 or self.net_bytes_sent > 0:
            lines.append("Network I/O Metrics (from /proc/net/dev):")
            if self.net_bytes_recv > 0:
                lines.append(
                    f"  Net received: {self.net_bytes_recv / (1024 * 1024):.2f} MB "
                    f"({self.net_recv_throughput_mbps:.2f} MB/s, {self.net_packets_recv:,} packets)"
                )
            if self.net_bytes_sent > 0:
                lines.append(
                    f"  Net sent:     {self.net_bytes_sent / (1024 * 1024):.2f} MB "
                    f"({self.net_send_throughput_mbps:.2f} MB/s, {self.net_packets_sent:,} packets)"
                )
            if self.net_bytes_recv > 0 and self.net_bytes_sent > 0:
                total_net_mb = (self.net_bytes_recv + self.net_bytes_sent) / (
                    1024 * 1024
                )
                lines.append(
                    f"  Total net:    {total_net_mb:.2f} MB ({self.net_total_throughput_mbps:.2f} MB/s)"
                )
            lines.append("")

        # Add Scheduler/Timer stats
        if self.scheduled_callback_samples > 0:
            lines.append("Scheduler Metrics:")
            lines.append(
                f"  Scheduled callbacks: avg={self.avg_scheduled_callbacks:.1f}, max={self.max_scheduled_callbacks}"
            )
            lines.append("")

        # Show which tasks/coroutines were suspended waiting for I/O
        if self.task_io_wait:
            # Sort by total I/O wait time
            top_io = sorted(
                self.task_io_wait.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # Filter out negligible entries
            significant_io = [(name, time) for name, time in top_io if time > 0.001]

            if significant_io:
                lines.append(
                    f"I/O Wait by Coroutine (% of {self.select_time:.3f}s I/O wait time):"
                )
                lines.append("")
                for func_name, io_time in significant_io:
                    io_pct = (
                        (io_time / self.select_time * 100)
                        if self.select_time > 0
                        else 0
                    )
                    lines.append(
                        f"  {func_name[:60]:<60} {io_time * 1000:>8.1f}ms ({io_pct:>5.1f}%)"
                    )
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


class MeasuredEventLoop(asyncio.SelectorEventLoop):
    """Event loop that measures time spent in select vs executing code.

    Instruments the event loop to track:
    - Total runtime (using loop.time())
    - Time spent in select/poll waiting for I/O (idle time)
    - Time spent executing callbacks (busy/CPU time)

    Based on approach from:
    https://stackoverflow.com/questions/54222794/how-do-i-monitor-how-busy-a-python-event-loop-is
    """

    def __init__(self, stats: EventLoopStats, *args, **kwargs):
        """Initialize measured event loop.

        Args:
                stats: EventLoopStats instance to populate with measurements
        """
        super().__init__(*args, **kwargs)
        self._stats = stats
        self._before_select: float | None = None
        self._start_time: float | None = None
        self._finalized = False  # Track if we've finalized

    def run_forever(self):
        """Run event loop and track total elapsed time."""
        self._start_time = self.time()

        # Capture thread stats at start
        self._stats.thread_count_start = threading.active_count()

        # Capture resource usage at start
        self._capture_rusage_start()

        # Capture I/O stats at start
        self._capture_io_stats_start()

        # Capture network stats at start
        self._capture_net_stats_start()

        try:
            super().run_forever()
        finally:
            if self._start_time is not None:
                self._stats.total_time = self.time() - self._start_time

            # Capture thread stats at end
            self._stats.thread_count_end = threading.active_count()

            # Capture resource usage at end and calculate deltas
            self._capture_rusage_end()

            # Capture I/O stats at end and calculate deltas
            self._capture_io_stats_end()

            # Capture network stats at end and calculate deltas
            self._capture_net_stats_end()

            self._stats.finalize()

    def run_until_complete(self, future):
        """Run event loop until future completes and track total time."""
        self._start_time = self.time()

        # Capture thread stats at start
        self._stats.thread_count_start = threading.active_count()

        # Capture resource usage at start
        self._capture_rusage_start()

        # Capture I/O stats at start
        self._capture_io_stats_start()

        # Capture network stats at start
        self._capture_net_stats_start()

        try:
            return super().run_until_complete(future)
        finally:
            if self._start_time is not None:
                self._stats.total_time = self.time() - self._start_time

            # Capture thread stats at end
            self._stats.thread_count_end = threading.active_count()

            # Capture resource usage at end and calculate deltas
            self._capture_rusage_end()

            # Capture I/O stats at end and calculate deltas
            self._capture_io_stats_end()

            # Capture network stats at end and calculate deltas
            self._capture_net_stats_end()

            # Finalize and print stats immediately (don't wait for close/atexit)
            if not self._finalized:
                self._finalized = True
                self._stats.finalize()

    def _run_once(self):
        """Override to capture time before select."""
        # Mark time before select/poll (for I/O wait calculation)
        self._before_select = self.time()

        # Run the iteration (ready callbacks + select + process events)
        super()._run_once()

    def _process_events(self, event_list):
        """Override to capture time after select (time spent in I/O wait)."""
        after_select = self.time()
        if self._before_select is not None:
            # Time spent in select/poll (I/O wait)
            io_wait_time = after_select - self._before_select
            self._stats.select_time += io_wait_time

            # Increment callback count (one iteration)
            self._stats.callback_count += 1

            # Track ready queue size (callbacks waiting to execute)
            try:
                ready_count = len(self._ready)
                self._stats.max_ready = max(self._stats.max_ready, ready_count)
                self._stats.total_ready += ready_count
                self._stats.ready_samples += 1
            except (AttributeError, TypeError):
                # _ready might not be accessible
                pass

            # Track file descriptor (socket) count
            try:
                # Get number of registered file descriptors from selector
                fd_count = len(self._selector.get_map())
                self._stats.max_fds = max(self._stats.max_fds, fd_count)
                self._stats.total_fds += fd_count
                self._stats.fd_samples += 1
            except (AttributeError, RuntimeError):
                # Selector might not be available or in invalid state
                pass

            # Track active task count and attribute I/O wait
            try:
                all_tasks = asyncio.all_tasks(self)
                active_task_list = [t for t in all_tasks if not t.done()]
                active_count = len(active_task_list)

                self._stats.max_tasks = max(self._stats.max_tasks, active_count)
                self._stats.total_tasks += active_count
                self._stats.task_samples += 1

                # Attribute I/O wait to active tasks
                if active_task_list and io_wait_time > 0:
                    # Split I/O wait time among active tasks
                    io_per_task = io_wait_time / active_count
                    for task in active_task_list:
                        func_name = self._get_task_name(task)
                        self._stats.task_io_wait[func_name] += io_per_task
                        self._stats.task_io_samples[func_name] += 1

            except RuntimeError:
                # Loop might be closed or in invalid state
                pass

            # Track thread count (for detecting thread pool usage)
            try:
                thread_count = threading.active_count()
                self._stats.max_threads = max(self._stats.max_threads, thread_count)
                self._stats.total_threads += thread_count
                self._stats.thread_samples += 1
            except Exception:
                # threading might fail in rare cases
                pass

            # Track scheduled callbacks (timers)
            try:
                scheduled_count = len(self._scheduled)
                self._stats.max_scheduled_callbacks = max(
                    self._stats.max_scheduled_callbacks, scheduled_count
                )
                self._stats.total_scheduled_callbacks += scheduled_count
                self._stats.scheduled_callback_samples += 1
            except (AttributeError, TypeError):
                # _scheduled might not be accessible
                pass

        # Process events (CPU attribution happens in _run_once after full iteration)
        super()._process_events(event_list)

    def _capture_rusage_start(self) -> None:
        """Capture resource usage at start of event loop."""
        try:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            self._stats.rusage_start = {
                "utime": rusage.ru_utime,
                "stime": rusage.ru_stime,
                "nvcsw": rusage.ru_nvcsw,
                "nivcsw": rusage.ru_nivcsw,
                "maxrss": rusage.ru_maxrss,
            }
        except Exception:
            # resource module might not be available on all platforms
            self._stats.rusage_start = {}

    def _capture_rusage_end(self) -> None:
        """Capture resource usage at end of event loop and calculate deltas."""
        try:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            self._stats.rusage_end = {
                "utime": rusage.ru_utime,
                "stime": rusage.ru_stime,
                "nvcsw": rusage.ru_nvcsw,
                "nivcsw": rusage.ru_nivcsw,
                "maxrss": rusage.ru_maxrss,
            }

            # Calculate deltas if we have start data
            if self._stats.rusage_start:
                self._stats.user_cpu_time = (
                    self._stats.rusage_end["utime"] - self._stats.rusage_start["utime"]
                )
                self._stats.system_cpu_time = (
                    self._stats.rusage_end["stime"] - self._stats.rusage_start["stime"]
                )
                self._stats.voluntary_ctx_switches = (
                    self._stats.rusage_end["nvcsw"] - self._stats.rusage_start["nvcsw"]
                )
                self._stats.involuntary_ctx_switches = (
                    self._stats.rusage_end["nivcsw"]
                    - self._stats.rusage_start["nivcsw"]
                )
                self._stats.max_rss_kb = self._stats.rusage_end["maxrss"]

        except Exception:
            # resource module might not be available on all platforms
            self._stats.rusage_end = {}

    def _capture_io_stats_start(self) -> None:
        """Capture I/O statistics from /proc/self/io at start of event loop."""
        try:
            with open("/proc/self/io") as f:
                io_stats = {}
                for line in f:
                    key, value = line.strip().split(": ")
                    io_stats[key] = int(value)
                self._stats.io_stats_start = io_stats
        except (FileNotFoundError, PermissionError, ValueError):
            # /proc/self/io might not be available on all platforms
            self._stats.io_stats_start = {}

    def _capture_io_stats_end(self) -> None:
        """Capture I/O statistics from /proc/self/io at end and calculate deltas."""
        try:
            with open("/proc/self/io") as f:
                io_stats = {}
                for line in f:
                    key, value = line.strip().split(": ")
                    io_stats[key] = int(value)
                self._stats.io_stats_end = io_stats

            # Calculate deltas if we have start data
            if self._stats.io_stats_start:
                self._stats.bytes_read = io_stats.get(
                    "read_bytes", 0
                ) - self._stats.io_stats_start.get("read_bytes", 0)
                self._stats.bytes_written = io_stats.get(
                    "write_bytes", 0
                ) - self._stats.io_stats_start.get("write_bytes", 0)
                self._stats.read_syscalls = io_stats.get(
                    "syscr", 0
                ) - self._stats.io_stats_start.get("syscr", 0)
                self._stats.write_syscalls = io_stats.get(
                    "syscw", 0
                ) - self._stats.io_stats_start.get("syscw", 0)

        except (FileNotFoundError, PermissionError, ValueError):
            # /proc/self/io might not be available on all platforms
            self._stats.io_stats_end = {}

    def _capture_net_stats_start(self) -> None:
        """Capture network statistics from /proc/net/dev at start of event loop."""
        try:
            net_stats = {}
            with open("/proc/net/dev") as f:
                lines = f.readlines()
                # Skip first two header lines
                for line in lines[2:]:
                    # Format: interface: bytes packets errs drop fifo frame compressed multicast ...
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    iface = parts[0].rstrip(":")
                    # Skip loopback
                    if iface == "lo":
                        continue
                    recv_bytes = int(parts[1])
                    recv_packets = int(parts[2])
                    send_bytes = int(parts[9])
                    send_packets = int(parts[10])
                    net_stats[iface] = (
                        recv_bytes,
                        recv_packets,
                        send_bytes,
                        send_packets,
                    )
            self._stats.net_stats_start = net_stats
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            # /proc/net/dev might not be available on all platforms
            self._stats.net_stats_start = {}

    def _capture_net_stats_end(self) -> None:
        """Capture network statistics from /proc/net/dev at end and calculate deltas."""
        try:
            net_stats = {}
            with open("/proc/net/dev") as f:
                lines = f.readlines()
                # Skip first two header lines
                for line in lines[2:]:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    iface = parts[0].rstrip(":")
                    # Skip loopback
                    if iface == "lo":
                        continue
                    recv_bytes = int(parts[1])
                    recv_packets = int(parts[2])
                    send_bytes = int(parts[9])
                    send_packets = int(parts[10])
                    net_stats[iface] = (
                        recv_bytes,
                        recv_packets,
                        send_bytes,
                        send_packets,
                    )
            self._stats.net_stats_end = net_stats

            # Calculate deltas if we have start data
            if self._stats.net_stats_start:
                total_recv_bytes = 0
                total_recv_packets = 0
                total_send_bytes = 0
                total_send_packets = 0

                for iface, (
                    recv_b_end,
                    recv_p_end,
                    send_b_end,
                    send_p_end,
                ) in net_stats.items():
                    if iface in self._stats.net_stats_start:
                        recv_b_start, recv_p_start, send_b_start, send_p_start = (
                            self._stats.net_stats_start[iface]
                        )
                        total_recv_bytes += recv_b_end - recv_b_start
                        total_recv_packets += recv_p_end - recv_p_start
                        total_send_bytes += send_b_end - send_b_start
                        total_send_packets += send_p_end - send_p_start

                self._stats.net_bytes_recv = total_recv_bytes
                self._stats.net_bytes_sent = total_send_bytes
                self._stats.net_packets_recv = total_recv_packets
                self._stats.net_packets_sent = total_send_packets

        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            # /proc/net/dev might not be available on all platforms
            self._stats.net_stats_end = {}

    @staticmethod
    def _get_task_name(task: asyncio.Task) -> str:
        """Extract meaningful name from asyncio task."""
        try:
            # Try to get coroutine
            coro = task.get_coro()

            # Get qualified name from coroutine
            if hasattr(coro, "__qualname__"):
                return coro.__qualname__

            # Try cr_code for function name
            if hasattr(coro, "cr_code"):
                return coro.cr_code.co_qualname or coro.cr_code.co_name

            # Parse from string representation
            coro_str = str(coro)
            # Extract from "<coroutine object ClassName.method_name at 0x...>"
            if "<coroutine object " in coro_str:
                name = coro_str.split("<coroutine object ")[1].split(" at ")[0]
                return name

            # Fallback to task name
            task_name = task.get_name()
            if task_name and task_name != "Task-" and not task_name.startswith("Task-"):
                return task_name

            return coro_str[:50]
        except (AttributeError, RuntimeError):
            return "unknown"


class MeasuredEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Event loop policy that creates measured event loops.

    All event loops created by this policy will track load statistics.
    """

    def __init__(self):
        """Initialize policy with shared stats tracker."""
        super().__init__()
        self.stats = EventLoopStats()

    def new_event_loop(self):
        """Create a new measured event loop."""
        # Use standard asyncio SelectorEventLoop with measurement
        return MeasuredEventLoop(self.stats)


def setup_event_loop_policy() -> None:
    """Set up event loop policy based on ENABLE_LOOP_STATS environment variable.

    If ENABLE_LOOP_STATS=1, installs MeasuredEventLoopPolicy that instruments
    the event loop to track CPU vs I/O time and automatically logs stats.

    Otherwise, uses standard uvloop for maximum performance.

    Example:
        >>> setup_event_loop_policy()
        >>> asyncio.run(main())
        >>> # If ENABLE_LOOP_STATS=1, stats are logged automatically
    """
    if os.environ.get("ENABLE_LOOP_STATS", "0") == "1":
        # Install measured event loop policy (standard asyncio with instrumentation)
        policy = MeasuredEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)
    else:
        # Use standard uvloop for production (no monitoring overhead)
        uvloop.install()


def new_event_loop() -> asyncio.AbstractEventLoop:
    """Create a new event loop respecting ENABLE_LOOP_STATS setting.

    Returns instrumented MeasuredEventLoop if ENABLE_LOOP_STATS=1,
    otherwise returns high-performance uvloop.

    Returns:
        New event loop instance (MeasuredEventLoop or uvloop)

    Example:
        >>> loop = new_event_loop()
        >>> asyncio.set_event_loop(loop)
        >>> loop.run_forever()
    """
    if os.environ.get("ENABLE_LOOP_STATS", "0") == "1":
        # Create instrumented event loop for profiling
        stats = EventLoopStats()
        return MeasuredEventLoop(stats)
    else:
        # Use high-performance uvloop for production
        return uvloop.new_event_loop()
