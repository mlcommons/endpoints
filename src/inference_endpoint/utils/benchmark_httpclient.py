#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HTTP client performance testing utility.

Benchmarks send/recv rate of the HTTPEndpointClient using uvloop.
Can auto-launch a MaxThroughputServer or connect to an external endpoint.

Usage (see all available args in --help):
    python -m inference_endpoint.utils.benchmark_httpclient -w 8 -c 512 -d 20
    python -m inference_endpoint.utils.benchmark_httpclient --endpoint http://host:8080/v1/chat/completions
    python -m inference_endpoint.utils.benchmark_httpclient --no-pin --track-memory

Sweep modes (-w, -c, -l accept ranges; endpoints always included):
    -w 4:12           every int in [4, 12]
    -c 100:500:100    start:stop:step  -> [100, 200, 300, 400, 500]
    -w 1:32::12       start:stop::N    -> 12 evenly-spaced points in [1, 32]
    -l 32,128,512     explicit values
    -w 1:32::12 -c 100:500::4          cartesian product sweep
    --full                             preset sweep of common worker counts x prompt lengths (non-streaming)
    --full --stream                    preset sweep of common worker counts x prompt lengths (streaming)
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import itertools
import os
import re
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass

from inference_endpoint.async_utils.transport.zmq.context import (
    ManagedZMQContext,
)
from inference_endpoint.core.types import Query, QueryResult
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.cpu_affinity import (
    compute_affinity_plan,
)
from inference_endpoint.endpoint_client.http_client import (
    HTTPEndpointClient,
)
from inference_endpoint.testing.max_throughput_server import (
    MaxThroughputServer,
    build_response,
)

# Suppress transformers "no framework found" warning (only tokenizers used)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@dataclass(slots=True)
class BenchmarkStats:
    """Snapshot of benchmark statistics."""

    sent: int = 0
    received: int = 0
    errors: int = 0
    send_elapsed_ns: int = 0  # Send phase duration
    total_elapsed_ns: int = 0  # Total duration including drain
    peak_inflight: int = 0  # Max observed in-flight count
    stall_ns: int = 0  # Client overhead: time blocked waiting for responses to drain
    sse_events_per_response: int = 1  # SSE events per response (set for streaming)
    # Per-interval samples (populated by LiveDisplay)
    send_rate_samples: list[float] | None = None
    recv_rate_samples: list[float] | None = None


@dataclass(slots=True)
class MemoryStats:
    """Memory statistics for a process."""

    pid: int
    rss_mb: float
    shm_mb: float


@dataclass(slots=True)
class SweepResult:
    """Result of a single benchmark run within a parameter sweep."""

    param_values: dict[str, int]  # swept param name -> value for this run
    stats: BenchmarkStats
    send_rate: float  # req/s (mean)
    recv_rate: float  # resp/s (mean)
    sse_rate: float  # SSE pkts/s (mean, streaming only)
    outstanding: int  # sent - received - errors
    error_rate: float  # errors/sent (%)
    stall_pct: float  # client overhead: % of send time blocked on drain (%)
    # Variation bounds from per-second samples
    send_rate_min: float = 0.0
    send_rate_max: float = 0.0
    recv_rate_min: float = 0.0
    recv_rate_max: float = 0.0
    sse_rate_min: float = 0.0
    sse_rate_max: float = 0.0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _linspace_int(start: int, stop: int, n: int) -> list[int]:
    """Generate *n* evenly-spaced integers from *start* to *stop* (inclusive).

    Intermediate values are rounded to the nearest integer, but *start* and
    *stop* are always included exactly.
    """
    if n <= 0:
        raise argparse.ArgumentTypeError(f"Number of points must be positive, got {n}")
    if n == 1:
        return [start]
    points: list[int] = []
    for i in range(n):
        points.append(round(start + (stop - start) * i / (n - 1)))
    # Deduplicate while preserving order (can happen with small ranges)
    return list(dict.fromkeys(points))


def int_or_range(value: str) -> list[int]:
    """Parse an integer, range, or list specification. Endpoints always included.

    Formats:
        8            single value
        4:12         every integer from 4 to 12
        100:500:100  start:stop:step  -> [100, 200, 300, 400, 500]
        1:32::12     start:stop::N    -> 12 evenly-spaced points in [1, 32]
        1,4,8,16     explicit values
    """
    # Detect comma-separated values: "1,4,8,16"
    if "," in value:
        try:
            return [int(v.strip()) for v in value.split(",") if v.strip()]
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid integer in comma-separated list {value!r}: {e}"
            ) from e

    # Detect linspace syntax: "start:end::N"
    if "::" in value:
        halves = value.split("::")
        if len(halves) != 2 or ":" not in halves[0]:
            raise argparse.ArgumentTypeError(
                f"Linspace syntax must be start:end::N, got {value!r}"
            )
        try:
            left = halves[0].split(":")
            start, stop = int(left[0]), int(left[1])
            num_points = int(halves[1])
        except (ValueError, IndexError) as e:
            raise argparse.ArgumentTypeError(
                f"Invalid integer in linspace spec {value!r}: {e}"
            ) from e
        if start > stop:
            raise argparse.ArgumentTypeError(
                f"Range start ({start}) must be <= stop ({stop})"
            )
        return _linspace_int(start, stop, num_points)

    parts = value.split(":")
    try:
        if len(parts) == 1:
            return [int(parts[0])]
        elif len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            if start > stop:
                raise argparse.ArgumentTypeError(
                    f"Range start ({start}) must be <= stop ({stop})"
                )
            return list(range(start, stop + 1))
        elif len(parts) == 3:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
            if step <= 0:
                raise argparse.ArgumentTypeError(f"Step must be positive, got {step}")
            if start > stop:
                raise argparse.ArgumentTypeError(
                    f"Range start ({start}) must be <= stop ({stop})"
                )
            result = list(range(start, stop + 1, step))
            if result[-1] != stop:
                result.append(stop)
            return result
        else:
            raise argparse.ArgumentTypeError(f"Too many ':' in range spec: {value!r}")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid integer in range spec {value!r}: {e}"
        ) from e


def collect_sweep_params(
    workers: list[int],
    connections: list[int],
    prompt_lengths: list[int],
    stream_intervals: list[int] | None = None,
) -> list[tuple[str, list[int]]]:
    """Collect parameters that have ranges (more than one value)."""
    candidates = [
        ("num_workers", workers),
        ("max_connections", connections),
        ("prompt_length", prompt_lengths),
    ]
    if stream_intervals is not None:
        candidates.append(("stream_interval", stream_intervals))
    return [(name, vals) for name, vals in candidates if len(vals) > 1]


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float) -> float:
    """Division with zero-denominator guard."""
    return numerator / denominator if denominator > 0 else 0.0


def _stall_pct(stats: BenchmarkStats) -> float:
    """Client-side stall time (blocked on drain) as a percentage of send phase."""
    if stats.send_elapsed_ns <= 0:
        return 0.0
    return _safe_div(stats.stall_ns, stats.send_elapsed_ns) * 100


def _compute_derived_stats(
    stats: BenchmarkStats,
) -> tuple[float, float, float, float, float, int]:
    """Compute derived metrics from raw benchmark stats.

    Returns (send_elapsed_sec, total_elapsed_sec, send_rate, recv_rate,
             sse_rate, outstanding).
    """
    send_elapsed_sec = stats.send_elapsed_ns / 1e9
    total_elapsed_sec = stats.total_elapsed_ns / 1e9
    return (
        send_elapsed_sec,
        total_elapsed_sec,
        _safe_div(stats.sent, send_elapsed_sec),
        _safe_div(stats.received, total_elapsed_sec),
        _safe_div(stats.received * stats.sse_events_per_response, total_elapsed_sec),
        stats.sent - stats.received - stats.errors,
    )


# ---------------------------------------------------------------------------
# /proc memory helpers
# ---------------------------------------------------------------------------


def get_process_memory(pid: int) -> MemoryStats | None:
    """Get RSS and shared memory for a process from /proc."""
    try:
        with open(f"/proc/{pid}/statm") as f:
            parts = f.read().split()
            # statm fields: size resident shared text lib data dt (in pages)
            page_size = os.sysconf("SC_PAGE_SIZE")
            rss_pages = int(parts[1])
            shared_pages = int(parts[2])
            return MemoryStats(
                pid=pid,
                rss_mb=(rss_pages * page_size) / (1024 * 1024),
                shm_mb=(shared_pages * page_size) / (1024 * 1024),
            )
    except (OSError, IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Live display
# ---------------------------------------------------------------------------


class LiveDisplay:
    """Live statistics display with optional memory tracking."""

    def __init__(
        self,
        stats_ref: BenchmarkStats,
        track_memory: bool = False,
        streaming: bool = False,
        interval: float = 1.0,
    ):
        self.stats = stats_ref
        self.track_memory = track_memory
        self.streaming = streaming
        self.interval = interval
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sent = 0
        self._last_received = 0
        self._last_time = time.monotonic()
        self._main_pid = os.getpid()
        self._worker_pids: list[int] = []

    def set_worker_pids(self, pids: list[int]) -> None:
        """Set worker PIDs for memory tracking."""
        self._worker_pids = pids

    def start(self) -> None:
        """Start the display thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the display thread."""
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        """Display loop."""
        while not self._shutdown.wait(self.interval):
            self._print_stats()
        # Final stats (partial interval — print but don't record sample)
        self._print_stats(record_sample=False)

    def _print_stats(self, record_sample: bool = True) -> None:
        """Print current statistics and optionally record samples."""
        now = time.monotonic()
        elapsed = now - self._last_time

        sent = self.stats.sent
        received = self.stats.received
        errors = self.stats.errors
        in_flight = sent - received - errors

        send_rate = (sent - self._last_sent) / elapsed if elapsed > 0 else 0
        recv_rate = (received - self._last_received) / elapsed if elapsed > 0 else 0

        self._last_sent = sent
        self._last_received = received
        self._last_time = now

        # Record per-interval samples for variation analysis
        # (skip partial intervals, e.g. the final stop() call)
        if record_sample:
            if self.stats.send_rate_samples is None:
                self.stats.send_rate_samples = []
            if self.stats.recv_rate_samples is None:
                self.stats.recv_rate_samples = []
            self.stats.send_rate_samples.append(send_rate)
            self.stats.recv_rate_samples.append(recv_rate)

        # Build output line
        line = f"[Stats] Send/s: {send_rate:>9,.0f} | " f"Recv/s: {recv_rate:>9,.0f} | "
        if self.streaming:
            sse_rate = recv_rate * self.stats.sse_events_per_response
            line += f"SSE-pkts/s: {sse_rate:>9,.0f} | "
        line += (
            f"InFlight: {in_flight:>8,} | "
            f"Recv: {received:>10,} | "
            f"Err: {errors:>5,}"
        )

        if self.track_memory:
            mem_info = self._get_memory_info()
            line += f" | {mem_info}"

        print(line, flush=True)

    def _get_memory_info(self) -> str:
        """Get memory info string."""
        main_mem = get_process_memory(self._main_pid)
        if main_mem is None:
            return "Mem: N/A"

        total_rss = main_mem.rss_mb
        total_shm = main_mem.shm_mb

        # Get worker memory if available
        for pid in self._worker_pids:
            worker_mem = get_process_memory(pid)
            if worker_mem:
                total_rss += worker_mem.rss_mb
                total_shm += worker_mem.shm_mb

        return f"RSS: {total_rss:>7.1f}MB | SHM: {total_shm:>7.1f}MB"


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


_SEND_BATCH = 32  # Max requests issued per event-loop yield


def build_prompt(length: int) -> str:
    """Build a prompt of specified length using repeated 'hello world '."""
    base = "hello world "
    repeats = (length // len(base)) + 1
    return (base * repeats)[:length]


def _create_client(
    endpoint_url: str,
    num_workers: int,
    max_connections: int,
    streaming: bool,
    prompt: str,
    enable_affinity: bool,
    verbose: bool = True,
    zmq_context: ManagedZMQContext | None = None,
) -> tuple:
    """Create an endpoint client and query data dict.

    Returns (client, query_data). Caller must shut down the client.
    """
    cpu_affinity_plan = None
    if enable_affinity:
        effective = num_workers if num_workers > 0 else -1
        tmp = HTTPClientConfig(endpoint_urls=[endpoint_url], workers=effective)
        cpu_affinity_plan = compute_affinity_plan(tmp.workers)
        if cpu_affinity_plan.loadgen_cpus:
            os.sched_setaffinity(os.getpid(), set(cpu_affinity_plan.loadgen_cpus))  # type: ignore[attr-defined]
        if verbose:
            print(f"CPU Affinity Plan ({tmp.workers} workers):")
            for line in cpu_affinity_plan.summary().split("\n"):
                print(f"  {line}")
    elif verbose:
        print("CPU Affinity: disabled")

    config = HTTPClientConfig(
        endpoint_urls=[endpoint_url],
        workers=num_workers if num_workers > 0 else -1,
        max_connections=max_connections if max_connections > 0 else -1,
        warmup_connections=0,
        worker_gc_mode="relaxed",
        log_level="CRITICAL",
        cpu_affinity=cpu_affinity_plan,
    )

    if verbose:
        print(
            f"Config: workers={config.workers}, "
            f"max_connections={config.max_connections}, stream={streaming}"
        )

    client = HTTPEndpointClient(config)
    query_data = {
        "prompt": prompt,
        "model": "benchmark-model",
        "max_completion_tokens": 100,
        "stream": streaming,
    }

    return client, query_data


def run_benchmark(
    endpoint_url: str,
    duration: float,
    num_workers: int,
    max_connections: int,
    prompt: str,
    track_memory: bool,
    streaming: bool = False,
    max_concurrency: int = 100_000,
    send_batch: int = _SEND_BATCH,
    enable_affinity: bool = False,
    sse_events_per_response: int = 1,
    max_total_time: float = 15.0,
) -> BenchmarkStats:
    """
    Run the HTTP client benchmark.

    Args:
        endpoint_url: Target endpoint URL
        duration: Benchmark duration in seconds
        num_workers: Number of worker processes (-1 for auto)
        max_connections: Max TCP connections (-1 for auto)
        prompt: Prompt string to send
        track_memory: Whether to track memory usage
        streaming: Whether to use streaming mode
        max_concurrency: Maximum in-flight requests for back-pressure
        send_batch: Max requests issued per event-loop yield; auto-tuned at startup
        sse_events_per_response: Number of SSE events per streaming response (for rate derivation)
        max_total_time: Hard cap on total run time (send + drain) in seconds

    Returns:
        Final benchmark statistics
    """
    # Save original affinity before _create_client pins us to loadgen CPUs.
    # Without this, subsequent sweep iterations see only the restricted set
    # and compute_affinity_plan detects fewer physical cores than exist.
    saved_affinity: set[int] | None = None
    if enable_affinity:
        try:
            saved_affinity = os.sched_getaffinity(os.getpid())  # type: ignore[attr-defined]
        except OSError:
            pass

    zmq_ctx_manager = ManagedZMQContext.scoped()
    zmq_ctx = zmq_ctx_manager.__enter__()

    client, query_data = _create_client(
        endpoint_url,
        num_workers,
        max_connections,
        streaming,
        prompt,
        enable_affinity,
        zmq_context=zmq_ctx,
    )
    loop = client.loop
    stats = BenchmarkStats(sse_events_per_response=sse_events_per_response)
    display = LiveDisplay(stats, track_memory=track_memory, streaming=streaming)

    if track_memory:
        worker_pids = list(client.worker_manager.worker_pids.values())
        display.set_worker_pids(worker_pids)

    display.start()

    # Suppress GC during measurement to avoid collection pauses
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()

    async def benchmark_main():
        """Main benchmark coroutine running on event loop."""
        nonlocal stats

        send_done = False
        receiver_done = asyncio.Event()
        start_ns = time.monotonic_ns()
        send_deadline = time.monotonic() + duration
        overall_deadline = time.monotonic() + max_total_time
        stall_timeout = 5.0
        last_recv_time = time.monotonic()

        qid = 0

        def _process_result(result):
            nonlocal last_recv_time
            last_recv_time = time.monotonic()
            if isinstance(result, QueryResult):
                if result.error:
                    stats.errors += 1
                else:
                    stats.received += 1

        async def sender():
            nonlocal qid, send_done
            while time.monotonic() < send_deadline and not receiver_done.is_set():
                # Back-pressure: wait if too many in-flight
                in_flight = stats.sent - stats.received - stats.errors
                if in_flight > stats.peak_inflight:
                    stats.peak_inflight = in_flight
                if in_flight >= max_concurrency:
                    stall_start = time.monotonic_ns()
                    await asyncio.sleep(0.0001)
                    stats.stall_ns += time.monotonic_ns() - stall_start
                    continue

                # Burst up to send_batch requests before yielding
                for _ in range(min(send_batch, max_concurrency - in_flight)):
                    query = Query(id=str(qid), data=query_data)
                    client.issue(query)
                    stats.sent += 1
                    qid += 1

                # Yield to let receiver process
                await asyncio.sleep(0)

            stats.send_elapsed_ns = time.monotonic_ns() - start_ns
            send_done = True

        async def receiver():
            nonlocal last_recv_time
            try:
                while True:
                    # Fast drain: poll all available results synchronously
                    result = client.poll()
                    if result is not None:
                        while result is not None:
                            _process_result(result)
                            result = client.poll()
                        # Deadline check once per drain batch
                        if last_recv_time > overall_deadline:
                            outstanding = stats.sent - stats.received - stats.errors
                            print(
                                f"\nTime limit ({max_total_time:.0f}s) reached, "
                                f"stopping ({outstanding:,} in-flight)"
                            )
                            return
                        continue

                    # Nothing queued — check termination before blocking
                    if send_done and (stats.received + stats.errors) >= stats.sent:
                        break
                    if (
                        send_done
                        and (time.monotonic() - last_recv_time) > stall_timeout
                    ):
                        print(f"\nStalled for {stall_timeout}s, stopping")
                        break
                    if time.monotonic() > overall_deadline:
                        outstanding = stats.sent - stats.received - stats.errors
                        print(
                            f"\nTime limit ({max_total_time:.0f}s) reached, "
                            f"stopping ({outstanding:,} in-flight)"
                        )
                        break

                    # Block until next response (event-driven, no CPU spin)
                    result = await client.recv()
                    if result is None:
                        break
                    _process_result(result)
            finally:
                receiver_done.set()

        # Run sender and receiver concurrently
        await asyncio.gather(sender(), receiver())

        stats.total_elapsed_ns = time.monotonic_ns() - start_ns

    try:
        future = asyncio.run_coroutine_threadsafe(benchmark_main(), loop)
        future.result()  # Block until complete
    except KeyboardInterrupt:
        print("\nInterrupted...")

    display.stop()

    # Restore GC state after measurement
    if gc_was_enabled:
        gc.enable()
    gc.collect()

    client.shutdown()
    zmq_ctx_manager.__exit__(None, None, None)

    # Restore original affinity so the next sweep iteration sees all CPUs
    if saved_affinity is not None:
        try:
            os.sched_setaffinity(os.getpid(), saved_affinity)  # type: ignore[attr-defined]
        except OSError:
            pass

    return stats


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


_FULL_WORKERS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
_FULL_PROMPT_LENGTHS = [
    1,
    32,
    128,
    512,
    1024,
    8192,
    16384,
    32768,
    65536,
    131072,
]

_FULL_STREAM_INTERVAL_PCTS = [0.0, 0.5, 1.0]


def _resolve_stream_intervals(output_length: int, fractions: list[float]) -> list[int]:
    """Map fractions to absolute stream_interval (chars-per-event) values, deduped and sorted.

    0.0 -> 1 (1 char/event, finest); 0.5 -> output_length//2; 1.0 -> output_length (1 event total).
    """
    vals: list[int] = []
    for f in fractions:
        if f <= 0.0:
            vals.append(1)
        else:
            vals.append(max(1, round(output_length * f)))
    return sorted(set(vals))


def _sse_events_per_response(output_length: int, stream_interval: int = 1) -> int:
    """Compute the number of SSE events the server sends per streaming response.

    ``ceil(output_length / stream_interval)`` content events, plus 1 finish
    event and 1 [DONE] sentinel.
    """
    return -(-output_length // stream_interval) + 2  # ceil division + 2


def _wait_for_port_free(host: str, port: int, timeout: float = 5.0) -> None:
    """Wait until *port* is bindable on *host* (old workers may linger in kernel)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass  # SO_REUSEPORT unavailable — _start_workers will report
            try:
                s.bind((host, port))
                return  # port is free
            except OSError:
                time.sleep(0.1)
    # Don't raise — proceed anyway and let _start_workers report the real error


def _restart_server(
    server: MaxThroughputServer,
    output_length: int,
    stream: bool,
    stream_interval: int = 1,
) -> None:
    """Stop, reconfigure response payload, and restart workers."""
    server.stop()
    # Wait for port to become available (old workers may linger in kernel)
    if server._port is not None:
        _wait_for_port_free(server.host, server._port)
    server.stream = stream
    server._response = build_response(output_length, stream, stream_interval)
    # Reset shutdown flags and worker lists so _start_workers can re-use them
    server._shutdown.clear()
    server._workers.clear()
    server._ready_events.clear()
    server._start_workers()


# ---------------------------------------------------------------------------
# Single-run mode
# ---------------------------------------------------------------------------


def run_single(
    args: argparse.Namespace,
    endpoint_url: str,
    server: MaxThroughputServer | None = None,
) -> None:
    """Run a single benchmark (original behavior)."""
    mode = "streaming" if args.streaming else "non-streaming"
    prompt_length = args.prompt_length[0]
    prompt = build_prompt(prompt_length)
    stream_interval = args.stream_interval[0]
    mode_info = f"Mode: {mode}  |  Prompt length: {len(prompt)} chars"
    if args.streaming and stream_interval > 1:
        mode_info += f"  |  Stream interval: {stream_interval}"
    print(mode_info)

    if server:
        _restart_server(server, prompt_length, args.streaming, stream_interval)

    epr = (
        _sse_events_per_response(prompt_length, stream_interval)
        if args.streaming
        else 1
    )

    print(f"\nStarting benchmark for {args.duration}s...")
    print("=" * 70)

    stats = run_benchmark(
        endpoint_url=endpoint_url,
        duration=args.duration,
        num_workers=args.workers[0],
        max_connections=args.max_connections[0],
        prompt=prompt,
        track_memory=args.track_memory,
        streaming=args.streaming,
        max_concurrency=args.max_concurrency,
        send_batch=args.send_batch,
        enable_affinity=args.pin,
        sse_events_per_response=epr,
        max_total_time=args.duration + 10,
    )

    send_elapsed, total_elapsed, send_rate, recv_rate, sse_rate, outstanding = (
        _compute_derived_stats(stats)
    )
    print("=" * 70)
    print("\nFinal Results:")
    print(f"  Send Duration:  {send_elapsed:.2f}s")
    print(f"  Total Duration: {total_elapsed:.2f}s")
    print(f"  Total Sent:     {stats.sent:,}")
    print(f"  Total Recv:     {stats.received:,}")
    print(f"  Errors:         {stats.errors:,}")
    print(f"  Outstanding:    {outstanding:,}")
    print(f"  Send Rate:      {send_rate:,.0f} req/s")
    print(f"  Recv Rate:      {recv_rate:,.0f} resp/s")
    if args.streaming:
        print(f"  SSE-pkts/s:     {sse_rate:,.0f}")
    print(f"  Peak InFlight:  {stats.peak_inflight:,} / {args.max_concurrency:,}")
    print(f"  Stall%:         {_stall_pct(stats):.1f}%")
    if outstanding > 0:
        print(f"  WARNING: {outstanding:,} queries did not complete")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------


def _make_sweep_result(pv: dict[str, int], stats: BenchmarkStats) -> SweepResult:
    """Build a SweepResult from param values and raw benchmark stats."""
    _, _, send_rate, recv_rate, sse_rate, outstanding = _compute_derived_stats(stats)
    sr = stats.send_rate_samples or []
    rr = stats.recv_rate_samples or []
    epr = stats.sse_events_per_response
    return SweepResult(
        param_values=pv,
        stats=stats,
        send_rate=send_rate,
        recv_rate=recv_rate,
        sse_rate=sse_rate,
        outstanding=outstanding,
        error_rate=_safe_div(stats.errors, stats.sent) * 100,
        stall_pct=_stall_pct(stats),
        send_rate_min=min(sr) if sr else send_rate,
        send_rate_max=max(sr) if sr else send_rate,
        recv_rate_min=min(rr) if rr else recv_rate,
        recv_rate_max=max(rr) if rr else recv_rate,
        sse_rate_min=min(rr) * epr if rr else sse_rate,
        sse_rate_max=max(rr) * epr if rr else sse_rate,
    )


def run_sweep(
    args: argparse.Namespace,
    sweeps: list[tuple[str, list[int]]],
    endpoint_url: str,
    server: MaxThroughputServer | None = None,
) -> None:
    """Run a parameter sweep (cartesian product), print summary, and plot.

    When ``args._stream_interval_pcts`` is set, stream_interval is resolved
    as a *dependent* inner loop per prompt_length rather than a static
    cartesian dimension. This avoids redundant runs for small prompt_lengths
    where the fractional intervals collapse (e.g. prompt_length=1 → [1]).
    """
    si_pcts: list[float] | None = getattr(args, "_stream_interval_pcts", None)

    sweep_names = [s[0] for s in sweeps]
    sweep_values = [s[1] for s in sweeps]
    combinations = list(itertools.product(*sweep_values))

    default_workers = args.workers[0]
    default_connections = args.max_connections[0]
    default_prompt_length = args.prompt_length[0]
    default_stream_interval = args.stream_interval[0]

    # Build effective sweep info (includes stream_interval when pct-based)
    if si_pcts is not None:
        effective_sweep_names = [*sweep_names, "stream_interval"]
        all_si: set[int] = set()
        for combo in combinations:
            pv = dict(zip(sweep_names, combo, strict=False))
            pl: int = pv.get("prompt_length", default_prompt_length)  # type: ignore[assignment]
            all_si.update(_resolve_stream_intervals(pl, si_pcts))
        effective_sweeps: list[tuple[str, list[int]]] = [
            *sweeps,
            ("stream_interval", sorted(all_si)),
        ]
        total_iterations = sum(
            len(
                _resolve_stream_intervals(
                    dict(zip(sweep_names, c, strict=False)).get(
                        "prompt_length",
                        default_prompt_length,  # type: ignore[arg-type]
                    ),
                    si_pcts,
                )
            )
            for c in combinations
        )
    else:
        effective_sweep_names = sweep_names
        effective_sweeps = list(sweeps)
        total_iterations = len(combinations)

    mode = "streaming" if args.streaming else "non-streaming"
    print(f"Mode: {mode}")
    for name, vals in effective_sweeps:
        suffix = ""
        if name == "stream_interval" and si_pcts is not None:
            pct_strs = ", ".join(f"{p:.0%}" for p in si_pcts)
            suffix = f"  (auto: {pct_strs} of prompt_length)"
        print(f"  {name} = {vals}{suffix}")
    print(f"Total: {total_iterations} iterations of {args.duration}s each\n")

    results: list[SweepResult] = []
    last_prompt_length: int | None = None
    last_stream_interval: int | None = None
    iteration = 0

    for combo in combinations:
        pv_base = dict(zip(sweep_names, combo, strict=False))

        workers: int = pv_base.get("num_workers", default_workers)  # type: ignore[assignment]
        connections: int = pv_base.get("max_connections", default_connections)  # type: ignore[assignment]
        prompt_length: int = pv_base.get("prompt_length", default_prompt_length)  # type: ignore[assignment]

        # Resolve stream_interval values for this prompt_length
        if si_pcts is not None:
            si_values = _resolve_stream_intervals(prompt_length, si_pcts)
        else:
            si_val: int = pv_base.get("stream_interval", default_stream_interval)  # type: ignore[assignment]
            si_values = [si_val]

        for stream_interval in si_values:
            iteration += 1
            pv = {**pv_base}
            if si_pcts is not None:
                pv["stream_interval"] = stream_interval

            label = ", ".join(f"{k}={v}" for k, v in pv.items())
            print(f"\n{'='*70}")
            print(f"  Sweep {iteration}/{total_iterations}: {label}")
            print(f"{'='*70}")

            # Restart server when prompt_length or stream_interval changes
            if server and (
                prompt_length != last_prompt_length
                or stream_interval != last_stream_interval
            ):
                _restart_server(server, prompt_length, args.streaming, stream_interval)
                last_prompt_length = prompt_length
                last_stream_interval = stream_interval

            prompt = build_prompt(prompt_length)
            epr = (
                _sse_events_per_response(prompt_length, stream_interval)
                if args.streaming
                else 1
            )

            stats = run_benchmark(
                endpoint_url=endpoint_url,
                duration=args.duration,
                num_workers=workers,
                max_connections=connections,
                prompt=prompt,
                track_memory=args.track_memory,
                streaming=args.streaming,
                max_concurrency=args.max_concurrency,
                send_batch=args.send_batch,
                enable_affinity=args.pin,
                sse_events_per_response=epr,
                max_total_time=args.duration + 10,
            )

            results.append(_make_sweep_result(pv, stats))

    print_sweep_summary(effective_sweep_names, results, streaming=args.streaming)
    generate_sweep_plot(effective_sweeps, results, args)


def print_sweep_summary(
    sweep_names: list[str],
    results: list[SweepResult],
    streaming: bool = False,
) -> None:
    """Print a formatted summary table of sweep results."""
    # Build dynamic param columns
    param_headers = [f"{n:>14}" for n in sweep_names]
    param_sep = " | ".join(param_headers)
    sse_hdr = f" | {'SSE-pkts/s':>12}" if streaming else ""
    header = (
        f"{param_sep} | {'Send Rate':>12} | {'Recv Rate':>12}{sse_hdr} | "
        f"{'Outstanding':>11} | {'Stall%':>7} | {'Errors':>8}"
    )
    width = len(header)
    print(f"\n{'='*width}")
    print(f"Sweep Summary: {', '.join(sweep_names)}")
    print(f"{'='*width}")
    print(header)
    print(f"{'-'*width}")
    for r in results:
        param_cols = " | ".join(f"{r.param_values[n]:>14,}" for n in sweep_names)
        sse_col = f" | {r.sse_rate:>12,.0f}" if streaming else ""
        print(
            f"{param_cols} | {r.send_rate:>12,.0f} | "
            f"{r.recv_rate:>12,.0f}{sse_col} | "
            f"{r.outstanding:>11,} | {r.stall_pct:>6.1f}% | {r.error_rate:>7.1f}%"
        )
    print(f"{'='*width}")


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _plot_config(args: argparse.Namespace) -> list[str]:
    """Return list of 'key=val' strings for non-swept run parameters."""
    si_pcts = getattr(args, "_stream_interval_pcts", None)
    return [
        f"duration={args.duration}",
        f"max_concurrency={args.max_concurrency}",
        *(["streaming=True"] if args.streaming else []),
        *(
            [f"stream_interval={args.stream_interval[0]}"]
            if args.streaming and si_pcts is None and args.stream_interval[0] > 1
            else []
        ),
        *(["pin=False"] if not args.pin else []),
    ]


def _fmt_si(v: float) -> str:
    """Format a value with SI suffixes for annotations."""
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"{v / 1e9:.2f}B" if v % 1e8 else f"{v / 1e9:.1f}B"
    if abs_v >= 1e6:
        return f"{v / 1e6:.2f}M" if v % 1e5 else f"{v / 1e6:.1f}M"
    if abs_v >= 1e3:
        return f"{v / 1e3:.1f}K" if v % 1e3 else f"{v / 1e3:.0f}K"
    return f"{v:,.0f}"


def _annotate_peak(ax: object, x: list[int], y: list[float], color: str) -> None:
    """Add a subtle peak label to an axes object."""
    if not y:
        return
    peak_idx = y.index(max(y))
    # Shift label left if peak is at rightmost point to avoid clipping
    is_rightmost = peak_idx == len(x) - 1
    ha = "right" if is_rightmost else "center"
    x_offset = -6 if is_rightmost else 0
    ax.annotate(  # type: ignore[attr-defined]
        _fmt_si(y[peak_idx]),
        xy=(x[peak_idx], y[peak_idx]),
        xytext=(x_offset, 8),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


def generate_sweep_plot(
    sweeps: list[tuple[str, list[int]]],
    results: list[SweepResult],
    args: argparse.Namespace,
) -> None:
    """Generate sweep plots (Send Rate + Recv Rate, plus SSE Rate when streaming).

    Layout adapts to the number of swept parameters:
      1 param:  1xN — single line per subplot, with peak annotation.
      2 params: 1xN — x-axis=param1, colored lines per param2 value.
      3 params: MxN facet grid — one row per param3 value, x-axis=param1,
                colored lines per param2 value.
      4 params: MxNxK facet grid — rows=param3, columns=param4.
    Where N = 2 (non-streaming) or 3 (streaming, adds SSE Rate).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("\nMatplotlib not installed. Skipping plot generation.")
        print("  Install with: pip install matplotlib")
        return

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass  # Fall back to default style

    si_fmt = ticker.FuncFormatter(lambda v, _pos: _fmt_si(v))
    marker_kw: dict[str, object] = {
        "marker": "o",
        "markersize": 5,
        "linewidth": 2,
        "zorder": 3,
    }

    # 1 param: x-axis only. 2 params: x + colored lines.
    # 3 params: x + colored lines + facet rows.
    # 4 params: x + colored lines + facet rows + facet columns.
    # Priority: x prefers workers, lines prefer prompt_length.
    swept = dict(sweeps)
    x_pref = ["num_workers", "max_connections", "prompt_length", "stream_interval"]
    line_pref = ["prompt_length", "stream_interval", "max_connections", "num_workers"]

    x_param = next((p for p in x_pref if p in swept), sweeps[0][0])
    remaining = [(n, v) for n, v in sweeps if n != x_param]

    line_param: str | None = None
    line_values: list[int] = []
    facet_param: str | None = None
    facet_values: list[int] = []
    facet_col_param: str | None = None
    facet_col_values: list[int] = []

    if remaining:
        line_param = next(
            (p for p in line_pref if p in dict(remaining)), remaining[0][0]
        )
        line_values = dict(remaining)[line_param]
        remaining = [(n, v) for n, v in remaining if n != line_param]
    if remaining:
        facet_param, facet_values = remaining[0]
        remaining = [(n, v) for n, v in remaining if n != facet_param]
    if remaining:
        facet_col_param, facet_col_values = remaining[0]

    xlabel = x_param.replace("_", " ").title()

    nrows = len(facet_values) if facet_param else 1
    ncols_facet = len(facet_col_values) if facet_col_param else 1
    metrics_per_cell = 3 if args.streaming else 2
    fig, axes = plt.subplots(
        nrows,
        ncols_facet * metrics_per_cell,
        figsize=(
            max(7 * metrics_per_cell, 7 * ncols_facet * metrics_per_cell // 2),
            max(5 * nrows + 1.5, 6),
        ),
        squeeze=False,
    )

    config = _plot_config(args)
    sweep_desc = ", ".join(f"{name}=[{vals[0]}..{vals[-1]}]" for name, vals in sweeps)
    subtitle = f"{sweep_desc}  |  {', '.join(config)}"

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    has_stall = any(r.stall_pct > 1 for r in results)

    # Use gradient colormap when line param has many values (>6)
    use_cmap = line_param is not None and len(line_values) > 6
    cmap = None
    norm = None
    if use_cmap:
        cmap = plt.get_cmap("viridis")
        lo, hi = max(min(line_values), 1), max(line_values)
        norm = (
            mcolors.LogNorm(vmin=lo, vmax=hi)
            if hi / lo > 10
            else mcolors.Normalize(vmin=lo, vmax=hi)
        )

    def _line_color(li: int, lv: int | None) -> str:
        if use_cmap and lv is not None:
            return cmap(norm(max(lv, 1)))  # type: ignore[misc]
        return cycle_colors[li % len(cycle_colors)]

    # Index results: (facet_row_val, facet_col_val, line_val, x_val) -> result
    result_idx: dict[tuple[int | None, int | None, int | None, int], SweepResult] = {}
    for r in results:
        fv = r.param_values.get(facet_param) if facet_param else None
        fcv = r.param_values.get(facet_col_param) if facet_col_param else None
        lv = r.param_values.get(line_param) if line_param else None
        result_idx[(fv, fcv, lv, r.param_values[x_param])] = r

    x_values = swept[x_param]
    facet_iter = facet_values if facet_param else [None]  # type: ignore
    facet_col_iter = facet_col_values if facet_col_param else [None]  # type: ignore
    mean_maxes: dict[int, float] = {}  # id(ax) -> max mean value

    for row, fv in enumerate(facet_iter):
        for col, fcv in enumerate(facet_col_iter):
            base = col * metrics_per_cell
            ax_send = axes[row, base]
            ax_recv = axes[row, base + 1]
            ax_sse = axes[row, base + 2] if args.streaming else None

            if facet_param and col == 0:
                facet_label = facet_param.replace("_", " ").title()
                ax_send.set_ylabel(f"{facet_label}={fv}\nreq/s", fontsize=10)
            elif col == 0:
                ax_send.set_ylabel("req/s", fontsize=10)
            ax_recv.set_ylabel("resp/s" if col == 0 else "", fontsize=10)
            if ax_sse is not None:
                ax_sse.set_ylabel("SSE-pkts/s" if col == 0 else "", fontsize=10)

            ax_stall: object | None = None
            if has_stall:
                ax_stall = ax_send.twinx()
                ax_stall.set_ylim(0, 100)  # type: ignore[union-attr]
                ax_stall.set_ylabel("stall %", fontsize=8, color="#cc4444", alpha=0.6)  # type: ignore[union-attr]
                ax_stall.tick_params(  # type: ignore[union-attr]
                    axis="y",
                    labelcolor="#cc4444",
                    labelsize=7,
                    length=3,
                )
                ax_stall.yaxis.set_major_locator(ticker.MaxNLocator(5))  # type: ignore[union-attr]

            line_iter = line_values if line_param else [None]  # type: ignore

            for li, lv in enumerate(line_iter):
                color = _line_color(li, lv)
                group = [
                    result_idx[(fv, fcv, lv, xv)]
                    for xv in x_values
                    if (fv, fcv, lv, xv) in result_idx
                ]
                if not group:
                    continue

                x = [r.param_values[x_param] for r in group]
                label = f"{line_param}={lv}" if (line_param and not use_cmap) else None

                n_lines = len(line_iter)
                band_alpha = 0.15 if n_lines <= 3 else max(0.03, 0.15 / (n_lines / 3))
                metric_axes = [
                    (ax_send, "send_rate", "send_rate_min", "send_rate_max"),
                    (ax_recv, "recv_rate", "recv_rate_min", "recv_rate_max"),
                ]
                if ax_sse is not None:
                    metric_axes.append(
                        (ax_sse, "sse_rate", "sse_rate_min", "sse_rate_max")
                    )
                for ax, attr, attr_min, attr_max in metric_axes:
                    y = [getattr(r, attr) for r in group]
                    y_lo = [getattr(r, attr_min) for r in group]
                    y_hi = [getattr(r, attr_max) for r in group]
                    ax.plot(x, y, color=color, label=label, **marker_kw)
                    ax.fill_between(
                        x,
                        y_lo,
                        y_hi,
                        color=color,
                        alpha=band_alpha,
                        linewidth=0,
                        zorder=2,
                    )
                    ax_id = id(ax)
                    mean_maxes[ax_id] = max(mean_maxes.get(ax_id, 0), max(y))
                    if len(line_iter) == 1:
                        _annotate_peak(ax, x, y, color)

                if ax_stall is not None:
                    stall_pcts = [r.stall_pct for r in group]
                    ax_stall.plot(  # type: ignore[attr-defined]
                        x,
                        stall_pcts,
                        color="#cc4444",
                        alpha=0.5,
                        linewidth=1.2,
                        linestyle="--",
                        zorder=1,
                    )
                    ax_stall.fill_between(  # type: ignore[attr-defined]
                        x,
                        0,
                        stall_pcts,
                        color="#cc4444",
                        alpha=0.03,
                        linewidth=0,
                        zorder=0,
                    )

            # Format all metric axes in this cell
            titles = [("Send Rate", ax_send), ("Recv Rate", ax_recv)]
            if ax_sse is not None:
                titles.append(("SSE Rate", ax_sse))
            for title, ax in titles:
                if row == 0:
                    col_prefix = (
                        f"{facet_col_param.replace('_', ' ').title()}={fcv}\n"
                        if facet_col_param
                        else ""
                    )
                    ax.set_title(f"{col_prefix}{title}", fontsize=12, pad=8)
                ax.set_xlabel(xlabel, fontsize=10)
                max_mean = mean_maxes.get(id(ax), 0)
                if max_mean > 0:
                    ax.set_ylim(bottom=0, top=max_mean * 1.18)
                else:
                    ax.set_ylim(bottom=0)
                ax.yaxis.set_major_formatter(si_fmt)
                ax.grid(True, alpha=0.15, linewidth=0.5)
                ax.tick_params(labelsize=9)
                if line_param and not use_cmap and col == 0:
                    ax.legend(fontsize=8, loc="best", framealpha=0.8)

    # Explicit margins; reserve right edge for colorbar when present
    left = 0.07
    right = 0.87 if use_cmap else 0.95
    fig.subplots_adjust(
        top=0.86,
        bottom=0.11,
        left=left,
        right=right,
        wspace=0.35,
    )

    # Center title/subtitle over the full figure
    fig.suptitle(
        "HTTP Client Benchmark Sweep",
        fontsize=14,
        fontweight="bold",
        x=0.5,
        y=0.97,
    )
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=9, color="0.4")

    if use_cmap:
        assert line_param is not None
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_label = line_param.replace("_", " ").title()
        cbar = fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            pad=0.03,
            aspect=30,
            shrink=0.9,
        )
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    # Build filename
    name_parts = [f"{name}_{vals[0]}-{vals[-1]}" for name, vals in sweeps]
    filename = f"/tmp/sweep_{'_x_'.join(name_parts)}_{'_'.join(config)}.png"

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {filename}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP client performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Allow values like "-1,2048" to be parsed as arg values, not flags.
    parser._negative_number_matcher = re.compile(r"^-\d")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Endpoint URL. If not provided, launches MaxThroughputServer at localhost:12345",
    )
    parser.add_argument(
        "--no-pin",
        action="store_false",
        dest="pin",
        help="Disable CPU affinity pinning for workers (enabled by default)",
    )
    parser.add_argument(
        "-l",
        "--prompt-length",
        "--prompt-len",
        type=int_or_range,
        default=[-1],
        dest="prompt_length",
        help="Prompt length in characters, or range (e.g. 500:2000:500). Default: 1000",
    )
    parser.add_argument(
        "-c",
        "--max-connections",
        type=int_or_range,
        default=[-1],
        help="Max TCP connections, or range (e.g. 100:500:100). -1 for auto. Default: -1",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int_or_range,
        default=[-1],
        help="Number of worker processes, or range (e.g. 4:12). -1 for auto. Default: -1",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=5.0,
        help="Benchmark duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--track-memory",
        action="store_true",
        help="Track memory/SHM usage (adds overhead)",
    )
    parser.add_argument(
        "--server-workers",
        type=int,
        default=4,
        help="Number of server workers if auto-launching (default: 4)",
    )
    parser.add_argument(
        "--stream",
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Use streaming mode for queries (default: non-streaming)",
    )
    parser.add_argument(
        "--stream-interval",
        type=int_or_range,
        default=[1],
        dest="stream_interval",
        help="Characters per SSE event (supports ranges). "
        "Total events = ceil(output_length / stream_interval). Default: 1",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100_000,
        dest="max_concurrency",
        help="Maximum in-flight requests for back-pressure (default: 100000)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Preset sweep of common worker counts x prompt lengths",
    )
    args = parser.parse_args()

    args._stream_interval_pcts = None

    if args.full:
        if args.workers == [-1]:
            args.workers = _FULL_WORKERS
        if args.prompt_length == [-1]:
            args.prompt_length = _FULL_PROMPT_LENGTHS
        if args.streaming and args.stream_interval == [1]:
            args._stream_interval_pcts = _FULL_STREAM_INTERVAL_PCTS

    if args.prompt_length == [-1]:
        args.prompt_length = [1000]

    sweeps = collect_sweep_params(
        args.workers,
        args.max_connections,
        args.prompt_length,
        stream_intervals=(
            args.stream_interval
            if args.streaming and args._stream_interval_pcts is None
            else None
        ),
    )

    gc.set_threshold(70000, 10, 100)

    import uvloop

    uvloop.install()

    server: MaxThroughputServer | None = None
    if args.endpoint:
        endpoint_url = args.endpoint
        print(f"Using external endpoint: {endpoint_url}")
    else:
        server = MaxThroughputServer(
            host="127.0.0.1",
            port=12345,
            num_workers=args.server_workers,
        )
        server.start()
        endpoint_url = f"{server.url}/v1/chat/completions"
        print(f"Launched MaxThroughputServer: {endpoint_url}")

    def sig_handler(signum, frame):
        print("\nReceived signal, shutting down...")
        if server:
            server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    args.send_batch = _SEND_BATCH

    try:
        if sweeps:
            run_sweep(args, sweeps, endpoint_url, server)
        else:
            run_single(args, endpoint_url, server)
    finally:
        if server:
            server.stop()
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
