# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible LLM API server with variable response lengths and rates.

Unlike :class:`MaxThroughputServer` (which returns identical pre-compiled responses
instantly for roofline testing), this server models realistic LLM inference:

* **Variable output lengths** — lognormal distribution, configurable mean + spread
* **Variable response rates** — lognormal distribution (responses/sec per request)
* **Per-chunk latency jitter** — lognormal inter-chunk delays in streaming mode

This makes it suitable for benchmarking client behaviour under realistic workloads
(request queuing, backpressure, tail-latency handling, etc.).

Usage::

    # Non-streaming with variable lengths
    python -m inference_endpoint.testing.variable_response_server --stats

    # Streaming with chunk jitter
    python -m inference_endpoint.testing.variable_response_server --stream --stats \\
        --output-len-mean 256 --response-rate-mean 50 --response-chunk-spread 0.2

    # Python — as a context manager
    with VariableResponseServer(port=0, num_workers=2) as srv:
        url = f"{srv.url}/v1/chat/completions"
        # ... send requests ...
"""

from __future__ import annotations

import argparse
import asyncio
import math
import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import os
import random
import signal
import socket
import threading
import time

import httptools
import uvloop

from .max_throughput_server import (
    RequestParser,
    _chunked,
    _counter_add,
    _counter_read,
    _sse_event,
    build_non_streaming_response,
)

# Shared counters (multiprocessing.Value for cross-process atomicity)
type _SyncInt = multiprocessing.sharedctypes.Synchronized[int]  # type: ignore[valid-type]

_req_counter: _SyncInt | None = None
_resp_counter: _SyncInt | None = None
_byte_counter: _SyncInt | None = None


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Convert mean + coefficient of variation to lognormal (mu, sigma).

    Args:
        mean: Desired mean of the distribution.
        cv: Coefficient of variation (sigma / mean).  0 = deterministic.

    Returns:
        (mu, sigma) parameters for ``random.lognormvariate``.
    """
    if cv <= 0:
        # Degenerate: always return mean (sigma=0 makes lognormvariate = e^mu)
        return math.log(mean), 0.0
    sigma2 = math.log(1.0 + cv * cv)
    mu = math.log(mean) - sigma2 / 2.0
    return mu, math.sqrt(sigma2)


class _TokenBucket:
    """Global rate limiter: enforces N responses/sec across all concurrent requests."""

    __slots__ = ("_interval", "_available_at")

    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._available_at = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        start = max(now, self._available_at)
        self._available_at = start + self._interval
        if (wait := start - now) > 0:
            await asyncio.sleep(wait)


class VariableResponseProtocol(asyncio.Protocol):
    """asyncio Protocol that samples output length + response time per request.

    Non-streaming: sleeps for ``1/rate`` seconds, then writes the response.
    Streaming: spreads chunks over ``1/rate`` seconds with optional jitter.

    Because ``data_received`` is synchronous, async work is dispatched via
    ``loop.create_task``.
    """

    __slots__ = (
        "transport",
        "_parser",
        "_request",
        "_loop",
        "_rng",
        "_bucket",
        "_stream",
        "_stream_interval",
        "_osl_mu",
        "_osl_sigma",
        "_osl_min",
        "_osl_max",
        "_inter_chunk_latency",
        "_chunk_cv",
        "_model",
    )

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        rng: random.Random,
        bucket: _TokenBucket,
        *,
        stream: bool,
        stream_interval: int,
        osl_mu: float,
        osl_sigma: float,
        osl_min: int,
        osl_max: int,
        inter_chunk_latency: float,
        chunk_cv: float,
    ):
        self.transport = None
        self._parser: httptools.HttpRequestParser | None = None
        self._request: RequestParser | None = None
        self._loop = loop
        self._rng = rng
        self._bucket = bucket
        self._stream = stream
        self._stream_interval = stream_interval
        self._osl_mu = osl_mu
        self._osl_sigma = osl_sigma
        self._osl_min = osl_min
        self._osl_max = osl_max
        self._inter_chunk_latency = inter_chunk_latency
        self._chunk_cv = chunk_cv
        self._model = "variable-resp"

    def connection_made(self, transport):
        self.transport = transport
        sock = transport.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._request = RequestParser()
        self._parser = httptools.HttpRequestParser(self._request)

    def data_received(self, data):
        parser = self._parser
        request = self._request
        if parser is None or request is None:
            return

        try:
            parser.feed_data(data)
        except httptools.HttpParserError:
            if self.transport:
                self.transport.close()
            return

        if request.done():
            _counter_add(_req_counter)

            # Sample per-request output length
            rng = self._rng
            osl = int(rng.lognormvariate(self._osl_mu, self._osl_sigma))
            osl = max(self._osl_min, min(osl, self._osl_max))

            # Dispatch async handler (rate governed by shared bucket)
            self._loop.create_task(self._handle_request(osl))

            # Reset for next request (HTTP/1.1 keep-alive)
            self._request = RequestParser()
            self._parser = httptools.HttpRequestParser(self._request)

    async def _handle_request(self, osl: int):
        """Handle a single request with rate-limited output."""
        transport = self.transport
        if transport is None or transport.is_closing():
            return

        try:
            if self._stream:
                await self._handle_streaming(transport, osl)
            else:
                await self._handle_non_streaming(transport, osl)
        except (RuntimeError, ConnectionError, OSError):
            # Client disconnected — transport closed mid-write. Ignore.
            return

    async def _handle_non_streaming(self, transport, osl: int):
        """Wait for rate-limit slot, then send complete response."""
        await self._bucket.acquire()

        if transport.is_closing():
            return

        response = build_non_streaming_response(osl, self._model)
        transport.write(response)
        _counter_add(_resp_counter)
        _counter_add(_byte_counter, len(response))

    async def _handle_streaming(self, transport, osl: int):
        """Wait for rate-limit slot, then stream chunks."""
        await self._bucket.acquire()

        interval = self._stream_interval
        num_events = math.ceil(osl / interval)
        created = int(time.time())

        # Streaming response headers
        headers = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: keep-alive\r\n\r\n"
        )
        transport.write(headers)
        total_bytes = len(headers)

        rng = self._rng
        chunk_cv = self._chunk_cv
        mean_delay = self._inter_chunk_latency
        model = self._model
        remaining = osl

        for _ in range(num_events):
            if transport.is_closing():
                return

            # Inter-chunk delay (simulates token generation time)
            if mean_delay > 0:
                if chunk_cv > 0:
                    sigma2 = math.log(1.0 + chunk_cv * chunk_cv)
                    mu = math.log(mean_delay) - sigma2 / 2.0
                    delay = rng.lognormvariate(mu, math.sqrt(sigma2))
                else:
                    delay = mean_delay
                await asyncio.sleep(delay)

            chars = min(interval, remaining)
            remaining -= chars
            event = _sse_event(model, created, {"content": "x" * chars})
            chunk = _chunked(event)
            transport.write(chunk)
            total_bytes += len(chunk)

        # Finish event + [DONE] sentinel + terminator
        if not transport.is_closing():
            finish = _sse_event(model, created, {}, "stop")
            tail = _chunked(finish + b"data: [DONE]\n\n") + b"0\r\n\r\n"
            transport.write(tail)
            total_bytes += len(tail)

        _counter_add(_resp_counter)
        _counter_add(_byte_counter, total_bytes)

    def connection_lost(self, exc):
        self.transport = None
        self._parser = None
        self._request = None


def _worker(
    wid: int,
    host: str,
    port: int,
    ready: multiprocessing.synchronize.Event,
    shutdown: multiprocessing.synchronize.Event,
    counters: tuple[_SyncInt | None, _SyncInt | None, _SyncInt | None],
    *,
    stream: bool,
    stream_interval: int,
    osl_mu: float,
    osl_sigma: float,
    osl_min: int,
    osl_max: int,
    rate_per_worker: float,
    inter_chunk_latency: float,
    chunk_cv: float,
):
    """Worker process entry point."""
    global _req_counter, _resp_counter, _byte_counter
    _req_counter, _resp_counter, _byte_counter = counters

    import gc

    gc.disable()
    uvloop.install()

    # Per-worker reproducible RNG seeded with worker id
    rng = random.Random(wid)

    async def run():
        loop = asyncio.get_running_loop()
        bucket = _TokenBucket(rate_per_worker)

        def protocol_factory():
            return VariableResponseProtocol(
                loop,
                rng,
                bucket,
                stream=stream,
                stream_interval=stream_interval,
                osl_mu=osl_mu,
                osl_sigma=osl_sigma,
                osl_min=osl_min,
                osl_max=osl_max,
                inter_chunk_latency=inter_chunk_latency,
                chunk_cv=chunk_cv,
            )

        server = await loop.create_server(
            protocol_factory,
            host,
            port,
            reuse_address=True,
            reuse_port=True,
            backlog=65535,
        )
        ready.set()
        while not shutdown.is_set():
            await asyncio.sleep(0.1)
        server.close()
        await server.wait_closed()

    try:
        asyncio.run(run())
    except Exception as exc:
        import sys

        print(
            f"[VariableResponseServer] Worker {wid} failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


class VariableResponseServer:
    """OpenAI-compatible server with variable response lengths and rates.

    Output length is sampled per request from a lognormal distribution.
    Total server throughput is capped at ``response_rate_mean`` resp/sec
    via per-worker token buckets.  In streaming mode, ``inter_chunk_latency``
    controls how fast chunks arrive (simulates token generation speed).

    Args:
        host: Bind address.
        port: Bind port (0 for auto-assign).
        output_len_mean: Mean output sequence length (chars).
        output_len_spread: Coefficient of variation for output length.
        output_len_min: Minimum output sequence length (chars).
        output_len_max: Maximum output sequence length (chars). None = 8 * mean.
        response_rate_mean: Total server response rate (responses/sec), split across workers.
        inter_chunk_latency: Mean delay between SSE chunks in seconds (streaming only).
            0 = no delay (chunks stream instantly). E.g. 0.02 = ~50 tokens/sec.
        response_chunk_spread: CoV for jitter on inter-chunk latency (streaming only).
        stream: SSE streaming mode.
        stream_interval: Characters per SSE event.
        num_workers: Worker processes (SO_REUSEPORT kernel load-balancing).
        stats: Enable live stats on stdout.
        stats_interval: Seconds between stats prints.
        quiet: Suppress startup banner.
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 12345,
        output_len_mean: int = 1000,
        output_len_spread: float = 0.5,
        output_len_min: int = 0,
        output_len_max: int | None = None,
        response_rate_mean: float = 100.0,
        inter_chunk_latency: float = 0.0,
        response_chunk_spread: float = 0.0,
        stream: bool = False,
        stream_interval: int = 1,
        num_workers: int = 10,
        stats: bool = False,
        stats_interval: float = 1.0,
        quiet: bool = False,
    ):
        self.host, self.port, self.num_workers = host, port, num_workers
        self.stream = stream
        self.stream_interval = stream_interval
        self.stats, self.stats_interval = stats, stats_interval
        self.quiet = quiet
        self._port: int | None = None
        self._shutdown = threading.Event()
        self._workers: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.synchronize.Event] = []
        self._worker_shutdown: multiprocessing.synchronize.Event | None = None
        self._stats_thread: threading.Thread | None = None

        # Pre-compute lognormal params
        self._osl_mu, self._osl_sigma = _lognormal_params(
            output_len_mean, output_len_spread
        )
        self._osl_min = output_len_min
        self._osl_max = (
            output_len_max if output_len_max is not None else 8 * output_len_mean
        )
        self._rate_per_worker = response_rate_mean / num_workers
        self._inter_chunk_latency = inter_chunk_latency
        self._chunk_cv = response_chunk_spread

        global _req_counter, _resp_counter, _byte_counter
        if stats:
            _req_counter = multiprocessing.Value("L", 0)  # type: ignore[assignment]
            _resp_counter = multiprocessing.Value("L", 0)  # type: ignore[assignment]
            _byte_counter = multiprocessing.Value("L", 0)  # type: ignore[assignment]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._port or self.port}"

    def _start_workers(self):
        if self.port == 0:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, 0))
            self._port = s.getsockname()[1]
            s.close()
        else:
            self._port = self.port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except (AttributeError, OSError) as e:
                    raise RuntimeError(
                        f"SO_REUSEPORT not available (required for multi-worker): {e}"
                    ) from e
                try:
                    s.bind((self.host, self._port))
                except OSError as e:
                    raise RuntimeError(
                        f"Port {self._port} is not available: {e}"
                    ) from e

        self._worker_shutdown = multiprocessing.Event()
        for i in range(self.num_workers):
            ready = multiprocessing.Event()
            self._ready_events.append(ready)
            p = multiprocessing.Process(
                target=_worker,
                args=(
                    i,
                    self.host,
                    self._port,
                    ready,
                    self._worker_shutdown,
                    (_req_counter, _resp_counter, _byte_counter),
                ),
                kwargs={
                    "stream": self.stream,
                    "stream_interval": self.stream_interval,
                    "osl_mu": self._osl_mu,
                    "osl_sigma": self._osl_sigma,
                    "osl_min": self._osl_min,
                    "osl_max": self._osl_max,
                    "rate_per_worker": self._rate_per_worker,
                    "inter_chunk_latency": self._inter_chunk_latency,
                    "chunk_cv": self._chunk_cv,
                },
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Poll for readiness — detect crashes early
        for i, evt in enumerate(self._ready_events):
            worker = self._workers[i]
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if evt.wait(timeout=0.2):
                    break
                if not worker.is_alive():
                    time.sleep(0.2)
                    msg = f"Worker {i} crashed (exit code {worker.exitcode})"
                    self._cleanup_workers()
                    raise RuntimeError(msg)
            else:
                if not evt.is_set():
                    msg = (
                        f"Worker {i} timed out after 30s waiting for ready "
                        f"signal (pid={worker.pid}, alive={worker.is_alive()})"
                    )
                    self._cleanup_workers()
                    raise RuntimeError(msg)

    def _stats_loop(self):
        last_req, last_resp, last_bytes, last_t = 0, 0, 0, time.monotonic()
        while not self._shutdown.is_set():
            time.sleep(self.stats_interval)
            now = time.monotonic()
            elapsed = now - last_t
            curr_req = _counter_read(_req_counter)
            curr_resp = _counter_read(_resp_counter)
            curr_bytes = _counter_read(_byte_counter)
            if elapsed > 0:
                print(
                    f"[Stats] Req/s: {(curr_req-last_req)/elapsed:>9,.0f} | "
                    f"Resp/s: {(curr_resp-last_resp)/elapsed:>9,.0f} | "
                    f"MB/s: {(curr_bytes-last_bytes)/elapsed/1e6:>8,.1f} | "
                    f"Total: {curr_resp:>10,}",
                    flush=True,
                )
            last_req, last_resp, last_bytes, last_t = (
                curr_req,
                curr_resp,
                curr_bytes,
                now,
            )

    def start(self):
        """Start server."""
        self._start_workers()

        if self.stats:
            self._stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            self._stats_thread.start()

        if not self.quiet:
            mode = "streaming" if self.stream else "non-streaming"
            print(
                f"VariableResponseServer @ {self.url} "
                f"(workers={self.num_workers}, mode={mode})"
            )

    def _cleanup_workers(self):
        """Terminate and join all workers, escalating to kill if needed."""
        for w in self._workers:
            if w.is_alive():
                w.terminate()
                w.join(timeout=1.0)
            if w.is_alive():
                w.kill()
                w.join(timeout=1.0)

    def stop(self):
        """Stop server."""
        self._shutdown.set()
        if self._worker_shutdown:
            self._worker_shutdown.set()
        for w in self._workers:
            w.join(timeout=2.0)
        self._cleanup_workers()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Variable-response test server (realistic LLM simulation)"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument(
        "--output-len-mean",
        type=int,
        default=1000,
        help="Mean output sequence length in chars (default: 1000)",
    )
    parser.add_argument(
        "--output-len-spread",
        type=float,
        default=0.5,
        help="Coefficient of variation for output length (default: 0.5)",
    )
    parser.add_argument(
        "--output-len-min",
        type=int,
        default=0,
        help="Minimum output sequence length in chars (default: 0)",
    )
    parser.add_argument(
        "--output-len-max",
        type=int,
        default=None,
        help="Maximum output sequence length in chars (default: 8 * output-len-mean)",
    )
    parser.add_argument(
        "--response-rate-mean",
        type=float,
        default=100.0,
        help="Total server response rate in resp/sec (default: 100.0). "
        "Split evenly across workers via per-worker token bucket.",
    )
    parser.add_argument(
        "--inter-chunk-latency",
        type=float,
        default=0.0,
        help="Mean delay between SSE chunks in seconds, streaming only (default: 0.0). "
        "E.g. 0.02 = ~50 tokens/sec per response.",
    )
    parser.add_argument(
        "--response-chunk-spread",
        type=float,
        default=0.0,
        help="CoV for inter-chunk latency, streaming only (default: 0.0)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="SSE streaming mode (default: non-streaming JSON)",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Characters per SSE event (default: 1)",
    )
    parser.add_argument("--num-workers", "-w", type=int, default=10)
    parser.add_argument("--stats", action="store_true", help="Enable live stats")
    args = parser.parse_args()

    server = VariableResponseServer(
        host=args.host,
        port=args.port,
        output_len_mean=args.output_len_mean,
        output_len_spread=args.output_len_spread,
        output_len_min=args.output_len_min,
        output_len_max=args.output_len_max,
        response_rate_mean=args.response_rate_mean,
        inter_chunk_latency=args.inter_chunk_latency,
        response_chunk_spread=args.response_chunk_spread,
        stream=args.stream,
        stream_interval=args.stream_interval,
        num_workers=args.num_workers,
        stats=args.stats,
    )

    _main_pid = os.getpid()

    def sig_handler(signum, frame):
        if os.getpid() != _main_pid:
            # Worker process — just exit, don't try to join siblings.
            os._exit(0)
        server.stop()
        os._exit(0)

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    server.start()
    print("Press Ctrl+C to stop...")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
