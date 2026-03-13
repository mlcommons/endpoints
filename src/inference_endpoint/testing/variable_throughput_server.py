# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible LLM API server with variable response lengths and rates.

Unlike :class:`MaxThroughputServer` (which returns identical pre-compiled responses
instantly for roofline testing), this server models realistic LLM inference:

* **Variable output lengths** — lognormal distribution, configurable mean + spread
* **Per-worker response rate** — token bucket rate limiter per worker process
* **First-chunk latency (TTFT)** — lognormal delay before first data
* **Per-chunk latency jitter** — lognormal inter-token delays in streaming mode

Two mutually exclusive timing modes:

* **Response-rate mode** (``--response-rate-mean``): controls total response time
  per request.  TPOT is derived from ``(1/rate - TTFT) / (num_chunks - 1)``.
* **Inter-token mode** (``--inter-token-latency``): controls per-token delay
  (TPOT) directly.  Actual inter-SSE-event delay = TPOT × stream_interval.

Usage::

    # Basic non-streaming
    python -m inference_endpoint.testing.variable_throughput_server --stats

    # Offline with response-rate jitter
    python -m inference_endpoint.testing.variable_throughput_server --stats \\
        --output-len-mean 1000 --output-len-spread 0.4 \\
        --response-rate-mean 10000 --response-rate-spread 2.0

    # Streaming with inter-token latency (ms) + TTFT (s)
    python -m inference_endpoint.testing.variable_throughput_server --stream --stats \\
        --stream-interval 2 \\
        --inter-token-latency 20 --inter-token-spread 0.05 \\
        --first-chunk-latency 0.1 --first-chunk-spread 0.02

    # Streaming with response-rate + TTFT
    python -m inference_endpoint.testing.variable_throughput_server --stream --stats \\
        --response-rate-mean 50 --response-rate-spread 0.2 \\
        --first-chunk-latency 0.5 --first-chunk-spread 0.1

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

try:
    from .max_throughput_server import (
        RequestParser,
        _chunked,
        _counter_add,
        _counter_read,
        _sse_event,
        build_non_streaming_response,
    )
except ImportError:
    from max_throughput_server import (  # type: ignore[no-redef]
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
    """Per-worker rate limiter: enforces N responses/sec within a single worker."""

    __slots__ = ("_interval", "_available_at")

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        self._interval = 1.0 / rate
        self._available_at = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        start = max(now, self._available_at)
        self._available_at = start + self._interval
        if (wait := start - now) > 0:
            await asyncio.sleep(wait)


def _sample_lognormal(
    rng: random.Random, mu: float, sigma: float, mean: float
) -> float:
    """Sample from lognormal, falling back to mean when sigma=0."""
    if sigma > 0:
        return rng.lognormvariate(mu, sigma)
    return mean


class VariableResponseProtocol(asyncio.Protocol):
    """asyncio Protocol that samples output length + response time per request.

    Two timing modes:
    - **rate**: Global token bucket controls overall throughput (responses/sec).
    - **icl**: Per-request inter-token latency, no global rate limit.

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
        "_semaphore",
        "_stream",
        "_stream_interval",
        "_osl_mu",
        "_osl_sigma",
        "_osl_min",
        "_osl_max",
        "_osl_mean",
        "_model",
        # Timing mode: "rate" or "icl"
        "_mode",
        # First-chunk latency (TTFT) params
        "_fcl_mu",
        "_fcl_sigma",
        "_fcl_mean",
        # Response-rate params (mode="rate")
        "_rate_mu",
        "_rate_sigma",
        "_rate_mean",
        # Inter-token-latency params (mode="icl")
        "_icl_mu",
        "_icl_sigma",
        "_icl_mean",
    )

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        rng: random.Random,
        bucket: _TokenBucket | None,
        semaphore: asyncio.Semaphore | None,
        *,
        stream: bool,
        stream_interval: int,
        osl_mu: float,
        osl_sigma: float,
        osl_min: int,
        osl_max: int,
        osl_mean: int,
        mode: str,
        fcl_mu: float,
        fcl_sigma: float,
        fcl_mean: float,
        rate_mu: float,
        rate_sigma: float,
        rate_mean: float,
        icl_mu: float,
        icl_sigma: float,
        icl_mean: float,
    ):
        self.transport = None
        self._parser: httptools.HttpRequestParser | None = None
        self._request: RequestParser | None = None
        self._loop = loop
        self._rng = rng
        self._bucket = bucket
        self._semaphore = semaphore
        self._stream = stream
        self._stream_interval = stream_interval
        self._osl_mu = osl_mu
        self._osl_sigma = osl_sigma
        self._osl_min = osl_min
        self._osl_max = osl_max
        self._osl_mean = osl_mean
        self._model = "variable-resp"
        self._mode = mode
        self._fcl_mu = fcl_mu
        self._fcl_sigma = fcl_sigma
        self._fcl_mean = fcl_mean
        self._rate_mu = rate_mu
        self._rate_sigma = rate_sigma
        self._rate_mean = rate_mean
        self._icl_mu = icl_mu
        self._icl_sigma = icl_sigma
        self._icl_mean = icl_mean

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

            # Dispatch async handler
            self._loop.create_task(self._handle_request(osl))

            # Reset for next request (HTTP/1.1 keep-alive)
            self._request = RequestParser()
            self._parser = httptools.HttpRequestParser(self._request)

    async def _handle_request(self, osl: int):
        """Handle a single request with sampled timing."""
        transport = self.transport
        if transport is None or transport.is_closing():
            return

        # Optional concurrency limit — wait for a slot
        sem = self._semaphore
        if sem is not None:
            await sem.acquire()

        try:
            rng = self._rng

            # Rate mode: global token bucket controls overall throughput
            if self._mode == "rate":
                if self._bucket:
                    await self._bucket.acquire()
                # Sample TTFT
                ttft = _sample_lognormal(
                    rng, self._fcl_mu, self._fcl_sigma, self._fcl_mean
                )
                # Derive TPOT from per-request sampled rate
                rate = _sample_lognormal(
                    rng, self._rate_mu, self._rate_sigma, self._rate_mean
                )
                total_time = 1.0 / rate if rate > 0 else 0.0
                if self._stream and osl > 1:
                    # Derive per-token TPOT: remaining time spread across output tokens.
                    # Streaming loop scales by tokens_per_chunk automatically.
                    tpot = max(0.0, (total_time - ttft) / (osl - 1))
                else:
                    # Non-streaming or single chunk: total_time is the full delay
                    ttft = total_time
                    tpot = 0.0
            else:  # mode == "icl" — per-request timing, no global rate
                # Sample TTFT
                ttft = _sample_lognormal(
                    rng, self._fcl_mu, self._fcl_sigma, self._fcl_mean
                )
                tpot = _sample_lognormal(
                    rng, self._icl_mu, self._icl_sigma, self._icl_mean
                )

            if self._stream:
                await self._handle_streaming(transport, osl, ttft, tpot)
            else:
                await self._handle_non_streaming(transport, osl, ttft)
        except (RuntimeError, ConnectionError, OSError):
            # Client disconnected — transport closed mid-write. Ignore.
            pass
        finally:
            if sem is not None:
                sem.release()

    async def _handle_non_streaming(self, transport, osl: int, delay: float):
        """Wait for delay (TTFT / total_time), then send complete response."""
        if delay > 0:
            await asyncio.sleep(delay)

        if transport.is_closing():
            return

        response = build_non_streaming_response(osl, self._model)
        transport.write(response)
        _counter_add(_resp_counter)
        _counter_add(_byte_counter, len(response))

    async def _handle_streaming(self, transport, osl: int, ttft: float, tpot: float):
        """Stream chunks with TTFT + per-token TPOT delays.

        tpot is the simulated per-token generation time (seconds).
        stream_interval is chars per SSE event (≈ tokens).
        Actual inter-event delay = tpot × stream_interval.
        """
        interval = self._stream_interval
        num_events = math.ceil(osl / interval)
        created = int(time.time())
        model = self._model

        # Pre-compile all chunk bytes upfront.
        chunks: list[bytes] = []
        chars_left = osl
        for _ in range(num_events):
            chars = min(interval, chars_left)
            chars_left -= chars
            event = _sse_event(model, created, {"content": "x" * chars})
            chunks.append(_chunked(event))

        finish = _sse_event(model, created, {}, "stop")
        tail = _chunked(finish + b"data: [DONE]\n\n") + b"0\r\n\r\n"

        # TTFT delay before first data
        if ttft > 0:
            await asyncio.sleep(ttft)

        if transport.is_closing():
            return

        # Headers
        headers = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: keep-alive\r\n\r\n"
        )
        transport.write(headers)
        total_bytes = len(headers)

        # Stream chunks with deadline-based TPOT timing.
        # Uses loop.time() (uvloop high-res monotonic) to avoid drift.
        loop = self._loop
        target = loop.time()

        for i, chunk in enumerate(chunks):
            if transport.is_closing():
                return

            # Delay = tpot × chars_per_event (chars ≈ tokens). Skip first — TTFT already applied.
            if i > 0 and tpot > 0:
                target += tpot * interval
                wait = target - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)

            transport.write(chunk)
            total_bytes += len(chunk)

        # Finish event + [DONE] sentinel + terminator
        if not transport.is_closing():
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
    osl_mean: int,
    mode: str,
    fcl_mu: float,
    fcl_sigma: float,
    fcl_mean: float,
    rate_per_worker: float,
    rate_mu: float,
    rate_sigma: float,
    rate_mean: float,
    icl_mu: float,
    icl_sigma: float,
    icl_mean: float,
    max_concurrency: int,
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
        bucket = _TokenBucket(rate_per_worker) if rate_per_worker > 0 else None
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

        def protocol_factory():
            return VariableResponseProtocol(
                loop,
                rng,
                bucket,
                semaphore,
                stream=stream,
                stream_interval=stream_interval,
                osl_mu=osl_mu,
                osl_sigma=osl_sigma,
                osl_min=osl_min,
                osl_max=osl_max,
                osl_mean=osl_mean,
                mode=mode,
                fcl_mu=fcl_mu,
                fcl_sigma=fcl_sigma,
                fcl_mean=fcl_mean,
                rate_mu=rate_mu,
                rate_sigma=rate_sigma,
                rate_mean=rate_mean,
                icl_mu=icl_mu,
                icl_sigma=icl_sigma,
                icl_mean=icl_mean,
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
    """OpenAI-compatible server with variable response lengths and per-request timing.

    Output length is sampled per request from a lognormal distribution.
    Timing is per-request with two mutually exclusive modes:

    * **Response-rate mode**: each request samples its own rate, TPOT is derived.
    * **Inter-token mode**: each request samples its own TPOT directly.

    Both modes support first-chunk latency (TTFT) with optional spread.

    Args:
        host: Bind address.
        port: Bind port (0 for auto-assign).
        output_len_mean: Mean output sequence length (chars).
        output_len_spread: Coefficient of variation for output length.
        output_len_min: Minimum output sequence length (chars).
        output_len_max: Maximum output sequence length (chars). None = 8 * mean.
        response_rate_mean: Per-request response rate mean (responses/sec). 0 = no rate mode.
        response_rate_spread: CoV for per-request response rate.
        inter_token_latency: Per-token delay (TPOT) mean in milliseconds. 0 = no ICL mode.
        inter_token_spread: CoV for per-chunk delay.
        first_chunk_latency: Mean TTFT in seconds. 0 = no TTFT delay.
        first_chunk_spread: CoV for TTFT.
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
        output_len_spread: float = 0.3,
        output_len_min: int = 0,
        output_len_max: int | None = None,
        response_rate_mean: float = 0.0,
        response_rate_spread: float = 0.0,
        inter_token_latency: float = 0.0,
        inter_token_spread: float = 0.0,
        first_chunk_latency: float = 0.0,
        first_chunk_spread: float = 0.2,
        stream: bool = False,
        stream_interval: int = 1,
        max_concurrency: int = 0,
        num_workers: int = 10,
        stats: bool = False,
        stats_interval: float = 1.0,
        quiet: bool = False,
    ):
        # Validate mutual exclusivity
        if response_rate_mean > 0 and inter_token_latency > 0:
            raise ValueError(
                "response_rate_mean and inter_token_latency are mutually exclusive. "
                "Use response-rate mode OR inter-token-latency mode, not both."
            )
        if num_workers <= 0:
            raise ValueError(f"num_workers must be > 0, got {num_workers}")
        if stream and stream_interval <= 0:
            raise ValueError(
                f"stream_interval must be > 0 in streaming mode, got {stream_interval}"
            )
        if response_rate_mean < 0:
            raise ValueError(
                f"response_rate_mean must be >= 0, got {response_rate_mean}"
            )

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

        # Pre-compute lognormal params for output length
        self._osl_mu, self._osl_sigma = _lognormal_params(
            output_len_mean, output_len_spread
        )
        self._osl_min = output_len_min
        self._osl_max = (
            output_len_max if output_len_max is not None else 2 * output_len_mean
        )
        self._osl_mean = output_len_mean

        # Determine timing mode
        if response_rate_mean > 0:
            self._mode = "rate"
        elif inter_token_latency > 0:
            self._mode = "icl"
        else:
            # Neither specified — no delays (instant response)
            self._mode = "rate"

        # First-chunk latency (TTFT) params
        if first_chunk_latency > 0:
            self._fcl_mu, self._fcl_sigma = _lognormal_params(
                first_chunk_latency, first_chunk_spread
            )
        else:
            self._fcl_mu, self._fcl_sigma = 0.0, 0.0
        self._fcl_mean = first_chunk_latency

        # Response-rate params (global bucket + per-request sampling)
        if response_rate_mean > 0:
            self._rate_per_worker = response_rate_mean / num_workers
            self._rate_mu, self._rate_sigma = _lognormal_params(
                response_rate_mean, response_rate_spread
            )
        else:
            self._rate_per_worker = 0.0
            self._rate_mu, self._rate_sigma = 0.0, 0.0
        self._rate_mean = response_rate_mean

        # Inter-token-latency params (convert ms → seconds for asyncio.sleep)
        icl_s = inter_token_latency / 1000.0
        if icl_s > 0:
            self._icl_mu, self._icl_sigma = _lognormal_params(icl_s, inter_token_spread)
        else:
            self._icl_mu, self._icl_sigma = 0.0, 0.0
        self._icl_mean = icl_s

        # Max concurrency per worker (0 = unlimited)
        if max_concurrency > 0:
            self._max_concurrency_per_worker = max(1, max_concurrency // num_workers)
        else:
            self._max_concurrency_per_worker = 0

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
                    "osl_mean": self._osl_mean,
                    "mode": self._mode,
                    "fcl_mu": self._fcl_mu,
                    "fcl_sigma": self._fcl_sigma,
                    "fcl_mean": self._fcl_mean,
                    "rate_per_worker": self._rate_per_worker,
                    "rate_mu": self._rate_mu,
                    "rate_sigma": self._rate_sigma,
                    "rate_mean": self._rate_mean,
                    "icl_mu": self._icl_mu,
                    "icl_sigma": self._icl_sigma,
                    "icl_mean": self._icl_mean,
                    "max_concurrency": self._max_concurrency_per_worker,
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
            timing = self._mode if self._mode == "icl" else "rate"
            print(
                f"VariableResponseServer @ {self.url} "
                f"(workers={self.num_workers}, mode={mode}, timing={timing})"
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
        default=0.3,
        help="Coefficient of variation for output length (default: 0.3)",
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
        help="Maximum output sequence length in chars (default: 2 * output-len-mean)",
    )

    # Timing mode A: response-rate
    parser.add_argument(
        "--response-rate-mean",
        type=float,
        default=0.0,
        help="Per-request response rate mean in resp/sec (default: 0 = disabled). "
        "Mutually exclusive with --inter-token-latency.",
    )
    parser.add_argument(
        "--response-rate-spread",
        type=float,
        default=0.0,
        help="CoV for per-request response rate (default: 0.0 = deterministic)",
    )

    # Timing mode B: inter-token-latency
    parser.add_argument(
        "--inter-token-latency",
        type=float,
        default=0.0,
        help="Per-token generation time (TPOT) in milliseconds (default: 0 = disabled). "
        "E.g. 20 = 20ms/token. Actual inter-SSE-event delay = TPOT × stream_interval. "
        "Mutually exclusive with --response-rate-mean.",
    )
    parser.add_argument(
        "--inter-token-spread",
        type=float,
        default=0.0,
        help="CoV for per-chunk delay (default: 0.0 = deterministic)",
    )

    # TTFT (first-chunk latency)
    parser.add_argument(
        "--first-chunk-latency",
        type=float,
        default=0.0,
        help="Mean first-chunk latency (TTFT) in seconds (default: 0.0). "
        "Applies in both streaming and non-streaming modes.",
    )
    parser.add_argument(
        "--first-chunk-spread",
        type=float,
        default=0.2,
        help="CoV for first-chunk latency (default: 0.2)",
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="Max concurrent requests per server (0 = unlimited). "
        "Split evenly across workers.",
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
        response_rate_spread=args.response_rate_spread,
        inter_token_latency=args.inter_token_latency,
        inter_token_spread=args.inter_token_spread,
        first_chunk_latency=args.first_chunk_latency,
        first_chunk_spread=args.first_chunk_spread,
        stream=args.stream,
        stream_interval=args.stream_interval,
        max_concurrency=args.max_concurrency,
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
