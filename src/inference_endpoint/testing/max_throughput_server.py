# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal OpenAI-compatible LLM API server for client roofline testing.

Returns fixed, pre-compiled HTTP responses for every request — mocks model inference.
This isolates the HTTP client's raw send/recv throughput from any server-side
compute overhead, making it a good target for roofline benchmarks and parameter sweeps
(see :mod:`inference_endpoint.utils.benchmark_httpclient`).

Usage::

    # CLI — non-streaming (offline) mode with live stats
    python -m inference_endpoint.testing.max_throughput_server --port 12345 --stats

    # CLI — SSE streaming mode, 50 characters per SSE event
    python -m inference_endpoint.testing.max_throughput_server --stream --stream-interval 50 --stats

    # Python — as a context manager in tests / scripts
    with MaxThroughputServer(port=0, num_workers=2) as srv:
        url = f"{srv.url}/v1/chat/completions"
        # ... send requests to *url* ...

    # Paired with the benchmark client
    python -m inference_endpoint.utils.benchmark_httpclient            # auto-launches server
    python -m inference_endpoint.utils.benchmark_httpclient -w 4:12    # worker sweep
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import os
import signal
import socket
import threading
import time

import httptools
import msgspec
import uvloop

# Shared counters (multiprocessing.Value for cross-process atomicity)
type _SyncInt = multiprocessing.sharedctypes.Synchronized[int]  # type: ignore[valid-type]

_req_counter: _SyncInt | None = None
_resp_counter: _SyncInt | None = None
_byte_counter: _SyncInt | None = None


def _counter_add(counter: _SyncInt | None, n: int = 1) -> None:
    """Atomically increment a shared counter (no-op if counter is None)."""
    if counter is not None:
        with counter.get_lock():
            counter.value += n


def _counter_read(counter: _SyncInt | None) -> int:
    """Atomically read a shared counter (returns 0 if counter is None)."""
    if counter is not None:
        with counter.get_lock():
            return counter.value
    return 0


def _chunked(data: bytes) -> bytes:
    """HTTP chunked transfer encoding."""
    return f"{len(data):x}\r\n".encode() + data + b"\r\n" if data else b"0\r\n\r\n"


def _sse_event(
    model: str, created: int, delta: dict, finish_reason: str | None = None
) -> bytes:
    """Encode a single SSE data frame."""
    return (
        b"data: "
        + msgspec.json.encode(
            {
                "id": "r",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        )
        + b"\n\n"
    )


def build_streaming_response(
    output_length: int = 160, stream_interval: int = 1, model: str = "max-tp"
) -> bytes:
    """Build complete SSE streaming response (chunked encoding).

    *stream_interval* controls how many characters are packed into each SSE
    event.  Total SSE content events = ``ceil(output_length / stream_interval)``.
    Each event is sent in its own HTTP chunked frame.

    Args:
        output_length: Total characters in the response.
        stream_interval: Characters per SSE event (1 = finest granularity).
        model: Model name in the response JSON.
    """
    headers = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Connection: keep-alive\r\n\r\n"
    )
    created = int(time.time())

    # Each SSE event carries `stream_interval` chars (last may be shorter)
    chunks: list[bytes] = []
    for i in range(0, output_length, stream_interval):
        chars = min(stream_interval, output_length - i)
        event = _sse_event(model, created, {"content": "x" * chars})
        chunks.append(_chunked(event))

    # Finish event + [DONE] sentinel in one final chunk
    finish = _sse_event(model, created, {}, "stop")
    chunks.append(_chunked(finish + b"data: [DONE]\n\n"))
    chunks.append(b"0\r\n\r\n")

    return headers + b"".join(chunks)


def build_non_streaming_response(
    output_length: int = 160, model: str = "max-tp"
) -> bytes:
    """Build complete non-streaming JSON response.

    Args:
        output_length: Number of characters in the response content.
        model: Model name in the response JSON.
    """
    body = msgspec.json.encode(
        {
            "id": "r",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "x" * output_length,
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": output_length,
                "total_tokens": output_length + 10,
            },
            "system_fingerprint": None,
        }
    )
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\nConnection: keep-alive\r\n\r\n"
        + body
    )


def build_response(
    output_length: int = 160,
    stream: bool = False,
    stream_interval: int = 1,
    model: str = "max-tp",
) -> bytes:
    """Build a pre-compiled HTTP response.

    Args:
        output_length: Characters in the response content.
        stream: SSE streaming (True) or single JSON (False).
        stream_interval: Characters per SSE event (streaming only).
        model: Model name in the response JSON.
    """
    if stream:
        return build_streaming_response(output_length, stream_interval, model)
    return build_non_streaming_response(output_length, model)


class RequestParser:
    """Minimal HTTP request parser callbacks."""

    __slots__ = ("body", "_parts", "_done")

    def __init__(self):
        self.body = b""
        self._parts = []
        self._done = False

    def on_message_begin(self):
        pass

    def on_url(self, url):
        pass

    def on_header(self, name, value):
        pass

    def on_headers_complete(self):
        pass

    def on_body(self, body):
        self._parts.append(body)

    def on_message_complete(self):
        self.body = b"".join(self._parts)
        self._done = True

    def done(self) -> bool:
        return self._done


class ServerProtocol(asyncio.Protocol):
    """High-performance asyncio Protocol with proper request buffering."""

    __slots__ = (
        "transport",
        "response",
        "response_size",
        "_parser",
        "_request",
    )

    def __init__(self, response: bytes):
        self.transport = None
        self.response = response
        self.response_size = len(response)
        self._request: RequestParser | None = None
        self._parser: httptools.HttpRequestParser | None = None

    def connection_made(self, transport):
        self.transport = transport
        sock = transport.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Initialize parser for this connection
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

        # Process request if complete
        if request.done():
            _counter_add(_req_counter)

            transport = self.transport
            if transport and not transport.is_closing():
                transport.write(self.response)  # type: ignore[union-attr]
                _counter_add(_resp_counter)
                _counter_add(_byte_counter, self.response_size)

            # Reset for next request (HTTP/1.1 keep-alive)
            self._request = RequestParser()
            self._parser = httptools.HttpRequestParser(self._request)

    def connection_lost(self, exc):
        self.transport = None
        self._parser = None
        self._request = None


def _worker(
    wid: int,
    host: str,
    port: int,
    response: bytes,
    ready: multiprocessing.synchronize.Event,
    shutdown: multiprocessing.synchronize.Event,
    counters: tuple[_SyncInt | None, _SyncInt | None, _SyncInt | None] = (
        None,
        None,
        None,
    ),
):
    """Worker process entry point."""
    global _req_counter, _resp_counter, _byte_counter
    _req_counter, _resp_counter, _byte_counter = counters

    import gc

    gc.disable()
    uvloop.install()

    async def run():
        loop = asyncio.get_running_loop()

        def protocol_factory():
            return ServerProtocol(response)

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
            f"[MaxThroughputServer] Worker {wid} failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


class MaxThroughputServer:
    """OpenAI-compatible stub server that returns pre-compiled responses.

    Designed for client roofline tests: every request receives the same
    fixed response bytes, so measured throughput reflects pure client +
    network overhead with zero server-side compute.

    Args:
        host: Bind address.
        port: Bind port (0 for auto-assign).
        output_length: Characters in the response content.
        stream: Return SSE streaming responses (default: non-streaming JSON).
        stream_interval: Characters per SSE event (only with *stream=True*).
            Total events = ceil(output_length / stream_interval).
            1 = one char per event (finest), output_length = one big event.
        num_workers: Worker processes (SO_REUSEPORT kernel load-balancing).
        stats: Enable live req/s and throughput stats on stdout.
        stats_interval: Seconds between stats prints.
        quiet: Suppress startup banner.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        output_length: int = 160,
        stream: bool = False,
        stream_interval: int = 1,
        num_workers: int = 4,
        stats: bool = False,
        stats_interval: float = 1.0,
        quiet: bool = False,
    ):
        self.host, self.port, self.num_workers = host, port, num_workers
        self.stream = stream
        self.stats, self.stats_interval = stats, stats_interval
        self.quiet = quiet
        self._port: int | None = None
        self._shutdown = threading.Event()
        self._workers: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.synchronize.Event] = []
        self._worker_shutdown: multiprocessing.synchronize.Event | None = None
        self._stats_thread: threading.Thread | None = None

        self._response = build_response(output_length, stream, stream_interval)

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
            # Fail fast if port is already in use or SO_REUSEPORT unavailable
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
                    self._response,
                    ready,
                    self._worker_shutdown,
                    (_req_counter, _resp_counter, _byte_counter),
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Poll for readiness — detect crashes early, allow slow starts
        for i, evt in enumerate(self._ready_events):
            worker = self._workers[i]
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if evt.wait(timeout=0.2):
                    break
                if not worker.is_alive():
                    # Brief pause to let stderr flush from dying worker
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
                    f"QPS: {(curr_resp-last_resp)/elapsed:>9,.0f} | "
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
                f"MaxThroughputServer @ {self.url} "
                f"(workers={self.num_workers}, mode={mode}, "
                f"response={len(self._response)}B)"
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
        # Give workers a chance to exit gracefully
        for w in self._workers:
            w.join(timeout=2.0)
        # Escalate: terminate, then kill any stragglers
        self._cleanup_workers()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Max throughput test server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument(
        "--output-length",
        type=int,
        default=4000,
        help="Characters in the response content",
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
        help="Characters per SSE event (only with --stream). "
        "Total events = ceil(output_length / stream_interval). "
        "1 = one char per event (finest), output_length = one big event.",
    )
    parser.add_argument("--num-workers", "-w", type=int, default=4)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    server = MaxThroughputServer(
        host=args.host,
        port=args.port,
        output_length=args.output_length,
        stream=args.stream,
        stream_interval=args.stream_interval,
        num_workers=args.num_workers,
        stats=args.stats,
    )

    _main_pid = os.getpid()

    def sig_handler(signum, frame):
        if os.getpid() != _main_pid:
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
