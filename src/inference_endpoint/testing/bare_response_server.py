# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Barebones HTTP server for benchmarking - responds immediately without parsing request.

Provides two variants:
- BareResponseServer: In-process async server (simpler, shares event loop)
- BareResponseServerProcess: Out-of-process server (isolates CPU load from client)
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket
from multiprocessing import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.context import SpawnProcess

# Default content size in tokens (approximated as characters for ASCII)
DEFAULT_RESPONSE_SIZE = 64

# Socket buffer sizes (10MB for high throughput)
_SOCKET_BUFFER_SIZE = 10 * 1024 * 1024


def _build_content(size: int) -> bytes:
    """Build response content of specified token count (1 char = ~1 token for ASCII)."""
    return b"x" * size


def _build_non_streaming_response(response_size: int = DEFAULT_RESPONSE_SIZE) -> bytes:
    """Build non-streaming JSON response with content of specified token count."""
    content = _build_content(response_size)
    # Full OpenAI-compatible response with all required fields
    body = (
        b'{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"test-model",'
        b'"choices":[{"index":0,"message":{"role":"assistant","content":"'
        + content
        + b'","refusal":null},"finish_reason":"stop"}],"usage":null,"system_fingerprint":null}'
    )
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: keep-alive\r\n"
        b"\r\n" + body
    )


def _build_streaming_response(
    num_chunks: int, response_size: int = DEFAULT_RESPONSE_SIZE
) -> bytes:
    """Build streaming SSE response with content of specified token count per chunk."""
    content = _build_content(response_size)
    # Full OpenAI-compatible SSE chunk with all required fields
    chunk = (
        b'{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"test-model",'
        b'"choices":[{"index":0,"delta":{"content":"'
        + content
        + b'"},"finish_reason":null}]}'
    )
    body = (b"data: " + chunk + b"\n\n") * num_chunks
    # Final chunk with finish_reason
    body += (
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"test-model",'
        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Connection: keep-alive\r\n"
        b"\r\n" + body
    )


def _configure_socket(sock: socket.socket) -> None:
    """Apply aggressive performance socket options."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SOCKET_BUFFER_SIZE)

    # Linux-specific optimizations
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass  # Not supported on all systems

    if hasattr(socket, "TCP_QUICKACK"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)

    sock.setblocking(False)


class BareResponseServer:
    """Zero-overhead HTTP server - fires response as soon as request header end is detected.

    Runs in the same process/event loop as the caller.

    Args:
        host: Bind address (default: 127.0.0.1)
        port: Bind port (default: 0 for auto-assign)
        streaming: Use SSE streaming response format
        num_chunks: Number of SSE chunks for streaming mode
        response_size: Size of content in tokens (approximated as characters for ASCII)
    """

    __slots__ = (
        "host",
        "port",
        "_server",
        "_actual_port",
        "request_count",
        "_response",
        "_handler_tasks",
    )

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        streaming: bool = False,
        num_chunks: int = 1,
        response_size: int = DEFAULT_RESPONSE_SIZE,
    ):
        self.host = host
        self.port = port
        self._server: asyncio.Server | None = None
        self._actual_port: int | None = None
        self.request_count = 0
        self._response = (
            _build_streaming_response(num_chunks, response_size)
            if streaming
            else _build_non_streaming_response(response_size)
        )
        self._handler_tasks: set[asyncio.Task] = set()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._actual_port or self.port}"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle connection - read request (headers + body), send pre-built response."""
        response = self._response  # Local reference for speed
        try:
            while True:
                # Read headers
                headers = await reader.readuntil(b"\r\n\r\n")

                # Parse Content-Length to consume request body (prevents protocol desync)
                content_length = 0
                for line in headers.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        content_length = int(line.split(b":", 1)[1].strip())
                        break

                # Consume request body if present
                if content_length > 0:
                    await reader.readexactly(content_length)

                self.request_count += 1
                writer.write(response)
                # Skip drain() for speed - let OS buffer handle it
        except (
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            asyncio.CancelledError,
            OSError,
        ):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Callback when client connects - track the handler task."""
        task = asyncio.create_task(self._handle(reader, writer))
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    async def start(self) -> None:
        """Start the server."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _configure_socket(sock)
        sock.bind((self.host, self.port))
        sock.listen(65535)
        self._server = await asyncio.start_server(self._on_client_connected, sock=sock)
        self._actual_port = sock.getsockname()[1]

    async def stop(self) -> None:
        """Stop the server and cleanup."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Cancel all pending handler tasks
        for task in self._handler_tasks:
            task.cancel()

        # Wait for all tasks to complete cancellation
        if self._handler_tasks:
            await asyncio.gather(*self._handler_tasks, return_exceptions=True)
        self._handler_tasks.clear()

    async def __aenter__(self) -> BareResponseServer:
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()


# =============================================================================
# Out-of-process server (isolates CPU load from client)
# =============================================================================


def _server_process_main(
    port_queue: Queue,
    stop_event: mp.Event,
    host: str,
    streaming: bool,
    num_chunks: int,
    response_size: int,
) -> None:
    """Entry point for server subprocess."""
    import uvloop

    async def run_server():
        server = BareResponseServer(
            host=host,
            port=0,
            streaming=streaming,
            num_chunks=num_chunks,
            response_size=response_size,
        )
        await server.start()
        port_queue.put(server._actual_port)

        # Wait for stop signal
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        await server.stop()

    uvloop.run(run_server())


class BareResponseServerProcess:
    """Out-of-process HTTP server - runs in separate process to isolate CPU load.

    Use this for accurate benchmarking where the server shouldn't compete
    with the client for CPU resources.

    Args:
        host: Bind address (default: 127.0.0.1)
        streaming: Use SSE streaming response format
        num_chunks: Number of SSE chunks for streaming mode
        response_size: Size of content in tokens (approximated as characters for ASCII)
    """

    __slots__ = (
        "host",
        "streaming",
        "num_chunks",
        "response_size",
        "_process",
        "_port",
        "_port_queue",
        "_stop_event",
    )

    def __init__(
        self,
        host: str = "127.0.0.1",
        streaming: bool = False,
        num_chunks: int = 1,
        response_size: int = DEFAULT_RESPONSE_SIZE,
    ):
        self.host = host
        self.streaming = streaming
        self.num_chunks = num_chunks
        self.response_size = response_size
        self._process: SpawnProcess | None = None
        self._port: int | None = None
        self._port_queue: Queue | None = None
        self._stop_event: mp.Event | None = None

    @property
    def url(self) -> str:
        if self._port is None:
            raise RuntimeError("Server not started")
        return f"http://{self.host}:{self._port}"

    def start(self) -> None:
        """Start server in separate process."""
        ctx = mp.get_context("spawn")
        self._port_queue = ctx.Queue()
        self._stop_event = ctx.Event()

        self._process = ctx.Process(
            target=_server_process_main,
            args=(
                self._port_queue,
                self._stop_event,
                self.host,
                self.streaming,
                self.num_chunks,
                self.response_size,
            ),
            daemon=True,
        )
        self._process.start()

        # Wait for port from subprocess
        self._port = self._port_queue.get(timeout=10.0)

    def stop(self) -> None:
        """Stop the server process."""
        if self._stop_event:
            self._stop_event.set()

        if self._process:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        self._process = None
        self._port = None

    def __enter__(self) -> BareResponseServerProcess:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    async def __aenter__(self) -> BareResponseServerProcess:
        self.start()
        return self

    async def __aexit__(self, *_) -> None:
        self.stop()
