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
from collections.abc import Callable
from multiprocessing import Queue
from multiprocessing.context import SpawnProcess

# Default content size in tokens (approximated as characters for ASCII)
DEFAULT_RESPONSE_SIZE = 64

# Socket buffer sizes (10MB for high throughput)
_SOCKET_BUFFER_SIZE = 10 * 1024 * 1024

# Pre-computed header markers
_HEADER_END = b"\r\n\r\n"
_CONTENT_LENGTH_LOWER = b"content-length:"


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
        b"Connection: close\r\n"
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


class BareResponseProtocol(asyncio.Protocol):
    """
    Raw Protocol implementation for HTTP response processing.
    - data_received(): Called when data arrives (no await overhead)
    - transport.write(): Direct write (no drain() await)

    Handles HTTP keep-alive by buffering and processing multiple requests
    per connection. Closes connection after response if Connection: close.
    """

    __slots__ = (
        "_transport",
        "_response",
        "_request_count_callback",
        "_buffer",
        "_waiting_for_body",
        "_body_remaining",
        "_close_after_response",
    )

    def __init__(
        self,
        response: bytes,
        request_count_callback: Callable[[], None] | None = None,
    ):
        self._response = response
        self._request_count_callback = request_count_callback
        self._transport: asyncio.Transport | None = None
        self._buffer = b""
        self._waiting_for_body = False
        self._body_remaining = 0
        # Check if response has Connection: close header
        self._close_after_response = b"Connection: close" in response

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Called when connection is established."""
        self._transport = transport

    def data_received(self, data: bytes) -> None:
        """
        Called when data is received from the socket.

        Processes complete HTTP requests from the buffer and sends responses.
        Handles keep-alive by processing multiple requests per connection.
        """
        self._buffer += data
        transport = self._transport
        response = self._response
        callback = self._request_count_callback

        # Process all complete requests in buffer
        while True:
            if self._waiting_for_body:
                # Waiting for request body
                if len(self._buffer) < self._body_remaining:
                    return  # Need more data

                # Consume body and send response
                self._buffer = self._buffer[self._body_remaining :]
                self._waiting_for_body = False
                self._body_remaining = 0

                if callback is not None:
                    callback()
                transport.write(response)
                if self._close_after_response:
                    transport.close()
                    return
                continue

            # Look for header end
            header_end = self._buffer.find(_HEADER_END)
            if header_end == -1:
                return  # Need more data

            # Extract headers (including \r\n\r\n)
            header_end += 4
            headers = self._buffer[:header_end]

            # Parse Content-Length (case-insensitive)
            content_length = 0
            headers_lower = headers.lower()
            cl_pos = headers_lower.find(_CONTENT_LENGTH_LOWER)
            if cl_pos != -1:
                # Find the value after "content-length:"
                value_start = cl_pos + len(_CONTENT_LENGTH_LOWER)
                value_end = headers.find(b"\r\n", value_start)
                if value_end != -1:
                    content_length = int(headers[value_start:value_end].strip())

            # Check if we have the full request
            request_end = header_end + content_length
            if len(self._buffer) < request_end:
                # Need to wait for body
                self._buffer = self._buffer[header_end:]
                self._waiting_for_body = True
                self._body_remaining = content_length
                return

            # Complete request - consume and respond
            self._buffer = self._buffer[request_end:]

            if callback is not None:
                callback()
            transport.write(response)
            if self._close_after_response:
                transport.close()
                return

    def connection_lost(self, exc: Exception | None) -> None:
        """Called when connection is closed."""
        self._transport = None
        self._buffer = b""


class BareResponseServer:
    """Zero-overhead HTTP server - fires response as soon as request is complete.

    Uses raw Protocol API for minimal overhead (no StreamReader/StreamWriter).
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

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._actual_port or self.port}"

    def _increment_request_count(self) -> None:
        """Callback to increment request count (passed to protocol)."""
        self.request_count += 1

    async def start(self) -> None:
        """Start the server."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _configure_socket(sock)
        sock.bind((self.host, self.port))
        sock.listen(65535)

        loop = asyncio.get_running_loop()
        self._server = await loop.create_server(
            lambda: BareResponseProtocol(self._response, self._increment_request_count),
            sock=sock,
        )
        self._actual_port = sock.getsockname()[1]

    async def stop(self) -> None:
        """Stop the server and cleanup."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> BareResponseServer:
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()


# =============================================================================
# Out-of-process server (isolates CPU load from client)
# =============================================================================


def _server_process_main(
    port_queue: Queue | None,
    stop_event: mp.Event,
    host: str,
    port: int,
    streaming: bool,
    num_chunks: int,
    response_size: int,
) -> None:
    """Entry point for server subprocess.

    Args:
        port_queue: Queue to report assigned port (only first worker uses this)
        stop_event: Event to signal shutdown
        host: Bind address
        port: Port to bind (0 = auto-assign, used by first worker)
        streaming: Use SSE streaming response format
        num_chunks: Number of SSE chunks for streaming mode
        response_size: Size of content in tokens
    """
    import uvloop

    async def run_server():
        server = BareResponseServer(
            host=host,
            port=port,
            streaming=streaming,
            num_chunks=num_chunks,
            response_size=response_size,
        )
        await server.start()

        # First worker reports port, others just start
        if port_queue is not None:
            port_queue.put(server._actual_port)

        # Wait for stop signal
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        await server.stop()

    uvloop.run(run_server())


class BareResponseServerProcess:
    """Out-of-process HTTP server - runs in separate process(es) to isolate CPU load.

    Use this for accurate benchmarking where the server shouldn't compete
    with the client for CPU resources.

    With num_workers > 1, spawns multiple processes that all bind to the same
    port using SO_REUSEPORT. The kernel load-balances connections across workers.

    Args:
        host: Bind address (default: 127.0.0.1)
        streaming: Use SSE streaming response format
        num_chunks: Number of SSE chunks for streaming mode
        response_size: Size of content in tokens (approximated as characters for ASCII)
        num_workers: Number of worker processes (default: 1)
    """

    __slots__ = (
        "host",
        "streaming",
        "num_chunks",
        "response_size",
        "num_workers",
        "_processes",
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
        num_workers: int = 1,
    ):
        self.host = host
        self.streaming = streaming
        self.num_chunks = num_chunks
        self.response_size = response_size
        self.num_workers = max(1, num_workers)
        self._processes: list[SpawnProcess] = []
        self._port: int | None = None
        self._port_queue: Queue | None = None
        self._stop_event: mp.Event | None = None

    @property
    def url(self) -> str:
        if self._port is None:
            raise RuntimeError("Server not started")
        return f"http://{self.host}:{self._port}"

    def start(self) -> None:
        """Start server in separate process(es).

        First worker binds to port 0 (OS assigns), reports port via queue.
        Additional workers bind to the same port using SO_REUSEPORT.
        """
        ctx = mp.get_context("spawn")
        self._port_queue = ctx.Queue()
        self._stop_event = ctx.Event()

        # Start first worker - it picks the port
        first_process = ctx.Process(
            target=_server_process_main,
            args=(
                self._port_queue,
                self._stop_event,
                self.host,
                0,  # port=0, OS assigns
                self.streaming,
                self.num_chunks,
                self.response_size,
            ),
            daemon=True,
        )
        first_process.start()
        self._processes.append(first_process)

        # Wait for port from first worker
        self._port = self._port_queue.get(timeout=10.0)

        # Start additional workers on the same port
        for _ in range(1, self.num_workers):
            process = ctx.Process(
                target=_server_process_main,
                args=(
                    None,  # No port queue needed
                    self._stop_event,
                    self.host,
                    self._port,  # Same port as first worker
                    self.streaming,
                    self.num_chunks,
                    self.response_size,
                ),
                daemon=True,
            )
            process.start()
            self._processes.append(process)

    def stop(self) -> None:
        """Stop all server processes."""
        if self._stop_event:
            self._stop_event.set()

        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        self._processes.clear()
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
