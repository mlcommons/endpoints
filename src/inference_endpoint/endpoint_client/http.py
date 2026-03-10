# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""HTTP transport types for the endpoint client module."""

from __future__ import annotations

import asyncio
import logging
import select
import socket
import ssl
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import httptools

logger = logging.getLogger(__name__)


class _SocketConfig:
    """
    Socket configuration for TCP connections.
    Applied to all sockets created by ConnectionPool managed by http.py APIS.
    Optimized for low-latency streaming workloads.
    """

    # Nagle's algorithm batches small packets to improve network efficiency
    # TCP_NODELAY disables Nagle's algorithm lower latency in both directions
    # Causes increased CPU usage due to more packets being sent
    TCP_NODELAY: int = 1

    # Quick ACK mode (Linux-specific)
    # Forces immediate acknowledgment of received packets
    # instead of the default delayed ACK behavior.
    TCP_QUICKACK: int = 1

    # Connection keepalive-probe settings for long-lived connections
    # client kernel sends probe, server's kernel ACKs - no application overhead
    SO_KEEPALIVE: int = 1  # Enable keepalive at socket level
    TCP_KEEPIDLE: int = 1  # Probe after 1s idle
    TCP_KEEPCNT: int = 1  # 1 failed probe = dead
    TCP_KEEPINTVL: int = 1  # 1s between probes

    # Make sure socket buffers are never the bottle neck
    # With HTTP/1.1, a TCP socket will only be used for a single request
    # Largest message size would be server response in Offline Mode
    # 4MB /4 bytes per token = 1M tokens in any given packet
    # TODO(vir): analyze workloads to better tune buffer sizes
    SO_RCVBUF: int = 1024 * 1024 * 4  # 4MB receive buffer
    SO_SNDBUF: int = 1024 * 1024 * 4  # 4MB send buffer

    # Linux-specific:
    # kernel closes socket if sent data not ACKed within timeout
    # ie. timeout on unACKed sent data
    TCP_USER_TIMEOUT: int = 0

    @classmethod
    def apply(cls, sock: socket.socket) -> None:
        """Apply configuration to the given socket."""
        # Low-latency optimizations for streaming
        sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, cls.TCP_NODELAY)

        # Connection keepalive settings for long-lived SSE connections
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, cls.SO_KEEPALIVE)

        # Fine-tune keepalive timing
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, cls.TCP_KEEPIDLE)
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, cls.TCP_KEEPINTVL)
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, cls.TCP_KEEPCNT)

        # Buffer size optimizations for streaming
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, cls.SO_RCVBUF)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, cls.SO_SNDBUF)

        # Enable Quick ACK mode
        if hasattr(socket, "TCP_QUICKACK"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, cls.TCP_QUICKACK)

        # Set idle connection timeout
        if hasattr(socket, "TCP_USER_TIMEOUT"):
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_USER_TIMEOUT, cls.TCP_USER_TIMEOUT
            )


class HttpResponseProtocol(asyncio.Protocol):
    """
    Minimal HTTP/1.1 response protocol using httptools.

    Uses llhttp (same C parser as Node.js) for parsing HTTP responses.
    Designed for connection reuse - call reset() between requests.
    """

    __slots__ = (
        "_loop",
        "_transport",
        "_parser",
        "_status_code",
        "_headers",
        "_body_chunks",
        "_should_close",
        "_headers_future",
        "_body_future",
        "_streaming",
        "_stream_chunks",
        "_stream_event",
        "_headers_complete",
        "_message_complete",
        "_connection_lost",
        "_exc",
    )

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._transport: asyncio.Transport | None = None
        self._parser: httptools.HttpResponseParser | None = None

        # Response state
        self._status_code: int = 0
        self._headers: dict[str, str] = {}
        self._body_chunks: list[bytes] = []
        self._should_close: bool = False

        # Async coordination
        self._headers_future: asyncio.Future | None = None
        self._body_future: asyncio.Future | None = None

        # Streaming state
        self._streaming: bool = False
        self._stream_chunks: list[bytes] = []
        self._stream_event: asyncio.Event = asyncio.Event()

        # Flags
        self._headers_complete: bool = False
        self._message_complete: bool = False
        self._connection_lost: bool = False
        self._exc: Exception | None = None

    def reset(self) -> None:
        """Reset protocol state for connection reuse."""
        # Lazy parser creation - will be created on first data_received()
        self._parser = None
        self._status_code = 0
        self._headers.clear()
        self._body_chunks.clear()
        self._should_close = False
        self._headers_future = None
        self._body_future = None
        self._streaming = False
        self._stream_chunks = []
        self._stream_event.clear()
        self._headers_complete = False
        self._message_complete = False
        self._exc = None
        # NOTE: Don't reset _connection_lost - that's transport state

    def _signal_stream_end(self) -> None:
        """Signal end of stream for streaming mode."""
        if self._streaming:
            self._stream_event.set()

    # -------------------------------------------------------------------------
    # asyncio.Protocol callbacks
    # -------------------------------------------------------------------------

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        """Called by asyncio when connection is established.

        Note: We intentionally narrow the transport type from BaseTransport to Transport
        for better type safety, as we know we're using TCP transports with specific features.
        """
        self._transport = transport
        self._parser = httptools.HttpResponseParser(self)

    def data_received(self, data: bytes) -> None:
        # Lazy parser creation for better reset() performance
        if self._parser is None:
            self._parser = httptools.HttpResponseParser(self)
        try:
            self._parser.feed_data(data)
        except httptools.HttpParserError as e:
            self._exc = e
            if self._headers_future and not self._headers_future.done():
                self._headers_future.set_exception(e)
            if self._body_future and not self._body_future.done():
                self._body_future.set_exception(e)

    def connection_lost(self, exc: Exception | None) -> None:
        self._connection_lost = True
        self._exc = exc

        # Complete any pending futures
        if self._headers_future and not self._headers_future.done():
            self._headers_future.set_exception(
                exc or ConnectionResetError("Connection closed before headers received")
            )

        if self._body_future and not self._body_future.done():
            if exc:
                self._body_future.set_exception(exc)
            elif not self._message_complete:
                self._body_future.set_exception(
                    ConnectionResetError("Connection closed before body complete")
                )
            else:
                self._body_future.set_result(b"".join(self._body_chunks))

        self._signal_stream_end()

    def eof_received(self) -> bool | None:
        """Handle server EOF (FIN packet).

        CRITICAL:
        Must mark connection as lost to prevent reuse of half-closed sockets.

        TCP half-closed behavior:
        - Server sends FIN → client receives EOF
        - Client can STILL WRITE (client→server direction still open)
        - But server won't respond (server→client closed)
        - If we reuse this connection: write succeeds, read HANGS FOREVER

        See: https://bugs.python.org/issue44805 (asyncio EOF detection on reused sockets)
        See: https://superuser.com/questions/298919/tcp-half-open-vs-half-closed
        """
        self._connection_lost = True
        self._signal_stream_end()  # Unblock any waiting iter_body()
        # Return True to keep transport open briefly for pending data processing
        return True

    # -------------------------------------------------------------------------
    # httptools callbacks
    # -------------------------------------------------------------------------

    def on_status(self, status: bytes) -> None:
        pass  # We get status code from on_headers_complete

    def on_header(self, name: bytes, value: bytes) -> None:
        # RFC 9110: Header names are ASCII tokens, values use UTF-8 with surrogateescape
        # for safe round-tripping of non-UTF-8 bytes (same as aiohttp)
        self._headers[name.decode("ascii").lower()] = value.decode(
            "utf-8", "surrogateescape"
        )

    def on_headers_complete(self) -> None:
        # Parser is always set when this callback is invoked by httptools
        assert self._parser is not None
        self._status_code = self._parser.get_status_code()
        # Check if server wants to close connection (Connection: close or HTTP/1.0)
        self._should_close = not self._parser.should_keep_alive()
        self._headers_complete = True

        if self._headers_future and not self._headers_future.done():
            self._headers_future.set_result((self._status_code, self._headers))

    def on_body(self, body: bytes) -> None:
        if self._streaming:
            # Streaming mode - append and signal
            self._stream_chunks.append(body)
            self._stream_event.set()
        else:
            # Buffered mode
            self._body_chunks.append(body)

    def on_message_complete(self) -> None:
        self._message_complete = True

        if self._body_future and not self._body_future.done():
            self._body_future.set_result(b"".join(self._body_chunks))

        self._signal_stream_end()

    def on_chunk_header(self) -> None:
        pass

    def on_chunk_complete(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def transport(self) -> asyncio.Transport | None:
        return self._transport

    @property
    def should_close(self) -> bool:
        """Whether connection should be closed after this response."""
        return self._should_close or self._connection_lost or self._exc is not None

    def write(self, data: bytes) -> None:
        """Write data to transport."""
        if self._transport:
            self._transport.write(data)

    async def read_headers(self) -> tuple[int, dict[str, str]]:
        """Wait for and return (status_code, headers)."""
        if self._headers_complete:
            return (self._status_code, self._headers)

        self._headers_future = self._loop.create_future()
        return await self._headers_future

    async def read_body(self) -> bytes:
        """Read entire response body."""
        if self._message_complete:
            return b"".join(self._body_chunks)

        self._body_future = self._loop.create_future()
        return await self._body_future

    async def iter_body(self) -> AsyncGenerator[list[bytes], None]:
        """
        Iterate over body chunks as they arrive.
        Yields list[bytes] containing all chunks available at each iteration.

        Will drain all available data synchronously before awaiting,
        to reduce event loop yields when data arrives faster than processing.
        """
        self._streaming = True

        # Yield any pre-buffered chunks
        if self._body_chunks:
            yield self._body_chunks
            self._body_chunks = []

        # If message already complete (sync parse), exit early
        if self._message_complete:
            return

        # Main streaming loop
        while True:
            # Sync drain: yield all available chunks without awaiting
            while self._stream_chunks:
                yield self._stream_chunks
                self._stream_chunks = []

            # Check termination after sync drain
            if self._message_complete or self._connection_lost:
                return

            # Only await when no data available
            self._stream_event.clear()
            await self._stream_event.wait()


class PooledConnection:
    """A pooled TCP connection with its protocol."""

    __slots__ = (
        "transport",
        "protocol",
        "created_at",
        "last_used",
        "in_use",
        "idle_time_on_acquire",
        "_fd",
        "_stale_poller",
    )

    def __init__(
        self,
        transport: asyncio.Transport,
        protocol: HttpResponseProtocol,
        created_at: float,
    ):
        self.transport = transport
        self.protocol = protocol
        self.created_at = created_at
        self.last_used = created_at
        self.in_use = True
        self.idle_time_on_acquire = 0.0

        # Cache fd for stale checks — stable for the lifetime of the connection
        sock = transport.get_extra_info("socket")
        self._fd: int = sock.fileno() if sock is not None else -1
        self._stale_poller: select.poll | None = None

    def is_alive(self) -> bool:
        """Check if the connection is still usable.

        Returns False if:
        - Transport is None (never connected or already cleaned up)
        - Protocol received EOF (server sent FIN - half-closed)
        - Transport is closing (local close initiated)
        """
        return (
            self.transport is not None
            and not self.protocol._connection_lost
            and not self.transport.is_closing()
        )

    def is_stale(self) -> bool:
        """Check if idle connection was closed by server (non-blocking probe).

        For idle HTTP keep-alive connections, there should be no pending data.
        If the socket is readable, it means the server sent FIN (EOF).

        Uses poll() instead of select() to avoid FD_SETSIZE limit on high fds.
        Poller is created lazily on first call and reused (fd is stable per connection).
        """
        # Skip stale check for recently-used connections
        # Server unlikely to close within 1 second of last use
        if time.monotonic() - self.last_used < 1.0:
            return False

        # Fast path: poller already registered from a previous call
        if self._stale_poller is not None:
            try:
                return bool(self._stale_poller.poll(0))
            except (OSError, ValueError):
                # fd closed or invalid — connection is dead, treat as stale
                return True

        # Slow path: first call — create poller and register fd
        if self._fd < 0:
            return True

        try:
            poller = select.poll()
            poller.register(self._fd, select.POLLIN | select.POLLERR | select.POLLHUP)
            self._stale_poller = poller
            return bool(poller.poll(0))

        except (OSError, ValueError):
            # fd closed or invalid — connection is dead, treat as stale
            return True


class ConnectionPool:
    """
    Connection pool for HTTP/1.1 connections with automatic limiting.

    Uses a LIFO stack for idle connections (to reuse hot connections).
    Uses OrderedDict for FIFO waiter queue when max_connections is reached.
    """

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        max_connections: int | None = None,  # None means no limit
        max_idle_time: float = 4.0,  # Discard connections idle longer than this
        ssl_context: ssl.SSLContext | None = None,
    ):
        self._host = host
        self._port = port
        self._loop = loop
        self._max_connections = max_connections
        self._max_idle_time = max_idle_time
        self._ssl_context = ssl_context
        # Connection tracking
        self._idle_stack: list[PooledConnection] = []
        self._all_connections: set[PooledConnection] = set()
        self._creating: int = 0

        # FIFO waiter queue using OrderedDict (O(1) operations)
        self._waiters: OrderedDict[asyncio.Future[None], None] = OrderedDict()

    def _try_get_idle(self) -> PooledConnection | None:
        """Try to get a usable idle connection, cleaning up dead ones."""
        now = time.monotonic()
        while self._idle_stack:
            conn = self._idle_stack.pop()
            # Proactively discard connections idle longer than max_idle_time
            # to avoid server keep-alive timeout races
            idle_time = now - conn.last_used
            if idle_time > self._max_idle_time:
                self._close_connection(conn)
                continue
            if conn.is_alive() and not conn.is_stale():
                conn.in_use = True
                conn.idle_time_on_acquire = idle_time
                conn.protocol.reset()
                return conn
            # Dead or stale connection, close and remove from tracking
            self._close_connection(conn)
        return None

    def _close_connection(self, conn: PooledConnection) -> None:
        """Close a connection and remove from tracking."""
        if conn.transport and not conn.transport.is_closing():
            conn.transport.close()
        self._all_connections.discard(conn)

    async def acquire(self) -> PooledConnection:
        """Acquire a connection from the pool, waiting if necessary."""
        # Fast path: try to get an idle connection
        if conn := self._try_get_idle():
            return conn

        # Check if we can create a new connection
        if self._can_create_connection():
            return await self._create_connection()

        # Must wait for a connection to become available
        return await self._wait_for_connection()

    def _can_create_connection(self) -> bool:
        """Check if we're allowed to create a new connection."""
        if self._max_connections is None:
            return True
        return len(self._all_connections) + self._creating < self._max_connections

    async def _wait_for_connection(self) -> PooledConnection:
        """Wait for an available connection slot."""
        while True:
            fut: asyncio.Future[None] = self._loop.create_future()
            self._waiters[fut] = None

            try:
                await fut
            finally:
                self._waiters.pop(fut, None)

            # Try to get an idle connection first
            if conn := self._try_get_idle():
                return conn

            # If slot available, create new
            if self._can_create_connection():
                return await self._create_connection()

            # Otherwise loop and wait again

    async def _create_connection(self) -> PooledConnection:
        """Create a new TCP connection."""
        self._creating += 1
        try:
            # Create protocol factory
            def protocol_factory() -> HttpResponseProtocol:
                return HttpResponseProtocol(self._loop)

            # Create connection without timeout
            transport, protocol = await self._loop.create_connection(
                protocol_factory,
                host=self._host,
                port=self._port,
                ssl=self._ssl_context,
            )

            # Apply/Override socket defaults
            if sock := transport.get_extra_info("socket"):
                _SocketConfig.apply(sock)

            conn = PooledConnection(
                transport=transport,
                protocol=protocol,
                created_at=time.monotonic(),
            )
            self._all_connections.add(conn)
            return conn

        finally:
            self._creating -= 1

    def release(self, conn: PooledConnection) -> None:
        """Return connection to pool for reuse and notify waiters (idempotent)."""
        if not conn.in_use:
            return

        # Must close if: dead, server requested close, or error occurred
        if not conn.is_alive() or conn.protocol.should_close:
            self._close_connection(conn)
            self._notify_waiter()
            return

        conn.in_use = False
        conn.last_used = time.monotonic()
        conn.protocol.reset()
        self._idle_stack.append(conn)
        self._notify_waiter()

    def _notify_waiter(self) -> None:
        """Wake up the first waiting acquirer (FIFO order)."""
        while self._waiters:
            waiter, _ = self._waiters.popitem(last=False)
            if not waiter.done():
                waiter.set_result(None)
                return

    async def warmup(self, count: int | None = None) -> int:
        """Pre-establish connections for warmup.

        Args:
            count: Number of connections to create. Defaults to max_connections.

        Returns:
            Number of connections successfully warmed up.
        """
        count = count if count is not None else self._max_connections
        if count is None:
            return 0
        if self._max_connections is not None and count > self._max_connections:
            raise ValueError(
                f"Cannot warmup more than max_connections "
                f"(requested: {count}, max: {self._max_connections})"
            )

        connections: list[PooledConnection] = []

        async def create_one() -> None:
            conn = await self.acquire()
            connections.append(conn)

        # ignore individual warmup exceptions
        _ = await asyncio.gather(
            *[create_one() for _ in range(count)],
            return_exceptions=True,
        )

        for conn in connections:
            self.release(conn)

        return len(self._idle_stack)

    async def close(self) -> None:
        """Close all connections and cancel pending waiters."""
        # Cancel all waiters
        for waiter in self._waiters:
            if not waiter.done():
                waiter.cancel()
        self._waiters.clear()

        # Close all connections
        for conn in self._all_connections:
            if conn.transport and not conn.transport.is_closing():
                conn.transport.close()
        self._all_connections.clear()
        self._idle_stack.clear()

    @property
    def idle_count(self) -> int:
        return len(self._idle_stack)

    @property
    def total_count(self) -> int:
        return len(self._all_connections)

    @property
    def in_use_count(self) -> int:
        return sum(1 for c in self._all_connections if c.in_use)

    @property
    def waiting_count(self) -> int:
        return len(self._waiters)


class HttpRequestTemplate:
    """HTTP/1.1 request template for an endpoint.

    Encapsulates the static portions of an HTTP request (request line, host header)
    that remain constant across requests to a given endpoint.

    Attributes:
        static_prefix: Pre-merged request line + host header bytes
    """

    __slots__ = ("static_prefix", "cached_headers")

    # Pre-encoded general headers
    HEADERS_STREAMING = (
        b"Content-Type: application/json\r\nAccept: text/event-stream\r\n"
    )
    HEADERS_NON_STREAMING = (
        b"Content-Type: application/json\r\nAccept: application/json\r\n"
    )

    def __init__(self, static_prefix: bytes):
        self.static_prefix = static_prefix
        self.cached_headers = b""

    @classmethod
    def from_url(cls, host: str, port: int, path: str = "/") -> HttpRequestTemplate:
        """
        Create an HttpRequestTemplate from URL components.

        Args:
            host: Target hostname
            port: Target port
            path: Request path (e.g., "/v1/chat/completions")

        Returns:
            HttpRequestTemplate ready for building requests
        """
        # Normalize empty path to "/" per HTTP/1.1 (RFC 7230)
        path = path or "/"
        request_line = f"POST {path} HTTP/1.1\r\n"

        # Host header is mandatory in HTTP/1.1 (RFC 7230 Section 5.4)
        # Port is omitted for default ports (80 for HTTP, 443 for HTTPS)
        if port in (80, 443):
            host_header = f"Host: {host}\r\n"
        else:
            host_header = f"Host: {host}:{port}\r\n"

        return cls(static_prefix=(request_line + host_header).encode("ascii"))

    def cache_headers(self, headers: dict[str, str]) -> None:
        """
        Pre-cache extra headers to avoid first-call encoding overhead.

        Call this during setup to prime the cache for headers that will be
        used repeatedly at runtime.

        Args:
            headers: Headers to pre-encode and cache
        """
        encoded = "".join(f"{k}: {v}\r\n" for k, v in headers.items()).encode(
            "utf-8", "surrogateescape"
        )
        if encoded not in self.cached_headers:
            self.cached_headers += encoded

    def build_request(
        self,
        body: bytes,
        streaming: bool,
        extra_headers: dict[str, str] | None = None,
    ) -> bytes:
        """
        Build a complete HTTP/1.1 request as raw bytes.

        Args:
            body: Request body bytes (JSON payload)
            streaming: True for SSE streaming, False for buffered response
            extra_headers: Optional additional headers (e.g., Authorization)

        Returns:
            Complete HTTP request in bytes.
        """
        content_type_headers = (
            self.HEADERS_STREAMING if streaming else self.HEADERS_NON_STREAMING
        )
        content_length = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")

        # Fast path: no extra headers
        if not extra_headers:
            return b"".join(
                [
                    self.static_prefix,
                    self.cached_headers,
                    content_type_headers,
                    content_length,
                    body,
                ]
            )

        # Slow path: extra headers are encoded per-call;
        # use cache_headers() at setup time for headers that repeat every request.
        extra = "".join(f"{k}: {v}\r\n" for k, v in extra_headers.items()).encode(
            "utf-8", "surrogateescape"
        )
        return b"".join(
            [
                self.static_prefix,
                self.cached_headers,
                content_type_headers,
                extra,
                content_length,
                body,
            ]
        )


@dataclass(slots=True)
class InFlightRequest:
    """State for a single HTTP request through its lifecycle:

    Attributes:
        query_id: Correlates response back to original Query.
        http_bytes: Serialized HTTP request for socket.write().
        is_streaming: Whether this is a streaming (SSE) request or not.
        connection: PooledConnection assigned to this request (set once request is fired).
    """

    query_id: str
    http_bytes: bytes
    is_streaming: bool
    connection: PooledConnection = field(default=None, repr=False)  # type: ignore[assignment]
