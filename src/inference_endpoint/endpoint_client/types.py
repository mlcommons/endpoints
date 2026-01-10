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

"""Type definitions for the endpoint client module."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

import httptools

from inference_endpoint.endpoint_client.configs import SocketConfig
from inference_endpoint.endpoint_client.timing_context import RequestTimingContext

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP Response Protocol
# =============================================================================


class HttpResponseProtocol(asyncio.Protocol):
    """
    Minimal HTTP/1.1 response protocol using httptools.

    Uses llhttp (same C parser as Node.js) for parsing HTTP responses.
    Designed for connection reuse - call reset() between requests.
    """

    __slots__ = (
        "_transport",
        "_parser",
        "_loop",
        # Response state
        "_status_code",
        "_headers",
        "_body_chunks",
        "_content_length",
        "_is_chunked",
        # Futures for async coordination
        "_headers_future",
        "_body_future",
        # Streaming state: deque+Event pair, guarded by _streaming flag
        "_streaming",
        "_chunk_deque",
        "_chunk_event",
        # Flags
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
        self._content_length: int = -1
        self._is_chunked: bool = False

        # Async coordination
        self._headers_future: asyncio.Future | None = None
        self._body_future: asyncio.Future | None = None
        # Streaming state: deque+Event pair (always set together via iter_body)
        # When _streaming is True, both _chunk_deque and _chunk_event are valid
        self._streaming: bool = False
        self._chunk_deque: deque[bytes | None] = deque()
        self._chunk_event: asyncio.Event = asyncio.Event()

        # Flags
        self._headers_complete: bool = False
        self._message_complete: bool = False
        self._connection_lost: bool = False
        self._exc: Exception | None = None

    def reset(self) -> None:
        """Reset protocol state for connection reuse."""
        self._parser = httptools.HttpResponseParser(self)
        self._status_code = 0
        self._headers.clear()
        self._body_chunks.clear()
        self._content_length = -1
        self._is_chunked = False
        self._headers_future = None
        self._body_future = None
        self._streaming = False
        self._chunk_deque.clear()
        self._chunk_event.clear()
        self._headers_complete = False
        self._message_complete = False
        self._exc = None

    # -------------------------------------------------------------------------
    # asyncio.Protocol callbacks
    # -------------------------------------------------------------------------

    def connection_made(self, transport: asyncio.Transport) -> None:
        self._transport = transport
        self._parser = httptools.HttpResponseParser(self)

    def data_received(self, data: bytes) -> None:
        if self._parser is None:
            return
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
            if exc:
                self._headers_future.set_exception(exc)
            else:
                self._headers_future.set_exception(
                    ConnectionResetError("Connection closed before headers received")
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

        # Signal end of stream for streaming mode
        if self._streaming:
            self._chunk_deque.append(None)
            self._chunk_event.set()

    def eof_received(self) -> bool | None:
        # Return False to close transport, True to keep open
        return False

    # -------------------------------------------------------------------------
    # httptools callbacks
    # -------------------------------------------------------------------------

    def on_status(self, status: bytes) -> None:
        pass  # We get status code from on_headers_complete

    def on_header(self, name: bytes, value: bytes) -> None:
        header_name = name.decode("latin-1").lower()
        header_value = value.decode("latin-1")
        self._headers[header_name] = header_value

        if header_name == "content-length":
            self._content_length = int(header_value)
        elif header_name == "transfer-encoding" and "chunked" in header_value.lower():
            self._is_chunked = True

    def on_headers_complete(self) -> None:
        self._status_code = self._parser.get_status_code()
        self._headers_complete = True

        if self._headers_future and not self._headers_future.done():
            self._headers_future.set_result((self._status_code, self._headers))

    def on_body(self, body: bytes) -> None:
        if self._streaming:
            # Streaming mode - push to deque and signal
            self._chunk_deque.append(body)
            self._chunk_event.set()
        else:
            # Buffered mode
            self._body_chunks.append(body)

    def on_message_complete(self) -> None:
        self._message_complete = True

        if self._body_future and not self._body_future.done():
            self._body_future.set_result(b"".join(self._body_chunks))

        # Signal end of stream for streaming mode
        if self._streaming:
            self._chunk_deque.append(None)
            self._chunk_event.set()

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

    async def iter_body(self) -> AsyncGenerator[bytes, None]:
        """Iterate over body chunks as they arrive."""
        # Enable streaming mode (deque+event already initialized in __init__)
        self._streaming = True

        # Yield any chunks already buffered
        for chunk in self._body_chunks:
            yield chunk
        self._body_chunks.clear()

        # If message already complete (sync parse), exit early
        if self._message_complete:
            return

        # Yield new chunks as they arrive
        while True:
            # Wait for data if deque is empty
            while not self._chunk_deque:
                self._chunk_event.clear()
                await self._chunk_event.wait()
            chunk = self._chunk_deque.popleft()
            if chunk is None:
                break
            yield chunk


# =============================================================================
# Connection Pool
# =============================================================================


class PooledConnection:
    """A pooled TCP connection with its protocol."""

    __slots__ = ("transport", "protocol", "created_at", "last_used", "in_use", "_id")

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
        self._id = id(self)  # Unique identifier for hashing

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PooledConnection):
            return NotImplemented
        return self._id == other._id

    def is_alive(self) -> bool:
        """Check if connection is still usable."""
        return (
            not self.protocol._connection_lost
            and self.transport is not None
            and not self.transport.is_closing()
        )

    def release(self) -> None:
        """Mark connection as available for reuse."""
        self.in_use = False
        self.last_used = time.monotonic()


class ConnectionPool:
    """
    Minimal async connection pool for HTTP/1.1.

    Optimized for single-host usage (all requests to same endpoint).
    Uses LIFO (stack) for connection reuse to favor hot connections.
    """

    __slots__ = (
        "_host",
        "_port",
        "_loop",
        "_socket_config",
        "_idle_stack",
        "_all_connections",
        "_max_connections",
        "_keepalive_timeout",
        "_creating",
    )

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        socket_config: SocketConfig | None = None,
        max_connections: int = 0,
        keepalive_timeout: float = 86400,
    ):
        self._host = host
        self._port = port
        self._loop = loop
        self._socket_config = socket_config
        self._max_connections = max_connections  # 0 = unlimited
        self._keepalive_timeout = keepalive_timeout

        self._idle_stack: list[PooledConnection] = []
        self._all_connections: set[PooledConnection] = set()
        self._creating: int = 0

    async def acquire(self) -> PooledConnection:
        """Get a connection from pool or create new one."""
        # Fast path: reuse from stack (LIFO for hot connections)
        while self._idle_stack:
            conn = self._idle_stack.pop()
            if conn.is_alive():
                conn.in_use = True
                conn.protocol.reset()
                return conn
            else:
                # Dead connection, remove from tracking
                self._all_connections.discard(conn)

        # Slow path: create new connection
        return await self._create_connection()

    async def _create_connection(self) -> PooledConnection:
        """Create a new TCP connection."""
        self._creating += 1
        try:
            # Create protocol instance
            protocol = HttpResponseProtocol(self._loop)

            # Use asyncio's create_connection which handles socket creation properly
            transport, _ = await self._loop.create_connection(
                lambda: protocol,
                host=self._host,
                port=self._port,
            )

            # Apply socket options after connection is established
            if self._socket_config is not None:
                sock = transport.get_extra_info("socket")
                if sock is not None:
                    self._socket_config.apply_to_socket(sock)

            now = time.monotonic()
            conn = PooledConnection(
                transport=transport,
                protocol=protocol,
                created_at=now,
            )
            self._all_connections.add(conn)
            return conn

        finally:
            self._creating -= 1

    def release(self, conn: PooledConnection) -> None:
        """Return connection to pool for reuse."""
        if not conn.is_alive():
            self._all_connections.discard(conn)
            return

        conn.release()
        self._idle_stack.append(conn)

    async def warmup(self, num_connections: int) -> int:
        """Pre-establish connections for warmup."""
        connections: list[PooledConnection] = []

        async def create_one():
            try:
                conn = await self._create_connection()
                connections.append(conn)
            except Exception as e:
                logger.debug(f"Warmup connection failed: {e}")

        await asyncio.gather(
            *[create_one() for _ in range(num_connections)],
            return_exceptions=True,
        )

        # Release all to pool
        for conn in connections:
            self.release(conn)

        return len(self._idle_stack)

    async def close(self) -> None:
        """Close all connections."""
        for conn in list(self._all_connections):
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


# =============================================================================
# HTTP Request Template
# =============================================================================


@dataclass(slots=True)
class HttpRequestTemplate:
    """
    Pre-computed HTTP/1.1 request parts for direct socket writes.

    These are computed once from the endpoint URL and reused for all requests,
    avoiding repeated string formatting and encoding overhead.

    Attributes:
        request_line: HTTP request line bytes (e.g., b"POST /v1/chat HTTP/1.1\\r\\n")
        host_header: HTTP Host header bytes, required for HTTP/1.1
    """

    request_line: bytes
    host_header: bytes

    @classmethod
    def from_url(cls, host: str, port: int, path: str) -> HttpRequestTemplate:
        """
        Create an HttpRequestTemplate from URL components.

        Pre-computes static HTTP/1.1 request components that remain constant
        across all requests.

        Args:
            host: Target hostname
            port: Target port
            path: Request path (e.g., "/v1/chat/completions")

        Returns:
            HttpRequestTemplate ready for building requests
        """
        request_line = f"POST {path} HTTP/1.1\r\n".encode("ascii")

        # Host header is mandatory in HTTP/1.1 (RFC 7230 Section 5.4)
        # Port is omitted for default ports (80 for HTTP, 443 for HTTPS)
        if port in (80, 443):
            host_header = f"Host: {host}\r\n".encode("ascii")
        else:
            host_header = f"Host: {host}:{port}\r\n".encode("ascii")

        return cls(request_line=request_line, host_header=host_header)

    def build_request(self, body: bytes, headers: dict[str, str]) -> bytes:
        """
        Build a complete HTTP/1.1 request as raw bytes.

        Constructs the wire format per RFC 7230:
            request-line CRLF
            *(header-field CRLF)
            CRLF
            message-body

        Args:
            body: Request body bytes (JSON payload)
            headers: Additional headers from the query (e.g., Content-Type, Authorization)

        Returns:
            Complete HTTP request ready for socket.write()
        """
        parts = [self.request_line, self.host_header]

        # Append additional headers from query
        for key, value in headers.items():
            parts.append(f"{key}: {value}\r\n".encode("latin-1"))

        # Content-Length is required for requests with a body (RFC 7230 Section 3.3.2)
        parts.append(b"Content-Length: ")
        parts.append(str(len(body)).encode("ascii"))
        parts.append(b"\r\n")

        # Empty line marks end of headers, followed by body
        parts.append(b"\r\n")
        parts.append(body)
        return b"".join(parts)


# =============================================================================
# Prepared Request
# =============================================================================


@dataclass(slots=True)
class PreparedRequest:
    """A prepared HTTP request ready to be sent.

    Attributes:
        query_id: Unique identifier for the request (matches Query.id).
        http_bytes: Pre-built HTTP request bytes ready for socket.write().
        timing: RequestTimingContext for overhead measurement.
        is_streaming: True if this is a streaming (SSE) request.
        process: Callback to handle the response (set after preparation).
        connection: PooledConnection (set after connection acquired).
    """

    query_id: str
    http_bytes: bytes
    timing: RequestTimingContext
    is_streaming: bool
    process: Callable[[], Any] | None = None
    connection: PooledConnection | None = field(default=None, repr=False)
