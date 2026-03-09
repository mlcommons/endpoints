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

"""Tests for HTTP API implementation."""

import asyncio
from urllib.parse import urlparse

import httptools
import pytest
import pytest_asyncio
from inference_endpoint.endpoint_client.http import (
    ConnectionPool,
    HttpRequestTemplate,
    HttpResponseProtocol,
)
from inference_endpoint.testing.echo_server import EchoServer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def echo_server():
    """Start EchoServer for integration tests (module-scoped for efficiency)."""
    server = EchoServer(host="127.0.0.1", port=0)
    server.start()
    yield server
    server.stop()


@pytest_asyncio.fixture
async def pool(echo_server):
    """Create ConnectionPool connected to echo_server, auto-cleanup on exit."""
    loop = asyncio.get_running_loop()
    parsed = urlparse(echo_server.url)
    pool = ConnectionPool(
        host=parsed.hostname,
        port=parsed.port,
        loop=loop,
        max_connections=4,
    )
    yield pool
    await pool.close()


@pytest.fixture
def template(echo_server):
    """Create HttpRequestTemplate for echo_server."""
    parsed = urlparse(echo_server.url)
    return HttpRequestTemplate.from_url(
        parsed.hostname, parsed.port, "/v1/chat/completions"
    )


# =============================================================================
# HttpRequestTemplate Tests
# =============================================================================


class TestHttpRequestTemplate:
    """Tests for HTTP request building."""

    def test_builds_valid_http_request(self):
        """Built request parses successfully with httptools."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "/v1/chat")
        body = b'{"model": "test", "messages": []}'

        request = template.build_request(body, streaming=False)

        # Parse with httptools (same parser as Node.js)
        parser_state = {"method": None, "url": None, "headers": {}, "body": b""}

        class Handler:
            def on_url(self, url):
                parser_state["url"] = url

            def on_header(self, name, value):
                parser_state["headers"][name.decode().lower()] = value.decode()

            def on_body(self, body):
                parser_state["body"] += body

        parser = httptools.HttpRequestParser(Handler())
        parser.feed_data(request)
        parser_state["method"] = parser.get_method()

        assert parser_state["method"] == b"POST"
        assert parser_state["url"] == b"/v1/chat"
        assert parser_state["headers"]["host"] == "localhost:8080"
        assert parser_state["headers"]["content-length"] == str(len(body))
        assert parser_state["body"] == body

    def test_host_header_omits_default_ports(self):
        """Ports 80/443 omitted from Host header per RFC 7230."""
        template_80 = HttpRequestTemplate.from_url("example.com", 80, "/")
        template_443 = HttpRequestTemplate.from_url("example.com", 443, "/")

        assert b"Host: example.com\r\n" in template_80.static_prefix
        assert b"Host: example.com\r\n" in template_443.static_prefix
        # Should NOT include port
        assert b":80" not in template_80.static_prefix
        assert b":443" not in template_443.static_prefix

    def test_empty_path_normalized_to_root(self):
        """Empty path normalized to '/' per HTTP/1.1."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "")

        assert b"POST / HTTP/1.1" in template.static_prefix

    def test_build_request_with_extra_headers(self):
        """Extra headers are included in built request."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "/v1/chat")
        body = b'{"model": "test"}'
        extra_headers = {
            "Authorization": "Bearer test-token-123",
            "X-Custom-Header": "custom-value",
        }

        request = template.build_request(
            body, streaming=False, extra_headers=extra_headers
        )

        # Parse with httptools to verify headers
        parser_state = {"headers": {}}

        class Handler:
            def on_url(self, url):
                pass

            def on_header(self, name, value):
                parser_state["headers"][name.decode().lower()] = value.decode()

            def on_body(self, body):
                pass

        parser = httptools.HttpRequestParser(Handler())
        parser.feed_data(request)

        assert parser_state["headers"]["authorization"] == "Bearer test-token-123"
        assert parser_state["headers"]["x-custom-header"] == "custom-value"
        # Standard headers should still be present
        assert parser_state["headers"]["host"] == "localhost:8080"
        assert parser_state["headers"]["content-type"] == "application/json"

    def test_extra_headers_are_cached(self):
        """Extra headers are cached after first use."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "/v1/chat")
        body = b'{"model": "test"}'
        extra_headers = {"Authorization": "Bearer token"}

        # Cache should be empty initially
        assert len(template._extra_headers_cache) == 0

        # First call should cache
        template.build_request(body, streaming=False, extra_headers=extra_headers)
        assert len(template._extra_headers_cache) == 1

        # Second call with same headers should reuse cache (no new entries)
        template.build_request(body, streaming=False, extra_headers=extra_headers)
        assert len(template._extra_headers_cache) == 1

        # Different headers should create new cache entry
        different_headers = {"Authorization": "Bearer different-token"}
        template.build_request(body, streaming=False, extra_headers=different_headers)
        assert len(template._extra_headers_cache) == 2

    def test_cache_headers_pre_caches(self):
        """cache_headers() pre-caches headers before runtime use."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "/v1/chat")
        headers_to_cache = {"Authorization": "Bearer pre-cached-token"}

        # Pre-cache headers
        template.cache_headers(headers_to_cache)
        assert len(template._extra_headers_cache) == 1

        # Verify cached value is correct encoding
        cache_key = frozenset(headers_to_cache.items())
        cached_bytes = template._extra_headers_cache[cache_key]
        assert b"Authorization: Bearer pre-cached-token\r\n" == cached_bytes

    def test_build_request_with_cache_headers_pre_caches(self):
        """build_request() with cache_headers() pre-caches headers before runtime use."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "/v1/chat")
        headers_to_cache = {"Authorization": "Bearer pre-cached-token"}
        body = b'{"model": "test"}'

        # Pre-cache headers
        template.cache_headers(headers_to_cache)
        request = template.build_request(body, streaming=False)
        assert b"Authorization: Bearer pre-cached-token\r\n" in request
        assert b"Host: localhost:8080" in request
        assert b"Content-Type: application/json" in request
        assert b"Content-Length: 17" in request
        assert b'{"model": "test"}' in request

    def test_build_request_without_extra_headers(self):
        """Request without extra headers uses fast path."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "/v1/chat")
        body = b'{"model": "test"}'

        # Build without extra headers
        request = template.build_request(body, streaming=False)

        # Cache should remain empty (fast path)
        assert len(template._extra_headers_cache) == 0

        # Verify request is valid
        assert b"POST /v1/chat HTTP/1.1" in request
        assert b"Host: localhost:8080" in request


# =============================================================================
# HttpResponseProtocol Tests
# =============================================================================


class TestHttpResponseProtocol:
    """Tests for HTTP response parsing."""

    def test_parses_200_response_with_body(self):
        """Parses standard HTTP 200 response."""
        loop = asyncio.new_event_loop()
        protocol = HttpResponseProtocol(loop)

        body = b'{"ok": true}'
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"\r\n" + body
        )

        # Simulate connection and data
        protocol.connection_made(MockTransport())
        protocol.data_received(response)

        assert protocol._status_code == 200
        assert protocol._headers["content-type"] == "application/json"
        assert protocol._headers["content-length"] == str(len(body))
        assert protocol._message_complete
        assert b"".join(protocol._body_chunks) == body
        assert protocol.transport is not None

        loop.close()

    def test_parses_chunked_transfer_encoding(self):
        """Parses chunked transfer encoding (used by SSE)."""
        loop = asyncio.new_event_loop()
        protocol = HttpResponseProtocol(loop)

        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"5\r\nhello\r\n"
            b"6\r\n world\r\n"
            b"0\r\n\r\n"
        )

        protocol.connection_made(MockTransport())
        protocol.data_received(response)

        assert protocol._message_complete
        assert b"".join(protocol._body_chunks) == b"hello world"

        loop.close()

    @pytest.mark.asyncio
    async def test_iter_body_yields_chunks(self):
        """iter_body yields batches of chunks as they arrive."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())

        # Send headers
        protocol.data_received(
            b"HTTP/1.1 200 OK\r\n" b"Transfer-Encoding: chunked\r\n" b"\r\n"
        )

        chunk_batches = []

        async def consume():
            async for chunk_batch in protocol.iter_body():
                chunk_batches.append(chunk_batch)

        # Start consumer
        consumer_task = asyncio.create_task(consume())

        # Feed chunks with small delays
        await asyncio.sleep(0.01)
        protocol.data_received(b"5\r\nhello\r\n")

        await asyncio.sleep(0.01)
        protocol.data_received(b"6\r\n world\r\n")

        await asyncio.sleep(0.01)
        protocol.data_received(b"0\r\n\r\n")

        await consumer_task

        # iter_body yields batches, flatten to check content
        all_chunks = [chunk for batch in chunk_batches for chunk in batch]
        assert b"hello" in all_chunks
        assert b" world" in all_chunks

    def test_reset_clears_state_for_reuse(self):
        """reset() clears state for connection reuse."""
        loop = asyncio.new_event_loop()
        protocol = HttpResponseProtocol(loop)

        # First response
        protocol.connection_made(MockTransport())
        protocol.data_received(
            b"HTTP/1.1 200 OK\r\n" b"Content-Length: 4\r\n" b"\r\n" b"test"
        )

        assert protocol._message_complete
        assert protocol._status_code == 200

        # Reset for next request
        protocol.reset()

        assert not protocol._message_complete
        assert protocol._status_code == 0
        assert protocol._headers == {}
        assert protocol._body_chunks == []

        loop.close()

    @pytest.mark.asyncio
    async def test_connection_lost_propagates_error(self):
        """connection_lost completes pending futures with error."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())

        # Start waiting for headers (won't complete)
        headers_task = asyncio.create_task(protocol.read_headers())

        # Let the task start waiting
        await asyncio.sleep(0.01)

        # Simulate connection lost before headers complete
        protocol.connection_lost(None)

        # Future should raise
        with pytest.raises(ConnectionResetError):
            await headers_task

        # Verify eof_received also marks connection as lost
        protocol2 = HttpResponseProtocol(loop)
        protocol2.connection_made(MockTransport())
        assert not protocol2._connection_lost
        result = protocol2.eof_received()
        assert protocol2._connection_lost is True
        assert result is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_error",
        [
            (OSError("broken pipe"), OSError),
            (None, ConnectionResetError),
        ],
        ids=["with_exception", "clean_close"],
    )
    async def test_connection_lost_before_body_complete(self, exc, expected_error):
        """connection_lost propagates errors to pending body_future."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())

        protocol.data_received(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")

        if exc is None:
            protocol.data_received(b"partial")  # partial body for clean close case

        body_task = asyncio.create_task(protocol.read_body())
        await asyncio.sleep(0)

        protocol.connection_lost(exc)

        with pytest.raises(expected_error):
            await body_task

    @pytest.mark.asyncio
    async def test_connection_lost_after_message_complete(self):
        """connection_lost(None) after message_complete resolves body future normally."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())

        body_content = b'{"ok": true}'
        # Send full response (headers + complete body)
        protocol.data_received(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: " + str(len(body_content)).encode() + b"\r\n"
            b"\r\n" + body_content
        )

        # Start read_body task
        body_task = asyncio.create_task(protocol.read_body())
        await asyncio.sleep(0.01)

        # Connection lost after message already complete
        protocol.connection_lost(None)

        result = await body_task
        assert result == body_content

    @pytest.mark.asyncio
    async def test_connection_lost_message_complete_body_future_pending(self):
        """connection_lost resolves body_future when message was already complete."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())
        # Manually set up state: message complete, body has data, but body_future is pending
        protocol._message_complete = True
        protocol._body_chunks = [b"test data"]
        protocol._body_future = loop.create_future()
        protocol.connection_lost(None)
        result = await protocol._body_future
        assert result == b"test data"

    @pytest.mark.asyncio
    async def test_data_received_parser_error(self):
        """Garbage data causes HttpParserError on pending headers future."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())

        # Start waiting for headers
        headers_task = asyncio.create_task(protocol.read_headers())
        await asyncio.sleep(0.01)

        # Feed garbage data that is not valid HTTP
        protocol.data_received(b"NOT HTTP AT ALL")

        with pytest.raises(httptools.HttpParserError):
            await headers_task

    @pytest.mark.asyncio
    async def test_data_received_parser_error_with_body_future(self):
        """Parser error propagates to body_future when headers already received."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())
        # Use chunked encoding so parser validates chunk framing
        protocol.data_received(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        # Start read_body (creates _body_future)
        body_task = asyncio.create_task(protocol.read_body())
        await asyncio.sleep(0)
        # Feed invalid chunk framing — triggers HttpParserError on body_future
        protocol.data_received(b"ZZZZ\r\nNOT A VALID CHUNK\r\n")
        with pytest.raises(httptools.HttpParserError):
            await body_task

    @pytest.mark.asyncio
    async def test_read_body_completes_on_message_complete(self):
        """read_body() future resolved by on_message_complete callback."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())
        # Send headers only
        protocol.data_received(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n")
        # Start body read BEFORE body arrives
        body_task = asyncio.create_task(protocol.read_body())
        await asyncio.sleep(0)
        assert not body_task.done()
        # Now send body — triggers on_message_complete → sets body_future result
        protocol.data_received(b"hello")
        result = await body_task
        assert result == b"hello"

    @pytest.mark.asyncio
    async def test_read_headers_fast_path(self):
        """read_headers() returns immediately if headers already parsed."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())
        # Feed complete response synchronously
        protocol.data_received(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
        # Headers already complete — should return instantly
        status, headers = await protocol.read_headers()
        assert status == 200


# =============================================================================
# ConnectionPool Tests
# =============================================================================


class TestConnectionPool:
    """Tests for connection pooling."""

    @pytest.mark.asyncio
    async def test_acquire_creates_and_reuses(self, pool):
        """acquire() creates new connection, then reuses from pool."""
        # First acquire creates connection
        conn1 = await pool.acquire()
        assert pool.total_count == 1
        assert pool.in_use_count == 1

        # Release returns to pool
        pool.release(conn1)
        assert pool.idle_count == 1
        assert pool.in_use_count == 0

        # Second acquire reuses
        conn2 = await pool.acquire()
        assert conn2 is conn1
        assert pool.total_count == 1

        pool.release(conn2)

    @pytest.mark.asyncio
    async def test_respects_max_connections(self, pool):
        """Pool blocks when max_connections reached."""
        # Acquire all 4 connections
        conns = [await pool.acquire() for _ in range(4)]
        assert pool.total_count == 4
        assert pool.in_use_count == 4

        # Next acquire should block
        acquire_task = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0.05)
        assert not acquire_task.done()
        assert pool.waiting_count == 1

        # Release one - waiter should get it
        pool.release(conns[0])
        await asyncio.sleep(0.01)
        assert acquire_task.done()
        assert pool.waiting_count == 0

        # Cleanup
        conn5 = await acquire_task
        pool.release(conn5)
        for c in conns[1:]:
            pool.release(c)

    @pytest.mark.asyncio
    async def test_discards_dead_connections(self, pool):
        """Pool discards connections with closed transport."""
        conn = await pool.acquire()

        # Simulate transport close
        conn.transport.close()
        pool.release(conn)

        # Should be discarded, not returned to idle
        assert pool.idle_count == 0
        assert pool.total_count == 0

    @pytest.mark.asyncio
    async def test_release_closes_if_should_close(self, pool):
        """Pool closes connection if protocol.should_close is True."""
        conn = await pool.acquire()

        # Simulate server sent Connection: close
        conn.protocol._should_close = True
        pool.release(conn)

        # Should be closed, not pooled
        assert pool.idle_count == 0
        assert pool.total_count == 0

    @pytest.mark.asyncio
    async def test_close_cleans_up_all(self, pool):
        """close() closes all connections and cancels waiters."""
        # Create some connections
        conn1 = await pool.acquire()
        _ = await pool.acquire()
        pool.release(conn1)

        # Use remaining slots and start a waiter
        _ = [await pool.acquire() for _ in range(2)]
        waiter_task = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0.01)

        # Close pool
        await pool.close()

        assert pool.total_count == 0
        assert pool.idle_count == 0
        assert waiter_task.cancelled() or waiter_task.done()

    @pytest.mark.asyncio
    async def test_stale_connection_discarded_on_acquire(self, pool):
        """_try_get_idle discards dead connections and creates fresh ones."""
        # Acquire and release a connection so it sits in the idle stack
        conn1 = await pool.acquire()
        pool.release(conn1)
        assert pool.idle_count == 1
        assert pool.total_count == 1

        # Simulate server-side close by marking protocol as connection lost
        # (is_alive() checks _connection_lost flag)
        conn1.protocol._connection_lost = True

        # Next acquire should discard the dead connection and create a new one
        conn2 = await pool.acquire()
        assert conn2 is not conn1
        assert pool.total_count == 1  # Old one removed, new one added

        pool.release(conn2)

    @pytest.mark.asyncio
    async def test_idle_timeout_discards_connection(self, echo_server):
        """Connections idle longer than max_idle_time are discarded on acquire."""
        loop = asyncio.get_running_loop()
        parsed = urlparse(echo_server.url)
        short_idle_pool = ConnectionPool(
            host=parsed.hostname,
            port=parsed.port,
            loop=loop,
            max_connections=4,
            max_idle_time=0.1,
        )
        try:
            # Acquire and release a connection
            conn1 = await short_idle_pool.acquire()
            short_idle_pool.release(conn1)
            assert short_idle_pool.idle_count == 1

            # Wait for the idle timeout to expire
            await asyncio.sleep(0.15)

            # Next acquire should discard the expired connection and create new
            conn2 = await short_idle_pool.acquire()
            assert conn2 is not conn1
            assert short_idle_pool.total_count == 1

            short_idle_pool.release(conn2)
        finally:
            await short_idle_pool.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "count,expected",
        [(3, 3), (None, 4)],
        ids=["explicit_count", "none_uses_max"],
    )
    async def test_warmup_creates_connections(self, pool, count, expected):
        """warmup() pre-establishes connections and returns them to idle."""
        result = await pool.warmup(count=count)

        assert result == expected
        assert pool.idle_count == expected
        assert pool.total_count == expected

    @pytest.mark.asyncio
    async def test_warmup_exceeding_max_raises(self, pool):
        """warmup() raises ValueError when count exceeds max_connections."""
        with pytest.raises(ValueError, match="max_connections"):
            await pool.warmup(count=5)

    @pytest.mark.asyncio
    async def test_pool_close_cancels_waiters(self, pool):
        """close() cancels all pending waiters and clears the waiter queue."""
        # Acquire all 4 connections to saturate the pool
        _conns = [await pool.acquire() for _ in range(4)]

        # Create 2 waiter tasks that will block on acquire
        task1 = asyncio.create_task(pool.acquire())
        task2 = asyncio.create_task(pool.acquire())

        # Let them register as waiters
        await asyncio.sleep(0.01)
        assert pool.waiting_count == 2

        # Close pool — should cancel waiters and clean up
        await pool.close()

        # Let cancellation propagate through task wrappers
        await asyncio.sleep(0)

        assert task1.done()
        assert task2.done()
        assert pool.waiting_count == 0

    @pytest.mark.asyncio
    async def test_release_idempotent(self, pool):
        """Releasing a connection twice is a no-op the second time."""
        conn = await pool.acquire()
        assert pool.in_use_count == 1

        # First release returns it to idle
        pool.release(conn)
        assert pool.idle_count == 1
        assert pool.in_use_count == 0

        # Second release is a no-op (conn.in_use is already False)
        pool.release(conn)
        assert pool.idle_count == 1  # Still 1, not 2
        assert pool.total_count == 1

    @pytest.mark.asyncio
    async def test_unlimited_pool(self, echo_server):
        """Pool with max_connections=None allows unlimited connections and warmup returns 0."""
        loop = asyncio.get_running_loop()
        parsed = urlparse(echo_server.url)
        unlimited_pool = ConnectionPool(
            host=parsed.hostname,
            port=parsed.port,
            loop=loop,
            max_connections=None,
        )
        try:
            # Unlimited pool allows arbitrary number of connections
            conns = [await unlimited_pool.acquire() for _ in range(8)]
            assert unlimited_pool.total_count == 8
            for c in conns:
                unlimited_pool.release(c)

            # warmup(None) with max_connections=None returns 0 (nothing to warm)
            result = await unlimited_pool.warmup(count=None)
            assert result == 0
        finally:
            await unlimited_pool.close()

    @pytest.mark.asyncio
    async def test_waiter_creates_new_connection(self, pool):
        """When waiter wakes up with no idle connections, it creates a new one."""
        # Acquire all 4, then close one (destroys it, doesn't idle it)
        conns = [await pool.acquire() for _ in range(4)]
        # Start waiter
        waiter_task = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0.01)
        assert pool.waiting_count == 1
        # Close transport on one conn (so it gets destroyed on release, not idled)
        conns[0].transport.close()
        pool.release(conns[0])  # destroyed, not idled — frees a slot
        # Waiter should create a NEW connection (no idle available)
        conn = await waiter_task
        assert conn is not conns[0]  # Different connection
        pool.release(conn)
        for c in conns[1:]:
            pool.release(c)

    @pytest.mark.asyncio
    async def test_is_stale_various_conditions(self, pool):
        """is_stale() returns correct results for different connection states."""
        import time as time_mod

        conn = await pool.acquire()
        pool.release(conn)

        # 1. Recently used — fast path returns False
        assert not conn.is_stale()

        # 2. Age past 1s, healthy connection — not stale
        conn.last_used = time_mod.monotonic() - 2.0
        assert not conn.is_stale()

        # 3. Transport is None — stale
        saved_transport = conn.transport
        conn.transport = None  # type: ignore[assignment]
        conn.last_used = time_mod.monotonic() - 2.0
        assert conn.is_stale()
        conn.transport = saved_transport  # restore

        # 4. get_extra_info("socket") returns None — stale
        conn.transport = saved_transport  # ensure transport is restored
        orig_get_extra = conn.transport.get_extra_info
        conn.transport.get_extra_info = lambda name, default=None: None  # type: ignore[assignment]
        conn.last_used = time_mod.monotonic() - 2.0
        assert conn.is_stale()
        conn.transport.get_extra_info = orig_get_extra  # type: ignore[assignment]

        # 5. fd < 0 — stale
        from unittest.mock import MagicMock

        mock_sock = MagicMock()
        mock_sock.fileno.return_value = -1
        conn.transport = saved_transport
        conn.transport.get_extra_info = (
            lambda name, default=None: mock_sock if name == "socket" else default
        )  # type: ignore[assignment]
        conn.last_used = time_mod.monotonic() - 2.0
        assert conn.is_stale()

        # 6. OSError from fileno/select — stale
        mock_sock2 = MagicMock()
        mock_sock2.fileno.side_effect = OSError("bad fd")
        conn.transport.get_extra_info = (
            lambda name, default=None: mock_sock2 if name == "socket" else default
        )  # type: ignore[assignment]
        conn.last_used = time_mod.monotonic() - 2.0
        assert conn.is_stale()

        conn.transport.get_extra_info = orig_get_extra  # type: ignore[assignment]


# =============================================================================
# Integration Tests
# =============================================================================


class TestHttpIntegration:
    """End-to-end tests against EchoServer."""

    @pytest.mark.asyncio
    async def test_non_streaming_roundtrip(self, pool, template):
        """Complete non-streaming request/response cycle."""
        body = b'{"model": "test", "messages": [{"role": "user", "content": "hello"}]}'
        request = template.build_request(body, streaming=False)

        conn = await pool.acquire()
        try:
            conn.protocol.write(request)

            status, headers = await conn.protocol.read_headers()
            assert status == 200
            assert "content-type" in headers

            response_body = await conn.protocol.read_body()
            assert len(response_body) > 0
        finally:
            pool.release(conn)

    @pytest.mark.asyncio
    async def test_streaming_sse_roundtrip(self, pool, template):
        """Complete streaming SSE request/response cycle."""
        body = b'{"model": "test", "stream": true, "messages": [{"role": "user", "content": "hello world"}]}'
        request = template.build_request(body, streaming=True)

        conn = await pool.acquire()
        try:
            conn.protocol.write(request)

            status, headers = await conn.protocol.read_headers()
            assert status == 200
            assert headers.get("content-type") == "text/event-stream"

            # Collect SSE chunks
            chunks = []
            async for chunk_batch in conn.protocol.iter_body():
                chunks.extend(chunk_batch)

            # Should have received SSE data
            combined = b"".join(chunks)
            assert b"data:" in combined
            assert b"[DONE]" in combined
        finally:
            pool.release(conn)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, pool, template):
        """Multiple concurrent requests through pool."""

        async def make_request(i: int) -> int:
            body = f'{{"model": "test", "messages": [{{"role": "user", "content": "req{i}"}}]}}'.encode()
            request = template.build_request(body, streaming=False)

            conn = await pool.acquire()
            try:
                conn.protocol.write(request)
                status, _ = await conn.protocol.read_headers()
                await conn.protocol.read_body()
                return status
            finally:
                pool.release(conn)

        # Run 10 concurrent requests (more than pool size of 4)
        results = await asyncio.gather(*[make_request(i) for i in range(10)])

        assert all(status == 200 for status in results)

    @pytest.mark.asyncio
    async def test_connection_reuse_across_requests(self, pool, template):
        """Same connection reused for sequential requests."""
        body = b'{"model": "test", "messages": [{"role": "user", "content": "test"}]}'
        request = template.build_request(body, streaming=False)

        # First request
        conn1 = await pool.acquire()
        conn1.protocol.write(request)
        await conn1.protocol.read_headers()
        await conn1.protocol.read_body()
        pool.release(conn1)

        # Second request should reuse same connection
        conn2 = await pool.acquire()
        assert conn2 is conn1

        conn2.protocol.write(request)
        status, _ = await conn2.protocol.read_headers()
        await conn2.protocol.read_body()
        pool.release(conn2)

        assert status == 200

    @pytest.mark.asyncio
    async def test_handles_error_response(self, pool, template):
        """Handles HTTP error responses gracefully."""
        # Send invalid JSON to trigger 400
        body = b"not valid json"
        request = template.build_request(body, streaming=False)

        conn = await pool.acquire()
        try:
            conn.protocol.write(request)

            status, _ = await conn.protocol.read_headers()
            response_body = await conn.protocol.read_body()

            assert status == 400
            assert b"error" in response_body
        finally:
            pool.release(conn)


# =============================================================================
# Helpers
# =============================================================================


class MockTransport:
    """Minimal mock transport for protocol tests."""

    def __init__(self):
        self._closing = False
        self._written = []

    def write(self, data: bytes) -> None:
        self._written.append(data)

    def close(self) -> None:
        self._closing = True

    def is_closing(self) -> bool:
        return self._closing

    def get_extra_info(self, name: str, default=None):
        return default
