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
        assert parser_state["headers"]["host"] == "localhost:8080"
        assert parser_state["headers"]["content-type"] == "application/json"

    def test_cache_headers_pre_caches(self):
        """cache_headers() pre-encodes headers and they appear in built request."""
        template = HttpRequestTemplate.from_url("localhost", 8080, "/v1/chat")
        body = b'{"model": "test"}'

        assert template.cached_headers == b""
        template.cache_headers({"Authorization": "Bearer pre-cached-token"})
        assert template.cached_headers == b"Authorization: Bearer pre-cached-token\r\n"

        # Cached headers appear in built request on fast path
        request = template.build_request(body, streaming=False)
        assert b"Authorization: Bearer pre-cached-token\r\n" in request
        assert b"Host: localhost:8080" in request
        assert b"Content-Type: application/json" in request
        assert b"Content-Length: 17" in request


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
        protocol.data_received(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")

        chunk_batches = []

        async def consume():
            async for chunk_batch in protocol.iter_body():
                chunk_batches.append(chunk_batch)

        # Start consumer
        consumer_task = asyncio.create_task(consume())

        # Feed chunks one at a time
        await asyncio.sleep(0)
        protocol.data_received(b"5\r\nhello\r\n")

        await asyncio.sleep(0)
        protocol.data_received(b"6\r\n world\r\n")

        await asyncio.sleep(0)
        protocol.data_received(b"0\r\n\r\n")

        await consumer_task

        # iter_body yields batches, flatten to check content
        all_chunks = [chunk for batch in chunk_batches for chunk in batch]
        assert b"hello" in all_chunks
        assert b" world" in all_chunks

    @pytest.mark.asyncio
    async def test_iter_body_pre_buffered_and_already_complete(self):
        """iter_body yields pre-buffered chunks and exits early if message complete."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())

        # Feed entire chunked response synchronously before calling iter_body
        protocol.data_received(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhello\r\n"
            b"0\r\n\r\n"
        )
        assert protocol._message_complete

        all_chunks = []
        async for batch in protocol.iter_body():
            all_chunks.extend(batch)
        assert b"hello" in all_chunks

    def test_reset_clears_state_for_reuse(self):
        """reset() clears state for connection reuse."""
        loop = asyncio.new_event_loop()
        protocol = HttpResponseProtocol(loop)

        # First response
        protocol.connection_made(MockTransport())
        protocol.data_received(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\ntest")

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
        await asyncio.sleep(0)

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
        "exc,expected_error,send_partial",
        [
            (OSError("broken pipe"), OSError, False),
            (None, ConnectionResetError, True),
        ],
        ids=["with_exception", "clean_close_partial"],
    )
    async def test_connection_lost_before_body_complete(
        self, exc, expected_error, send_partial
    ):
        """connection_lost propagates errors to pending body_future."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())
        protocol.data_received(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
        if send_partial:
            protocol.data_received(b"partial")
        body_task = asyncio.create_task(protocol.read_body())
        await asyncio.sleep(0)
        protocol.connection_lost(exc)
        with pytest.raises(expected_error):
            await body_task

    @pytest.mark.asyncio
    async def test_connection_lost_with_complete_message(self):
        """connection_lost after message_complete resolves body future normally."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())
        protocol.data_received(
            b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n" + b'{"ok": true}'
        )
        body_task = asyncio.create_task(protocol.read_body())
        await asyncio.sleep(0)
        protocol.connection_lost(None)
        assert await body_task == b'{"ok": true}'

        # Manually pending body_future with _message_complete=True
        p2 = HttpResponseProtocol(loop)
        p2.connection_made(MockTransport())
        p2._message_complete, p2._body_chunks = True, [b"test data"]
        p2._body_future = loop.create_future()
        p2.connection_lost(None)
        assert await p2._body_future == b"test data"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "setup_headers,feed_garbage",
        [
            (False, b"NOT HTTP AT ALL"),
            (True, b"ZZZZ\r\nNOT A VALID CHUNK\r\n"),
        ],
        ids=["headers_future", "body_future"],
    )
    async def test_data_received_parser_error(self, setup_headers, feed_garbage):
        """HttpParserError propagates to the pending headers or body future."""
        loop = asyncio.get_running_loop()
        protocol = HttpResponseProtocol(loop)
        protocol.connection_made(MockTransport())
        if setup_headers:
            protocol.data_received(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            )
            task = asyncio.create_task(protocol.read_body())
        else:
            task = asyncio.create_task(protocol.read_headers())
        await asyncio.sleep(0)
        protocol.data_received(feed_garbage)
        with pytest.raises(httptools.HttpParserError):
            await task

    @pytest.mark.asyncio
    async def test_read_body_and_headers_fast_paths(self):
        """read_body() resolves via on_message_complete; read_headers() returns
        immediately when headers already parsed."""
        loop = asyncio.get_running_loop()
        # read_body: headers arrive first, body later
        p1 = HttpResponseProtocol(loop)
        p1.connection_made(MockTransport())
        p1.data_received(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n")
        body_task = asyncio.create_task(p1.read_body())
        await asyncio.sleep(0)
        assert not body_task.done()
        p1.data_received(b"hello")
        assert await body_task == b"hello"
        # read_headers fast path: full response already parsed
        p2 = HttpResponseProtocol(loop)
        p2.connection_made(MockTransport())
        p2.data_received(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
        status, _ = await p2.read_headers()
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
        await asyncio.sleep(0)
        assert not acquire_task.done()
        assert pool.waiting_count == 1

        # Release one - waiter should get it
        pool.release(conns[0])
        await asyncio.sleep(0)
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
    async def test_stale_connection_discarded_on_acquire(self, pool):
        """_try_get_idle discards dead connections and creates fresh ones."""
        conn1 = await pool.acquire()
        pool.release(conn1)
        conn1.protocol._connection_lost = True
        conn2 = await pool.acquire()
        assert conn2 is not conn1 and pool.total_count == 1
        pool.release(conn2)

    @pytest.mark.asyncio
    async def test_idle_timeout_discards_connection(self, echo_server):
        """Connections idle longer than max_idle_time are discarded on acquire."""
        loop = asyncio.get_running_loop()
        parsed = urlparse(echo_server.url)
        p = ConnectionPool(
            host=parsed.hostname,
            port=parsed.port,
            loop=loop,
            max_connections=4,
            max_idle_time=0.01,
        )
        try:
            conn1 = await p.acquire()
            p.release(conn1)
            await asyncio.sleep(0.05)
            conn2 = await p.acquire()
            assert conn2 is not conn1 and p.total_count == 1
            p.release(conn2)
        finally:
            await p.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "count,expected",
        [(3, 3), (None, 4), (5, ValueError)],
        ids=["explicit_count", "none_uses_max", "exceeds_max"],
    )
    async def test_warmup(self, pool, count, expected):
        """warmup() pre-establishes connections, raises on over-limit."""
        if expected is ValueError:
            with pytest.raises(ValueError, match="max_connections"):
                await pool.warmup(count=count)
        else:
            result = await pool.warmup(count=count)
            assert result == expected
            assert pool.idle_count == expected
            assert pool.total_count == expected

    @pytest.mark.asyncio
    async def test_close_cancels_waiters(self, pool):
        """close() cancels all pending waiters and clears the waiter queue."""
        _conns = [await pool.acquire() for _ in range(4)]
        tasks = [asyncio.create_task(pool.acquire()) for _ in range(2)]
        await asyncio.sleep(0)
        assert pool.waiting_count == 2
        await pool.close()
        await asyncio.sleep(0)
        assert all(t.done() for t in tasks)
        assert pool.waiting_count == 0

    @pytest.mark.asyncio
    async def test_release_idempotent(self, pool):
        """Releasing a connection twice is a no-op the second time."""
        conn = await pool.acquire()
        pool.release(conn)
        pool.release(conn)
        assert pool.idle_count == 1
        assert pool.total_count == 1

    @pytest.mark.asyncio
    async def test_unlimited_pool(self, echo_server):
        """Pool with max_connections=None allows unlimited connections."""
        loop = asyncio.get_running_loop()
        parsed = urlparse(echo_server.url)
        p = ConnectionPool(
            host=parsed.hostname,
            port=parsed.port,
            loop=loop,
            max_connections=None,
        )
        try:
            conns = [await p.acquire() for _ in range(8)]
            assert p.total_count == 8
            for c in conns:
                p.release(c)
            # warmup(None) with unlimited pool returns 0
            assert await p.warmup(count=None) == 0
        finally:
            await p.close()

    @pytest.mark.asyncio
    async def test_waiter_creates_new_connection(self, pool):
        """When waiter wakes with no idle connections, it creates a new one."""
        conns = [await pool.acquire() for _ in range(4)]
        waiter_task = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        assert pool.waiting_count == 1
        # Destroy connection — destroyed, not idled — frees a slot
        conns[0].transport.close()
        pool.release(conns[0])
        conn = await waiter_task
        assert conn is not conns[0]
        pool.release(conn)
        for c in conns[1:]:
            pool.release(c)

    @pytest.mark.asyncio
    async def test_is_stale_various_conditions(self, pool):
        """is_stale() returns correct results for different connection states."""
        import time as time_mod
        from unittest.mock import MagicMock, patch

        conn = await pool.acquire()
        pool.release(conn)
        old_ts = time_mod.monotonic() - 2.0
        mock_poller = MagicMock()

        # Recently used — skip stale check entirely
        assert not conn.is_stale()

        # Fast path: poller already cached, healthy (poll returns empty) — not stale
        conn.last_used = old_ts
        conn._stale_poller = mock_poller
        mock_poller.poll.return_value = []
        assert not conn.is_stale()

        # Fast path: server sent FIN (poll returns events) — stale
        mock_poller.poll.return_value = [(5, 1)]
        assert conn.is_stale()

        # Fast path: poll raises OSError — stale
        mock_poller.poll.side_effect = OSError("bad fd")
        assert conn.is_stale()

        # Slow path: no poller, valid fd — creates poller and caches it
        conn._stale_poller = None
        conn.last_used = old_ts
        with patch("inference_endpoint.endpoint_client.http.select") as mock_select:
            mock_poll_inst = MagicMock()
            mock_select.poll.return_value = mock_poll_inst
            mock_select.POLLIN = 1
            mock_select.POLLERR = 8
            mock_select.POLLHUP = 16
            mock_poll_inst.poll.return_value = []
            assert not conn.is_stale()
            assert conn._stale_poller is mock_poll_inst

        # Slow path: no poller, fd < 0 — stale
        conn._stale_poller = None
        conn._fd = -1
        assert conn.is_stale()

        # Slow path: no poller, register raises OSError — stale
        conn._fd = 999
        conn._stale_poller = None
        with patch("inference_endpoint.endpoint_client.http.select") as mock_select:
            mock_select.poll.side_effect = OSError("bad")
            assert conn.is_stale()


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
