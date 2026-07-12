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

"""Unit tests for the worker's pre-response connection-reset retry.

Covers ``Worker._read_headers_with_retry`` — the guard that re-issues a request
on a fresh connection when the server closes the socket before sending any
response byte (the idle keep-alive race that otherwise zeroes accuracy samples
on single-stream localhost servers such as llama.cpp ``-np 1``).
"""

from types import SimpleNamespace

import pytest
from inference_endpoint.core.types import QueryResult
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http import InFlightRequest
from inference_endpoint.endpoint_client.worker import Worker


@pytest.mark.unit
def test_transport_retries_default_is_opt_in():
    """Default is 0 (fail-fast) so the retry never changes behavior unless a
    config opts in explicitly (e.g. the edge-agentic accuracy reference)."""
    assert HTTPClientConfig().transport_max_retries == 0


class _FakeProtocol:
    """Protocol whose ``read_headers`` replays a scripted result sequence."""

    def __init__(self, results, *, bytes_received: bool = False, body: bytes = b"{}"):
        # results: list where each item is either the sentinel "reset" (raise
        # ConnectionResetError) or an int status code to return.
        self._results = list(results)
        self.written: list[bytes] = []
        # bytes_received distinguishes a true zero-byte close (retryable) from a
        # partial-response close (server may already have processed the request).
        self.bytes_received = bytes_received
        self._body = body

    async def read_headers(self):
        outcome = self._results.pop(0)
        if outcome == "reset":
            raise ConnectionResetError("Connection closed before headers received")
        return outcome, {}

    async def read_body(self) -> bytes:
        return self._body

    def write(self, data: bytes) -> None:
        self.written.append(data)


class _FakeConn:
    def __init__(self, protocol: _FakeProtocol):
        self.protocol = protocol
        self.in_use = True


class _FakePool:
    """Records releases and hands out queued connections on ``acquire``."""

    def __init__(self, acquire_queue: list[_FakeConn]):
        self._acquire_queue = list(acquire_queue)
        self.released: list[_FakeConn] = []
        self.acquired: list[_FakeConn] = []

    def release(self, conn) -> None:
        self.released.append(conn)

    async def acquire(self) -> _FakeConn:
        conn = self._acquire_queue.pop(0)
        self.acquired.append(conn)
        return conn


class _InUsePool:
    """Fake pool that models ``in_use`` so a double-release is observably a
    no-op the second time (mirrors ``ConnectionPool.release`` idempotency).

    ``effective_releases`` counts only releases that actually transitioned a
    connection out of use; ``release_calls`` counts every call. A correct
    ``finally``-block double-release therefore shows two calls but one effect.
    """

    def __init__(self, acquire_queue: list[_FakeConn]):
        self._acquire_queue = list(acquire_queue)
        self.release_calls: list[_FakeConn] = []
        self.effective_releases: list[_FakeConn] = []
        self.acquired: list[_FakeConn] = []

    def release(self, conn) -> None:
        self.release_calls.append(conn)
        if not conn.in_use:
            return
        conn.in_use = False
        self.effective_releases.append(conn)

    async def acquire(self) -> _FakeConn:
        conn = self._acquire_queue.pop(0)
        conn.in_use = True
        self.acquired.append(conn)
        return conn


class _FakeResponses:
    """Captures results the worker sends back to the main rank."""

    def __init__(self) -> None:
        self.sent: list = []

    def send(self, item) -> None:
        self.sent.append(item)


class _FakeAdapter:
    """Minimal adapter: decodes a non-streaming body into a QueryResult."""

    @staticmethod
    def decode_response(body: bytes, query_id: str):
        return QueryResult(id=query_id)


def _make_worker(pool, max_retries: int, *, responses=None):
    """Build a Worker with only the attributes the retry path touches."""
    worker = Worker.__new__(Worker)
    worker._pool = pool  # type: ignore[assignment]
    worker.http_config = SimpleNamespace(  # type: ignore[assignment]
        transport_max_retries=max_retries
    )
    worker._active_tasks = set()  # type: ignore[attr-defined]
    worker._responses = responses  # type: ignore[assignment]
    worker._adapter = _FakeAdapter  # type: ignore[assignment]
    worker._handle_error_calls = []  # type: ignore[attr-defined]

    async def _record_error(query_id, error):
        worker._handle_error_calls.append((query_id, error))

    worker._handle_error = _record_error  # type: ignore[method-assign]
    return worker


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retry_reissues_on_fresh_connection_after_reset():
    """A pre-response reset is retried on a fresh connection and succeeds."""
    dead = _FakeConn(_FakeProtocol(["reset"]))
    fresh = _FakeConn(_FakeProtocol([200]))
    pool = _FakePool(acquire_queue=[fresh])
    worker = _make_worker(pool, max_retries=2)

    req = InFlightRequest(
        query_id="q1",
        http_bytes=b"POST /v1/chat/completions HTTP/1.1\r\n\r\n",
        is_streaming=True,
        connection=dead,
    )

    status = await worker._read_headers_with_retry(req)

    assert status == 200
    assert req.connection is fresh
    # Dead connection discarded exactly once; request re-written on the fresh one.
    assert pool.released == [dead]
    assert fresh.protocol.written == [req.http_bytes]
    assert worker._handle_error_calls == []
    # The recovery is counted on the request so the success carries a marker.
    assert req.transport_retries == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retry_exhausted_emits_error_and_returns_none():
    """When every attempt resets, retries are exhausted and an error is emitted."""
    dead = _FakeConn(_FakeProtocol(["reset"]))
    also_dead = _FakeConn(_FakeProtocol(["reset"]))
    pool = _FakePool(acquire_queue=[also_dead])
    worker = _make_worker(pool, max_retries=1)

    req = InFlightRequest(
        query_id="q2",
        http_bytes=b"POST /x HTTP/1.1\r\n\r\n",
        is_streaming=False,
        connection=dead,
    )

    status = await worker._read_headers_with_retry(req)

    assert status is None
    # Both connections were tried and discarded.
    assert pool.released == [dead, also_dead]
    assert len(worker._handle_error_calls) == 1
    assert worker._handle_error_calls[0][0] == "q2"
    assert isinstance(worker._handle_error_calls[0][1], ConnectionResetError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retries_disabled_emits_error_without_reacquiring():
    """transport_max_retries=0 => no re-issue; the reset surfaces immediately."""
    dead = _FakeConn(_FakeProtocol(["reset"]))
    pool = _FakePool(acquire_queue=[])  # acquire must never be called
    worker = _make_worker(pool, max_retries=0)

    req = InFlightRequest(
        query_id="q3",
        http_bytes=b"POST /x HTTP/1.1\r\n\r\n",
        is_streaming=False,
        connection=dead,
    )

    status = await worker._read_headers_with_retry(req)

    assert status is None
    assert pool.acquired == []
    assert pool.released == [dead]
    assert len(worker._handle_error_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_headers_ok_first_try_never_touches_pool():
    """No reset => no release/acquire churn, status returned directly."""
    conn = _FakeConn(_FakeProtocol([200]))
    pool = _FakePool(acquire_queue=[])
    worker = _make_worker(pool, max_retries=2)

    req = InFlightRequest(
        query_id="q4",
        http_bytes=b"POST /x HTTP/1.1\r\n\r\n",
        is_streaming=False,
        connection=conn,
    )

    status = await worker._read_headers_with_retry(req)

    assert status == 200
    assert req.connection is conn
    assert pool.acquired == []
    assert pool.released == []
    assert worker._handle_error_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fresh_conn_attached_before_write_so_failing_write_is_not_leaked():
    """If the re-issue write fails, req.connection already points to the fresh
    connection so the caller's finally-block can release it (no leak)."""

    class _RaisingWriteProtocol(_FakeProtocol):
        def write(self, data: bytes) -> None:
            raise ConnectionResetError("write on closed transport")

    dead = _FakeConn(_FakeProtocol(["reset"]))
    fresh = _FakeConn(_RaisingWriteProtocol([200]))
    pool = _FakePool(acquire_queue=[fresh])
    worker = _make_worker(pool, max_retries=2)

    req = InFlightRequest(
        query_id="q5",
        http_bytes=b"POST /x HTTP/1.1\r\n\r\n",
        is_streaming=False,
        connection=dead,
    )

    with pytest.raises(ConnectionError):
        await worker._read_headers_with_retry(req)

    # Fresh connection is attached before the failing write, so the request
    # still owns a releasable connection (the old one was already released).
    assert req.connection is fresh
    assert pool.released == [dead]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_retries_then_success():
    """Two consecutive resets are recovered when max_retries allows it."""
    dead1 = _FakeConn(_FakeProtocol(["reset"]))
    dead2 = _FakeConn(_FakeProtocol(["reset"]))
    fresh = _FakeConn(_FakeProtocol([200]))
    pool = _FakePool(acquire_queue=[dead2, fresh])
    worker = _make_worker(pool, max_retries=2)

    req = InFlightRequest(
        query_id="q6",
        http_bytes=b"POST /x HTTP/1.1\r\n\r\n",
        is_streaming=False,
        connection=dead1,
    )

    status = await worker._read_headers_with_retry(req)

    assert status == 200
    assert req.connection is fresh
    assert req.transport_retries == 2
    # Both dead connections discarded in order; each re-issue re-wrote the body.
    assert pool.released == [dead1, dead2]
    assert pool.acquired == [dead2, fresh]
    assert dead2.protocol.written == [req.http_bytes]
    assert fresh.protocol.written == [req.http_bytes]
    assert worker._handle_error_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partial_response_close_is_not_retried():
    """A close AFTER some response bytes arrived is not a zero-byte close: the
    server may already have processed the request, so it surfaces as an error
    rather than a (server-side non-idempotent) re-issue."""
    partial = _FakeConn(_FakeProtocol(["reset"], bytes_received=True))
    pool = _FakePool(acquire_queue=[])  # acquire must never be called
    worker = _make_worker(pool, max_retries=2)

    req = InFlightRequest(
        query_id="q7",
        http_bytes=b"POST /x HTTP/1.1\r\n\r\n",
        is_streaming=False,
        connection=partial,
    )

    status = await worker._read_headers_with_retry(req)

    assert status is None
    assert pool.acquired == []
    assert pool.released == [partial]
    assert req.transport_retries == 0
    assert len(worker._handle_error_calls) == 1
    assert isinstance(worker._handle_error_calls[0][1], ConnectionResetError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_response_double_release_is_idempotent():
    """Driving ``_process_response`` end-to-end after a recovered reset: the
    non-streaming body handler releases the fresh connection, then the
    ``finally`` block releases it again. With an ``in_use``-tracking pool the
    second release is a no-op, and the sent result carries the retry count."""
    dead = _FakeConn(_FakeProtocol(["reset"]))
    fresh = _FakeConn(_FakeProtocol([200]))
    pool = _InUsePool(acquire_queue=[fresh])
    responses = _FakeResponses()
    worker = _make_worker(pool, max_retries=2, responses=responses)

    req = InFlightRequest(
        query_id="q8",
        http_bytes=b"POST /x HTTP/1.1\r\n\r\n",
        is_streaming=False,
        connection=dead,
    )

    await worker._process_response(req)

    # fresh is released twice (body handler + finally) but only takes effect once.
    assert pool.release_calls.count(fresh) == 2
    assert pool.effective_releases.count(fresh) == 1
    assert fresh.in_use is False
    # The result reached the main rank tagged with the recovered retry.
    assert len(responses.sent) == 1
    result = responses.sent[0]
    assert isinstance(result, QueryResult)
    assert result.id == "q8"
    assert result.transport_retries == 1
    assert worker._handle_error_calls == []
