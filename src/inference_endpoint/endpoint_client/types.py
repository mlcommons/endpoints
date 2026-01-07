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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aiohttp import hdrs
from aiohttp.client import ClientSession
from aiohttp.client_reqrep import ClientRequest, ClientResponse, RequestInfo
from aiohttp.streams import StreamReader
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from inference_endpoint.endpoint_client.timing_context import RequestTimingContext


@dataclass(slots=True)
class HttpRequestTemplate:
    """
    Pre-computed HTTP/1.1 request parts for direct socket writes.

    These are computed once from the endpoint URL and reused for all requests,
    avoiding repeated string formatting and encoding overhead.

    Attributes:
        request_line: HTTP request line bytes (e.g., b"POST /v1/chat HTTP/1.1\\r\\n")
        host_header: HTTP Host header bytes, required for HTTP/1.1
        request_info: aiohttp RequestInfo for response association
        connection_request: Cached ClientRequest used as key for TCP connection pool
    """

    request_line: bytes
    host_header: bytes
    request_info: RequestInfo
    connection_request: ClientRequest

    @classmethod
    def from_url(
        cls,
        url: URL,
        loop: asyncio.AbstractEventLoop,
        session: ClientSession,
    ) -> HttpRequestTemplate:
        """
        Create an HttpRequestTemplate from a URL.

        Pre-computes static HTTP/1.1 request components that remain constant
        across all requests. This avoids repeated string formatting and
        encoding on every request in the hot path.

        Args:
            url: Target endpoint URL
            loop: Event loop for ClientRequest
            session: aiohttp session for connection pool key

        Returns:
            HttpRequestTemplate ready for building requests

        Components computed:
            - request_line: "POST /path HTTP/1.1\\r\\n" encoded as bytes
            - host_header: "Host: hostname[:port]\\r\\n" (required by HTTP/1.1 RFC 7230)
            - request_info: aiohttp metadata for response/request association
            - connection_request: ClientRequest used as connection pool lookup key
        """
        path = url.raw_path_qs or "/"
        request_line = f"POST {path} HTTP/1.1\r\n".encode("ascii")

        # Host header is mandatory in HTTP/1.1 (RFC 7230 Section 5.4)
        # Port is omitted for default ports (80 for HTTP, 443 for HTTPS)
        host = url.raw_host or "localhost"
        if url.port and url.port not in (80, 443):
            host_header = f"Host: {host}:{url.port}\r\n".encode("ascii")
        else:
            host_header = f"Host: {host}\r\n".encode("ascii")

        request_info = RequestInfo(
            url=url,
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=url,
        )

        # aiohttp's TCPConnector uses ClientRequest as the connection pool key.
        # The connector hashes (host, port, ssl) from the request to lookup/store
        # connections. By reusing the same ClientRequest instance for both warmup
        # and actual requests, we ensure connections warmed up during startup are
        # found in the pool during request handling.
        connection_request = ClientRequest(
            method=hdrs.METH_POST,
            url=url,
            loop=loop,
            response_class=ClientResponse,
            session=session,
            ssl=False,
        )

        return cls(
            request_line=request_line,
            host_header=host_header,
            request_info=request_info,
            connection_request=connection_request,
        )

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

        Header handling:
            - Host: Always included from template (required by HTTP/1.1)
            - Content-Length: Computed from body size (required for POST)
            - Other headers: Passed through from query.headers
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


@dataclass(slots=True)
class PreparedRequest:
    """A prepared HTTP request ready to be sent.

    Attributes:
        query_id: Unique identifier for the request (matches Query.id).
        http_bytes: Pre-built HTTP request bytes ready for socket.write().
        timing: RequestTimingContext for overhead measurement.
        is_streaming: True if this is a streaming (SSE) request.
        url: Target URL for the request.
        request_info: aiohttp RequestInfo for response association.
        process: Callback to handle the response (set after preparation).
        response: ResponseData container (set after headers received).
        connection: aiohttp connection (set after connection acquired).
    """

    query_id: str
    http_bytes: bytes
    timing: RequestTimingContext
    is_streaming: bool
    url: URL
    request_info: RequestInfo
    process: Callable[[], Any] | None = None
    response: ResponseData | None = field(default=None, repr=False)
    connection: Any | None = field(default=None, repr=False)

    def release(self) -> None:
        """Release connection back to pool."""
        if self.connection is not None:
            self.connection.release()
            self.connection = None

    def __del__(self) -> None:
        """Release connection on garbage collection."""
        self.release()


@dataclass(slots=True)
class ResponseData:
    """
    Container for HTTP response data from aiohttp protocol.

    Wraps the raw response components from aiohttp's ResponseHandler.read()
    into a minimal structure for processing. Provides async methods to
    consume the response body.

    Attributes:
        status: HTTP status code (e.g., 200, 404, 500)
        reason: HTTP reason phrase (e.g., "OK", "Not Found")
        headers: Response headers as case-insensitive multidict
        content: StreamReader for async body consumption
    """

    status: int
    reason: str
    headers: CIMultiDictProxy
    content: StreamReader

    async def read(self) -> bytes:
        """Read entire response body as bytes."""
        return await self.content.read()

    async def text(self, encoding: str = "utf-8") -> str:
        """Read response body as decoded text."""
        return (await self.content.read()).decode(encoding)
