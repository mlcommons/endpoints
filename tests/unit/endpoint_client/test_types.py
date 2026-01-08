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

"""Tests for endpoint_client types module.

Uses h11 library to validate HTTP/1.1 wire format compliance per RFC 7230.
"""

import h11
from inference_endpoint.endpoint_client.types import HttpRequestTemplate

# =============================================================================
# HTTP Request Building Tests (RFC 7230 Compliance)
# =============================================================================


class TestHttpRequestBuilding:
    """Test HTTP/1.1 request construction using h11 for RFC 7230 validation."""

    def _parse_request(self, request_bytes: bytes) -> h11.Request:
        """Parse HTTP request bytes using h11, returns parsed request."""
        conn = h11.Connection(h11.SERVER)
        conn.receive_data(request_bytes)
        event = conn.next_event()
        assert isinstance(event, h11.Request)
        return event

    def test_http_request_rfc7230_compliance(self):
        """Test HTTP/1.1 request format per RFC 7230: method, headers, body."""
        template = HttpRequestTemplate(
            request_line=b"POST /v1/chat/completions HTTP/1.1\r\n",
            host_header=b"Host: localhost:8080\r\n",
        )

        body = b'{"model": "test", "messages": []}'
        headers = {"Content-Type": "application/json"}
        request_bytes = template.build_request(body, headers)

        # Validate with h11
        parsed = self._parse_request(request_bytes)
        assert parsed.method == b"POST"
        assert parsed.target == b"/v1/chat/completions"

        # Verify required headers (RFC 7230)
        headers_dict = dict(parsed.headers)
        assert headers_dict[b"host"] == b"localhost:8080"  # Host required
        assert headers_dict[b"content-type"] == b"application/json"
        assert int(headers_dict[b"content-length"]) == len(
            body
        )  # Content-Length matches

    def test_http_request_edge_cases(self):
        """Test edge cases: empty body, UTF-8 multi-byte, header passthrough."""
        template = HttpRequestTemplate(
            request_line=b"POST /api HTTP/1.1\r\n",
            host_header=b"Host: localhost\r\n",
        )

        # Empty body -> Content-Length: 0
        empty_request = template.build_request(b"", {})
        parsed = self._parse_request(empty_request)
        assert dict(parsed.headers)[b"content-length"] == b"0"

        # UTF-8 multi-byte: Content-Length is byte count
        utf8_body = '{"text": "日本語"}'.encode()
        utf8_request = template.build_request(utf8_body, {})
        parsed = self._parse_request(utf8_request)
        assert int(dict(parsed.headers)[b"content-length"]) == len(utf8_body)

        # Custom headers passed through
        custom_request = template.build_request(b"{}", {"X-Custom": "value"})
        parsed = self._parse_request(custom_request)
        assert dict(parsed.headers)[b"x-custom"] == b"value"
