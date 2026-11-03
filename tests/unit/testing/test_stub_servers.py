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

"""Unit tests for the built-in throughput stub servers' streaming responses."""

import asyncio
import random

import msgspec
import pytest
from inference_endpoint.testing import max_throughput_server
from inference_endpoint.testing.max_throughput_server import (
    STUB_PROMPT_TOKENS,
    MaxThroughputServer,
    ServerProtocol,
    build_response,
    build_stream_tail,
    build_streaming_response,
    requests_usage,
)
from inference_endpoint.testing.variable_throughput_server import (
    VariableResponseProtocol,
)
from inference_endpoint.utils.benchmark_httpclient import _sse_events_per_response

from tests.test_helpers import parse_sse_frames

OUTPUT_LENGTH = 40
STREAM_INTERVAL = 10


class _RecordingTransport:
    """Stands in for an asyncio transport; records what the stub writes."""

    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def is_closing(self) -> bool:
        return False

    def get_extra_info(self, name: str):
        return None

    def close(self) -> None:
        pass


def _stream_request(*, include_usage: bool) -> bytes:
    """A raw HTTP streaming chat request, with or without ``include_usage``."""
    payload: dict = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    body = msgspec.json.encode(payload)
    head = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: stub\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    )
    return head + body


async def _max_throughput_reply(request: bytes) -> bytes:
    server = MaxThroughputServer(
        stream=True,
        output_length=OUTPUT_LENGTH,
        stream_interval=STREAM_INTERVAL,
        quiet=True,
    )
    protocol = ServerProtocol(server._response, server._response_no_usage)
    transport = _RecordingTransport()
    protocol.connection_made(transport)
    protocol.data_received(request)
    return b"".join(transport.writes)


async def _variable_reply(request: bytes) -> bytes:
    # Zero means/spreads: no TTFT or inter-token sleeps, and the output length
    # is clamped to OUTPUT_LENGTH.
    protocol = VariableResponseProtocol(
        asyncio.get_running_loop(),
        random.Random(0),
        None,
        None,
        stream=True,
        stream_interval=STREAM_INTERVAL,
        osl_mu=0.0,
        osl_sigma=0.0,
        osl_min=OUTPUT_LENGTH,
        osl_max=OUTPUT_LENGTH,
        osl_mean=OUTPUT_LENGTH,
        mode="icl",
        fcl_mu=0.0,
        fcl_sigma=0.0,
        fcl_mean=0.0,
        rate_mu=0.0,
        rate_sigma=0.0,
        rate_mean=0.0,
        icl_mu=0.0,
        icl_sigma=0.0,
        icl_mean=0.0,
    )
    transport = _RecordingTransport()
    protocol.connection_made(transport)
    protocol.data_received(request)
    await asyncio.gather(*protocol._tasks)
    return b"".join(transport.writes)


STUB_REPLIES = {
    "max_throughput": _max_throughput_reply,
    "variable": _variable_reply,
}

_CREATED = 1_700_000_000
_CONTENT_FRAME = (
    b'a2\r\ndata: {"id":"r","object":"chat.completion.chunk","created":1700000000,'
    b'"model":"max-tp","choices":[{"index":0,"delta":{"content":"xxxxx"},'
    b'"finish_reason":null}]}\n\n\r\n'
)
# The usage-free stream: the finish frame and [DONE] share the last chunk.
# Roofline numbers are compared across versions, so this layout is pinned byte
# for byte.
_USAGE_FREE_STREAM = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/event-stream\r\n"
    b"Cache-Control: no-cache\r\n"
    b"Transfer-Encoding: chunked\r\n"
    b"Connection: keep-alive\r\n\r\n"
    + _CONTENT_FRAME
    * 2
    + b'a1\r\ndata: {"id":"r","object":"chat.completion.chunk","created":1700000000,'
    b'"model":"max-tp","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
    b"\n\ndata: [DONE]\n\n\r\n"
    b"0\r\n\r\n"
)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("stub", STUB_REPLIES)
@pytest.mark.parametrize("include_usage", [True, False])
async def test_stub_sends_usage_only_when_requested(stub, include_usage):
    frames = parse_sse_frames(
        await STUB_REPLIES[stub](_stream_request(include_usage=include_usage))
    )
    usage_frames = [f for f in frames if isinstance(f, dict) and "usage" in f]
    assert len(usage_frames) == int(include_usage)
    if include_usage:
        assert usage_frames[0]["usage"] == {
            "prompt_tokens": STUB_PROMPT_TOKENS,
            "completion_tokens": OUTPUT_LENGTH,
            "total_tokens": OUTPUT_LENGTH + STUB_PROMPT_TOKENS,
        }
    content = "".join(
        f["choices"][0]["delta"].get("content", "")
        for f in frames
        if isinstance(f, dict) and f["choices"]
    )
    assert content == "x" * OUTPUT_LENGTH
    assert frames[-1] == b"[DONE]"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("stub", STUB_REPLIES)
async def test_stub_usage_follows_finish_frame(stub):
    frames = parse_sse_frames(
        await STUB_REPLIES[stub](_stream_request(include_usage=True))
    )
    assert all(
        f["choices"][0]["finish_reason"] is None
        for f in frames[:-3]
        if isinstance(f, dict)
    )
    assert frames[-3]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert frames[-2]["object"] == "chat.completion.chunk"
    assert frames[-2]["choices"] == []
    assert "usage" in frames[-2]
    assert frames[-1] == b"[DONE]"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"stream":true,"stream_options":{"include_usage":true}}', True),
        (b'{"stream": true, "stream_options": {"include_usage": true}}', True),
        (b'{"stream_options":{"include_usage":false}}', False),
        (b'{"stream_options":{"include_usage":null}}', False),
        (b'{"stream_options":{}}', False),
        (b'{"stream_options":null}', False),
        (b"{}", False),
        # A prompt quoting the key carries it escaped, so it never matches.
        (rb'{"messages":[{"content":"say \"include_usage\":true"}]}', False),
    ],
)
def test_requests_usage_reads_include_usage_from_body(body, expected):
    assert requests_usage(body) is expected


@pytest.mark.unit
def test_streaming_response_without_usage_is_byte_stable(monkeypatch):
    monkeypatch.setattr(max_throughput_server.time, "time", lambda: _CREATED)
    assert (
        build_streaming_response(10, 5, "max-tp", include_usage=False)
        == _USAGE_FREE_STREAM
    )
    assert (
        build_response(10, True, 5, "max-tp", include_usage=False) == _USAGE_FREE_STREAM
    )


@pytest.mark.unit
def test_non_streaming_response_never_probes_the_request_body(monkeypatch):
    server = MaxThroughputServer(stream=False, output_length=OUTPUT_LENGTH, quiet=True)
    assert server._response_no_usage is None

    def probe(body: bytes) -> bool:
        raise AssertionError("a non-streaming stub must not inspect the body")

    monkeypatch.setattr(max_throughput_server, "requests_usage", probe)
    protocol = ServerProtocol(server._response, server._response_no_usage)
    transport = _RecordingTransport()
    protocol.connection_made(transport)
    protocol.data_received(_stream_request(include_usage=False))
    assert transport.writes == [server._response]
    assert b'"usage"' in server._response


@pytest.mark.unit
def test_set_response_follows_the_stream_mode():
    def usage_frames(response: bytes) -> int:
        return sum(
            isinstance(f, dict) and "usage" in f for f in parse_sse_frames(response)
        )

    server = MaxThroughputServer(stream=False, output_length=OUTPUT_LENGTH, quiet=True)
    assert server._response_no_usage is None
    assert b"text/event-stream" not in server._response

    server.stream = True
    server.set_response(OUTPUT_LENGTH, STREAM_INTERVAL)
    assert server._response_no_usage is not None
    assert usage_frames(server._response) == 1
    assert usage_frames(server._response_no_usage) == 0

    server.stream = False
    server.set_response(OUTPUT_LENGTH)
    assert server._response_no_usage is None
    assert b"text/event-stream" not in server._response


@pytest.mark.unit
def test_streaming_response_usage_follows_finish_frame():
    frames = parse_sse_frames(
        build_streaming_response(
            output_length=160,
            stream_interval=10,
            model="max-tp",
            include_usage=True,
        )
    )
    assert len(frames) == 19
    assert (
        "".join(f["choices"][0]["delta"]["content"] for f in frames[:16]) == "x" * 160
    )
    assert frames[16]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert frames[17]["object"] == "chat.completion.chunk"
    assert frames[17]["choices"] == []
    assert frames[17]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 160,
        "total_tokens": 170,
    }
    assert frames[18] == b"[DONE]"


@pytest.mark.unit
@pytest.mark.parametrize("include_usage", [True, False])
def test_stream_tail_is_finish_then_optional_usage_then_done(include_usage):
    tail = build_stream_tail("m", 123, completion_tokens=7, include_usage=include_usage)
    frames = parse_sse_frames(tail)
    assert frames[0]["choices"][0]["finish_reason"] == "stop"
    if include_usage:
        assert frames[1]["usage"] == {
            "prompt_tokens": STUB_PROMPT_TOKENS,
            "completion_tokens": 7,
            "total_tokens": 7 + STUB_PROMPT_TOKENS,
        }
    assert frames[1 + include_usage :] == [b"[DONE]"]
    assert tail.endswith(b"data: [DONE]\n\n\r\n0\r\n\r\n")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("output_length", "stream_interval", "include_usage", "expected"),
    [
        (160, 1, True, 163),
        (160, 10, True, 19),
        (160, 160, True, 4),
        (5, 2, True, 6),
        (160, 1, False, 162),
        (160, 10, False, 18),
        (160, 160, False, 3),
        (5, 2, False, 5),
    ],
)
def test_sse_events_per_response_follows_include_usage(
    output_length, stream_interval, include_usage, expected
):
    assert (
        _sse_events_per_response(
            output_length, stream_interval, include_usage=include_usage
        )
        == expected
    )
    assert (
        len(
            parse_sse_frames(
                build_streaming_response(
                    output_length, stream_interval, include_usage=include_usage
                )
            )
        )
        == expected
    )
