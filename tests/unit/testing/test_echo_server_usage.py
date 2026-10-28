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

"""Unit tests for the echo server's token usage reporting."""

import json

import aiohttp
import pytest
from inference_endpoint.openai.completions_adapter import OpenAITextCompletionsAdapter

from tests.test_helpers import parse_sse_frames

PROMPT = "hello  big   world"  # 18 chars; streamed as "hello big world" (15)
ERR = b"Stream options can only be defined when `stream=True`."
INCLUDE = {"stream": True, "stream_options": {"include_usage": True}}


async def _post(url, payload) -> tuple[int, bytes]:
    async with aiohttp.ClientSession() as s, s.post(url, json=payload) as r:
        return r.status, await r.read()


def _chat(content=PROMPT, **extra):
    return {"model": "m", "messages": [{"role": "user", "content": content}], **extra}


def _url(server, path="/v1/chat/completions"):
    return f"{server.url}{path}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_non_streaming_reports_usage(mock_http_echo_server):
    status, body = await _post(_url(mock_http_echo_server), _chat())
    assert (
        status == 200 and json.loads(body)["choices"][0]["message"]["content"] == PROMPT
    )
    assert json.loads(body)["usage"] == {
        "prompt_tokens": 18,
        "completion_tokens": 18,
        "total_tokens": 36,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_streaming_usage_counts_emitted_text(mock_http_echo_server):
    frames = parse_sse_frames(
        (await _post(_url(mock_http_echo_server), _chat(**INCLUDE)))[1]
    )
    assert (
        "".join(f["choices"][0]["delta"]["content"] for f in frames[:-3])
        == "hello big world"
    )
    assert frames[-3]["choices"][0]["finish_reason"] == "stop"
    assert frames[-2]["choices"] == []
    assert frames[-2]["usage"] == {
        "prompt_tokens": 18,
        "completion_tokens": 15,
        "total_tokens": 33,
    }
    assert frames[-1] == b"[DONE]"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_max_osl_usage_counts_text_actually_sent(mock_http_echo_server):
    mock_http_echo_server.set_max_osl(8)  # "ab cd" -> "ab cdab " -> streamed "ab cdab"
    frames = parse_sse_frames(
        (await _post(_url(mock_http_echo_server), _chat("ab cd", **INCLUDE)))[1]
    )
    assert frames[-2]["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [{}, {"stream_options": None}, {"stream_options": {"include_usage": False}}],
)
async def test_streaming_without_include_usage_sends_no_usage(
    mock_http_echo_server, extra
):
    frames = parse_sse_frames(
        (await _post(_url(mock_http_echo_server), _chat(stream=True, **extra)))[1]
    )
    assert frames[-1] == b"[DONE]" and not any("usage" in f for f in frames[:-1])


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload", "expected"),
    [
        ("/v1/chat/completions", _chat(stream_options={"include_usage": True}), 400),
        (
            "/v1/chat/completions",
            _chat(stream=False, stream_options={"include_usage": True}),
            400,
        ),
        ("/v1/chat/completions", _chat(stream=False, stream_options={}), 200),
        (
            "/v1/completions",
            {"model": "m", "prompt": "hi", "stream_options": {"include_usage": True}},
            400,
        ),
    ],
)
async def test_stream_options_without_stream_is_rejected(
    mock_http_echo_server, path, payload, expected
):
    status, body = await _post(_url(mock_http_echo_server, path), payload)
    assert status == expected and (ERR in body) is (expected == 400)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_report_usage_false_never_sends_usage(mock_http_echo_server_factory):
    server = mock_http_echo_server_factory(report_usage=False)
    assert json.loads((await _post(_url(server), _chat()))[1])["usage"] is None
    server.set_max_osl(0)  # every output becomes ""
    frames = parse_sse_frames((await _post(_url(server), _chat(**INCLUDE)))[1])
    assert [f["choices"][0]["finish_reason"] for f in frames[:-1]] == ["stop"]
    assert frames[-1] == b"[DONE]"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_text_completions_route_reports_usage(mock_http_echo_server):
    url = _url(mock_http_echo_server, "/v1/completions")
    body = (await _post(url, {"model": "m", "prompt": "hi there"}))[1]
    assert (
        OpenAITextCompletionsAdapter.decode_response(
            body, "q"
        ).get_response_output_string()
        == "hi there"
    )
    assert json.loads(body)["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 8,
        "total_tokens": 16,
    }
    frames = parse_sse_frames(
        (await _post(url, {"model": "m", "prompt": [11, 22, 333], **INCLUDE}))[1]
    )
    assert [f["object"] for f in frames[:-1]] == ["text_completion"] * 5
    assert "".join(f["choices"][0]["text"] for f in frames[:3]) == "11 22 333"
    assert frames[3]["choices"][0]["finish_reason"] == "stop"
    assert frames[4]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 9,
        "total_tokens": 12,
    }
