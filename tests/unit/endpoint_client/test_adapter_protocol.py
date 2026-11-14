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

"""Tests for HttpRequestAdapter: the encode_query contract and parse_sse_chunk."""

import inspect
import logging
from typing import Any

import msgspec
import pytest
from inference_endpoint.config.schema import ModelParams
from inference_endpoint.core.types import Query
from inference_endpoint.dataset_manager.transforms import Transform
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter
from inference_endpoint.endpoint_client.config import ADAPTER_MAP
from inference_endpoint.openai.completions_adapter import OpenAITextCompletionsAdapter
from inference_endpoint.openai.openai_adapter import OpenAIAdapter
from inference_endpoint.openai.openai_msgspec_adapter import OpenAIMsgspecAdapter
from inference_endpoint.sglang.adapter import SGLangGenerateAdapter
from inference_endpoint.videogen.adapter import VideoGenAdapter


class _SimpleAdapter(HttpRequestAdapter):
    """Minimal adapter: raises DecodeError for b'BAD', returns {"x": 1} otherwise."""

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        return []

    @classmethod
    def encode_query(cls, query: Any, *, include_usage: bool) -> bytes:
        return b""

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> Any:
        return None

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> Any:
        if json_bytes.strip() == b"BAD":
            raise msgspec.DecodeError("bad frame")
        return {"x": 1}


_ADAPTERS = [
    OpenAIMsgspecAdapter,
    OpenAITextCompletionsAdapter,
    SGLangGenerateAdapter,
    VideoGenAdapter,
    OpenAIAdapter,
]


@pytest.mark.unit
def test_adapter_list_covers_every_api_type():
    names = {f"{a.__module__}.{a.__qualname__}" for a in _ADAPTERS}
    assert set(ADAPTER_MAP.values()) <= names


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter", [HttpRequestAdapter, *_ADAPTERS], ids=lambda a: a.__name__
)
def test_encode_query_requires_keyword_only_include_usage(adapter):
    param = inspect.signature(adapter.encode_query).parameters["include_usage"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, data",
    [
        (OpenAIAdapter, {"model": "m", "prompt": "hi", "stream": True}),
        (SGLangGenerateAdapter, {"input_tokens": [1, 2], "stream": True}),
        (VideoGenAdapter, {"prompt": "a cat"}),
    ],
    ids=["openai-test-only", "sglang", "videogen"],
)
def test_adapters_without_usage_requests_ignore_include_usage(adapter, data):
    query = Query(id="q", data=data)
    assert adapter.encode_query(query, include_usage=True) == adapter.encode_query(
        query, include_usage=False
    )


@pytest.mark.unit
def test_parse_sse_chunk_skips_bad_frame_and_keeps_valid():
    """A malformed SSE frame is skipped; surrounding valid frames are preserved."""
    buffer = b'data: {"ok":1}\n\ndata: BAD\n\ndata: {"ok":1}\n\n'
    end_pos = len(buffer)
    result = _SimpleAdapter.parse_sse_chunk(buffer, end_pos)
    assert result == [{"x": 1}, {"x": 1}]


_LOGGER = "inference_endpoint.endpoint_client.adapter_protocol"


class _StrictAdapter(_SimpleAdapter):
    """Decodes frames as dict[str, int] so bad frames raise real msgspec errors."""

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> Any:
        return msgspec.json.decode(json_bytes, type=dict[str, int])


def _sse(*frames: bytes) -> bytes:
    return b"".join(b"data: " + f + b"\n\n" for f in frames)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad", [b"{not json}", b'{"x":"s"}'], ids=["decode-error", "validation-error"]
)
def test_parse_sse_chunk_keeps_frames_after_bad_frame(bad, caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    buffer = _sse(b'{"x":1}', bad, b'{"x":2}', b'{"x":3}')
    result = _StrictAdapter.parse_sse_chunk(buffer, len(buffer))
    assert result == [{"x": 1}, {"x": 2}, {"x": 3}]
    assert len(caplog.records) == 1


@pytest.mark.unit
def test_parse_sse_chunk_warns_once_per_call(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    buffer = _sse(b"{not json}", b'{"x":1}', b'{"x":"s"}')
    assert _StrictAdapter.parse_sse_chunk(buffer, len(buffer)) == [{"x": 1}]
    assert [r.getMessage() for r in caplog.records] == [
        "skipping 2 malformed SSE frame(s) (DecodeError)"
    ]


@pytest.mark.unit
def test_parse_sse_chunk_clean_batch_logs_nothing(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    buffer = _sse(b'{"x":1}', b'{"x":2}')
    assert _StrictAdapter.parse_sse_chunk(buffer, len(buffer)) == [{"x": 1}, {"x": 2}]
    assert caplog.records == []
