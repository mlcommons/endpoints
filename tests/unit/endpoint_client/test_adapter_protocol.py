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

"""Tests for HttpRequestAdapter.parse_sse_chunk."""

from typing import Any

import msgspec
import pytest
from inference_endpoint.config.schema import ModelParams
from inference_endpoint.dataset_manager.transforms import Transform
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter
from inference_endpoint.sglang.adapter import SGLangGenerateAdapter


class _SimpleAdapter(HttpRequestAdapter):
    """Decode JSON objects with integer values."""

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        return []

    @classmethod
    def encode_query(cls, query: Any) -> bytes:
        return b""

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> Any:
        return None

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> Any:
        return msgspec.json.decode(json_bytes, type=dict[str, int])


@pytest.mark.unit
@pytest.mark.parametrize("bad_frame", [b'{"ok":}', b'{"ok":"bad"}'])
def test_parse_sse_chunk_skips_bad_frame_and_keeps_valid(bad_frame):
    """A malformed SSE frame is skipped; surrounding valid frames are preserved."""
    buffer = b'data: {"ok":1}\n\ndata: ' + bad_frame + b'\n\ndata: {"ok":2}\n\n'
    end_pos = len(buffer)
    result = _SimpleAdapter.parse_sse_chunk(buffer, end_pos)
    assert result == [{"ok": 1}, {"ok": 2}]


@pytest.mark.unit
@pytest.mark.parametrize("finish_reason", [None, {"type": "stop"}])
@pytest.mark.parametrize("latency", [{}, {"e2e_latency": None}, {"e2e_latency": 0.1}])
@pytest.mark.parametrize("retractions", [0, 2])
def test_sglang_response_metadata(finish_reason, latency, retractions):
    response = msgspec.json.encode(
        {
            "text": "Hello",
            "output_ids": [10],
            "meta_info": {
                "id": "sample",
                "finish_reason": finish_reason,
                "prompt_tokens": 3,
                "weight_version": "v1",
                "num_retractions": retractions,
                "completion_tokens": 1,
                "cached_tokens": 0,
                **latency,
            },
        }
    )
    result = SGLangGenerateAdapter.decode_response(response, "sample")
    assert result.get_response_output_string() == "Hello"
    assert result.metadata == {"token_ids": [10], "n_tokens": 1}
    frame = b"data: " + response + b"\n\n"
    chunks = SGLangGenerateAdapter.parse_sse_chunk(frame, len(frame))
    assert len(chunks) == 1
    assert chunks[0].text == "Hello"
    assert chunks[0].token_delta == [10]
    assert chunks[0].total_completion_tokens == 1
    assert chunks[0].has_retractions is (retractions > 0)
