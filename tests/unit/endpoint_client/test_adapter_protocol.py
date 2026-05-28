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


class _SimpleAdapter(HttpRequestAdapter):
    """Minimal adapter: raises DecodeError for b'BAD', returns {"x": 1} otherwise."""

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
        if json_bytes.strip() == b"BAD":
            raise msgspec.DecodeError("bad frame")
        return {"x": 1}


@pytest.mark.unit
def test_parse_sse_chunk_skips_bad_frame_and_keeps_valid():
    """A malformed SSE frame is skipped; surrounding valid frames are preserved."""
    buffer = b'data: {"ok":1}\n\ndata: BAD\n\ndata: {"ok":1}\n\n'
    end_pos = len(buffer)
    result = _SimpleAdapter.parse_sse_chunk(buffer, end_pos)
    assert result == [{"x": 1}, {"x": 1}]
