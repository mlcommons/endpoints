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

"""Integration tests: VideoGenAdapter encode/decode round-trip against mock trtllm-serve."""

import base64
import json
import urllib.request
from urllib.error import HTTPError

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.videogen.adapter import VideoGenAdapter
from pydantic import ValidationError

from .conftest import DUMMY_VIDEO_BYTES, MockTrtllmServe, MockTrtllmServeError


def _post(url: str, body: bytes) -> tuple[int, bytes]:
    """Synchronous HTTP POST returning (status_code, response_body)."""
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except HTTPError as e:
        return e.code, e.read()


@pytest.mark.integration
class TestVideoGenAdapterRoundTrip:
    """Verify encode_query → HTTP POST → decode_response against mock trtllm-serve."""

    def test_successful_generation_returns_video_bytes(
        self, mock_trtllm_serve: MockTrtllmServe
    ) -> None:
        query = Query(id="q1", data={"prompt": "a golden retriever running on a beach"})
        request_bytes = VideoGenAdapter.encode_query(query)

        status, content = _post(
            f"{mock_trtllm_serve.url}/v1/videos/generations", request_bytes
        )
        assert status == 200

        result = VideoGenAdapter.decode_response(content, query.id)
        assert result.id == "q1"
        assert result.error is None
        assert "video_bytes" in result.metadata
        decoded = base64.b64decode(result.metadata["video_bytes"])
        assert decoded == DUMMY_VIDEO_BYTES

    def test_request_default_response_format_is_video_path(
        self, mock_trtllm_serve: MockTrtllmServe
    ) -> None:
        query = Query(id="q2", data={"prompt": "ocean waves at sunset"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["response_format"] == "video_path"

    def test_request_accuracy_mode_asks_for_video_bytes(
        self, mock_trtllm_serve: MockTrtllmServe
    ) -> None:
        query = Query(
            id="q2b",
            data={"prompt": "ocean waves at sunset", "response_format": "video_bytes"},
        )
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["response_format"] == "video_bytes"

    def test_request_carries_mlperf_defaults(
        self, mock_trtllm_serve: MockTrtllmServe
    ) -> None:
        query = Query(id="q3", data={"prompt": "ocean waves at sunset"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["fps"] == 16
        assert payload["seconds"] == pytest.approx(5.0)
        assert payload["size"] == "720x1280"
        assert payload["num_inference_steps"] == 20
        assert payload["guidance_scale"] == pytest.approx(4.0)
        assert payload["seed"] == 42


@pytest.mark.integration
class TestVideoGenAdapterErrorHandling:
    def test_http_500_response_raises_on_decode(
        self, mock_trtllm_serve_error: MockTrtllmServeError
    ) -> None:
        query = Query(id="q4", data={"prompt": "a cat"})
        status, content = _post(
            f"{mock_trtllm_serve_error.url}/v1/videos/generations",
            VideoGenAdapter.encode_query(query),
        )
        assert status == 500
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            VideoGenAdapter.decode_response(content, query.id)

    def test_malformed_json_response_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            VideoGenAdapter.decode_response(b"not json at all", "q5")

    def test_missing_video_bytes_field_raises_validation_error(self) -> None:
        bad_body = b'{"video_id": "vid_001"}'  # missing video_bytes
        with pytest.raises(ValidationError):
            VideoGenAdapter.decode_response(bad_body, "q6")

    def test_missing_video_id_field_raises_validation_error(self) -> None:
        bad_body = b'{"video_bytes": "AAEC"}'  # missing video_id
        with pytest.raises(ValidationError):
            VideoGenAdapter.decode_response(bad_body, "q7")
