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

"""Unit tests for VideoGenAdapter and VideoGenAccumulator (trtllm-native API)."""

import json

import pytest
from inference_endpoint.config.schema import ModelParams
from inference_endpoint.core.types import APIType, Query, QueryResult
from inference_endpoint.endpoint_client.accumulator_protocol import (
    SSEAccumulatorProtocol,
)
from inference_endpoint.videogen.adapter import VideoGenAccumulator, VideoGenAdapter
from inference_endpoint.videogen.types import VideoPayloadResponse


@pytest.mark.unit
class TestVideoGenAdapter:
    def test_encode_query_produces_valid_json(self):
        query = Query(id="q1", data={"prompt": "a golden retriever running"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["prompt"] == "a golden retriever running"

    def test_encode_query_always_requests_video_bytes(self):
        """MLPerf runs perf+accuracy in one pass — always request video_bytes."""
        query = Query(id="q1", data={"prompt": "ocean waves"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["response_format"] == "video_bytes"

    def test_encode_query_uses_mlperf_defaults(self):
        query = Query(id="q1", data={"prompt": "ocean waves"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["fps"] == 16
        assert payload["seconds"] == pytest.approx(5.0)
        assert payload["size"] == "720x1280"
        assert payload["num_inference_steps"] == 20
        assert payload["guidance_scale"] == pytest.approx(4.0)
        assert payload["guidance_scale_2"] == pytest.approx(3.0)
        assert payload["seed"] == 42

    def test_encode_query_allows_override_via_data(self):
        query = Query(
            id="q1",
            data={"prompt": "test", "seed": 99, "fps": 24, "guidance_scale": 6.0},
        )
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["seed"] == 99
        assert payload["fps"] == 24
        assert payload["guidance_scale"] == pytest.approx(6.0)

    def test_encode_query_includes_negative_prompt(self):
        query = Query(id="q1", data={"prompt": "test", "negative_prompt": "blurry"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["negative_prompt"] == "blurry"

    def test_encode_query_omits_negative_prompt_when_absent(self):
        """exclude_none=True so server can apply its own default."""
        query = Query(id="q1", data={"prompt": "test"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert "negative_prompt" not in payload

    def test_encode_query_includes_latent_path(self):
        query = Query(
            id="q1", data={"prompt": "test", "latent_path": "/lustre/fixed_latent.pt"}
        )
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert payload["latent_path"] == "/lustre/fixed_latent.pt"

    def test_encode_query_omits_latent_path_when_absent(self):
        query = Query(id="q1", data={"prompt": "test"})
        payload = json.loads(VideoGenAdapter.encode_query(query))
        assert "latent_path" not in payload

    def test_encode_query_missing_prompt_raises(self):
        query = Query(id="q1", data={"seed": 42})
        with pytest.raises(KeyError):
            VideoGenAdapter.encode_query(query)

    def test_decode_response_returns_video_bytes_in_metadata(self):
        resp = VideoPayloadResponse(
            video_id="video_abc123",
            video_bytes="dGVzdCB2aWRlbyBjb250ZW50",
        )
        result = VideoGenAdapter.decode_response(resp.model_dump_json().encode(), "q1")
        assert isinstance(result, QueryResult)
        assert result.id == "q1"
        assert result.error is None
        assert result.metadata["video_bytes"] == "dGVzdCB2aWRlbyBjb250ZW50"

    def test_decode_response_video_id_in_output(self):
        resp = VideoPayloadResponse(video_id="vid_xyz", video_bytes="AAEC")
        result = VideoGenAdapter.decode_response(resp.model_dump_json().encode(), "q1")
        assert result.response_output.output == "vid_xyz"

    def test_decode_sse_message_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            VideoGenAdapter.decode_sse_message(b"{}")

    def test_dataset_transforms_returns_empty_list(self):
        params = ModelParams()
        assert VideoGenAdapter.dataset_transforms(params) == []

    def test_default_route_is_trtllm_native(self):
        assert APIType.VIDEOGEN.default_route() == "/v1/videos/generations"


@pytest.mark.unit
class TestVideoGenAccumulator:
    def test_add_chunk_always_returns_none(self):
        acc = VideoGenAccumulator(query_id="q1", stream_all_chunks=True)
        assert acc.add_chunk("anything") is None
        assert acc.add_chunk(None) is None

    def test_get_final_output_returns_query_result_with_id(self):
        acc = VideoGenAccumulator(query_id="q1", stream_all_chunks=False)
        result = acc.get_final_output()
        assert isinstance(result, QueryResult)
        assert result.id == "q1"

    def test_satisfies_sse_accumulator_protocol(self):
        assert isinstance(VideoGenAccumulator("q1", False), SSEAccumulatorProtocol)
