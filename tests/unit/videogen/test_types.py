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

"""Unit tests for videogen Pydantic wire models (trtllm-serve native API)."""

import pytest
from inference_endpoint.videogen.types import (
    VideoPathRequest,
    VideoPathResponse,
    VideoPayloadResponse,
)
from pydantic import ValidationError


class TestVideoPathRequest:
    @pytest.mark.unit
    def test_defaults(self):
        req = VideoPathRequest(prompt="a cat")
        assert req.negative_prompt == ""
        assert req.size == "720x1280"
        assert req.seconds == pytest.approx(5.0)
        assert req.fps == 16
        assert req.num_inference_steps == 20
        assert req.guidance_scale == pytest.approx(4.0)
        assert req.guidance_scale_2 == pytest.approx(3.0)
        assert req.seed == 42
        assert req.output_format == "auto"
        assert req.response_format == "video_path"

    @pytest.mark.unit
    def test_custom_values(self):
        req = VideoPathRequest(
            prompt="ocean waves",
            seed=99,
            fps=24,
            size="640x480",
            guidance_scale=6.0,
            num_inference_steps=50,
        )
        assert req.seed == 99
        assert req.fps == 24
        assert req.size == "640x480"
        assert req.guidance_scale == pytest.approx(6.0)
        assert req.num_inference_steps == 50

    @pytest.mark.unit
    def test_response_format_video_bytes(self):
        req = VideoPathRequest(prompt="test", response_format="video_bytes")
        assert req.response_format == "video_bytes"

    @pytest.mark.unit
    def test_invalid_output_format_rejected(self):
        with pytest.raises(ValidationError):
            VideoPathRequest(prompt="test", output_format="invalid")

    @pytest.mark.unit
    def test_invalid_response_format_rejected(self):
        with pytest.raises(ValidationError):
            VideoPathRequest(prompt="test", response_format="invalid")


class TestVideoPathResponse:
    @pytest.mark.unit
    def test_basic(self):
        resp = VideoPathResponse(video_id="vid_001", video_path="/lustre/vid_001.mp4")
        assert resp.video_id == "vid_001"
        assert resp.video_path == "/lustre/vid_001.mp4"

    @pytest.mark.unit
    def test_roundtrip_json(self):
        resp = VideoPathResponse(video_id="vid_abc", video_path="/data/vid_abc.mp4")
        restored = VideoPathResponse.model_validate_json(resp.model_dump_json())
        assert restored.video_id == resp.video_id
        assert restored.video_path == resp.video_path


class TestVideoPayloadResponse:
    @pytest.mark.unit
    def test_basic(self):
        resp = VideoPayloadResponse(video_id="vid_001", video_bytes="AAEC")
        assert resp.video_id == "vid_001"
        assert resp.video_bytes == "AAEC"

    @pytest.mark.unit
    def test_roundtrip_json(self):
        resp = VideoPayloadResponse(video_id="vid_abc", video_bytes="dGVzdA==")
        restored = VideoPayloadResponse.model_validate_json(resp.model_dump_json())
        assert restored.video_id == resp.video_id
        assert restored.video_bytes == resp.video_bytes
