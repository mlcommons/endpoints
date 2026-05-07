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

"""Wire models for trtllm-serve POST /v1/videos/generations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoPathRequest(BaseModel):
    """Request body for POST /v1/videos/generations.

    Matches trtllm-serve's VideoGenerationRequest. All fields have MLPerf defaults
    so that only `prompt` is required from the dataset.

    `response_format` defaults to "video_path"; callers wanting accuracy-mode
    behaviour must inject `response_format="video_bytes"` into `query.data`
    via the dataset (the adapter does not derive it from `benchmark_mode`).
    """

    prompt: str
    negative_prompt: str | None = Field(
        default=None,
        description=(
            "Text describing what to avoid. None means the field is omitted "
            "from the JSON payload so trtllm-serve can apply its model default. "
            "The bundled MLPerf prompts dataset carries the canonical negative "
            "prompt per row."
        ),
    )
    size: str = Field(default="720x1280", description="Frame size in 'WxH' format.")
    seconds: float = Field(
        default=5.0,
        description="Video duration. 81 frames @ ~16.2 fps = 5 s (MLPerf standard).",
    )
    fps: int = Field(default=16, description="Frames per second (MLPerf: 16).")
    num_inference_steps: int = Field(
        default=20, description="Denoising steps (MLPerf: 20)."
    )
    guidance_scale: float = Field(
        default=4.0, description="CFG guidance scale (MLPerf: 4.0)."
    )
    guidance_scale_2: float = Field(
        default=3.0,
        description="Secondary guidance scale for null-text CFG (MLPerf: 3.0).",
    )
    seed: int = Field(default=42, description="Random seed (MLPerf: 42).")
    latent_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to a pre-computed latent tensor (.pt file) on shared "
            "storage accessible to the server (e.g. Lustre). When provided, the "
            "server uses this tensor as the initial denoising noise instead of "
            "sampling random noise. MLPerf uses a fixed latent for reproducibility."
        ),
    )
    output_format: Literal["mp4", "avi", "auto"] = "auto"
    response_format: Literal["video_bytes", "video_path"] = "video_path"


class VideoPathResponse(BaseModel):
    """Response body from trtllm-serve when response_format='video_path'.

    The server saves the encoded video to its local filesystem (e.g. Lustre)
    and returns the absolute path instead of video bytes.
    Used in perf mode: MLPerf defines query completion as server finishing
    generation, so bytes do not need to cross the wire.

    `extra="forbid"` so a server returning unexpected fields (or both
    video_path and video_bytes) fails loudly at deserialisation.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str
    video_path: str


class VideoPayloadResponse(BaseModel):
    """Response body from trtllm-serve when response_format='video_bytes'.

    Used in accuracy mode. video_bytes is the base64-encoded video content,
    which the accuracy evaluator decodes to score quality (e.g. FVD).
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str
    video_bytes: str  # base64-encoded video content
