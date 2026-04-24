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

"""Adapter for the WAN 2.2 trtllm-serve POST /v1/videos/generations endpoint."""

import json
from typing import TYPE_CHECKING, Any

from inference_endpoint.core.types import (
    Query,
    QueryResult,
    StreamChunk,
    TextModelOutput,
)
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from .types import VideoPathRequest, VideoPathResponse, VideoPayloadResponse

if TYPE_CHECKING:
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import Transform


class VideoGenAdapter(HttpRequestAdapter):
    """Adapter for trtllm-serve POST /v1/videos/generations.

    Supports both server response formats via query.data["response_format"]:
    - "video_bytes" (default): server returns base64-encoded video content.
      Suitable for accuracy evaluation in a single pass.
    - "video_path": server saves video to shared storage (Lustre) and returns
      only the file path. Avoids 3-5 MB payloads over HTTP + ZMQ per request.
    """

    @classmethod
    def dataset_transforms(cls, model_params: "ModelParams") -> "list[Transform]":
        return []

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Serialise query.data to VideoPathRequest JSON bytes.

        Only `prompt` is required. All other fields fall back to MLPerf defaults
        defined in VideoPathRequest but can be overridden via query.data.
        Pass response_format="video_path" in query.data to request a Lustre path
        instead of inline video bytes.
        """
        data = query.data
        if "prompt" not in data:
            raise KeyError(
                f"'prompt' not found in query.data keys: {list(data.keys())}"
            )
        req = VideoPathRequest(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", ""),
            size=data.get("size", "720x1280"),
            seconds=data.get("seconds", 5.0),
            fps=data.get("fps", 16),
            num_inference_steps=data.get("num_inference_steps", 20),
            guidance_scale=data.get("guidance_scale", 4.0),
            guidance_scale_2=data.get("guidance_scale_2", 3.0),
            seed=data.get("seed", 42),
            output_format=data.get("output_format", "auto"),
            response_format=data.get("response_format", "video_bytes"),
        )
        return req.model_dump_json().encode()

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """Deserialise trtllm-serve response JSON bytes to QueryResult.

        Dispatches on the response shape:
        - "video_bytes" response: metadata["video_bytes"] holds the base64 payload.
        - "video_path" response: metadata["video_path"] holds the Lustre file path.
        """
        raw = json.loads(response_bytes)
        if "video_bytes" in raw:
            resp = VideoPayloadResponse.model_validate(raw)
            return QueryResult(
                id=query_id,
                response_output=TextModelOutput(output=resp.video_id),
                metadata={"video_bytes": resp.video_bytes},
            )
        resp = VideoPathResponse.model_validate(raw)
        return QueryResult(
            id=query_id,
            response_output=TextModelOutput(output=resp.video_id),
            metadata={"video_path": resp.video_path},
        )

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> str:
        raise NotImplementedError("WAN 2.2 does not use SSE streaming")


class VideoGenAccumulator:
    """No-op SSE accumulator satisfying SSEAccumulatorProtocol.

    WAN 2.2 uses non-streaming HTTP. This class exists only to satisfy
    the HTTPClientConfig.accumulator type contract.
    """

    def __init__(self, query_id: str, stream_all_chunks: bool) -> None:
        self.query_id = query_id
        # stream_all_chunks is intentionally ignored: WAN 2.2 is non-streaming.

    def add_chunk(self, delta: Any) -> StreamChunk | None:
        return None

    def get_final_output(self) -> QueryResult:
        return QueryResult(id=self.query_id)
