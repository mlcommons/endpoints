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

"""Adapter for trtllm-serve's POST /v1/videos/generations endpoint."""

import json
from typing import TYPE_CHECKING, Any

from inference_endpoint.core.types import (
    Query,
    QueryResult,
    StreamChunk,
    TextModelOutput,
)
from inference_endpoint.dataset_manager.transforms import ColumnFilter
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from .types import VideoPathRequest, VideoPathResponse, VideoPayloadResponse

if TYPE_CHECKING:
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import Transform


class VideoGenAdapter(HttpRequestAdapter):
    """Adapter for trtllm-serve POST /v1/videos/generations.

    `response_format` is read from `query.data` (default "video_path") and
    is *not* derived from BenchmarkConfig.benchmark_mode. Callers that want
    accuracy-mode bytes must inject `response_format="video_bytes"` into the
    dataset rows — typically via an `AddStaticColumns` transform.
    """

    @classmethod
    def dataset_transforms(cls, model_params: "ModelParams") -> "list[Transform]":
        # ColumnFilter rejects unknown columns at dataset-load time so typos
        # (e.g. "negitive_prompt") fail loud instead of silently falling back
        # to server-side defaults.
        request_fields = list(VideoPathRequest.model_fields.keys())
        return [
            ColumnFilter(
                required_columns=["prompt"],
                optional_columns=[f for f in request_fields if f != "prompt"],
            ),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Serialise query.data to VideoPathRequest JSON bytes.

        Only `prompt` is required. All other fields fall back to defaults on
        VideoPathRequest but can be overridden via query.data. Streaming is
        not supported — `stream=True` raises.
        """
        data = query.data
        if "prompt" not in data:
            raise KeyError(
                f"'prompt' not found in query.data keys: {list(data.keys())}"
            )
        if data.get("stream"):
            raise ValueError(
                "VideoGenAdapter is non-streaming; remove `stream` from query.data."
            )
        known = VideoPathRequest.model_fields.keys()
        req = VideoPathRequest.model_validate({k: data[k] for k in known if k in data})
        # exclude_none so optional fields with value None fall back to
        # server-side defaults; fields explicitly set in query.data
        # (e.g. negative_prompt from the bundled JSONL) are forwarded.
        return req.model_dump_json(exclude_none=True).encode()

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """Deserialise trtllm-serve response JSON bytes to QueryResult.

        Dispatches on the response shape:
        - video_bytes response (str payload): metadata["video_bytes"].
        - video_path response: metadata["video_path"].
        """
        raw = json.loads(response_bytes)
        # Truthiness check, not key presence: a server that returns
        # `"video_bytes": null` belongs in the video_path branch.
        if isinstance(raw.get("video_bytes"), str):
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
        raise NotImplementedError("VideoGenAdapter does not use SSE streaming")


class VideoGenAccumulator:
    """SSE accumulator stub for HTTPClientConfig contract.

    Video generation requests are non-streaming HTTP, so this class should
    never be exercised. `get_final_output` raises rather than returning an
    empty `QueryResult`, because the worker's SSE path swallows the
    `NotImplementedError` from `decode_sse_message` and would otherwise
    surface zero-output queries as successful.
    """

    def __init__(self, query_id: str, stream_all_chunks: bool) -> None:
        self.query_id = query_id
        # stream_all_chunks is intentionally ignored: non-streaming endpoint.

    def add_chunk(self, delta: Any) -> StreamChunk | None:
        return None

    def get_final_output(self) -> QueryResult:
        raise RuntimeError(
            "VideoGenAccumulator.get_final_output called: video generation is "
            "non-streaming — check HTTPClientConfig.streaming and query.data['stream']."
        )
