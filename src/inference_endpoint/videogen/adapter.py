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
import os
from pathlib import Path
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

_BINARY_FALLBACK_ENV = "INFERENCE_ENDPOINT_VIDEOGEN_FALLBACK_DIR"

if TYPE_CHECKING:
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import Transform


def _video_file_suffix(body: bytes) -> str | None:
    """Return ".mp4" or ".avi" for a raw video body, else None."""
    if len(body) >= 8 and body[4:8] == b"ftyp":
        return ".mp4"
    if len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"AVI ":
        return ".avi"
    return None


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
    def encode_query(cls, query: Query, *, include_usage: bool) -> bytes:
        """Serialise query.data to VideoPathRequest JSON bytes.

        Only `prompt` is required. All other fields fall back to defaults on
        VideoPathRequest but can be overridden via query.data. Streaming is
        not supported — `stream=True` raises. `include_usage` is ignored: the
        endpoint reports no token usage.
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
        return cls._to_server_wire(req)

    # The server's VideoGenerationRequest forbids unknown top-level fields
    # (HTTP 422), so client-only names are renamed or moved into extra_params.
    _RENAME_TO_SERVER = {"output_format": "format"}
    # The only extra_params Wan 2.2 A14B accepts; others get HTTP 400.
    _INTO_EXTRA_PARAMS = ("boundary_ratio", "guidance_scale_2")
    # "path" returns {id, output_path}; "file" returns the video file.
    _RESPONSE_FORMAT_MAP = {"video_path": "path", "video_bytes": "file"}

    @classmethod
    def _to_server_wire(cls, req: VideoPathRequest) -> bytes:
        """Translate the client request model onto the server's wire schema."""
        body = req.model_dump(exclude_none=True)

        if "latent_path" in body:
            raise ValueError(
                "latent_path is not part of the trtllm-serve request schema; set "
                "TRTLLM_VIDEO_FIXED_LATENT_PATH in the server environment instead."
            )

        for src, dst in cls._RENAME_TO_SERVER.items():
            if src in body:
                body[dst] = body.pop(src)

        extra = {key: body.pop(key) for key in cls._INTO_EXTRA_PARAMS if key in body}
        if extra:
            body["extra_params"] = extra

        rf = body.get("response_format")
        if rf in cls._RESPONSE_FORMAT_MAP:
            body["response_format"] = cls._RESPONSE_FORMAT_MAP[rf]

        return json.dumps(body).encode()

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """Deserialise trtllm-serve response to QueryResult.

        A video file (returned for response_format="file") is saved to
        $INFERENCE_ENDPOINT_VIDEOGEN_FALLBACK_DIR and returned as a path.
        Anything else is parsed as JSON.
        """
        suffix = _video_file_suffix(response_bytes)
        if suffix is not None:
            return cls._decode_binary_response(response_bytes, query_id, suffix)
        return cls._decode_json_response(response_bytes, query_id)

    @classmethod
    def _normalize_server_job(cls, raw: dict) -> dict:
        """Map trtllm-serve's OpenAI-style VideoJob onto this client's shape.

        The server returns a VideoJob (openai_protocol.py: id, object, status,
        created_at, model, prompt, size, fps, duration, output_path,
        output_paths, timing_metrics, ...). VideoPathResponse declares
        extra="forbid" and wants video_id/video_path, so an unmapped VideoJob
        fails with "4 validation errors" and the sample is scored as an error
        even though the server generated the video.

        Bodies already in the client shape pass through unchanged.
        """
        if "video_id" in raw or "video_bytes" in raw:
            return raw
        if "id" not in raw:
            return raw

        status = raw.get("status")
        if raw.get("error"):
            raise ValueError(f"server reported video job error: {raw['error']}")
        if status is not None and status != "completed":
            raise ValueError(f"video job not completed (status={status!r})")

        path = raw.get("output_path")
        if not path:
            paths = raw.get("output_paths") or []
            path = paths[0] if paths else None
        if not path:
            raise ValueError(
                "server VideoJob carried no output_path/output_paths; "
                "response_format='path' may be disabled server-side"
            )

        return {"video_id": raw["id"], "video_path": path}

    @classmethod
    def _decode_json_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        raw = cls._normalize_server_job(json.loads(response_bytes))
        # Truthiness check, not key presence: a server that returns
        # `"video_bytes": null` belongs in the video_path branch.
        if isinstance(raw.get("video_bytes"), str):
            resp_bytes = VideoPayloadResponse.model_validate(raw)
            return QueryResult(
                id=query_id,
                metadata={
                    "video_id": resp_bytes.video_id,
                    "video_bytes": resp_bytes.video_bytes,
                },
            )
        resp_path = VideoPathResponse.model_validate(raw)
        # Mirror video_path into response_output so the event log carries it
        # to the accuracy scorer (VBench reads videos by path).
        return QueryResult(
            id=query_id,
            response_output=TextModelOutput(output=resp_path.video_path),
            metadata={
                "video_id": resp_path.video_id,
                "video_path": resp_path.video_path,
            },
        )

    @classmethod
    def _decode_binary_response(
        cls, response_bytes: bytes, query_id: str, suffix: str = ".mp4"
    ) -> QueryResult:
        fallback_dir = os.environ.get(_BINARY_FALLBACK_ENV)
        if not fallback_dir:
            raise RuntimeError(
                f"trtllm-serve returned binary video bytes "
                f"({len(response_bytes)} B), but {_BINARY_FALLBACK_ENV} is not "
                "set. Point it at a shared-filesystem directory so the adapter "
                "can persist responses for downstream scoring."
            )
        out_path = Path(fallback_dir) / f"{query_id}{suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response_bytes)
        return QueryResult(
            id=query_id,
            response_output=TextModelOutput(output=str(out_path)),
            metadata={"video_id": query_id, "video_path": str(out_path)},
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
