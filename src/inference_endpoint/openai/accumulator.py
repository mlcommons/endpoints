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

"""OpenAI SSE stream accumulator implementation."""

from typing import Any

from inference_endpoint.core.types import QueryResult, StreamChunk, TextModelOutput
from inference_endpoint.endpoint_client.accumulator_protocol import (
    SSEAccumulatorProtocol,
)
from inference_endpoint.openai.types import SSEChoice


class OpenAISSEAccumulator(SSEAccumulatorProtocol):
    """Accumulator for OpenAI-compatible SSE streaming responses."""

    def __init__(self, query_id: str, stream_all_chunks: bool):
        self.output_chunks: list[str] = []
        self.reasoning_chunks: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason: str | None = None

        self.first_chunk_sent = False
        self.query_id = query_id
        self.stream_all_chunks = stream_all_chunks

    def add_chunk(self, choice: SSEChoice | None) -> StreamChunk | None:
        if not isinstance(choice, SSEChoice):
            return None

        if choice.finish_reason:
            self._finish_reason = choice.finish_reason

        delta = choice.delta
        if delta is None:
            return None

        # Accumulate tool_calls partials (streamed as incremental JSON fragments)
        if delta.tool_calls:
            for partial in delta.tool_calls:
                idx = partial.get("index", 0)
                tc = self._tool_calls.setdefault(
                    idx, {"type": "function", "function": {"arguments": ""}}
                )
                if partial.get("id"):
                    tc["id"] = partial["id"]
                if partial.get("type"):
                    tc["type"] = partial["type"]
                fn = partial.get("function") or {}
                if fn.get("name"):
                    tc["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    tc["function"]["arguments"] += fn["arguments"]

        content = None
        if delta.content:
            self.output_chunks.append(delta.content)
            content = delta.content
        elif delta.reasoning_content or delta.reasoning:
            rc = delta.reasoning_content or delta.reasoning
            self.reasoning_chunks.append(rc)  # type: ignore[arg-type]
            content = rc
        else:
            return None

        if content is not None and (
            self.stream_all_chunks or not self.first_chunk_sent
        ):
            chunk = StreamChunk(
                id=self.query_id,
                response_chunk=content,
                metadata={
                    "first_chunk": not self.first_chunk_sent,
                },
            )
            self.first_chunk_sent = True
            return chunk
        else:
            return None

    def get_final_output(self) -> QueryResult:
        if self.reasoning_chunks:
            resp_reasoning: list[str] = [self.reasoning_chunks[0]]
            if len(self.reasoning_chunks) > 1:
                resp_reasoning.append("".join(self.reasoning_chunks[1:]))
            text_output = TextModelOutput(
                output="".join(self.output_chunks),
                reasoning=resp_reasoning,
            )
        elif self.output_chunks:
            resp_output: list[str] = [self.output_chunks[0]]
            if len(self.output_chunks) > 1:
                resp_output.append("".join(self.output_chunks[1:]))
            text_output = TextModelOutput(output=resp_output, reasoning=None)
        else:
            text_output = TextModelOutput(output=[], reasoning=None)

        metadata: dict[str, Any] = {
            "first_chunk": not self.first_chunk_sent,
            "final_chunk": True,
        }
        if self._finish_reason:
            metadata["finish_reason"] = self._finish_reason
        if self._tool_calls:
            metadata["tool_calls"] = [
                self._tool_calls[i] for i in sorted(self._tool_calls)
            ]

        return QueryResult(
            id=self.query_id,
            response_output=text_output,
            metadata=metadata,
        )
