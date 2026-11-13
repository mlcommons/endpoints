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
from inference_endpoint.openai.types import SSEChoice, SSEDelta


class OpenAISSEAccumulator(SSEAccumulatorProtocol):
    """Accumulator for OpenAI-compatible SSE streaming responses."""

    def __init__(self, query_id: str, stream_all_chunks: bool):
        self.output_chunks: list[str] = []
        self.reasoning_chunks: list[str] = []
        self.tool_call_chunks: list[tuple[dict[str, Any], ...]] = []
        self._finish_reason: str | None = None

        self._first_delta: SSEDelta | None = None
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

        if delta.tool_calls:
            self.tool_call_chunks.append(tuple(delta.tool_calls))

        rc = delta.reasoning_content or delta.reasoning
        if rc:
            self.reasoning_chunks.append(rc)
        if delta.content:
            self.output_chunks.append(delta.content)
        content = (rc or "") + (delta.content or "")
        if not self.first_chunk_sent and (content or delta.tool_calls):
            self._first_delta = delta
        if not content and delta.tool_calls and not self.first_chunk_sent:
            # Pure tool-call delta with no text: emit a zero-length sentinel so
            # RECV_FIRST / TTFT fires for agentic responses that have no content.
            sentinel = StreamChunk(
                id=self.query_id,
                response_chunk="",
                metadata={"first_chunk": True},
            )
            self.first_chunk_sent = True
            return sentinel
        elif not content:
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
        first = self._first_delta
        first_content = (first.content or "") if first else ""
        first_reasoning = (
            (first.reasoning_content or first.reasoning or "") if first else ""
        )
        tool_calls = None
        if self.tool_call_chunks:
            tool_calls = tuple(self.tool_call_chunks)
            if first is not None and not first.tool_calls:
                tool_calls = ((), *tool_calls)

        text_output = TextModelOutput(
            output=_with_first_chunk(self.output_chunks, first_content),
            reasoning=_with_first_chunk(self.reasoning_chunks, first_reasoning) or None,
            tool_calls=tool_calls,
        )

        metadata: dict[str, Any] = {
            "first_chunk": not self.first_chunk_sent,
            "final_chunk": True,
        }
        if self._finish_reason:
            metadata["finish_reason"] = self._finish_reason
        _content, _reasoning, merged_tool_calls = text_output.as_message_parts()
        if merged_tool_calls:
            metadata["tool_calls"] = list(merged_tool_calls)

        return QueryResult(
            id=self.query_id,
            response_output=text_output,
            metadata=metadata,
        )


def _with_first_chunk(chunks: list[str], first: str) -> tuple[str, ...]:
    """Keep the first delta's contribution separate from the accumulated tail."""
    if not chunks:
        return ()
    tail_start = 1 if first else 0
    if len(chunks) == tail_start:
        return (first,)
    return first, "".join(chunks[tail_start:])
