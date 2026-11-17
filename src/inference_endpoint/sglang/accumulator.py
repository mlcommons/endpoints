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

"""SGLang SSE stream accumulator implementation."""

from typing import Any

from inference_endpoint.core.types import (
    COMPLETION_TOKENS_KEY,
    PROMPT_TOKENS_KEY,
    QueryResult,
    StreamChunk,
    TextModelOutput,
    put_server_usage,
    server_usage_count,
)
from inference_endpoint.endpoint_client.accumulator_protocol import (
    SSEAccumulatorProtocol,
)
from inference_endpoint.sglang.types import SGLangSSEDelta


class SGLangSSEAccumulator(SSEAccumulatorProtocol):
    """Accumulator for SGLang /generate SSE streaming responses."""

    def __init__(self, query_id: str, stream_all_chunks: bool):
        self.text = ""
        self.token_ids: list[int] = []
        self.total_tokens = 0
        self.prompt_tokens: int | None = None
        # The last raw prompt_tokens validated; frames repeat one value.
        self._raw_prompt_tokens: object = None
        # Set by the first decoded frame (each carries meta_info counts); until
        # then the initial total_tokens of 0 is not a server-reported count.
        self._usage_seen = False
        self.retraction_occurred = False

        self.first_chunk_sent = False
        self.query_id = query_id
        self.stream_all_chunks = stream_all_chunks

    def add_chunk(self, delta: SGLangSSEDelta) -> StreamChunk | None:
        if not isinstance(delta, SGLangSSEDelta):
            return None

        # Captured before the unchanged-count early return so a stream whose
        # completion count never moves (e.g. an abort) still reports its counts.
        # Only a changed raw value is validated; a frame without a usable
        # prompt_tokens keeps the earlier count.
        self._usage_seen = True
        raw = delta.prompt_tokens
        if raw is not None and raw != self._raw_prompt_tokens:
            self._raw_prompt_tokens = raw
            count = server_usage_count(raw, PROMPT_TOKENS_KEY)
            if count is not None:
                self.prompt_tokens = count

        if delta.total_completion_tokens == self.total_tokens:
            return None

        # In SGLang /generate, the .text field is the total accumulated text, not
        # a difference, so we'll need to compute the diff for the StreamChunk
        content_diff = ""
        if len(delta.text) > (start_idx := len(self.text)):
            content_diff = delta.text[start_idx:]
        self.text = delta.text
        self.token_ids.extend(delta.token_delta)
        self.total_tokens = delta.total_completion_tokens
        if delta.has_retractions:
            # For now, we won't be handling retractions if they occur, but we will
            # report it as part of the metadata if it does happen.
            self.retraction_occurred = True

        if content_diff and (self.stream_all_chunks or not self.first_chunk_sent):
            metadata = {
                "first_chunk": not self.first_chunk_sent,
                "final_chunk": False,
                "retraction_occurred": delta.has_retractions,
            }
            chunk = StreamChunk(
                id=self.query_id,
                response_chunk=content_diff,
                metadata=metadata,
            )
            self.first_chunk_sent = True
            return chunk
        else:
            return None

    def get_final_output(self) -> QueryResult:
        metadata: dict[str, Any] = {
            "first_chunk": not self.first_chunk_sent,
            "final_chunk": True,
            "retraction_occurred": self.retraction_occurred,
            "token_ids": self.token_ids,
        }
        if self._usage_seen:
            put_server_usage(
                metadata,
                self.prompt_tokens,
                server_usage_count(self.total_tokens, COMPLETION_TOKENS_KEY),
            )
        return QueryResult(
            id=self.query_id,
            response_output=TextModelOutput(output=self.text),
            metadata=metadata,
        )
