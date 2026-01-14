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

from inference_endpoint.core.types import QueryResult, StreamChunk
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
        self.retraction_occurred = False

        self.first_chunk_sent = False
        self.query_id = query_id
        self.stream_all_chunks = stream_all_chunks

    def add_chunk(self, delta: SGLangSSEDelta) -> StreamChunk | None:
        if not isinstance(delta, SGLangSSEDelta):
            return None

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
                "n_tokens": len(delta.token_delta),
            }
            chunk = StreamChunk(
                id=self.query_id,
                response_chunk=content_diff,
                is_complete=False,
                metadata=metadata,
            )
            self.first_chunk_sent = True
            return chunk
        else:
            return None

    def get_final_output(self) -> QueryResult:
        return QueryResult(
            id=self.query_id,
            response_output=self.text,
            metadata={
                "first_chunk": not self.first_chunk_sent,
                "final_chunk": True,
                "retraction_occurred": self.retraction_occurred,
                "n_tokens": self.total_tokens,
                "token_ids": self.token_ids,
            },
        )
