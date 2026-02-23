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

import logging

from inference_endpoint.core.types import QueryResult, StreamChunk, TextModelOutput
from inference_endpoint.endpoint_client.accumulator_protocol import (
    SSEAccumulatorProtocol,
)
from inference_endpoint.openai.types import SSEDelta as OpenAISSEDelta

logger = logging.getLogger(__name__)


class OpenAISSEAccumulator(SSEAccumulatorProtocol):
    """Accumulator for OpenAI-compatible SSE streaming responses."""

    def __init__(self, query_id: str, stream_all_chunks: bool):
        self.output_chunks: list[str] = []
        self.reasoning_chunks: list[str] = []

        self.first_chunk_sent = False
        self.query_id = query_id
        self.stream_all_chunks = stream_all_chunks

    def add_chunk(self, delta: OpenAISSEDelta) -> StreamChunk | None:
        if not isinstance(delta, OpenAISSEDelta):
            return None

        content = None
        if delta.content:
            self.output_chunks.append(delta.content)
            content = delta.content
        elif delta.reasoning:
            self.reasoning_chunks.append(delta.reasoning)
            content = delta.reasoning
        else:
            return None

        if content is not None and (
            self.stream_all_chunks or not self.first_chunk_sent
        ):
            chunk = StreamChunk(
                id=self.query_id,
                response_chunk=content,
                is_complete=False,
                metadata={
                    "first_chunk": not self.first_chunk_sent,
                    "final_chunk": False,
                },
            )
            self.first_chunk_sent = True
            return chunk
        else:
            return None

    def get_final_output(self) -> QueryResult:
        if self.reasoning_chunks:
            # If there are reasoning chunks, then the first chunk received
            # is the first reasoning chunk. The rest of the reasoning chunks,
            # as well as the output chunks can be joined together.
            resp_reasoning: list[str] = [self.reasoning_chunks[0]]
            if len(self.reasoning_chunks) > 1:
                resp_reasoning.append("".join(self.reasoning_chunks[1:]))
            text_output = TextModelOutput(
                output="".join(self.output_chunks),
                reasoning=resp_reasoning,
            )
        elif self.output_chunks:
            # If there are only output chunks, the first chunk is used for
            # TTFT calculations. The rest are joined together.
            resp_output: list[str] = [self.output_chunks[0]]
            if len(self.output_chunks) > 1:
                resp_output.append("".join(self.output_chunks[1:]))
            text_output = TextModelOutput(output=resp_output, reasoning=None)
        else:
            text_output = TextModelOutput(output=[], reasoning=None)
        return QueryResult(
            id=self.query_id,
            response_output=text_output,
            metadata={
                "first_chunk": not self.first_chunk_sent,
                "final_chunk": True,
            },
        )
