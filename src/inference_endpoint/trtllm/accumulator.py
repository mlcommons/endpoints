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

"""TRT-LLM SSE stream accumulator implementation.

Uses the same dict output format as OpenAI ({"output": [first_chunk, rest]})
so that the metrics reporter can compute TPOT from streaming responses.
"""

from inference_endpoint.core.types import QueryResult, StreamChunk, TextModelOutput
from inference_endpoint.endpoint_client.accumulator_protocol import (
    SSEAccumulatorProtocol,
)
from inference_endpoint.openai.types import SSEDelta


class TRTLLMSSEAccumulator(SSEAccumulatorProtocol):
    """Accumulator for TRT-LLM SSE streaming responses.

    TRT-LLM uses OpenAI-compatible SSE format, so deltas are SSEDelta objects.
    Output is returned as {"output": [first_chunk, rest_joined]} to support
    TPOT calculation in the metrics reporter.
    """

    def __init__(self, query_id: str, stream_all_chunks: bool):
        self.output_chunks: list[str] = []
        self.n_tokens: int = 0

        self.first_chunk_sent = False
        self.query_id = query_id
        self.stream_all_chunks = stream_all_chunks

    def add_chunk(self, delta: SSEDelta) -> StreamChunk | None:
        if not isinstance(delta, SSEDelta):
            return None

        content = delta.content
        if not content:
            return None

        self.output_chunks.append(content)
        self.n_tokens += 1

        if self.stream_all_chunks or not self.first_chunk_sent:
            chunk = StreamChunk(
                id=self.query_id,
                response_chunk=content,
                metadata={
                    "first_chunk": not self.first_chunk_sent,
                    "final_chunk": False,
                },
            )
            self.first_chunk_sent = True
            return chunk
        return None

    def get_final_output(self) -> QueryResult:
        # TextModelOutput.output carries [first_chunk, rest_joined] (post-init
        # converts the list to a tuple): element 0 is the first token and
        # element 1 the remainder, mirroring the OpenAI accumulator.
        if self.output_chunks:
            resp_output: list[str] = [self.output_chunks[0]]
            if len(self.output_chunks) > 1:
                resp_output.append("".join(self.output_chunks[1:]))
        else:
            resp_output = []

        return QueryResult(
            id=self.query_id,
            response_output=TextModelOutput(output=resp_output),
            metadata={
                "first_chunk": not self.first_chunk_sent,
                "final_chunk": True,
                "n_tokens": self.n_tokens,
            },
        )
