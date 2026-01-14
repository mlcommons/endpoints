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

"""Protocol definition for SSE stream accumulators."""

from typing import Any, Protocol

from inference_endpoint.core.types import QueryResult, StreamChunk


class SSEAccumulatorProtocol(Protocol):
    """
    Protocol for Server-Sent Events (SSE) stream accumulators.

    Accumulators collect streaming SSE deltas from inference endpoints and
    produce StreamChunk objects for intermediate results and a final QueryResult.

    Implementations internally track whether the first chunk has been emitted
    to support TTFT (Time-To-First-Token) measurements. When stream_all_chunks
    is disabled, only the first chunk is emitted via add_chunk().
    """

    def __init__(self, query_id: str, stream_all_chunks: bool) -> None:
        """
        Initialize the accumulator.

        Args:
            query_id: Unique identifier for the request being accumulated.
            stream_all_chunks: If True, emit all chunks; if False, only first chunk.
        """
        pass

    def add_chunk(self, delta: Any) -> StreamChunk | None:
        """
        Process an SSE delta and optionally emit a StreamChunk.

        Args:
            delta: API-specific SSE delta object (e.g., OpenAISSEDelta, SGLangSSEDelta).

        Returns:
            StreamChunk if content should be emitted, None otherwise.
            Returns None for empty deltas, or after first chunk when
            stream_all_chunks=False (TTFT-only mode).
        """
        pass

    def get_final_output(self) -> QueryResult:
        """
        Return the final accumulated result after stream completion.

        Called after all SSE deltas have been processed. Returns a QueryResult
        containing the complete accumulated response and metadata.

        Returns:
            QueryResult with the complete response output.
        """
        pass
