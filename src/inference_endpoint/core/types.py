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

"""
Core type definitions for the MLPerf Inference Endpoint Benchmarking System.

This module defines the basic data structures used throughout the system.
"""

import time
import uuid
from enum import Enum
from typing import Any

import msgspec


class QueryStatus(Enum):
    """Status of a query in its lifecycle.

    Query state transitions typically follow:
    PENDING -> RUNNING -> COMPLETED (or FAILED)

    Attributes:
        PENDING: Query created but not yet sent to endpoint.
        RUNNING: Query sent to endpoint, awaiting response.
        COMPLETED: Query finished successfully with response.
        FAILED: Query failed due to error (timeout, server error, etc.).
        CANCELLED: Query was cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_OUTPUT_DICT_TYPE = dict[str, str | list[str]]
_OUTPUT_RESULT_TYPE = str | tuple[str, ...] | _OUTPUT_DICT_TYPE | None


class Query(msgspec.Struct, kw_only=True):
    """Represents a single inference query to be sent to an endpoint.

    A Query encapsulates all information needed to make an HTTP request to
    an inference endpoint, including the request payload and any custom headers.

    This is the primary unit of work in the benchmarking system. Each Query
    is tracked through its complete lifecycle from creation to completion.

    Attributes:
        id: Unique identifier for this query (auto-generated UUID).
        data: Request payload as a dictionary (typically contains prompt, model, etc.).
        headers: HTTP headers to include in the request (e.g., authorization).
        created_at: Timestamp when query was created (seconds since epoch).

    Example:
        >>> query = Query(
        ...     data={"prompt": "Hello", "model": "Qwen/Qwen3-8B", "max_tokens": 100},
        ...     headers={"Authorization": "Bearer token123"},
        ... )
    """

    id: str = msgspec.field(default_factory=lambda: str(uuid.uuid4()))
    data: dict[str, Any] = msgspec.field(default_factory=dict)
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    created_at: float = msgspec.field(default_factory=time.time)


class QueryResult(msgspec.Struct, tag="query_result", kw_only=True, frozen=True):
    """Result of a completed inference query.

    Represents the outcome of processing a Query, including the response text,
    metadata, and any error information. The completed_at timestamp is
    automatically set to ensure accurate timing measurements.

    This struct is frozen (immutable) to prevent accidental modification of
    benchmark results, which is critical for reproducibility and fairness.

    Attributes:
        id: Query identifier (matches the originating Query.id).
        response_output: Generated text response from the endpoint (None if error).
                         Can be a string, or a tuple of strings. If it is a string,
                         it is assumed to be a non-streaming response. If it is a
                         tuple of strings, it is assumed to be a streamed response,
                         where the first element is the first chunk, which will not
                         be included in the TPOT measurements.
        metadata: Additional response metadata (token counts, model info, etc.).
        error: Error message if query failed (None if successful).
        completed_at: High-resolution timestamp (nanoseconds, monotonic clock).
                      Auto-set in __post_init__ to prevent tampering.

    Note:
        The completed_at field is intentionally set internally to prevent
        benchmark result manipulation. Users must not override this timestamp.
    """

    id: str = ""
    response_output: _OUTPUT_RESULT_TYPE = None
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)
    error: str | None = None
    completed_at: int = msgspec.UNSET

    def __post_init__(self):
        """Set completion timestamp automatically.

        This method is called during struct initialization and forcibly sets
        the completed_at timestamp using the monotonic clock. This ensures
        timing measurements cannot be manipulated by callers.

        Note:
            Uses msgspec.structs.force_setattr to bypass frozen=True protection.
        """
        # Disallow user setting completed_at time to prevent cheating.
        # Timestamp must be generated internally
        # Note that this will also be regenerated during encode+decode. This is
        # intentional, since timestamps in child and parent processes may be different
        # due to how monotonic_ns works.
        msgspec.structs.force_setattr(self, "completed_at", time.monotonic_ns())

        # A list can be passed on, but we need to convert it to a tuple to maintain immutability,
        # and for serialization to work properly.
        if isinstance(self.response_output, list):
            msgspec.structs.force_setattr(
                self, "response_output", tuple(self.response_output)
            )
        elif isinstance(self.response_output, dict):
            for k, v in self.response_output.items():
                if isinstance(v, list):
                    self.response_output[k] = tuple(v)


class StreamChunk(msgspec.Struct, tag="stream_chunk", kw_only=True):
    """A single chunk from a streaming inference response.

    Streaming responses are sent incrementally as the model generates text.
    Each StreamChunk represents one piece of the generation, enabling real-time
    display and accurate Time-To-First-Token (TTFT) measurements.

    Multiple StreamChunks with the same id collectively form the complete response.
    The is_complete flag indicates the final chunk in the sequence.

    Attributes:
        id: Query identifier (matches the originating Query.id).
        response_chunk: Partial response text for this chunk (delta, not cumulative).
        is_complete: True if this is the final chunk, False for intermediate chunks.
        metadata: Additional metadata for this chunk (timing, token info, etc.).

    Example:
        Streaming "Hello World" might produce:
        >>> StreamChunk(id="q1", response_chunk="Hello", is_complete=False)
        >>> StreamChunk(id="q1", response_chunk=" World", is_complete=True)
    """

    id: str = ""
    response_chunk: str = ""
    is_complete: bool = False
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


# Type aliases for clarity
QueryId = str
DatasetId = str
EndpointId = str
