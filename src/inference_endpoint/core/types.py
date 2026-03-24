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


class APIType(str, Enum):
    OPENAI = "openai"
    SGLANG = "sglang"

    def default_route(self) -> str:
        match self:
            case APIType.OPENAI:
                return "/v1/chat/completions"
            case APIType.SGLANG:
                return "/generate"
            case _:
                raise ValueError(f"Invalid API type: {self}")


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


OUTPUT_ELEM_TYPE = str | tuple[str, ...]
"""Type for a single output or reasoning value: string (non-streaming) or tuple of strings (streaming)."""


class TextModelOutput(
    msgspec.Struct,
    tag=True,
    kw_only=True,
    frozen=True,
    omit_defaults=True,
    array_like=True,
    gc=False,
):  # type: ignore[call-arg]
    """Structured output from a text model.

    Supports main output and optional reasoning (e.g. chain-of-thought).
    Each field may be a string (non-streaming) or tuple of strings (streaming chunks).

    Attributes:
        output: Main model output. Defaults to empty string.
        reasoning: Optional reasoning trace. Defaults to None.
    """

    output: OUTPUT_ELEM_TYPE = ""
    reasoning: OUTPUT_ELEM_TYPE | None = None

    def __post_init__(self):
        """Convert list to tuple for output and reasoning to preserve immutability."""
        if isinstance(self.output, list):
            msgspec.structs.force_setattr(self, "output", tuple(self.output))
        if self.reasoning is not None and isinstance(self.reasoning, list):
            msgspec.structs.force_setattr(self, "reasoning", tuple(self.reasoning))

    def __str__(self) -> str:
        """Return the full output as a single string (joins tuple chunks if streaming)."""
        parts = []
        if self.reasoning:
            if isinstance(self.reasoning, str):
                parts.append(self.reasoning)
            elif isinstance(self.reasoning, tuple):
                parts.extend(self.reasoning)

        if self.output:
            if isinstance(self.output, str):
                parts.append(self.output)
            elif isinstance(self.output, tuple):
                parts.extend(self.output)

        # NOTE: Not sure how output is formatted - there *might* need to be a space or separator between
        # reasoning and output depending on the accumulator / API.
        return "".join(parts)

    def text_after_first_chunk(self) -> str:
        """Return the full output text excluding the first chunk.

        For TPOT calculation: token_count(text_after_first_chunk) gives the
        number of tokens generated after the first chunk, which is the TPOT
        denominator.

        For non-streaming (str fields), there is no "first chunk" concept so
        this returns an empty string.
        """
        parts: list[str] = []
        if self.reasoning:
            if isinstance(self.reasoning, tuple) and len(self.reasoning) > 1:
                parts.extend(self.reasoning[1:])
            # str reasoning: single chunk, skip entirely (it IS the first chunk)
        if self.output:
            if isinstance(self.output, str):
                # Non-streaming: if reasoning was present and was the first chunk,
                # include the full output. Otherwise no first chunk to skip.
                if parts or (self.reasoning and isinstance(self.reasoning, tuple)):
                    parts.append(self.output)
            elif isinstance(self.output, tuple):
                if parts or self.reasoning:
                    # First chunk was in reasoning; include all output chunks.
                    parts.extend(self.output)
                elif len(self.output) > 1:
                    # No reasoning; first chunk is output[0], skip it.
                    parts.extend(self.output[1:])
        return "".join(parts)


OUTPUT_TYPE = TextModelOutput


class PromptData(
    msgspec.Struct,
    tag=True,
    kw_only=True,
    frozen=True,
    omit_defaults=True,
    array_like=True,
    gc=False,
):  # type: ignore[call-arg]
    """Prompt input data attached to ISSUED events for ISL computation.

    Exactly one of ``text`` or ``token_ids`` should be set:
    - ``text``: raw prompt string (OpenAI path) — requires tokenization for ISL.
    - ``token_ids``: pre-tokenized token ID list (SGLang/Harmonize path) — ISL is len().

    Attributes:
        text: Raw prompt string. Set when the adapter sends text prompts.
        token_ids: Pre-computed token IDs. Set when the adapter pre-tokenizes (e.g. SGLang).
    """

    text: str | None = None
    token_ids: tuple[int, ...] | None = None


class ErrorData(
    msgspec.Struct,
    tag=True,
    kw_only=True,
    frozen=True,
    omit_defaults=True,
    array_like=True,
    gc=False,
):  # type: ignore[call-arg]
    """Structured error information.

    Attributes:
        error_type: Name of error. If possible, should be a qualified error type (e.g. "msgspec.DecodeError")..
        error_message: Optional human-readable message. Defaults to empty string.
    """

    error_type: str
    error_message: str = ""

    def __str__(self) -> str:
        """Human-readable string: 'type: message' if message present, else 'type'."""
        return (
            f"{self.error_type}: {self.error_message}"
            if self.error_message
            else self.error_type
        )


class Query(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    array_like=True,
    omit_defaults=True,
    gc=False,
):  # type: ignore[call-arg]
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

    Note:
        gc=False: Safe because data/headers are simple key-value pairs without cycles.
        Do NOT store self-referential or cyclic structures in data/headers fields.

        array_like=True: Encodes as array instead of object (e.g., ["id", {...}, {...}, 0.0]
        instead of {"id": ..., "data": ..., ...}). Provides ~6-50% size reduction and
        ~6-29% ser/des speedup for ZMQ transport depending on payload size.

        omit_defaults=True: Fields with default values are omitted during encoding,
        further reducing message size for queries with empty headers.
    """

    id: str = msgspec.field(default_factory=lambda: str(uuid.uuid4()))
    data: dict[str, Any] = msgspec.field(default_factory=dict)
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    created_at: float = msgspec.field(default_factory=time.time)


# gc=False: audit 2026-03: metadata dict is only ever read, never mutated after construction.
class QueryResult(
    msgspec.Struct,
    tag="query_result",
    kw_only=True,
    frozen=True,
    array_like=True,
    omit_defaults=True,
    gc=False,
):  # type: ignore[call-arg]
    """Result of a completed inference query.

    AT-RISK (gc=False): Has mutable container field `metadata`. Any change that
    mutates `metadata` after construction or stores this struct in a container
    referenced by this struct must be audited; if so, remove gc=False.

    Represents the outcome of processing a Query, including the response text,
    metadata, and any error information. The completed_at timestamp is
    automatically set to ensure accurate timing measurements.

    This struct is frozen (immutable) to prevent accidental modification of
    benchmark results, which is critical for reproducibility and fairness.

    Attributes:
        id: Query identifier (matches the originating Query.id).
        response_output: Generated response from the endpoint (None if error).
                         Prefer TextModelOutput; str is supported but will be deprecated.
        metadata: Additional response metadata (token counts, model info, etc.).
        error: Structured error if query failed (None if successful).
        completed_at: High-resolution timestamp (nanoseconds, monotonic clock).
                      Auto-set in __post_init__ to prevent tampering.

    Note:
        The completed_at field is intentionally set internally to prevent
        benchmark result manipulation. Users must not override this timestamp.

        gc=False: Safe because metadata contains only scalar key-value pairs.
        Do NOT store cyclic references in metadata or response_output fields.

        omit_defaults=True: Fields with static defaults (ie. those NOT using default_factory)
        are omitted if value equals default.

        array_like=True: Encodes as array instead of object (e.g. ["id", "chunk", false, {}]
        instead of {"id": ..., "response_chunk": ..., ...}). Reduces payload size.
    """

    id: str = ""
    response_output: OUTPUT_TYPE | None = None
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)
    error: ErrorData | None = None
    completed_at: int | msgspec.UnsetType = msgspec.UNSET

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

    def get_response_output_string(self) -> str:
        """Get the response output as a string."""
        if isinstance(self.response_output, TextModelOutput):
            return str(self.response_output)
        elif isinstance(self.response_output, str):
            return self.response_output
        else:
            return "<EMPTY>"


class StreamChunk(
    msgspec.Struct,
    tag="stream_chunk",
    frozen=True,
    kw_only=True,
    array_like=True,
    omit_defaults=True,
    gc=False,
):  # type: ignore[call-arg]
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

    Note:
        gc=False: Safe because metadata contains only scalar key-value pairs.
        Do NOT store cyclic references in metadata field.

        omit_defaults=True: Fields with static defaults (ie. those NOT using default_factory)
        are omitted if value equals default.

        array_like=True: Encodes as array instead of object (e.g. ["id", "chunk", false, {}]
        instead of {"id": ..., "response_chunk": ..., ...}). Reduces payload size.
    """

    id: str = ""
    response_chunk: str = ""
    is_complete: bool = False
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


# Type aliases for clarity
QueryId = str
DatasetId = str
EndpointId = str
