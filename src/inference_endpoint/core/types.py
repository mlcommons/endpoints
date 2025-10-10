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
    """Status of a query in the system."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Query(msgspec.Struct, kw_only=True):
    """Represents a single query to be processed."""

    id: str = msgspec.field(default_factory=lambda: str(uuid.uuid4()))
    data: dict[str, Any] = msgspec.field(default_factory=dict)
    headers: dict[str, str] = msgspec.field(default_factory=dict)
    created_at: float = msgspec.field(default_factory=time.time)


class QueryResult(msgspec.Struct, tag="query_result", kw_only=True):
    """Result of a completed query."""

    id: str = ""
    response_output: str | None = None
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)
    error: str | None = None
    completed_at: float = msgspec.field(default_factory=time.time)


class StreamChunk(msgspec.Struct, tag="stream_chunk", kw_only=True):
    """A chunk of streaming response."""

    id: str = ""
    response_chunk: str = ""
    is_complete: bool = False
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


# Type aliases for clarity
QueryId = str
DatasetId = str
EndpointId = str
