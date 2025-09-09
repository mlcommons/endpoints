"""
Core type definitions for the MLPerf Inference Endpoint Benchmarking System.

This module defines the basic data structures used throughout the system.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryStatus(Enum):
    """Status of a query in the system."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Query:
    """Represents a single query to be processed."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    max_tokens: int = (
        100  # TODO: This is a token count - should we have text count instead?
    )
    temperature: float = 0.7
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    status: QueryStatus = QueryStatus.PENDING
    created_at: float | None = None
    modalities: list[str] = field(default_factory=lambda: ["text"])
    response_format: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            import time

            self.created_at = time.time()

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError("to_json is not implemented for Query")

    @classmethod
    def from_json(cls, json_str: dict[str, Any]) -> "Query":
        raise NotImplementedError("from_json is not implemented for Query")


@dataclass
class ChatCompletionQuery(Query):
    """Represents a single query to be processed."""

    prompt: str = (
        ""  # TODO for now a single prompt, but we can replace wiht a list of messages
    )

    headers: dict[str, str] = field(
        default_factory=lambda: {
            "Content-Type": "application/json",
            # TODO(vir): expose this config via __post_init__
            "Authorization": "Bearer dummy",
        }
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.prompt},
            ],
            "stream": self.stream,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "modalities": self.modalities,
            "response_format": self.response_format,
        }

    @classmethod
    def from_json(cls, json_value: dict[str, Any]) -> "ChatCompletionQuery":
        # Extract prompt from messages if present
        prompt = ""
        if "messages" in json_value and len(json_value["messages"]) > 0:
            # Find the last user message
            for msg in reversed(json_value["messages"]):
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break

        return ChatCompletionQuery(
            id=json_value.get("id", str(uuid.uuid4())),
            model=json_value.get("model", ""),
            prompt=prompt,
            stream=json_value.get("stream"),
            max_tokens=json_value.get("max_completion_tokens"),
            temperature=json_value.get("temperature"),
            modalities=json_value.get("modalities"),
            response_format=json_value.get("response_format"),
        )


@dataclass
class QueryResult:
    """Result of a completed query."""

    query_id: str
    response_output: str = ""
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed_at: float | None = None

    def __post_init__(self) -> None:
        if self.completed_at is None:
            import time

            self.completed_at = time.time()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.query_id,
            "choices": [
                {"message": {"role": "assistant", "content": self.response_output}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    @classmethod
    def from_json(cls, json_value: dict[str, Any]) -> "QueryResult":
        """Parse QueryResult from JSON in OpenAI format."""
        # Check if this is an OpenAI response format
        if "choices" in json_value and json_value["choices"]:
            choice = json_value["choices"][0]

            # Handle chat completion format
            if "message" in choice and "content" in choice["message"]:
                return QueryResult(
                    query_id=json_value.get("id", ""),
                    response_output=choice["message"]["content"],
                )
            # Handle text completion format
            elif "text" in choice:
                return QueryResult(
                    query_id=json_value.get("id", ""),
                    response_output=choice["text"],
                )
            else:
                raise ValueError(
                    "Invalid OpenAI response format: missing content in choices"
                )

        # Handle error responses
        elif "error" in json_value:
            error_info = json_value["error"]
            error_msg = (
                error_info.get("message", str(error_info))
                if isinstance(error_info, dict)
                else str(error_info)
            )
            return QueryResult(
                query_id=json_value.get("id", ""),
                response_output="",
                error=error_msg,
            )

        else:
            raise ValueError("Unrecognized response format")


@dataclass
class StreamChunk:
    """A chunk of streaming response."""

    query_id: str
    response_chunk: str = ""
    is_complete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# Type aliases for clarity
QueryId = str
DatasetId = str
EndpointId = str
