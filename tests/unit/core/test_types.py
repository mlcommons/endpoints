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
Unit tests for core type serialization using msgspec.msgpack.

Tests verify that Query, QueryResult, and StreamChunk can be properly
serialized and deserialized with various field combinations.
"""

import time

import msgspec
import pytest
from inference_endpoint.core.types import (
    ErrorData,
    Query,
    QueryResult,
    StreamChunk,
    TextModelOutput,
)


class TestErrorData:
    """Test ErrorData string representation."""

    def test_error_data_str_with_message(self):
        """str(ErrorData) is 'type: message' when error_message is non-empty."""
        err = ErrorData(error_type="ValueError", error_message="invalid value")
        assert str(err) == "ValueError: invalid value"

    def test_error_data_str_without_message(self):
        """str(ErrorData) is error_type only when error_message is empty."""
        err = ErrorData(error_type="TimeoutError", error_message="")
        assert str(err) == "TimeoutError"


class TestQuerySerialization:
    """Test Query msgspec.msgpack serialization with various field combinations."""

    def test_query_empty_defaults(self):
        """Test Query with all default values serializes correctly."""
        query = Query()

        # Serialize and deserialize
        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        # Verify fields
        assert decoded.id == query.id
        assert decoded.data == {}
        assert decoded.headers == {}
        assert decoded.created_at == query.created_at
        assert isinstance(decoded.id, str)
        assert len(decoded.id) > 0  # UUID should be non-empty

    def test_query_with_simple_data(self):
        """Test Query with basic data dict."""
        query = Query(
            data={"prompt": "Hello, world!", "model": "gpt-4", "max_tokens": 100}
        )

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.data == {
            "prompt": "Hello, world!",
            "model": "gpt-4",
            "max_tokens": 100,
        }
        assert decoded.headers == {}

    def test_query_with_headers(self):
        """Test Query with custom headers."""
        query = Query(
            data={"prompt": "Test"},
            headers={
                "Authorization": "Bearer token123",
                "Content-Type": "application/json",
            },
        )

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.headers == {
            "Authorization": "Bearer token123",
            "Content-Type": "application/json",
        }

    def test_query_with_complex_data(self):
        """Test Query with nested and complex data structures."""
        query = Query(
            data={
                "prompt": "Complex prompt",
                "model": "gpt-4",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.0,
                },
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                "stream": True,
                "n": 1,
            }
        )

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.data["parameters"]["temperature"] == 0.7
        assert len(decoded.data["messages"]) == 2
        assert decoded.data["stream"] is True

    def test_query_with_custom_id(self):
        """Test Query with custom ID."""
        custom_id = "custom-query-id-12345"
        query = Query(id=custom_id, data={"test": "value"})

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.id == custom_id

    def test_query_with_timestamp(self):
        """Test Query with custom timestamp."""
        custom_time = 1234567890.123456
        query = Query(created_at=custom_time)

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.created_at == custom_time

    def test_query_all_fields_populated(self):
        """Test Query with all fields fully populated."""
        query = Query(
            id="test-query-001",
            data={"prompt": "Full test", "max_tokens": 50},
            headers={"X-Custom": "header-value"},
            created_at=1700000000.0,
        )

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.id == "test-query-001"
        assert decoded.data == {"prompt": "Full test", "max_tokens": 50}
        assert decoded.headers == {"X-Custom": "header-value"}
        assert decoded.created_at == 1700000000.0

    def test_query_multiple_roundtrips(self):
        """Test Query survives multiple serialization roundtrips."""
        original = Query(
            data={"test": "data"},
            headers={"auth": "token"},
        )

        # First roundtrip
        encoded1 = msgspec.msgpack.encode(original)
        decoded1 = msgspec.msgpack.decode(encoded1, type=Query)

        # Second roundtrip
        encoded2 = msgspec.msgpack.encode(decoded1)
        decoded2 = msgspec.msgpack.decode(encoded2, type=Query)

        # Verify all fields remain consistent
        assert decoded2.id == original.id
        assert decoded2.data == original.data
        assert decoded2.headers == original.headers
        assert decoded2.created_at == original.created_at


class TestQueryResultSerialization:
    """Test QueryResult msgspec.msgpack serialization with various field combinations."""

    def test_query_result_minimal(self):
        """Test QueryResult with minimal required fields."""
        result = QueryResult(id="query-123")

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.id == "query-123"
        assert decoded.response_output is None
        assert decoded.metadata == {}
        assert decoded.error is None
        # completed_at will be different after decode due to __post_init__
        assert isinstance(decoded.completed_at, int)

    def test_query_result_with_string_response(self):
        """Test QueryResult with string response output."""
        result = QueryResult(
            id="query-456",
            response_output=TextModelOutput(
                output="This is a complete response from the model."
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.response_output == TextModelOutput(
            output="This is a complete response from the model."
        )

    def test_query_result_with_tuple_response(self):
        """Test QueryResult with TextModelOutput (tuple output, streaming chunks)."""
        result = QueryResult(
            id="query-789",
            response_output=TextModelOutput(
                output=("First chunk", "Second chunk", "Final chunk"),
                reasoning=None,
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == (
            "First chunk",
            "Second chunk",
            "Final chunk",
        )
        assert decoded.response_output.reasoning is None

    def test_query_result_with_list_response_converts_to_tuple(self):
        """Test TextModelOutput converts list output to tuple in __post_init__."""
        result = QueryResult(
            id="query-list",
            response_output=TextModelOutput(
                output=["Chunk 1", "Chunk 2", "Chunk 3"],
                reasoning=None,
            ),
        )

        assert isinstance(result.response_output, TextModelOutput)
        assert result.response_output.output == ("Chunk 1", "Chunk 2", "Chunk 3")

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("Chunk 1", "Chunk 2", "Chunk 3")

    def test_query_result_with_dict_response_output_only(self):
        """Test QueryResult with TextModelOutput (output only, list converted to tuple)."""
        result = QueryResult(
            id="query-dict",
            response_output=TextModelOutput(
                output=["First chunk", "rest of output"],
                reasoning=None,
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("First chunk", "rest of output")
        assert decoded.response_output.reasoning is None

    def test_query_result_with_dict_response_output_and_reasoning(self):
        """Test QueryResult with TextModelOutput (output and reasoning)."""
        result = QueryResult(
            id="query-dict-reasoning",
            response_output=TextModelOutput(
                output="Final output text",
                reasoning=["First reasoning chunk", "rest of reasoning"],
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == "Final output text"
        assert decoded.response_output.reasoning == (
            "First reasoning chunk",
            "rest of reasoning",
        )

    def test_query_result_with_dict_response_empty_output(self):
        """Test QueryResult with TextModelOutput (empty output list -> tuple)."""
        result = QueryResult(
            id="query-dict-empty",
            response_output=TextModelOutput(output=[], reasoning=None),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ()
        assert decoded.response_output.reasoning is None

    def test_query_result_with_metadata(self):
        """Test QueryResult with comprehensive metadata."""
        result = QueryResult(
            id="query-meta",
            response_output=TextModelOutput(output="Response text"),
            metadata={
                "model": "gpt-4",
                "tokens_used": 150,
                "finish_reason": "stop",
                "latency_ms": 234.5,
                "cache_hit": False,
                "provider": "openai",
            },
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.metadata["model"] == "gpt-4"
        assert decoded.metadata["tokens_used"] == 150
        assert decoded.metadata["finish_reason"] == "stop"
        assert decoded.metadata["latency_ms"] == 234.5
        assert decoded.metadata["cache_hit"] is False

    def test_query_result_with_error(self):
        """Test QueryResult with ErrorData."""
        result = QueryResult(
            id="query-error",
            error=ErrorData(
                error_type="TimeoutError",
                error_message="Connection timeout after 30 seconds",
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.error is not None
        assert decoded.error.error_type == "TimeoutError"
        assert decoded.error.error_message == "Connection timeout after 30 seconds"
        assert decoded.response_output is None

    def test_query_result_with_error_and_partial_response(self):
        """Test QueryResult with both error and partial response."""
        result = QueryResult(
            id="query-partial",
            response_output=TextModelOutput(output="Partial response before error"),
            error=ErrorData(
                error_type="ConnectionError",
                error_message="Server disconnected during streaming",
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.response_output == TextModelOutput(
            output="Partial response before error"
        )
        assert decoded.error is not None
        assert decoded.error.error_message == "Server disconnected during streaming"

    def test_query_result_all_fields_populated(self):
        """Test QueryResult with all fields fully populated."""
        result = QueryResult(
            id="query-full",
            response_output=TextModelOutput(
                output=("Chunk 1", "Chunk 2"),
                reasoning=None,
            ),
            metadata={
                "model": "llama-2-70b",
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
            error=None,
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.id == "query-full"
        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("Chunk 1", "Chunk 2")
        assert decoded.metadata["total_tokens"] == 150
        assert decoded.error is None

    def test_query_result_immutability(self):
        """Test QueryResult is frozen and cannot be modified."""
        result = QueryResult(
            id="query-frozen", response_output=TextModelOutput(output="Original text")
        )

        with pytest.raises(AttributeError):
            result.response_output = "Modified text"

        with pytest.raises(AttributeError):
            result.error = ErrorData(error_type="x", error_message="y")

    def test_query_result_completed_at_auto_set(self):
        """Test QueryResult completed_at is automatically set in __post_init__."""
        before = time.monotonic_ns()
        result = QueryResult(id="query-timestamp")
        after = time.monotonic_ns()

        # completed_at should be set between before and after
        assert before <= result.completed_at <= after
        assert isinstance(result.completed_at, int | float)

    def test_query_result_empty_string_response(self):
        """Test QueryResult with empty string response."""
        result = QueryResult(
            id="query-empty", response_output=TextModelOutput(output="")
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.response_output == TextModelOutput(output="")

    def test_query_result_empty_tuple_response(self):
        """Test QueryResult with TextModelOutput (empty tuple output)."""
        result = QueryResult(
            id="query-empty-tuple",
            response_output=TextModelOutput(output=(), reasoning=None),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ()

    def test_query_result_multiple_roundtrips(self):
        """Test QueryResult survives multiple serialization roundtrips."""
        original = QueryResult(
            id="query-roundtrip",
            response_output=TextModelOutput(
                output=("Chunk A", "Chunk B"),
                reasoning=None,
            ),
            metadata={"tokens": 42},
        )

        encoded1 = msgspec.msgpack.encode(original)
        decoded1 = msgspec.msgpack.decode(encoded1, type=QueryResult)

        encoded2 = msgspec.msgpack.encode(decoded1)
        decoded2 = msgspec.msgpack.decode(encoded2, type=QueryResult)

        assert decoded2.id == original.id
        assert isinstance(decoded2.response_output, TextModelOutput)
        assert decoded2.response_output.output == original.response_output.output
        assert decoded2.metadata == original.metadata


class TestStreamChunkSerialization:
    """Test StreamChunk msgspec.msgpack serialization with various field combinations."""

    def test_stream_chunk_minimal(self):
        """Test StreamChunk with all default values."""
        chunk = StreamChunk()

        encoded = msgspec.msgpack.encode(chunk)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.id == ""
        assert decoded.response_chunk == ""
        assert decoded.metadata == {}

    def test_stream_chunk_with_basic_content(self):
        """Test StreamChunk with basic response content."""
        chunk = StreamChunk(
            id="query-123", response_chunk="Hello, this is a chunk of text."
        )

        encoded = msgspec.msgpack.encode(chunk)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.id == "query-123"
        assert decoded.response_chunk == "Hello, this is a chunk of text."

    def test_stream_chunk_first_chunk(self):
        """Test StreamChunk representing first chunk with metadata."""
        chunk = StreamChunk(
            id="query-456",
            response_chunk="First token",
            metadata={"first_chunk": True, "latency_ns": 1234567},
        )

        encoded = msgspec.msgpack.encode(chunk)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.metadata["first_chunk"] is True
        assert decoded.metadata["latency_ns"] == 1234567

    def test_stream_chunk_with_comprehensive_metadata(self):
        """Test StreamChunk with detailed metadata."""
        chunk = StreamChunk(
            id="query-meta",
            response_chunk=" next token",
            metadata={
                "model": "llama-2-70b",
                "chunk_index": 5,
                "tokens_so_far": 50,
                "timestamp_ns": 1700000000000000,
                "first_chunk": False,
            },
        )

        encoded = msgspec.msgpack.encode(chunk)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.metadata["chunk_index"] == 5
        assert decoded.metadata["tokens_so_far"] == 50
        assert decoded.metadata["first_chunk"] is False

    def test_stream_chunk_empty_response(self):
        """Test StreamChunk with empty response text."""
        chunk = StreamChunk(id="query-empty", response_chunk="")

        encoded = msgspec.msgpack.encode(chunk)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.response_chunk == ""

    def test_stream_chunk_special_characters(self):
        """Test StreamChunk with special characters and unicode."""
        chunk = StreamChunk(
            id="query-unicode", response_chunk="Hello 世界! 🚀 Special chars: \n\t\r"
        )

        encoded = msgspec.msgpack.encode(chunk)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.response_chunk == "Hello 世界! 🚀 Special chars: \n\t\r"

    def test_stream_chunk_all_fields_populated(self):
        """Test StreamChunk with all fields fully populated."""
        chunk = StreamChunk(
            id="query-full-chunk",
            response_chunk="Complete chunk text",
            metadata={
                "model": "gpt-4",
                "finish_reason": "stop",
                "total_tokens": 100,
            },
        )

        encoded = msgspec.msgpack.encode(chunk)
        decoded = msgspec.msgpack.decode(encoded, type=StreamChunk)

        assert decoded.id == "query-full-chunk"
        assert decoded.response_chunk == "Complete chunk text"
        assert decoded.metadata["finish_reason"] == "stop"

    def test_stream_chunk_multiple_roundtrips(self):
        """Test StreamChunk survives multiple serialization roundtrips."""
        original = StreamChunk(
            id="query-roundtrip",
            response_chunk="Test chunk",
            metadata={"index": 1},
        )

        # First roundtrip
        encoded1 = msgspec.msgpack.encode(original)
        decoded1 = msgspec.msgpack.decode(encoded1, type=StreamChunk)

        # Second roundtrip
        encoded2 = msgspec.msgpack.encode(decoded1)
        decoded2 = msgspec.msgpack.decode(encoded2, type=StreamChunk)

        # Verify all fields remain consistent
        assert decoded2.id == original.id
        assert decoded2.response_chunk == original.response_chunk
        assert decoded2.metadata == original.metadata


class TestQueryResultWorkerPatterns:
    """Test QueryResult serialization patterns used by worker.py (TextModelOutput)."""

    def test_query_result_reasoning_chunks_pattern(self):
        """Test the exact pattern used when worker has reasoning chunks."""
        reasoning_chunks = ["Let me think...", " step by step", " to solve this"]
        output_chunks = ["The answer", " is", " 42"]

        resp_reasoning = [reasoning_chunks[0]]
        if len(reasoning_chunks) > 1:
            resp_reasoning.append("".join(reasoning_chunks[1:]))

        result = QueryResult(
            id="query-reasoning-pattern",
            response_output=TextModelOutput(
                output="".join(output_chunks),
                reasoning=resp_reasoning,
            ),
            metadata={"first_chunk": False, "final_chunk": True},
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == "The answer is 42"
        assert decoded.response_output.reasoning == (
            "Let me think...",
            " step by step to solve this",
        )
        assert len(decoded.response_output.reasoning) == 2

    def test_query_result_output_only_pattern(self):
        """Test the exact pattern used when worker has only output chunks."""
        output_chunks = ["Hello", " world", "!"]

        resp_output = [output_chunks[0]]
        if len(output_chunks) > 1:
            resp_output.append("".join(output_chunks[1:]))

        result = QueryResult(
            id="query-output-pattern",
            response_output=TextModelOutput(output=resp_output, reasoning=None),
            metadata={"first_chunk": False, "final_chunk": True},
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("Hello", " world!")
        assert len(decoded.response_output.output) == 2
        assert decoded.response_output.reasoning is None

    def test_query_result_no_chunks_pattern(self):
        """Test the exact pattern used when worker has no chunks."""
        result = QueryResult(
            id="query-no-chunks",
            response_output=TextModelOutput(output=[], reasoning=None),
            metadata={"first_chunk": True, "final_chunk": True},
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ()
        assert decoded.response_output.reasoning is None

    def test_query_result_single_reasoning_chunk(self):
        """Test pattern when there's only one reasoning chunk."""
        result = QueryResult(
            id="query-single-reasoning",
            response_output=TextModelOutput(
                output="Answer",
                reasoning=["Quick thought"],
            ),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.reasoning == ("Quick thought",)
        assert len(decoded.response_output.reasoning) == 1

    def test_query_result_single_output_chunk(self):
        """Test pattern when there's only one output chunk."""
        result = QueryResult(
            id="query-single-output",
            response_output=TextModelOutput(output=["SingleResponse"], reasoning=None),
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("SingleResponse",)
        assert len(decoded.response_output.output) == 1


class TestTextAfterFirstChunk:
    """Test TextModelOutput.text_after_first_chunk() for all reasoning/output combos."""

    @pytest.mark.parametrize(
        "reasoning, output, expected",
        [
            # No reasoning, str output (non-streaming)
            (None, "abc", ""),
            # No reasoning, tuple output (streaming)
            (None, ("a", "b", "c"), "bc"),
            # No reasoning, single-chunk tuple
            (None, ("a",), ""),
            # No reasoning, empty tuple
            (None, (), ""),
            # Str reasoning (is the first chunk), str output (non-streaming)
            ("think", "abc", ""),
            # Str reasoning (is the first chunk), tuple output
            ("think", ("a", "b"), "ab"),
            # Tuple reasoning (multi), str output
            (("t1", "t2"), "abc", "t2abc"),
            # Tuple reasoning (multi), tuple output
            (("t1", "t2"), ("a", "b"), "t2ab"),
            # Single-element tuple reasoning (is the first chunk), str output
            (("t1",), "abc", "abc"),
            # Single-element tuple reasoning (is the first chunk), tuple output
            (("t1",), ("a", "b"), "ab"),
            # Falsy str reasoning (empty string), tuple output — treated as no reasoning
            ("", ("a", "b"), "b"),
            # Empty tuple reasoning, tuple output — treated as no reasoning
            ((), ("a", "b"), "b"),
            # No reasoning, empty str output
            (None, "", ""),
        ],
        ids=[
            "no_reasoning-str_output",
            "no_reasoning-tuple_output",
            "no_reasoning-single_chunk",
            "no_reasoning-empty_tuple",
            "str_reasoning-str_output",
            "str_reasoning-tuple_output",
            "multi_reasoning-str_output",
            "multi_reasoning-tuple_output",
            "single_reasoning-str_output",
            "single_reasoning-tuple_output",
            "empty_str_reasoning-tuple_output",
            "empty_tuple_reasoning-tuple_output",
            "no_reasoning-empty_str_output",
        ],
    )
    def test_text_after_first_chunk(self, reasoning, output, expected):
        tmo = TextModelOutput(output=output, reasoning=reasoning)
        assert tmo.text_after_first_chunk() == expected


class TestMixedTypeSerialization:
    """Test serialization of mixed type combinations and edge cases."""

    def test_serialize_list_of_queries(self):
        """Test serializing a list of Query objects."""
        queries = [
            Query(data={"prompt": "Query 1"}),
            Query(data={"prompt": "Query 2"}),
            Query(data={"prompt": "Query 3"}),
        ]

        encoded = msgspec.msgpack.encode(queries)
        decoded = msgspec.msgpack.decode(encoded, type=list[Query])

        assert len(decoded) == 3
        assert decoded[0].data["prompt"] == "Query 1"
        assert decoded[2].data["prompt"] == "Query 3"

    def test_serialize_list_of_query_results(self):
        """Test serializing a list of QueryResult objects."""
        results = [
            QueryResult(id="r1", response_output=TextModelOutput(output="Response 1")),
            QueryResult(id="r2", response_output=TextModelOutput(output="Response 2")),
            QueryResult(
                id="r3",
                error=ErrorData(
                    error_type="RuntimeError", error_message="Error in query 3"
                ),
            ),
        ]

        encoded = msgspec.msgpack.encode(results)
        decoded = msgspec.msgpack.decode(encoded, type=list[QueryResult])

        assert len(decoded) == 3
        assert decoded[0].response_output == TextModelOutput(output="Response 1")
        assert decoded[2].error is not None
        assert decoded[2].error.error_type == "RuntimeError"
        assert decoded[2].error.error_message == "Error in query 3"

    def test_serialize_list_of_stream_chunks(self):
        """Test serializing a list of StreamChunk objects."""
        chunks = [
            StreamChunk(
                id="q1", response_chunk="First", metadata={"first_chunk": True}
            ),
            StreamChunk(id="q1", response_chunk=" second"),
            StreamChunk(id="q1", response_chunk=" final"),
        ]

        encoded = msgspec.msgpack.encode(chunks)
        decoded = msgspec.msgpack.decode(encoded, type=list[StreamChunk])

        assert len(decoded) == 3
        assert decoded[0].metadata.get("first_chunk") is True

    def test_query_result_with_nested_metadata(self):
        """Test QueryResult with deeply nested metadata and TextModelOutput."""
        result = QueryResult(
            id="query-nested",
            response_output=TextModelOutput(
                output=["First chunk", "remaining output"],
                reasoning=["Reasoning process"],
            ),
            metadata={
                "model_info": {
                    "name": "gpt-4",
                    "version": "2024-01",
                    "parameters": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "breakdown": [5, 5, 10, 10],
                },
                "tags": ["production", "high-priority"],
            },
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.metadata["model_info"]["parameters"]["temperature"] == 0.7
        assert decoded.metadata["usage"]["breakdown"] == [5, 5, 10, 10]
        assert "production" in decoded.metadata["tags"]
        assert isinstance(decoded.response_output, TextModelOutput)
        assert decoded.response_output.output == ("First chunk", "remaining output")
        assert decoded.response_output.reasoning == ("Reasoning process",)

    def test_query_with_none_values_in_data(self):
        """Test Query with None values in data dict."""
        query = Query(
            data={
                "prompt": "Test",
                "optional_param": None,
                "another_param": None,
            }
        )

        encoded = msgspec.msgpack.encode(query)
        decoded = msgspec.msgpack.decode(encoded, type=Query)

        assert decoded.data["optional_param"] is None
        assert decoded.data["another_param"] is None

    def test_large_payload_serialization(self):
        """Test serialization of large payloads."""
        large_text = "A" * 10000  # 10KB of text

        query = Query(data={"prompt": large_text})
        encoded_query = msgspec.msgpack.encode(query)
        decoded_query = msgspec.msgpack.decode(encoded_query, type=Query)
        assert decoded_query.data["prompt"] == large_text

        result = QueryResult(
            id="large", response_output=TextModelOutput(output=large_text)
        )
        encoded_result = msgspec.msgpack.encode(result)
        decoded_result = msgspec.msgpack.decode(encoded_result, type=QueryResult)
        assert decoded_result.response_output == TextModelOutput(output=large_text)

    def test_numeric_types_in_metadata(self):
        """Test various numeric types in metadata."""
        result = QueryResult(
            id="numeric-test",
            response_output=TextModelOutput(output="Text"),
            metadata={
                "int_value": 42,
                "float_value": 3.14159,
                "large_int": 9999999999999999,
                "negative": -123.456,
                "zero": 0,
                "zero_float": 0.0,
            },
        )

        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=QueryResult)

        assert decoded.metadata["int_value"] == 42
        assert abs(decoded.metadata["float_value"] - 3.14159) < 0.00001
        assert decoded.metadata["large_int"] == 9999999999999999
        assert decoded.metadata["negative"] == -123.456
        assert decoded.metadata["zero"] == 0
