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

"""Unit tests for TRT-LLM adapter and accumulator."""

import json

import msgspec
import pytest
from inference_endpoint.config.schema import ModelParams, StreamingMode
from inference_endpoint.core.types import Query, TextModelOutput
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnFilter,
    ColumnRemap,
)
from inference_endpoint.openai.types import SSEDelta
from inference_endpoint.trtllm.accumulator import TRTLLMSSEAccumulator
from inference_endpoint.trtllm.adapter import TRTLLMAdapter
from inference_endpoint.trtllm.types import TRTLLMChatRequest

# ============================================================================
# Adapter Tests
# ============================================================================


class TestTRTLLMAdapter:
    def test_encode_query_produces_valid_json(self):
        """Test that encode_query produces valid JSON with prompt_token_ids and messages."""
        query = Query(
            id="test-123",
            data={
                "prompt_token_ids": [1, 2, 3, 4, 5],
                "model": "test-model",
                "max_tokens": 512,
                "stream": False,
            },
        )
        encoded = TRTLLMAdapter.encode_query(query)
        parsed = json.loads(encoded)

        assert parsed["prompt_token_ids"] == [1, 2, 3, 4, 5]
        assert parsed["model"] == "test-model"
        assert parsed["max_tokens"] == 512
        assert "messages" in parsed
        assert parsed["messages"] == []

    def test_encode_query_defaults(self):
        """Test that encode_query uses correct defaults."""
        query = Query(
            id="test-456",
            data={"prompt_token_ids": [10, 20, 30]},
        )
        encoded = TRTLLMAdapter.encode_query(query)
        parsed = json.loads(encoded)

        assert parsed["prompt_token_ids"] == [10, 20, 30]
        assert parsed["model"] == "no-model-name"
        assert parsed["max_tokens"] == 1024
        # stream=False is the default but omit_defaults=True means it won't be in JSON
        # unless it differs from the struct default

    def test_encode_query_with_optional_params(self):
        """Test that optional parameters are included when provided."""
        query = Query(
            id="test-789",
            data={
                "prompt_token_ids": [1, 2, 3],
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "min_tokens": 10,
                "skip_special_tokens": True,
                "stop_token_ids": [2, 3],
            },
        )
        encoded = TRTLLMAdapter.encode_query(query)
        parsed = json.loads(encoded)

        assert parsed["temperature"] == 0.7
        assert parsed["top_p"] == 0.9
        assert parsed["top_k"] == 50
        assert parsed["min_tokens"] == 10
        assert parsed["skip_special_tokens"] is True
        assert parsed["stop_token_ids"] == [2, 3]

    def test_encode_query_omits_none_params(self):
        """Test that None optional parameters are omitted (omit_defaults)."""
        query = Query(
            id="test-omit",
            data={"prompt_token_ids": [1, 2, 3]},
        )
        encoded = TRTLLMAdapter.encode_query(query)
        parsed = json.loads(encoded)

        assert "temperature" not in parsed
        assert "top_p" not in parsed
        assert "top_k" not in parsed
        assert "min_tokens" not in parsed
        assert "skip_special_tokens" not in parsed
        assert "stop_token_ids" not in parsed

    def test_encode_query_missing_prompt_token_ids_raises(self):
        """Test that missing prompt_token_ids raises KeyError."""
        query = Query(id="test-err", data={"model": "test"})
        with pytest.raises(KeyError):
            TRTLLMAdapter.encode_query(query)

    def test_decode_response_with_openai_format(self):
        """Test decode_response with mock OpenAI-format response bytes."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!",
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
            "system_fingerprint": None,
        }
        response_bytes = json.dumps(response_data).encode()

        result = TRTLLMAdapter.decode_response(response_bytes, "query-123")

        assert result.id == "query-123"
        assert result.response_output == "Hello, world!"
        assert result.metadata["n_tokens"] == 3
        assert result.metadata["finish_reason"] == "stop"

    def test_decode_response_without_refusal(self):
        """Test decode_response when refusal field is missing (TRT-LLM responses)."""
        response_data = {
            "id": "chatcmpl-trtllm",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from TRT-LLM!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 4,
                "total_tokens": 9,
            },
            "system_fingerprint": None,
        }
        response_bytes = json.dumps(response_data).encode()

        result = TRTLLMAdapter.decode_response(response_bytes, "query-trtllm")

        assert result.id == "query-trtllm"
        assert result.response_output == "Hello from TRT-LLM!"
        assert result.metadata["n_tokens"] == 4

    def test_decode_response_without_usage(self):
        """Test decode_response when usage is not present."""
        response_data = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Response text",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": None,
            "system_fingerprint": None,
        }
        response_bytes = json.dumps(response_data).encode()

        result = TRTLLMAdapter.decode_response(response_bytes, "query-456")

        assert result.id == "query-456"
        assert result.response_output == "Response text"
        assert "n_tokens" not in result.metadata
        assert result.metadata["finish_reason"] == "length"

    def test_decode_sse_message(self):
        """Test decode_sse_message with mock SSE chunk bytes."""
        sse_data = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        sse_bytes = json.dumps(sse_data).encode()

        delta = TRTLLMAdapter.decode_sse_message(sse_bytes)

        assert isinstance(delta, SSEDelta)
        assert delta.content == "Hello"

    def test_decode_sse_message_empty_content(self):
        """Test decode_sse_message with empty content (role-only message)."""
        sse_data = {"choices": [{"delta": {}, "finish_reason": None}]}
        sse_bytes = json.dumps(sse_data).encode()

        delta = TRTLLMAdapter.decode_sse_message(sse_bytes)

        assert isinstance(delta, SSEDelta)
        # Absent content decodes to None on the current SSEDelta (omit_defaults);
        # the accumulator treats None and "" identically (`if not content`).
        assert not delta.content

    def test_dataset_transforms_returns_correct_pipeline(self):
        """Test dataset_transforms returns ColumnRemap, ColumnFilter, AddStaticColumns."""
        model_params = ModelParams(
            name="test-model",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=2048,
            streaming=StreamingMode.ON,
        )

        transforms = TRTLLMAdapter.dataset_transforms(model_params)

        assert len(transforms) == 3
        assert isinstance(transforms[0], ColumnRemap)
        assert isinstance(transforms[1], ColumnFilter)
        assert isinstance(transforms[2], AddStaticColumns)

    def test_dataset_transforms_metadata_values(self):
        """Test that dataset_transforms produces correct metadata."""
        model_params = ModelParams(
            name="my-model",
            temperature=0.5,
            top_p=0.95,
            max_new_tokens=4096,
            streaming=StreamingMode.ON,
        )

        transforms = TRTLLMAdapter.dataset_transforms(model_params)
        add_static = transforms[2]
        assert isinstance(add_static, AddStaticColumns)

        assert add_static.data["stream"] is True
        assert add_static.data["max_tokens"] == 4096
        assert add_static.data["temperature"] == 0.5
        assert add_static.data["top_p"] == 0.95
        assert add_static.data["model"] == "my-model"

    def test_dataset_transforms_no_model_name(self):
        """Test that model is not included in metadata when name is None."""
        model_params = ModelParams(max_new_tokens=1024)

        transforms = TRTLLMAdapter.dataset_transforms(model_params)
        add_static = transforms[2]
        assert isinstance(add_static, AddStaticColumns)

        assert "model" not in add_static.data

    def test_dataset_transforms_column_filter(self):
        """Test that ColumnFilter has correct required and optional columns."""
        model_params = ModelParams()

        transforms = TRTLLMAdapter.dataset_transforms(model_params)
        col_filter = transforms[1]
        assert isinstance(col_filter, ColumnFilter)

        assert col_filter.required_columns == ["prompt_token_ids"]
        assert col_filter.optional_columns == ["model"]


# ============================================================================
# Accumulator Tests
# ============================================================================


class TestTRTLLMSSEAccumulator:
    def test_first_chunk_emits_stream_chunk(self):
        """Test that the first chunk with content emits a StreamChunk (for TTFT)."""
        acc = TRTLLMSSEAccumulator(query_id="q1", stream_all_chunks=False)
        delta = SSEDelta(content="Hello")

        chunk = acc.add_chunk(delta)

        assert chunk is not None
        assert chunk.id == "q1"
        assert chunk.response_chunk == "Hello"
        assert chunk.metadata["first_chunk"] is True
        assert chunk.metadata["final_chunk"] is False

    def test_subsequent_chunks_suppressed_without_stream_all(self):
        """Test that only the first chunk is emitted when stream_all_chunks=False."""
        acc = TRTLLMSSEAccumulator(query_id="q2", stream_all_chunks=False)

        first = acc.add_chunk(SSEDelta(content="Hello"))
        second = acc.add_chunk(SSEDelta(content=" world"))
        third = acc.add_chunk(SSEDelta(content="!"))

        assert first is not None
        assert second is None
        assert third is None

    def test_stream_all_chunks_emits_every_chunk(self):
        """Test that all chunks are emitted when stream_all_chunks=True."""
        acc = TRTLLMSSEAccumulator(query_id="q3", stream_all_chunks=True)

        chunks = []
        for text in ["Hello", " world", "!"]:
            chunk = acc.add_chunk(SSEDelta(content=text))
            chunks.append(chunk)

        assert all(c is not None for c in chunks)
        assert chunks[0].metadata["first_chunk"] is True
        assert chunks[1].metadata["first_chunk"] is False
        assert chunks[2].metadata["first_chunk"] is False

    def test_get_final_output_returns_dict_for_tpot(self):
        """Test that get_final_output returns dict format for TPOT calculation."""
        acc = TRTLLMSSEAccumulator(query_id="q4", stream_all_chunks=False)
        acc.add_chunk(SSEDelta(content="Hello"))
        acc.add_chunk(SSEDelta(content=" world"))
        acc.add_chunk(SSEDelta(content="!"))

        result = acc.get_final_output()

        assert result.id == "q4"
        # Output is {"output": (first_chunk, rest_joined)} for TPOT
        # Lists are converted to tuples by QueryResult.__post_init__
        assert isinstance(result.response_output, TextModelOutput)
        assert result.response_output.output == ("Hello", " world!")
        assert result.metadata["n_tokens"] == 3
        assert result.metadata["final_chunk"] is True

    def test_empty_delta_handling(self):
        """Test that empty deltas are ignored."""
        acc = TRTLLMSSEAccumulator(query_id="q5", stream_all_chunks=True)

        chunk = acc.add_chunk(SSEDelta(content=""))
        assert chunk is None

        chunk = acc.add_chunk(SSEDelta())
        assert chunk is None

    def test_non_sse_delta_ignored(self):
        """Test that non-SSEDelta objects are ignored."""
        acc = TRTLLMSSEAccumulator(query_id="q6", stream_all_chunks=True)

        chunk = acc.add_chunk("not a delta")  # type: ignore[arg-type]
        assert chunk is None

    def test_get_final_output_empty_stream(self):
        """Test get_final_output with no chunks added."""
        acc = TRTLLMSSEAccumulator(query_id="q7", stream_all_chunks=False)

        result = acc.get_final_output()

        assert result.id == "q7"
        assert isinstance(result.response_output, TextModelOutput)
        assert result.response_output.output == ()  # empty
        assert result.metadata["n_tokens"] == 0
        assert result.metadata["first_chunk"] is True  # Never sent
        assert result.metadata["final_chunk"] is True

    def test_token_count_matches_content_chunks(self):
        """Test that n_tokens counts only non-empty content chunks."""
        acc = TRTLLMSSEAccumulator(query_id="q8", stream_all_chunks=False)

        acc.add_chunk(SSEDelta(content=""))  # Empty, ignored
        acc.add_chunk(SSEDelta(content="one"))
        acc.add_chunk(SSEDelta(content=""))  # Empty, ignored
        acc.add_chunk(SSEDelta(content="two"))
        acc.add_chunk(SSEDelta(content="three"))

        result = acc.get_final_output()
        assert result.metadata["n_tokens"] == 3
        assert result.response_output.output == ("one", "twothree")

    def test_single_chunk_output_format(self):
        """Test that a single chunk produces a tuple with one element."""
        acc = TRTLLMSSEAccumulator(query_id="q9", stream_all_chunks=False)
        acc.add_chunk(SSEDelta(content="only"))

        result = acc.get_final_output()
        assert result.response_output.output == ("only",)


# ============================================================================
# Types Tests
# ============================================================================


class TestTRTLLMTypes:
    def test_chat_request_serialization(self):
        """Test TRTLLMChatRequest serializes correctly with msgspec."""
        request = TRTLLMChatRequest(
            messages=[],
            prompt_token_ids=[1, 2, 3],
            model="test-model",
            max_tokens=512,
            temperature=0.7,
            stream=True,
        )
        encoder = msgspec.json.Encoder()
        encoded = encoder.encode(request)
        parsed = json.loads(encoded)

        assert parsed["prompt_token_ids"] == [1, 2, 3]
        assert parsed["model"] == "test-model"
        assert parsed["max_tokens"] == 512
        assert parsed["temperature"] == 0.7
        assert parsed["stream"] is True
        assert parsed["messages"] == []

    def test_chat_request_omit_defaults(self):
        """Test that default values are omitted in serialization."""
        request = TRTLLMChatRequest(
            messages=[],
            prompt_token_ids=[1, 2, 3],
        )
        encoder = msgspec.json.Encoder()
        encoded = encoder.encode(request)
        parsed = json.loads(encoded)

        # These have non-None defaults and should be omitted when matching default
        assert "temperature" not in parsed
        assert "top_p" not in parsed
        assert "top_k" not in parsed
        assert "min_tokens" not in parsed
