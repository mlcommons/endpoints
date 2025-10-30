# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Unit tests for core types.

These tests verify the basic data structures work correctly.
"""

from inference_endpoint.core.types import (
    Query,
    QueryResult,
    QueryStatus,
    StreamChunk,
)
from inference_endpoint.openai.openai_adapter import OpenAIAdapter


class TestQuery:
    """Test the Query dataclass."""

    def test_query_creation(self) -> None:
        """Test creating a basic query."""
        payload = {
            "prompt": "Test prompt",
            "model": "test-model",
            "max_completion_tokens": 100,
        }
        query = OpenAIAdapter.to_endpoint_request(
            Query(id="test-123", data=payload)
        ).model_dump(mode="json")
        assert query["messages"][0]["content"] == "Test prompt"
        # TODO : remove this once we have a way to handle the assistant message
        # assert query["messages"][1]["content"] == "You are a helpful assistant."
        assert query["model"] == "test-model"
        assert query["max_completion_tokens"] == 100
        assert query["temperature"] == 0.7  # default value
        # assert query["created_at"] is not None

    def test_query_store_load(self) -> None:
        """Test creating a basic query."""
        payload = {
            "prompt": "Test prompt",
            "model": "test-model",
            "max_completion_tokens": 100,
            "temperature": 0.7,
        }

        query_loaded = OpenAIAdapter.to_endpoint_request(
            Query(id="test-123", data=payload)
        )
        assert query_loaded.messages[0].root.content == payload["prompt"]
        # TODO : remove this once we have a way to handle the assistant message
        # assert query_loaded.messages[1].root.content == "You are a helpful assistant."
        assert query_loaded.model.root == payload["model"]
        assert query_loaded.max_completion_tokens == payload["max_completion_tokens"]
        assert query_loaded.temperature == payload["temperature"]

    def test_query_defaults(self) -> None:
        """Test query with minimal parameters."""
        query = Query(id="test-123", data={"prompt": ""})
        assert "prompt" in query.data and query.data["prompt"] == ""
        assert query.id == "test-123"
        assert query.created_at is not None


class TestQueryResult:
    """Test the QueryResult dataclass."""

    def test_query_result_creation(self) -> None:
        """Test creating a query result."""
        result = QueryResult(id="test-123", response_output="Test response")

        assert result.id == "test-123"
        assert result.response_output == "Test response"
        assert result.error is None
        assert result.completed_at is not None


class TestStreamChunk:
    """Test the StreamChunk dataclass."""

    def test_stream_chunk_creation(self) -> None:
        """Test creating a stream chunk."""
        chunk = StreamChunk(id="test-123", response_chunk="partial", is_complete=False)

        assert chunk.id == "test-123"
        assert chunk.response_chunk == "partial"
        assert chunk.is_complete is False
        assert chunk.metadata == {}


class TestQueryStatus:
    """Test the QueryStatus enum."""

    def test_status_values(self) -> None:
        """Test that all expected status values exist."""
        assert QueryStatus.PENDING.value == "pending"
        assert QueryStatus.RUNNING.value == "running"
        assert QueryStatus.COMPLETED.value == "completed"
        assert QueryStatus.FAILED.value == "failed"
        assert QueryStatus.CANCELLED.value == "cancelled"
