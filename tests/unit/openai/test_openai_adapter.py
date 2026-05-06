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

"""Unit tests for OpenAIAdapter tool serialization."""

import json

import msgspec
import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.openai.openai_adapter import OpenAIAdapter

_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
}

_TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q": "test"}'},
    }
]


@pytest.mark.unit
def test_tool_definitions_forwarded():
    """tools array in query.data is present in the encoded request."""
    messages = [
        {"role": "user", "content": "Find something"},
    ]
    query = Query(
        id="q1",
        data={
            "model": "test-model",
            "messages": messages,
            "tools": [_TOOL_DEF],
            "max_completion_tokens": 128,
            "stream": False,
        },
    )
    encoded = OpenAIAdapter.encode_query(query)
    payload = json.loads(encoded)

    assert "tools" in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == "search"


@pytest.mark.unit
def test_tool_use_messages_roundtrip():
    """Full tool-use message sequence encodes and decodes without data loss."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Find something"},
        {"role": "assistant", "content": None, "tool_calls": _TOOL_CALLS},
        {"role": "tool", "content": "search result", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "Here is the answer"},
    ]
    query = Query(
        id="q1",
        data={
            "model": "test-model",
            "messages": messages,
            "tools": [_TOOL_DEF],
            "max_completion_tokens": 128,
            "stream": False,
        },
    )
    encoded = OpenAIAdapter.encode_query(query)
    payload = json.loads(encoded)

    msgs = payload["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # assistant tool-dispatch: content is None (Pydantic model_dump includes None fields)
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"] == _TOOL_CALLS
    assert msgs[2].get("content") is None
    # tool result
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == "call_1"
    assert msgs[3]["content"] == "search result"
    # terminal assistant
    assert msgs[4]["content"] == "Here is the answer"


@pytest.mark.unit
def test_encode_request_produces_valid_json_bytes():
    """encode_request returns bytes that msgspec can decode back."""
    messages = [{"role": "user", "content": "Hello"}]
    query = Query(
        id="q2",
        data={
            "model": "m",
            "messages": messages,
            "max_completion_tokens": 64,
            "stream": False,
        },
    )
    request = OpenAIAdapter.to_endpoint_request(query)
    encoded = OpenAIAdapter.encode_request(request)

    assert isinstance(encoded, bytes)
    decoded = msgspec.json.decode(encoded)
    assert decoded["messages"][0]["role"] == "user"


@pytest.mark.unit
def test_no_tools_key_when_absent():
    """When query.data has no 'tools', the encoded payload has tools=None."""
    messages = [{"role": "user", "content": "Hello"}]
    query = Query(
        id="q3",
        data={
            "model": "m",
            "messages": messages,
            "max_completion_tokens": 64,
            "stream": False,
        },
    )
    encoded = OpenAIAdapter.encode_query(query)
    payload = json.loads(encoded)

    # Pydantic model_dump includes None fields; tools must be None when not supplied
    assert payload.get("tools") is None
