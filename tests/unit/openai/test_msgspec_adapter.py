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

"""Unit tests for OpenAIMsgspecAdapter with tool call fields."""

import json

import msgspec
import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.openai.openai_msgspec_adapter import (
    OpenAIMsgspecAdapter,
    _chat_message_from_dict,
)
from inference_endpoint.openai.types import ChatMessage


@pytest.mark.unit
def test_chat_message_tool_calls_serialised():
    """tool_calls field is included in the JSON output when non-None."""
    tool_calls = [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }
    ]
    msg = ChatMessage(role="assistant", tool_calls=tool_calls)
    encoded = msgspec.json.encode(msg)
    decoded = json.loads(encoded)
    assert decoded["role"] == "assistant"
    assert decoded["tool_calls"] == tool_calls
    assert "content" not in decoded  # omit_defaults=True, None omitted


@pytest.mark.unit
def test_chat_message_tool_call_id_serialised():
    """tool_call_id field is included in the JSON output when non-None."""
    msg = ChatMessage(role="tool", content="result", tool_call_id="call_0")
    encoded = msgspec.json.encode(msg)
    decoded = json.loads(encoded)
    assert decoded["role"] == "tool"
    assert decoded["content"] == "result"
    assert decoded["tool_call_id"] == "call_0"


@pytest.mark.unit
def test_to_endpoint_request_preserves_tool_calls():
    """to_endpoint_request forwards tool_calls in the messages array."""
    tool_calls = [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q": "test"}'},
        }
    ]
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "content": "answer", "tool_call_id": "call_0"},
        {"role": "assistant", "content": "Done"},
    ]
    query = Query(
        id="q1",
        data={
            "model": "test-model",
            "messages": messages,
        },
    )
    request = OpenAIMsgspecAdapter.to_endpoint_request(query)
    encoded = msgspec.json.encode(request)
    payload = json.loads(encoded)

    msgs = payload["messages"]
    # assistant tool-dispatch row
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"] == tool_calls
    assert "content" not in msgs[1]
    # tool result row
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "call_0"
    assert msgs[2]["content"] == "answer"
    # terminal assistant row
    assert msgs[3]["content"] == "Done"


@pytest.mark.unit
def test_backward_compat_plain_messages_unchanged():
    """Plain user/assistant messages encode identically to before the change."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    query = Query(
        id="q2",
        data={"model": "m", "messages": messages},
    )
    request = OpenAIMsgspecAdapter.to_endpoint_request(query)
    encoded = msgspec.json.encode(request)
    payload = json.loads(encoded)

    for i, msg in enumerate(payload["messages"]):
        assert msg["role"] == messages[i]["role"]
        assert msg["content"] == messages[i]["content"]
        assert "tool_calls" not in msg
        assert "tool_call_id" not in msg


@pytest.mark.unit
def test_chat_message_from_dict_all_fields():
    """_chat_message_from_dict forwards all four optional fields."""
    tool_calls = [
        {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]
    msg = _chat_message_from_dict(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "tool_call_id": None,
        }
    )
    assert msg.role == "assistant"
    assert msg.content is None
    assert msg.tool_calls == tool_calls
    assert msg.tool_call_id is None


@pytest.mark.unit
def test_chat_message_content_optional():
    """ChatMessage accepts content=None for tool-dispatching assistant turns."""
    msg = ChatMessage(role="assistant", tool_calls=[])
    assert msg.content is None
