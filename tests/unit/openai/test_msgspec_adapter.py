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
import pandas as pd
import pytest
from inference_endpoint.core.types import Query, TextModelOutput
from inference_endpoint.openai.openai_msgspec_adapter import (
    OpenAIMsgspecAdapter,
    _chat_message_from_dict,
)
from inference_endpoint.openai.types import (
    STREAM_OPTIONS_INCLUDE_USAGE,
    ChatMessage,
    SSEChoice,
    SSEDelta,
    SSEMessage,
    requests_multiple_choices,
)


def _make_reasoning_response(
    field_name: str,
    value: str,
    extra_message_fields: dict | None = None,
) -> bytes:
    message: dict = {"role": "assistant", "content": "answer", field_name: value}
    if extra_message_fields:
        message.update(extra_message_fields)
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()


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
    request = OpenAIMsgspecAdapter.to_endpoint_request(query, include_usage=False)
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
    request = OpenAIMsgspecAdapter.to_endpoint_request(query, include_usage=False)
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


@pytest.mark.unit
def test_from_endpoint_response_populates_tool_calls_in_text_output():
    """Non-streaming response with tool_calls populates TextModelOutput.tool_calls."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
    ]
    response_bytes = json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": None,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "system_fingerprint": None,
        }
    ).encode()

    result = OpenAIMsgspecAdapter.decode_response(response_bytes, "q1")

    assert isinstance(result.response_output, TextModelOutput)
    assert result.response_output.tool_calls is not None
    assert len(result.response_output.tool_calls) == 1
    assert result.response_output.tool_calls[0]["function"]["name"] == "search"
    # metadata path unchanged
    assert result.metadata.get("tool_calls") == tool_calls


@pytest.mark.unit
def test_seed_forwarded_in_metadata():
    """seed from ModelParams is propagated into the request metadata dict."""
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import AddStaticColumns

    model_params = ModelParams(name="test-model", seed=42)
    transforms = OpenAIMsgspecAdapter.dataset_transforms(model_params)
    injector = next(t for t in transforms if isinstance(t, AddStaticColumns))
    assert injector.data.get("seed") == 42


@pytest.mark.unit
def test_seed_none_when_not_set():
    """seed key is present but None in metadata when ModelParams.seed is not configured."""
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import AddStaticColumns

    model_params = ModelParams(name="test-model")
    transforms = OpenAIMsgspecAdapter.dataset_transforms(model_params)
    injector = next(t for t in transforms if isinstance(t, AddStaticColumns))
    assert injector.data.get("seed") is None


@pytest.mark.unit
def test_max_completion_tokens_always_emitted():
    """max_new_tokens always has a value (default 1024), so the metadata always
    carries max_completion_tokens — matching OpenAIAdapter (no conditional guard)."""
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import AddStaticColumns

    mp = ModelParams(name="test-model", max_new_tokens=256)
    transforms = OpenAIMsgspecAdapter.dataset_transforms(mp)
    injector = next(t for t in transforms if isinstance(t, AddStaticColumns))
    assert injector.data["max_completion_tokens"] == 256


@pytest.mark.unit
def test_seed_in_request_payload():
    """seed passes through to_endpoint_request when present in query data."""
    query = Query(
        id="q-seed",
        data={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "seed": 99,
        },
    )
    request = OpenAIMsgspecAdapter.to_endpoint_request(query, include_usage=False)
    encoded = msgspec.json.encode(request)
    payload = json.loads(encoded)
    assert payload.get("seed") == 99


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ["reasoning", "reasoning_content"],
    ids=["vllm", "sglang"],
)
def test_from_endpoint_response_reasoning_field(field_name: str):
    """Non-streaming reasoning is captured regardless of server field name."""
    result = OpenAIMsgspecAdapter.decode_response(
        _make_reasoning_response(field_name, "thinking..."), "q1"
    )
    assert isinstance(result.response_output, TextModelOutput)
    assert result.response_output.reasoning == "thinking..."
    assert result.metadata.get("reasoning_content") == "thinking..."


@pytest.mark.unit
def test_from_endpoint_response_reasoning_both_fields_raises():
    """Both reasoning fields populated simultaneously triggers an assertion error."""
    response = _make_reasoning_response(
        "reasoning_content",
        "thinking via sglang",
        extra_message_fields={"reasoning": "thinking via vllm"},
    )
    with pytest.raises(AssertionError):
        OpenAIMsgspecAdapter.decode_response(response, "q1")


@pytest.mark.unit
def test_chat_template_kwargs_forwarded_when_set():
    query = Query(
        id="q3",
        data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"thinking": True, "preserve_thinking": True},
        },
    )
    request = OpenAIMsgspecAdapter.to_endpoint_request(query, include_usage=False)
    payload = json.loads(msgspec.json.encode(request))
    assert payload["chat_template_kwargs"] == {
        "thinking": True,
        "preserve_thinking": True,
    }


@pytest.mark.unit
def test_chat_template_kwargs_omitted_when_unset():
    query = Query(
        id="q4",
        data={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    request = OpenAIMsgspecAdapter.to_endpoint_request(query, include_usage=False)
    payload = json.loads(msgspec.json.encode(request))
    assert "chat_template_kwargs" not in payload


@pytest.mark.unit
def test_dataset_transforms_includes_chat_template_kwargs():
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import AddStaticColumns

    mp = ModelParams(
        name="m", chat_template_kwargs={"thinking": True, "preserve_thinking": True}
    )
    transforms = OpenAIMsgspecAdapter.dataset_transforms(mp)
    add_static = next(t for t in transforms if isinstance(t, AddStaticColumns))
    assert add_static.data["chat_template_kwargs"] == {
        "thinking": True,
        "preserve_thinking": True,
    }


@pytest.mark.unit
def test_dataset_transforms_preserve_chat_template_kwargs_dict():
    from inference_endpoint.config.schema import ModelParams

    chat_template_kwargs = {"thinking": True, "preserve_thinking": True}
    mp = ModelParams(name="m", chat_template_kwargs=chat_template_kwargs)
    df = pd.DataFrame({"prompt": ["hi"]})

    for transform in OpenAIMsgspecAdapter.dataset_transforms(mp):
        df = transform(df)

    row = df.to_dict(orient="records")[0]
    assert row["chat_template_kwargs"] == chat_template_kwargs

    request = OpenAIMsgspecAdapter.to_endpoint_request(
        Query(id="q5", data=row), include_usage=False
    )
    payload = json.loads(msgspec.json.encode(request))
    assert payload["chat_template_kwargs"] == chat_template_kwargs


@pytest.mark.unit
def test_messages_shadow_prompt_in_request():
    """'messages' takes precedence over 'prompt' in the emitted request.

    A sample carrying both fields sends only 'messages'; the 'prompt' is dropped
    entirely. This is why _check_unsaltable() rejects such samples — salting the
    ignored 'prompt' would not change the payload and would silently fail to bust
    the KV cache.
    """
    query = Query(
        id="q6",
        data={
            "model": "m",
            "messages": [{"role": "user", "content": "authoritative"}],
            "prompt": "[deadbeefdeadbeef] shadowed",
        },
    )
    request = OpenAIMsgspecAdapter.to_endpoint_request(query, include_usage=False)
    payload = json.loads(msgspec.json.encode(request))

    assert [m["content"] for m in payload["messages"]] == ["authoritative"]
    assert "shadowed" not in json.dumps(payload)


_CHAT = {
    "model": "m",
    "system": "be brief",
    "prompt": "hi",
    "max_completion_tokens": 8,
    "temperature": 0.0,
}


def _chat_body(include_usage: bool, **data) -> dict:
    query = Query(id="q", data={**_CHAT, **data})
    return json.loads(
        OpenAIMsgspecAdapter.encode_query(query, include_usage=include_usage)
    )


@pytest.mark.unit
def test_streaming_body_requests_usage_when_enabled():
    query = Query(id="q", data={**_CHAT, "stream": True})
    request = OpenAIMsgspecAdapter.to_endpoint_request(query, include_usage=True)
    assert request.stream_options is STREAM_OPTIONS_INCLUDE_USAGE
    assert _chat_body(True, stream=True)["stream_options"] == {"include_usage": True}


# The exact body sent when usage is not requested.
_MAIN_CHAT_BODY = {
    True: b'{"model":"m","messages":[{"role":"system","content":"be brief"},'
    b'{"role":"user","content":"hi"}],"temperature":0.0,'
    b'"max_completion_tokens":8,"stream":true}',
    False: b'{"model":"m","messages":[{"role":"system","content":"be brief"},'
    b'{"role":"user","content":"hi"}],"temperature":0.0,'
    b'"max_completion_tokens":8,"stream":false}',
}


@pytest.mark.unit
@pytest.mark.parametrize("stream", [True, False])
def test_body_matches_main_when_usage_disabled(stream):
    query = Query(id="q", data={**_CHAT, "stream": stream})
    assert (
        OpenAIMsgspecAdapter.encode_query(query, include_usage=False)
        == _MAIN_CHAT_BODY[stream]
    )


@pytest.mark.unit
@pytest.mark.parametrize("data", [{"stream": False}, {}], ids=["stream-false", "unset"])
def test_non_streaming_body_never_requests_usage(data):
    assert "stream_options" not in _chat_body(True, **data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "n, expected",
    [(2, "absent"), (2.0, "absent"), (1, {"include_usage": True})],
)
def test_n_gt_1_row_omits_stream_options(n, expected):
    body = _chat_body(True, stream=True, n=n)
    assert body["n"] == n
    assert body.get("stream_options", "absent") == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "n, expected",
    [
        (2, True),
        (2.5, True),
        (1, False),
        (1.0, False),
        (0, False),
        (None, False),
        (True, False),
        ("2", False),
        (float("nan"), False),
    ],
)
def test_requests_multiple_choices(n, expected):
    assert requests_multiple_choices(n) is expected


_USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
_HI = SSEChoice(delta=SSEDelta(content="Hi"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "frame, expected",
    [
        pytest.param({"choices": [{"delta": {"content": "Hi"}}]}, _HI, id="choice"),
        pytest.param(
            {"choices": [{"delta": {"content": "Hi"}}], "usage": None},
            _HI,
            id="openai-null-usage",
        ),
        pytest.param(
            {"choices": [], "usage": _USAGE},
            SSEMessage(usage=_USAGE),
            id="usage-trailer",
        ),
        pytest.param(
            {"choices": [{"delta": {"content": "Hi"}}], "usage": _USAGE},
            SSEMessage(choices=(_HI,), usage=_USAGE),
            id="litellm",
        ),
        pytest.param(
            {"choices": [], "usage": {"prompt_tokens": None, "completion_tokens": 3}},
            SSEMessage(usage={"prompt_tokens": None, "completion_tokens": 3}),
            id="null-prompt-tokens",
        ),
        pytest.param(
            {"choices": [], "usage": {}}, SSEMessage(usage={}), id="empty-usage-object"
        ),
        pytest.param(
            {"choices": [{"delta": {"content": "Hi"}}], "usage": "x"},
            SSEMessage(choices=(_HI,), usage="x"),
            id="malformed-usage",
        ),
        pytest.param(
            {"choices": [{"index": 1, "delta": {"content": "Hi"}}]},
            _HI,
            id="second-choice-index-no-usage",
        ),
        pytest.param(
            {"choices": [{"delta": {"content": "Hi"}}, {"index": 1, "delta": {}}]},
            _HI,
            id="two-choices-no-usage",
        ),
        pytest.param(
            {
                "choices": [{"delta": {"content": "Hi"}}, {"index": 1, "delta": {}}],
                "usage": _USAGE,
            },
            SSEMessage(choices=(_HI, SSEChoice(delta=SSEDelta())), usage=_USAGE),
            id="two-choices-with-usage",
        ),
        pytest.param({"choices": []}, None, id="sglang-empty-choices"),
        pytest.param({"choices": [], "usage": None}, None, id="empty-null-usage"),
    ],
)
def test_decode_sse_message_usage_carrier(frame, expected):
    assert (
        OpenAIMsgspecAdapter.decode_sse_message(msgspec.json.encode(frame)) == expected
    )


@pytest.mark.unit
def test_decode_sse_message_common_frame_is_the_bare_choice():
    decoded = OpenAIMsgspecAdapter.decode_sse_message(
        b'{"choices":[{"index":0,"delta":{"content":"Hi"}}],"usage":null}'
    )
    assert type(decoded) is SSEChoice


def _chat_body_with(usage: object, n_choices: int = 1) -> bytes:
    body = json.loads(_make_reasoning_response("reasoning", "r"))
    body["choices"] = [{**body["choices"][0], "index": i} for i in range(n_choices)]
    body["usage"] = usage
    return json.dumps(body).encode()


@pytest.mark.unit
@pytest.mark.parametrize(
    "usage", [{}, "x", [], {"prompt_tokens": 7.0, "completion_tokens": -3}]
)
def test_non_streaming_malformed_usage_succeeds(usage):
    result = OpenAIMsgspecAdapter.decode_response(_chat_body_with(usage), "q1")
    assert result.error is None
    assert result.response_output.output == "answer"
    assert result.metadata["finish_reason"] == "stop"
    assert "prompt_tokens" not in result.metadata
    assert "completion_tokens" not in result.metadata


@pytest.mark.unit
def test_non_streaming_n_gt_1_has_no_counts():
    result = OpenAIMsgspecAdapter.decode_response(
        _chat_body_with(_USAGE, n_choices=2), "q1"
    )
    assert result.response_output.output == "answer"
    assert "prompt_tokens" not in result.metadata
    assert "completion_tokens" not in result.metadata


@pytest.mark.unit
@pytest.mark.parametrize(
    "usage, expected",
    [(_USAGE, {"prompt_tokens": 7, "completion_tokens": 3}), (None, {})],
    ids=["usage", "no-usage"],
)
def test_decode_response_copies_usage(usage, expected):
    body = json.loads(_make_reasoning_response("reasoning", "r"))
    if usage is not None:
        body["usage"] = usage
    md = OpenAIMsgspecAdapter.decode_response(json.dumps(body).encode(), "q1").metadata
    assert {
        k: md[k] for k in ("prompt_tokens", "completion_tokens") if k in md
    } == expected
