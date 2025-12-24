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

"""ASGI application for the echo server.

This module is loaded by Granian in the subprocess. It provides a minimal,
high-performance ASGI application for echoing requests.
"""

import os
import time
import uuid
from typing import Any

import orjson

# Read configuration from environment variables (set by parent process)
_MAX_OSL: int | None = None
_RESPONSE_MODULE: str | None = None
_RESPONSE_DATA: Any = None
_RESPONSE_FUNC: Any = None


def _init_config():
    """Initialize configuration from environment variables."""
    global _MAX_OSL, _RESPONSE_MODULE, _RESPONSE_DATA, _RESPONSE_FUNC

    max_osl_str = os.environ.get("_ECHO_SERVER_MAX_OSL", "")
    _MAX_OSL = int(max_osl_str) if max_osl_str else None

    _RESPONSE_MODULE = os.environ.get("_ECHO_SERVER_RESPONSE_MODULE")
    response_data_str = os.environ.get("_ECHO_SERVER_RESPONSE_DATA")
    if response_data_str:
        _RESPONSE_DATA = orjson.loads(response_data_str)


def _get_response(request: str) -> str:
    """Get response for request (echo by default)."""
    if _RESPONSE_FUNC is not None:
        return _RESPONSE_FUNC(request)
    return request


def _apply_max_osl(response: str) -> str:
    """Apply maximum output sequence length if configured."""
    if _MAX_OSL is None or len(response) == 0:
        return response

    if len(response) > _MAX_OSL:
        return response[:_MAX_OSL]
    else:
        # Repeat until we reach max_osl
        repeated = response * (_MAX_OSL // len(response) + 1)
        return repeated[:_MAX_OSL]


async def _read_body(receive) -> bytes:
    """Read the full request body."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    return body


async def _send_json_response(send, data: dict, status: int = 200):
    """Send a JSON response."""
    body = orjson.dumps(data)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _send_sse_response(
    send, request_id: str, model: str, content: str, prompt_tokens: int = 0
):
    """Send a streaming SSE response in OpenAI-compatible format."""
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/event-stream"),
                (b"cache-control", b"no-cache"),
                (b"connection", b"keep-alive"),
            ],
        }
    )

    created = int(time.time())

    # First chunk: role announcement (required by vLLM benchmark)
    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    await send(
        {
            "type": "http.response.body",
            "body": f"data: {orjson.dumps(first_chunk).decode()}\n\n".encode(),
            "more_body": True,
        }
    )

    # Stream content word by word
    words = content.split() if content else []
    output_tokens = len(words)  # Approximate: 1 word ≈ 1 token for counting

    for i, word in enumerate(words):
        chunk_content = f" {word}" if i > 0 else word
        chunk_data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": None,
            "choices": [
                {"index": 0, "delta": {"content": chunk_content}, "finish_reason": None}
            ],
        }
        await send(
            {
                "type": "http.response.body",
                "body": f"data: {orjson.dumps(chunk_data).decode()}\n\n".encode(),
                "more_body": True,
            }
        )

    # Send final chunk with finish_reason and usage (OpenAI streaming format)
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        },
    }
    await send(
        {
            "type": "http.response.body",
            "body": f"data: {orjson.dumps(final_chunk).decode()}\n\n".encode(),
            "more_body": True,
        }
    )

    # Send [DONE] marker
    await send(
        {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False}
    )


def _extract_text_content(content: str | list) -> str:
    """Extract text from content (handles both string and list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI multimodal format: [{"type": "text", "text": "..."}]
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return " ".join(texts)
    return ""


async def _handle_chat_completions(scope, receive, send):
    """Handle /v1/chat/completions endpoint."""
    try:
        body = await _read_body(receive)
        payload = orjson.loads(body)

        # Extract user message content
        raw_request = ""
        messages = payload.get("messages", [])
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                raw_request = _extract_text_content(content)
                break

        if not messages:
            await _send_json_response(
                send,
                {"error": "Request must contain at least one message"},
                status=400,
            )
            return

        request_id = payload.get("id", str(uuid.uuid4()))
        model = payload.get("model", "echo-model")
        is_streaming = payload.get("stream", False)

        # Generate response
        raw_response = _get_response(raw_request)
        raw_response = _apply_max_osl(raw_response)
        prompt_tokens = len(raw_request.split())

        if is_streaming:
            await _send_sse_response(
                send, request_id, model, raw_response, prompt_tokens
            )
        else:
            # Non-streaming response (OpenAI format compatible with msgspec adapter)
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": raw_response,
                            "refusal": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(raw_request.split()),
                    "completion_tokens": len(raw_response.split()),
                    "total_tokens": len(raw_request.split())
                    + len(raw_response.split()),
                },
                "system_fingerprint": None,
            }
            await _send_json_response(send, response)

    except orjson.JSONDecodeError as e:
        await _send_json_response(send, {"error": f"Invalid JSON: {e}"}, status=400)
    except Exception as e:
        await _send_json_response(
            send, {"error": f"error encountered: {e}"}, status=400
        )


async def _handle_echo(scope, receive, send):
    """Handle /echo endpoint."""
    try:
        body = await _read_body(receive)

        # Parse body
        try:
            json_payload = orjson.loads(body) if body else None
        except orjson.JSONDecodeError:
            json_payload = None

        raw_payload = body.decode() if body else ""

        # Build request data
        path = scope.get("path", "")
        query_string = scope.get("query_string", b"").decode()
        headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}

        request_data = {
            "method": scope.get("method", ""),
            "url": f"{scope.get('scheme', 'http')}://{headers.get('host', '')}{path}{'?' + query_string if query_string else ''}",
            "endpoint": path,
            "query_params": dict(
                item.split("=", 1) for item in query_string.split("&") if "=" in item
            )
            if query_string
            else {},
            "headers": headers,
            "json_payload": json_payload,
            "raw_payload": raw_payload,
            "timestamp": time.time(),
        }

        echo_response = {
            "echo": True,
            "request": request_data,
            "message": "Request payload echoed back successfully",
        }

        await _send_json_response(send, echo_response)

    except Exception as e:
        await _send_json_response(
            send, {"error": f"error encountered: {e}"}, status=400
        )


async def _handle_not_found(scope, receive, send):
    """Handle 404 for unknown paths."""
    await _send_json_response(
        send, {"error": "Not found", "path": scope.get("path", "")}, status=404
    )


async def _handle_health(scope, receive, send):
    """Handle /health endpoint for vLLM compatibility."""
    await _send_json_response(send, {"status": "ok"})


async def _handle_models(scope, receive, send):
    """Handle /v1/models endpoint for vLLM compatibility."""
    await _send_json_response(
        send,
        {
            "object": "list",
            "data": [
                {
                    "id": "echo-model",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "echo-server",
                }
            ],
        },
    )


async def app(scope, receive, send):
    """Main ASGI application entry point."""
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                _init_config()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
        return

    if scope["type"] != "http":
        return

    path = scope.get("path", "")
    method = scope.get("method", "")

    if method == "POST" and path == "/v1/chat/completions":
        await _handle_chat_completions(scope, receive, send)
    elif method == "POST" and path == "/echo":
        await _handle_echo(scope, receive, send)
    elif method == "GET" and path == "/health":
        await _handle_health(scope, receive, send)
    elif method == "GET" and path == "/v1/models":
        await _handle_models(scope, receive, send)
    else:
        await _handle_not_found(scope, receive, send)


def create_app_factory():
    """Factory function for Granian to create the ASGI app."""
    _init_config()
    return app
