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

"""HTTP Echo Server for testing inference endpoint clients."""

import argparse
import asyncio
import inspect
import json
import logging
import threading
import time
import uuid
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web

from inference_endpoint.core.types import QueryResult, TextModelOutput
from inference_endpoint.openai.openai_adapter import OpenAIAdapter
from inference_endpoint.openai.openai_types_gen import CreateChatCompletionRequest
from inference_endpoint.utils.logging import setup_logging

RequestHandler = Callable[[web.Request], web.Response | Awaitable[web.Response]]


def _include_usage(payload: dict[str, Any]) -> bool:
    """Whether a request asks for a trailing usage frame in its stream.

    Follows vLLM's truthiness rule: any truthy ``stream_options`` on a
    non-streaming request is rejected, so ``stream_options: {}`` is accepted.

    Raises:
        ValueError: If ``stream_options`` is set without ``stream``.
    """
    stream_options = payload.get("stream_options")
    if stream_options and not payload.get("stream"):
        raise ValueError("Stream options can only be defined when `stream=True`.")
    return bool((stream_options or {}).get("include_usage"))


class HTTPServer:
    @property
    @abstractmethod
    def url(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class EchoServer(HTTPServer):
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_osl: int | None = None,
        request_handler: RequestHandler | None = None,
        report_usage: bool = True,
    ):
        self.host = host
        self.port = port  # If 0, will auto-assign available port
        self.max_osl = max_osl
        self._request_handler = request_handler
        # False models a server that never reports token usage.
        self.report_usage = report_usage
        self._actual_port = None  # Store the actual port after binding

        self.app = None
        self.runner = None
        self.site = None
        self._server_thread = None
        self._loop = None
        self._shutdown_event = threading.Event()
        self._port_ready_event = threading.Event()  # Signal when port is ready
        self.logger = logging.getLogger(__name__)

    @property
    def url(self):
        """Get the server URL with the actual port."""
        port = self._actual_port or self.port
        return f"http://{self.host}:{port}"

    def set_max_osl(self, max_osl: int):
        self.max_osl = max_osl

    def get_max_osl(self):
        """
        Retrieve the current maximum output sequence length (OSL) setting.

        Returns:
            int: The maximum length allowed for output sequences in the echo server.
        """
        return self.max_osl

    def _usage(
        self, prompt_tokens: int, completion_tokens: int
    ) -> dict[str, int] | None:
        """Build an OpenAI usage object, or None when usage is not reported."""
        if not self.report_usage:
            return None
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _apply_max_osl(self, text: str) -> str:
        """Truncate or repeat ``text`` to exactly ``max_osl`` characters."""
        if self.max_osl is not None and len(text) > 0:
            if len(text) > self.max_osl:
                text = text[: self.max_osl]
            else:
                text = text * (self.max_osl // len(text) + 1)
                text = text[: self.max_osl]
        return text

    @staticmethod
    async def _read_json_payload(request: web.Request) -> dict[str, Any]:
        if request.content_type == "application/json":
            return await request.json()
        return json.loads(await request.text())

    def get_response(self, request: str) -> str:
        """
        Return the input request string as the response.

        This method serves as a simple echo mechanism, returning the exact request string unchanged. It can be overridden in subclasses to provide custom response generation logic.

        Args:
            request (str): The input request string to be echoed back.

        Returns:
            str: The input request string passed through unmodified
        """
        return request

    async def _dispatch(self, request: web.Request) -> web.Response | None:
        """Call the custom request_handler if set; return None to fall through to defaults."""
        if self._request_handler is None:
            return None
        result = self._request_handler(request)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _handle_echo_request(self, request: web.Request) -> web.Response:
        """
        Handle a generic HTTP request and return a JSON response that echoes all request details.

        Captures and logs comprehensive request information including method, URL, endpoint, query parameters, headers, and payload (both JSON and raw formats). Designed for testing and debugging network interactions.

        Returns a standardized JSON response containing the full request details and a success message.
        """
        custom = await self._dispatch(request)
        if custom is not None:
            return custom

        # Extract request data
        endpoint = request.path
        query_params = dict(request.query)
        headers = dict(request.headers)

        # Get request body
        try:
            if request.content_type == "application/json":
                json_payload = await request.json()
                raw_payload = json.dumps(json_payload)
            else:
                raw_payload = await request.text()
                try:
                    json_payload = json.loads(raw_payload)
                except (json.JSONDecodeError, TypeError):
                    json_payload = None
        except Exception:
            json_payload = None
            raw_payload = ""

        request_data = {
            "method": request.method,
            "url": str(request.url),
            "endpoint": endpoint,
            "query_params": query_params,
            "headers": headers,
            "json_payload": json_payload,
            "raw_payload": raw_payload,
            "timestamp": time.time(),
        }
        self.logger.info(f"Request data: {request_data}")

        # Default: echo back the request
        echo_response = {
            "echo": True,
            "request": request_data,
            "message": "Request payload echoed back successfully",
        }
        self.logger.info(f"Echo response: {echo_response}")

        return web.json_response(
            echo_response,
            status=200,
        )

    async def _handle_streaming_response(
        self,
        id: str,
        request: web.Request,
        model: str,
        content: str,
        *,
        prompt_tokens: int,
        include_usage: bool,
        text_completion: bool = False,
    ) -> web.StreamResponse:
        """
        Handle a streaming completion response using Server-Sent Events (SSE).

        Streams the response word by word, simulating an OpenAI-compatible endpoint. Creates an SSE-formatted stream with individual word chunks, a final finish chunk, an optional usage chunk (vLLM order) and the ``[DONE]`` marker.

        Args:
            id (str): Unique identifier for the streaming response.
            request (web.Request): The original web request.
            model (str): Model name echoed in every frame.
            content (str): The content to be streamed.
            prompt_tokens (int): Prompt token count reported in the usage frame.
            include_usage (bool): Whether the request asked for a usage frame. The frame is also suppressed when ``report_usage`` is False.
            text_completion (bool): Frame as ``/v1/completions`` (``text_completion`` objects with ``text`` choices) instead of chat completion chunks.

        Returns:
            web.StreamResponse: A streaming HTTP response with chunked SSE data.

        Raises:
            Any underlying exceptions that occur during streaming, which will be logged.
        """
        object_type = "text_completion" if text_completion else "chat.completion.chunk"

        def frame(choices: list[dict[str, Any]], **extra: Any) -> bytes:
            chunk = {
                "id": id,
                "object": object_type,
                "created": int(time.time()),
                "model": model,
                "choices": choices,
                **extra,
            }
            return f"data: {json.dumps(chunk)}\n\n".encode()

        def choice(text: str, finish_reason: str | None) -> dict[str, Any]:
            if text_completion:
                return {"index": 0, "text": text, "finish_reason": finish_reason}
            delta = {"content": text} if text else {}
            return {"index": 0, "delta": delta, "finish_reason": finish_reason}

        try:
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            await response.prepare(request)
            raw_response = self.get_response(content)

            # Send content in chunks (word by word for echo server)
            words = raw_response.split() if raw_response else []

            # Send chunks
            completion_tokens = 0
            for i, word in enumerate(words):
                # Add space before word (except first)
                chunk_content = f" {word}" if i > 0 else word

                await response.write(frame([choice(chunk_content, None)]))
                completion_tokens += len(chunk_content)

            # Send final chunk with finish_reason
            await response.write(frame([choice("", "stop")]))

            usage = self._usage(prompt_tokens, completion_tokens)
            if include_usage and usage is not None:
                await response.write(frame([], usage=usage))
            await response.write(b"data: [DONE]\n\n")
            return response
        except Exception as e:
            self.logger.error(f"Error handling streaming response: {e}")
            raise e
        finally:
            pass

    async def _handle_echo_chat_completions_request(
        self, request: web.Request
    ) -> web.Response:
        """
        Handle an incoming HTTP request to the OpenAI chat completions endpoint.

        Processes the request by extracting the first message content, generating a response via `get_response()`, and returning either a streaming Server-Sent Events (SSE) response or a standard JSON response based on the request configuration.

        Supports handling JSON payloads, enforcing maximum output sequence length, and converting the response to OpenAI-compatible format. Logs request and response details for debugging.

        Raises:
            ValueError: If no messages are present in the request payload.
        """
        custom = await self._dispatch(request)
        if custom is not None:
            return custom

        # Get request body
        try:
            json_payload = await self._read_json_payload(request)
            include_usage = _include_usage(json_payload)
            completion_request = CreateChatCompletionRequest(**json_payload)
            raw_request = ""
            if completion_request.messages and len(completion_request.messages) > 0:
                for message in completion_request.messages:
                    if str(message.root.role.value) == "user":
                        content = message.root.content
                        # Convert content to string - handle various content types
                        raw_request = str(content) if content is not None else ""
                        break
            else:
                raise ValueError("Request must contain at least one message")
            id = json_payload.get("id", str(uuid.uuid4()))
            raw_response = self.get_response(raw_request)
            self.logger.debug(
                f"Content of request: {raw_request} - response : {raw_response}"
            )
            raw_response = self._apply_max_osl(raw_response)

            # Check if this is a streaming request
            self.logger.debug(f"Streaming response: {completion_request.stream}\n")
            if completion_request.stream is True:
                # Return SSE (Server-Sent Events) format for streaming
                return await self._handle_streaming_response(
                    id,
                    request,
                    str(completion_request.model.root),
                    raw_response,
                    prompt_tokens=len(raw_request),
                    include_usage=include_usage,
                )
            else:
                # Non-streaming: return QueryResult
                response = QueryResult(
                    id=id,
                    response_output=TextModelOutput(output=raw_response),
                )
                echo_response = OpenAIAdapter.to_endpoint_response(response).model_dump(
                    mode="json"
                )
                echo_response["id"] = id
                echo_response["usage"] = self._usage(
                    len(raw_request), len(raw_response)
                )
                self.logger.debug(f"Echo response (non-streaming): {echo_response}")
                return web.json_response(echo_response, status=200)

        except Exception as e:
            # A catch-all exception handler to help debug the issue without bringing down the server
            self.logger.error(f"Error handling chat completions request: {e}")
            return web.json_response(
                {"error": f"error encountered : {str(e)}"},
                status=400,
            )

    async def _handle_echo_completions_request(
        self, request: web.Request
    ) -> web.StreamResponse:
        """
        Handle an incoming HTTP request to the OpenAI text completions endpoint (``/v1/completions``).

        Echoes the prompt back as the completion text: a ``str`` prompt is returned as-is and a token-ID list is joined with spaces. Returns either an SSE stream or a JSON ``text_completion`` body, with the same usage and ``stream_options`` behavior as the chat completions handler.

        Raises:
            ValueError: If the request payload has no prompt.
        """
        custom = await self._dispatch(request)
        if custom is not None:
            return custom

        try:
            json_payload = await self._read_json_payload(request)
            include_usage = _include_usage(json_payload)
            prompt = json_payload.get("prompt")
            if prompt is None:
                raise ValueError("Request must contain a prompt")
            raw_request = (
                prompt
                if isinstance(prompt, str)
                else " ".join(str(token) for token in prompt)
            )
            id = json_payload.get("id", str(uuid.uuid4()))
            model = str(json_payload.get("model", ""))
            raw_response = self._apply_max_osl(self.get_response(raw_request))

            if json_payload.get("stream") is True:
                return await self._handle_streaming_response(
                    id,
                    request,
                    model,
                    raw_response,
                    prompt_tokens=len(prompt),
                    include_usage=include_usage,
                    text_completion=True,
                )
            return web.json_response(
                {
                    "id": id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "text": raw_response, "finish_reason": "stop"}
                    ],
                    "usage": self._usage(len(prompt), len(raw_response)),
                },
                status=200,
            )

        except Exception as e:
            # A catch-all exception handler to help debug the issue without bringing down the server
            self.logger.error(f"Error handling completions request: {e}")
            return web.json_response(
                {"error": f"error encountered : {str(e)}"},
                status=400,
            )

    def _run_server(self):
        """Run the server in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())

    def _register_routes(self, app: "web.Application") -> None:
        """Register HTTP routes on the aiohttp app.

        Subclasses can override to swap out the OpenAI-shaped routes for
        a different wire contract while reusing the lifecycle plumbing.
        """
        app.router.add_post(
            "/v1/chat/completions", self._handle_echo_chat_completions_request
        )
        app.router.add_post("/v1/completions", self._handle_echo_completions_request)
        app.router.add_post("/echo", self._handle_echo_request)

    async def _start_server(self):
        """Start the HTTP server."""
        try:
            # Create the web application
            self.app = web.Application()
            self._register_routes(self.app)

            # Start the server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            # Create TCP site with backlog
            # NOTE(vir): 10k for test_massive_concurrency integration tests
            self.site = web.TCPSite(self.runner, self.host, self.port, backlog=10000)
            await self.site.start()

            # Get the actual port if we used port 0
            if self.port == 0:
                # Get the actual port assigned by the OS
                server_socket = self.site._server.sockets[0]
                self._actual_port = server_socket.getsockname()[1]
            else:
                self._actual_port = self.port

            self.logger.info(
                f"==========================\nServer started at {self.url}\n==========================",
            )

            # Signal that the port is ready
            self._port_ready_event.set()

            # Wait for shutdown signal
            while not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Server error: {e}")

        finally:  # Clean up
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()

    def start(self):
        """Start the server in a background thread."""
        self.logger.info("Starting HTTP Echo server...")
        self._server_thread = threading.Thread(target=self._run_server)
        self._server_thread.daemon = False  # Changed to False so main thread can wait
        self._server_thread.start()

        # Wait for the server to be ready and port to be assigned
        if self._port_ready_event.wait(timeout=5.0):
            self.logger.info(f"Server ready on port {self._actual_port}")
        else:
            raise RuntimeError("Server failed to start within timeout")

    def stop(self):
        """Stop the HTTP Echo server."""
        self.logger.info("Stopping HTTP Echo server...")
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._server_thread:
            self._server_thread.join(timeout=0.2)
        self.logger.info("HTTP Echo server stopped")


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="HTTP Echo Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the echo server with default settings
  echo_server

  # Show version
  echo_server --version

  # Run the echo server on port 8080
  echo_server --port 8080
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument(
        "--host", type=str, help="hostname/address to bind to", default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, help="port to bind to", default=12345)
    parser.add_argument(
        "--no-usage",
        action="store_true",
        help="never report token usage (models a server without usage)",
    )

    return parser


def main():
    """

      curl http://localhost:12345/v1/chat/completions   -H "Content-Type: application/json"   -d '{
      "model": "gpt-4o", "id" : "123",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "What is the capital of France?"
        }
      ]
    }'

    """

    #
    setup_logging()
    parser = create_parser()
    args = parser.parse_args()

    server = None
    try:
        server = EchoServer(
            host=args.host, port=args.port, report_usage=not args.no_usage
        )
        server.start()

        # Wait for the server thread to finish
        server.logger.info("Server is running. Press Ctrl+C to stop...")
        server._server_thread.join()

    except KeyboardInterrupt:
        server.logger.warning("\nKeyboard interrupt received, stopping server...")
        if server:
            server.stop()
    except Exception as e:
        if server:
            server.logger.error(f"Error starting server: {e}")
        else:
            print(f"Error starting server: {e}")


if __name__ == "__main__":
    main()
