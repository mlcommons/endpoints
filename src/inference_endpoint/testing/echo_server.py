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

"""HTTP Echo Server for testing inference endpoint clients."""

import argparse
import asyncio
import json
import logging
import threading
import time
import uuid
from abc import abstractmethod

from aiohttp import web

from inference_endpoint.core.types import QueryResult
from inference_endpoint.openai.openai_adapter import OpenAIAdapter
from inference_endpoint.openai.openai_types_gen import CreateChatCompletionRequest


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
        self, *, host: str = "127.0.0.1", port: int = 0, max_osl: int | None = None
    ):
        self.host = host
        self.port = port  # If 0, will auto-assign available port
        self.max_osl = max_osl
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

    async def _handle_echo_request(self, request: web.Request) -> web.Response:
        """
        Handle a generic HTTP request and return a JSON response that echoes all request details.

        Captures and logs comprehensive request information including method, URL, endpoint, query parameters, headers, and payload (both JSON and raw formats). Designed for testing and debugging network interactions.

        Returns a standardized JSON response containing the full request details and a success message.
        """
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
        completion_request: CreateChatCompletionRequest,
        content: str,
    ) -> web.StreamResponse:
        """
        Handle a streaming chat completion response using Server-Sent Events (SSE).

        Streams the response word by word, simulating an OpenAI-compatible chat completion endpoint. Creates an SSE-formatted stream with individual word chunks and a final completion marker.

        Args:
            id (str): Unique identifier for the streaming response.
            request (web.Request): The original web request.
            completion_request (CreateChatCompletionRequest): The parsed chat completion request.
            content (str): The content to be streamed.

        Returns:
            web.StreamResponse: A streaming HTTP response with chunked SSE data.

        Raises:
            Any underlying exceptions that occur during streaming, which will be logged.
        """
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
            for i, word in enumerate(words):
                # Add space before word (except first)
                chunk_content = f" {word}" if i > 0 else word

                chunk_data = {
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": str(completion_request.model.root),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_content},
                            "finish_reason": None,
                        }
                    ],
                }

                await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode())

            # Send final chunk with finish_reason
            final_chunk = {
                "id": id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": str(completion_request.model.root),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
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

        # Get request body
        try:
            if request.content_type == "application/json":
                json_payload = await request.json()
            else:
                raw_payload = await request.text()
                json_payload = json.loads(raw_payload)
            completion_request = CreateChatCompletionRequest(**json_payload)
            if completion_request.messages and len(completion_request.messages) > 0:
                for message in completion_request.messages:
                    if str(message.root.role.value) == "user":
                        raw_request = message.root.content
                        break
            else:
                raise ValueError("Request must contain at least one message")
            id = json_payload.get("id", str(uuid.uuid4()))
            raw_response = self.get_response(raw_request)
            self.logger.debug(
                f"Content of request: {raw_request} - response : {raw_response}"
            )
            if self.max_osl is not None and len(raw_response) > 0:
                # if max_osl is specified, it can be either larger or smaller than the length of the prompt
                # if max_osl is larger, we can repeate the prompt until we reach the max_osl
                if len(raw_response) > self.max_osl:
                    raw_response = raw_response[: self.max_osl]
                # if max_osl is smaller, we can truncate the prompt
                else:
                    raw_response = raw_response * (
                        self.max_osl // len(raw_response) + 1
                    )
                    raw_response = raw_response[: self.max_osl]

            # Check if this is a streaming request
            self.logger.debug(f"Streaming response: {completion_request.stream}\n")
            if completion_request.stream is True:
                # Return SSE (Server-Sent Events) format for streaming
                return await self._handle_streaming_response(
                    id, request, completion_request, raw_response
                )
            else:
                # Non-streaming: return QueryResult as before
                response = QueryResult(
                    id=id,
                    response_output=raw_response,
                )
                echo_response = OpenAIAdapter.to_openai_response(response).model_dump(
                    mode="json"
                )
                echo_response["id"] = id
                self.logger.debug(f"Echo response (non-streaming): {echo_response}")
                return web.json_response(echo_response, status=200)

        except Exception as e:
            # A catch-all exception handler to help debug the issue without bringing down the server
            self.logger.error(f"Error handling chat completions request: {e}")
            return web.json_response(
                {"error": f"error encountered : {str(e)}"},
                status=400,
            )

    def _run_server(self):
        """Run the server in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())

    async def _start_server(self):
        """Start the HTTP server."""
        try:
            # Create the web application
            self.app = web.Application()

            self.app.router.add_post(
                "/v1/chat/completions", self._handle_echo_chat_completions_request
            )
            self.app.router.add_post("/echo", self._handle_echo_request)

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
    from inference_endpoint.utils.logging import setup_logging

    setup_logging()
    parser = create_parser()
    args = parser.parse_args()

    server = None
    try:
        server = EchoServer(host=args.host, port=args.port)
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
