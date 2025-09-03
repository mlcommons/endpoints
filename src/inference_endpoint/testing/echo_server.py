import argparse
import asyncio
import json
import logging
import threading
import time

from aiohttp import web

from inference_endpoint.core.types import ChatCompletionQuery, QueryResult


class EchoServer:
    def __init__(self, host: str = "localhost", port: int = 12345):
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}"
        self.app = None
        self.runner = None
        self.site = None
        self._server_thread = None
        self._loop = None
        self._shutdown_event = threading.Event()
        self.logger = logging.getLogger(__name__)

    async def _handle_echo_request(self, request: web.Request) -> web.Response:
        """Handle incoming HTTP requests and echo back the payload."""
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

    async def _handle_echo_chat_completions_request(
        self, request: web.Request
    ) -> web.Response:
        """Handle incoming HTTP OpenAI chat completions requests and echo back a QueryResult payload."""
        # Extract request data
        endpoint = request.path
        query_params = dict(request.query)
        headers = dict(request.headers)

        # Get request body
        try:
            if request.content_type == "application/json":
                json_payload = await request.json()
            else:
                raw_payload = await request.text()
                json_payload = json.loads(raw_payload)
            completion_request = ChatCompletionQuery.from_json(json_payload)
            response = QueryResult(
                query_id=completion_request.id,
                response_output=completion_request.prompt,
            )
        except Exception as e:
            # A catch-all exception handler to help debug the issue without bringing down the server
            return web.json_response(
                {"error": f"error encountered : {str(e)}"},
                status=400,
            )

        request_data = {
            "method": request.method,
            "url": str(request.url),
            "endpoint": endpoint,
            "query_params": query_params,
            "headers": headers,
            "json_payload": response.to_json(),
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

    def _run_server(self):
        """Run the server in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._start_server())
        except Exception as e:
            print(f"Server error: {e}")

    async def _start_server(self):
        """Start the HTTP server."""
        # Create the web application
        self.app = web.Application()

        self.app.router.add_post(
            "/v1/chat/completions", self._handle_echo_chat_completions_request
        )
        self.app.router.add_post("/v1/completions", self._handle_echo_request)

        # Start the server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        self.logger.info(
            f"==========================\nServer started at {self.url}\n==========================",
        )

        # Wait for shutdown signal
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

        # Clean up
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

        # Delay for the server to start before returning
        time.sleep(0.5)

    def stop(self):
        """Stop the HTTP Echo server."""
        self.logger.info("Stopping HTTP Echo server...")
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._server_thread:
            self._server_thread.join(timeout=2)
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
        "--host", type=str, help="hostname/address to bind to", default="localhost"
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
        print("Server is running. Press Ctrl+C to stop...")
        server._server_thread.join()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping server...")
        if server:
            server.stop()
    except Exception as e:
        if server:
            server.logger.error(f"Error starting server: {e}")
        else:
            print(f"Error starting server: {e}")


if __name__ == "__main__":
    main()
