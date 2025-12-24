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

"""HTTP Echo Server for testing inference endpoint clients.

High-performance implementation using Granian (Rust-based ASGI server) running
in a separate process for maximum throughput.
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from abc import abstractmethod
from typing import Any

import orjson


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


def _find_free_port() -> int:
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for server to be ready by polling the port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(0.05)
    return False


class EchoServer(HTTPServer):
    """High-performance HTTP echo server using Granian.

    Spawns Granian (Rust-based ASGI server) in a separate subprocess for maximum
    throughput. Provides OpenAI-compatible chat completions endpoint with
    streaming support.

    Example:
        server = EchoServer(port=0)  # Auto-assign port
        server.start()
        print(f"Server at {server.url}")
        # ... run tests ...
        server.stop()
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_osl: int | None = None,
        workers: int | None = None,
    ):
        self.host = host
        self.port = port
        self.max_osl = max_osl
        # Default to 4 workers - sweet spot for echo server
        # More workers = more IPC overhead for simple I/O-bound tasks
        self.workers = workers if workers is not None else 8
        self._actual_port: int | None = None

        self._process: subprocess.Popen | None = None
        self.logger = logging.getLogger(__name__)

        # For subclass support - can be overridden
        self._response_func_module: str | None = None
        self._response_func_data: Any = None

    @property
    def url(self):
        """Get the server URL with the actual port."""
        port = self._actual_port or self.port
        return f"http://{self.host}:{port}"

    def set_max_osl(self, max_osl: int):
        """Set maximum output sequence length."""
        self.max_osl = max_osl
        # Note: Changes don't affect running server - restart required

    def get_max_osl(self) -> int | None:
        """Get current maximum output sequence length setting."""
        return self.max_osl

    def get_response(self, request: str) -> str:
        """Return the input request string as the response.

        This method serves as a simple echo mechanism. Override in subclasses
        to provide custom response generation logic.

        Args:
            request: The input request string to be echoed back.

        Returns:
            The input request string passed through unmodified.
        """
        return request

    def start(self):
        """Start the server in a background subprocess."""
        self.logger.info("Starting Granian HTTP Echo server...")

        # Find a free port if port=0
        if self.port == 0:
            self._actual_port = _find_free_port()
        else:
            self._actual_port = self.port

        # Set up environment variables for the ASGI app
        env = os.environ.copy()
        env["_ECHO_SERVER_MAX_OSL"] = str(self.max_osl) if self.max_osl else ""
        if self._response_func_module:
            env["_ECHO_SERVER_RESPONSE_MODULE"] = self._response_func_module
        if self._response_func_data is not None:
            env["_ECHO_SERVER_RESPONSE_DATA"] = orjson.dumps(
                self._response_func_data
            ).decode()

        # Build granian command
        cmd = [
            sys.executable,
            "-m",
            "granian",
            "inference_endpoint.testing._echo_asgi:create_app_factory",
            "--interface",
            "asgi",
            "--host",
            self.host,
            "--port",
            str(self._actual_port),
            "--workers",
            str(self.workers),
            "--backlog",
            "65535",
            "--factory",
            "--no-ws",
        ]

        # Start subprocess
        # Use DEVNULL to avoid pipe buffer deadlocks with many workers
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Create new process group for clean shutdown
        )

        # Wait for server to be ready
        if not _wait_for_server(self.host, self._actual_port, timeout=30.0):
            self.stop()
            raise RuntimeError(
                f"Server failed to start within timeout on {self.host}:{self._actual_port}"
            )

        self.logger.info(f"Server ready at {self.url}")

    def stop(self):
        """Stop the HTTP Echo server."""
        self.logger.info("Stopping Granian HTTP Echo server...")

        if self._process:
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                self._process.wait(timeout=1.0)

            self._process = None

        self.logger.info("Granian HTTP Echo server stopped")


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
    Run the echo server from command line.

    Example curl:
      curl http://localhost:12345/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
        "model": "gpt-4o", "id" : "123",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the capital of France?"}
        ]
      }'
    """
    from inference_endpoint.utils.logging import setup_logging

    setup_logging()
    parser = create_parser()
    args = parser.parse_args()

    server = None
    try:
        server = EchoServer(host=args.host, port=args.port)
        server.start()

        server.logger.info("Server is running. Press Ctrl+C to stop...")
        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        if server:
            server.logger.warning("\nKeyboard interrupt received, stopping server...")
            server.stop()
    except Exception as e:
        if server:
            server.logger.error(f"Error starting server: {e}")
        else:
            print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
