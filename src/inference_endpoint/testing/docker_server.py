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

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time

import requests


class DockerServer:
    """
    Class to manage the lifecycle of a server running in a Docker container.

    Example:
        docker_server = DockerServer(hf_model_name="meta-llama/Llama-3.1-8B-Instruct")
        docker_server.start()
        url = docker_server.url  # e.g., "http://localhost:8000"
        # ... run tests or inference requests ...
        docker_server.stop()
    """

    def __init__(
        self,
        hf_model_name: str,
        timeout_seconds: int = 10 * 60,
        user_cmd: str | None = None,
        port: int = 8000,
    ):
        self.logger = logging.getLogger(__name__)
        self.container_start_timeout_seconds = (
            60 * 30  # Server takes a long time to start, so we give it 30 minutes
        )
        self.hf_model_name = hf_model_name
        self.timeout_seconds = timeout_seconds
        self.user_cmd = user_cmd
        self.port = port
        self.hf_home = os.getenv("HF_HOME")
        self.hf_token = os.getenv("HF_TOKEN")
        self.container_id: str | None = None
        self._url = f"http://localhost:{port}"
        if not self.hf_home:
            raise OSError("HF_HOME environment variable is not set")
        if not self.hf_token:
            raise OSError("HF_TOKEN environment variable is not set")

    @property
    def url(self) -> str:
        return self._url

    def _get_docker_cmd(self, user_cmd: str | None = None) -> list:
        """
        Build the docker run command for server.
        If user_cmd is given, uses the user command to start the server.
        Otherwise, uses built-in default hardcoded command.
        """
        if user_cmd is None:
            return []

        # Compose docker run base parts
        cmd = ["docker", "run", "-d"]
        cmd.extend(shlex.split(user_cmd))

        return cmd

    def start(self, timeout_seconds: int = 10 * 60) -> None:
        """
        Start the Docker container and wait until the server becomes healthy.
        Raises RuntimeError on failure.
        """
        # breakpoint()
        docker_cmd = self._get_docker_cmd(self.user_cmd)
        self.logger.info(f"Starting docker container... {docker_cmd}")
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.container_start_timeout_seconds,
            )
            self.container_id = result.stdout.strip()
            self.logger.info(f"Container started with ID: {self.container_id}")

            # Wait for server to become healthy
            health_url = f"{self._url}/health"
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                self.logger.info("Checking for server health...")
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        self.logger.info("server is healthy and ready")
                        return
                except requests.exceptions.RequestException:
                    pass  # Server not ready yet

                time.sleep(5)

                # Check if container is still running
                check_cmd = [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    self.container_id,
                ]
                check_result = subprocess.run(
                    check_cmd, capture_output=True, text=True, timeout=10
                )
                if check_result.stdout.strip() != "true":
                    # Get container logs for debugging
                    logs_cmd = ["docker", "logs", self.container_id]
                    logs_result = subprocess.run(
                        logs_cmd, capture_output=True, text=True, timeout=10
                    )
                    self.logger.error(f"Container logs:\n{logs_result.stdout}")
                    raise RuntimeError(
                        f"Docker container stopped unexpectedly. Logs:\n{logs_result.stdout}"
                    )
            raise RuntimeError(
                f"server did not become healthy within {timeout_seconds} seconds"
            )
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Timeout while starting Docker container: {e}")
            self.stop()
            raise RuntimeError(f"Docker container startup timed out: {e}") from e
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker command failed: {e.stderr}")
            self.stop()
            raise RuntimeError(f"Failed to start Docker container: {e.stderr}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error with Docker container: {e}")
            self.stop()
            raise

    def stop(self) -> None:
        """
        Stop and remove the Docker container if running.
        """
        if self.container_id:
            self.logger.info(f"Stopping and removing container {self.container_id}...")
            try:
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                subprocess.run(
                    ["docker", "rm", "-f", self.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.logger.info("Docker container stopped and removed")
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    f"Timeout stopping/removing container {self.container_id}"
                )
            finally:
                self.container_id = None

    def __enter__(self):
        self.start(timeout_seconds=self.timeout_seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get_info(self) -> dict[str, str | None]:
        """
        Get server information.
        Returns:
            dict: { 'url': str, 'container_id': str }
        """
        return {"url": self.url, "container_id": self.container_id}


def main():
    """
    Ad-hoc manual test: Launch DockerServer with configurable parameters.
    Requires the following env vars to be set:
      HF_HOME: path to HuggingFace model cache
      HF_TOKEN: HuggingFace token

    Usage:
      python docker_server.py --model meta-llama/Llama-3.1-8B-Instruct --port 8000 --tp 1 --host 0.0.0.0
    """
    parser = argparse.ArgumentParser(
        description="Launch a Docker container running an inference server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to expose the server on (default: 8000)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for server startup (default: 600)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    hf_home = os.getenv("HF_HOME")
    hf_token = os.getenv("HF_TOKEN")

    sglang_user_cmd = (
        f"--gpus all --shm-size 32g --net host "
        f"-v {hf_home}:/root/.cache/huggingface "
        f"--env HF_TOKEN={hf_token} --ipc=host "
        f"lmsysorg/sglang:latest python3 -m sglang.launch_server "
        f"--model-path {args.model} --host {args.host} --port {args.port} --tp-size {args.tp}"
    )

    server = None
    try:
        server = DockerServer(
            hf_model_name=args.model,
            timeout_seconds=args.timeout,
            user_cmd=sglang_user_cmd,
            port=args.port,
        )
        logger.info("Starting DockerServer...")
        logger.info(f"Model: {args.model}")
        logger.info(f"Host: {args.host}, Port: {args.port}, TP: {args.tp}")
        server.start()
        logger.info(f"Server running at {server.url}")
        logger.info(f"Container ID: {server.container_id}")
        logger.info("Server is running. Press Ctrl+C to stop...")

        # Keep the server running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Failed to launch DockerServer: {e}")
        sys.exit(1)
    finally:
        if server:
            logger.info("Stopping DockerServer...")
            server.stop()


if __name__ == "__main__":
    main()
