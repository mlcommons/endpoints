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

"""Worker process manager for HTTP endpoint client."""

import asyncio
import logging
from multiprocessing import Process

import zmq.asyncio

from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import worker_main
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages the lifecycle of worker processes."""

    def __init__(
        self,
        http_config: HTTPClientConfig,
        aiohttp_config: AioHttpConfig,
        zmq_config: ZMQConfig,
        zmq_context: zmq.asyncio.Context,
    ):
        """Initialize worker manager."""
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config
        self.zmq_config = zmq_config
        self.zmq_context = zmq_context
        self.workers: list[Process] = []
        self.worker_pids: dict[int, int] = {}  # worker_id -> pid
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize workers and ZMQ infrastructure."""
        readiness_socket = ZMQPullSocket(
            self.zmq_context,
            self.zmq_config.zmq_readiness_queue_addr,
            self.zmq_config,
            bind=True,
        )

        initialization_succeeded = False
        try:
            logger.debug(f"Starting {self.http_config.num_workers} worker processes")

            # Spawn worker processes
            for i in range(self.http_config.num_workers):
                worker = self._spawn_worker(i)
                self.workers.append(worker)
                self.worker_pids[i] = worker.pid

            # Wait for all workers to signal readiness
            ready_count = 0
            while ready_count < self.http_config.num_workers:
                worker_id = await asyncio.wait_for(
                    readiness_socket.receive(),
                    timeout=self.http_config.worker_initialization_timeout,
                )
                if worker_id is not None:
                    ready_count += 1
                    logger.debug(
                        f"Worker {worker_id} ready ({ready_count}/{self.http_config.num_workers})"
                    )

            logger.debug(f"{ready_count}/{self.http_config.num_workers} workers ready")
            initialization_succeeded = True

        except TimeoutError as e:
            raise TimeoutError(
                f"Workers failed to initialize within "
                f"{self.http_config.worker_initialization_timeout} seconds."
            ) from e
        finally:
            readiness_socket.close()
            if not initialization_succeeded and self.workers:
                await self.shutdown()

    def _spawn_worker(self, worker_id: int) -> Process:
        """Spawn a single worker process."""
        process = Process(
            target=worker_main,
            args=(
                worker_id,
                self.http_config,
                self.aiohttp_config,
                self.zmq_config,
            ),
            daemon=True,
        )
        process.start()
        return process

    async def shutdown(self) -> None:
        """Graceful shutdown of all workers."""
        self._shutdown_event.set()

        # Send SIGTERM for graceful shutdown
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()

        await asyncio.sleep(self.http_config.worker_graceful_shutdown_wait)

        # Force kill remaining workers
        for worker in self.workers:
            if worker.is_alive():
                worker.kill()

        # Join all workers
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    worker.join, timeout=self.http_config.worker_force_kill_timeout
                )
                for worker in self.workers
            )
        )
