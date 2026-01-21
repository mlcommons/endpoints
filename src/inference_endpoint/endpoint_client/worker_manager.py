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

"""Worker manager for spawning and managing worker processes."""

import asyncio
import logging
from multiprocessing import Process

from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.cpu_affinity import set_cpu_affinity
from inference_endpoint.endpoint_client.transport import WorkerPoolTransport
from inference_endpoint.endpoint_client.worker import worker_main

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages worker processes and IPC transports.

    Creates and owns:
    - WorkerPoolTransport for IPC
    - Worker processes
    """

    def __init__(
        self,
        http_config: HTTPClientConfig,
        loop: asyncio.AbstractEventLoop,
    ):
        """Initialize worker manager.

        Args:
            http_config: HTTP client configuration.
            loop: Event loop for transport registration.
        """
        self.http_config = http_config

        # Create pool transport via factory
        self.pool_transport: WorkerPoolTransport = (
            http_config.worker_pool_transport.create(
                loop, num_workers=http_config.num_workers
            )
        )

        # Worker processes
        self.workers: list[Process] = []
        self.worker_pids: dict[int, int] = {}

    async def initialize(self) -> None:
        """Initialize transports and spawn workers."""
        initialization_succeeded = False
        try:
            logger.debug(f"Starting {self.http_config.num_workers} worker processes")

            # Spawn workers with connector
            connector = self.pool_transport.worker_connector
            for i in range(self.http_config.num_workers):
                process = self._spawn_worker(i, connector)
                self.workers.append(process)
                self.worker_pids[i] = process.pid

            # Apply CPU affinity after all workers are started
            self._pin_workers()

            # Wait for all workers to signal readiness
            await self.pool_transport.wait_for_workers_ready(
                timeout=self.http_config.worker_initialization_timeout
            )

            logger.debug(f"All {self.http_config.num_workers} workers ready")
            initialization_succeeded = True

        except TimeoutError as e:
            raise TimeoutError(
                f"Workers failed to initialize within {self.http_config.worker_initialization_timeout}s"
            ) from e

        finally:
            if not initialization_succeeded and self.workers:
                await self.shutdown()

    def _spawn_worker(self, worker_id: int, connector) -> Process:
        """Spawn a worker process."""
        process = Process(
            target=worker_main,
            args=(
                worker_id,
                connector,
                self.http_config,
            ),
            daemon=True,
        )
        process.start()
        return process

    def _pin_workers(self) -> None:
        """Pin workers using the AffinityPlan from config.

        Each worker gets all hyperthreads of its assigned physical core.
        """
        plan = self.http_config.cpu_affinity
        if plan is None or not plan.worker_cpu_sets:
            return

        for worker_id, pid in self.worker_pids.items():
            cpus = plan.get_worker_cpus(worker_id)
            if cpus:
                set_cpu_affinity(pid=pid, cpus=set(cpus))
                logger.debug(f"Worker {worker_id} (pid {pid}) pinned to CPUs {cpus}")

    async def shutdown(self) -> None:
        """Shutdown workers and transports."""
        # Terminate workers
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()

        await asyncio.sleep(self.http_config.worker_graceful_shutdown_wait)

        # Force kill remaining
        for worker in self.workers:
            if worker.is_alive():
                worker.kill()

        # Join all
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    worker.join, timeout=self.http_config.worker_force_kill_timeout
                )
                for worker in self.workers
            )
        )

        # Cleanup pool transport
        self.pool_transport.cleanup()
