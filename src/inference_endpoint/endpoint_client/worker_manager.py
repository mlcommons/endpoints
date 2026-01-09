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

from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
)
from inference_endpoint.endpoint_client.transport import WorkerPoolTransport
from inference_endpoint.endpoint_client.worker import worker_main
from inference_endpoint.utils.cpu_affinity import (
    AVAILABLE_CPUS,
    get_cpus_sorted_by_numa_preference,
    set_cpu_affinity,
)

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
        aiohttp_config: AioHttpConfig,
        loop: asyncio.AbstractEventLoop,
    ):
        """Initialize worker manager.

        Args:
            http_config: HTTP client configuration.
            aiohttp_config: aiohttp session configuration.
            loop: Event loop for transport registration.
        """
        self.http_config = http_config
        self.aiohttp_config = aiohttp_config

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
                self.aiohttp_config,
            ),
            daemon=True,
        )
        process.start()
        return process

    def _pin_workers(self) -> None:
        """
        Pin workers to CPU cores based on config:
         - "auto": distribute workers across available CPUs
         - list[int]: pin workers to specified cores (round-robin)
         - None or falsy: disable CPU affinity override
        """
        if not self.http_config.cpu_affinity:
            return

        match self.http_config.cpu_affinity:
            case "auto":
                cpu_list = get_cpus_sorted_by_numa_preference()
            case list():
                cpu_list = sorted(set(self.http_config.cpu_affinity) & AVAILABLE_CPUS)
            case _:
                return

        # assign CPU affinity round-robin among available CPUs
        if not cpu_list:
            logger.warning("No available CPUs for worker pinning")
            return

        for worker_id, pid in self.worker_pids.items():
            cpus = {cpu_list[worker_id % len(cpu_list)]}
            set_cpu_affinity(pid=pid, cpus=cpus)

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
