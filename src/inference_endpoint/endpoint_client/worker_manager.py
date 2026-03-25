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
import time
from multiprocessing import Process

from inference_endpoint.async_utils.transport import WorkerPoolTransport
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.cpu_affinity import set_cpu_affinity
from inference_endpoint.endpoint_client.worker import worker_main

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages worker processes and IPC transports.

    Creates and owns:
    - WorkerPoolTransport for IPC (created via transport_config.transport_class)
    - Worker processes

    Transport context is managed internally by the transport implementation.
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

        # Transport creates its own context internally
        assert http_config.transport is not None
        transport_cls = http_config.transport.transport_class
        self.pool_transport: WorkerPoolTransport = transport_cls.create(
            loop, http_config.workers, config=http_config.transport
        )

        # Worker processes
        self.workers: list[Process] = []
        self.worker_pids: dict[int, int] = {}

    async def initialize(self) -> None:
        """Initialize transports and spawn workers."""
        initialization_succeeded = False
        try:
            logger.debug(f"Starting {self.http_config.workers} worker processes")

            # Spawn workers with connector
            connector = self.pool_transport.worker_connector
            for i in range(self.http_config.workers):
                process = self._spawn_worker(i, connector)
                self.workers.append(process)
                assert (
                    process.pid is not None
                ), "Worker process should have a PID after spawning"
                self.worker_pids[i] = process.pid

            # Apply CPU affinity after all workers are started
            self._pin_workers()

            # Wait for workers with periodic liveness checks
            await self._wait_for_workers_with_liveness_check()

            logger.debug(f"All {self.http_config.workers} workers ready")
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

    async def _wait_for_workers_with_liveness_check(self) -> None:
        """Wait for workers, checking liveness at 10% intervals."""
        timeout = self.http_config.worker_initialization_timeout
        check_interval = timeout * 0.10 if timeout else 1.0
        start = time.monotonic()

        while True:
            # Check for dead workers
            dead = [w for w in self.workers if not w.is_alive()]
            if dead:
                raise RuntimeError(
                    f"Worker(s) died during init: PIDs {[w.pid for w in dead]}"
                )

            # Check remaining time
            elapsed = time.monotonic() - start
            remaining = timeout - elapsed if timeout else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Workers failed to initialize")

            # Try to wait with short timeout (25% of total, or remaining time)
            try:
                wait_time = (
                    min(check_interval, remaining) if remaining else check_interval
                )
                await self.pool_transport.wait_for_workers_ready(timeout=wait_time)
                return  # All ready
            except TimeoutError:
                continue  # Loop to check liveness again

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
