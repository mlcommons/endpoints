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

"""Service subprocess launcher with ready-check synchronization.

Launches service subprocesses (EventLoggerService, MetricsAggregatorService)
via ``python -m`` and waits for each to signal readiness over ZMQ before
returning. Uses the same ReadyCheckReceiver/send_ready_signal primitives as
the worker pool transport.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from dataclasses import dataclass, field

from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.ready_check import ReadyCheckReceiver

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a service subprocess to launch."""

    module: str
    """Python module path (e.g., 'inference_endpoint.async_utils.services.event_logger')."""

    args: list[str] = field(default_factory=list)
    """Additional CLI arguments for the service."""


class ServiceLauncher:
    """Launches service subprocesses and waits for ready signals.

    Usage::

        launcher = ServiceLauncher(zmq_context)
        procs = await launcher.launch([
            ServiceConfig(
                module="inference_endpoint.async_utils.services.event_logger",
                args=["--log-dir", "/tmp/logs", "--socket-dir", socket_dir,
                      "--socket-name", "events"],
            ),
        ], timeout=30.0)

        # ... run benchmark ...

        ServiceLauncher.wait_for_exit(procs, timeout=60.0)
    """

    def __init__(self, zmq_context: ManagedZMQContext) -> None:
        self._zmq_ctx = zmq_context

    async def launch(
        self,
        services: list[ServiceConfig],
        timeout: float | None = 30.0,
    ) -> list[subprocess.Popen]:
        """Spawn service subprocesses and wait for all to signal readiness.

        Each service receives ``--readiness-path`` and ``--readiness-id`` CLI
        arguments. After initialization, the service sends a ready signal via
        ``send_ready_signal()`` using the same socket_dir as the launcher.

        Args:
            services: List of ServiceConfig describing each service to launch.
            timeout: Maximum total seconds to wait for all services to become ready.

        Returns:
            List of Popen handles (one per service).

        Raises:
            TimeoutError: If services don't signal readiness within timeout.
        """
        if not services:
            return []

        readiness_path = f"svc_ready_{uuid.uuid4().hex[:8]}"
        receiver = ReadyCheckReceiver(
            readiness_path, self._zmq_ctx, count=len(services)
        )

        procs: list[subprocess.Popen] = []
        try:
            for i, svc in enumerate(services):
                cmd = [
                    sys.executable,
                    "-m",
                    svc.module,
                    *svc.args,
                    "--readiness-path",
                    readiness_path,
                    "--readiness-id",
                    str(i),
                ]
                logger.info("Launching service: %s (id=%d)", svc.module, i)
                proc = subprocess.Popen(cmd)
                procs.append(proc)

            await receiver.wait(timeout=timeout)
            logger.info("All %d services ready", len(services))

        except (TimeoutError, Exception) as e:
            # Check if any subprocess crashed (provides a better error message)
            for proc in procs:
                rc = proc.poll()
                if rc is not None and rc != 0:
                    raise RuntimeError(
                        f"Service pid={proc.pid} exited with code {rc} "
                        f"during startup"
                    ) from e
            for proc in procs:
                proc.kill()
            receiver.close()
            raise

        return procs

    @staticmethod
    def wait_for_exit(
        procs: list[subprocess.Popen],
        timeout: float | None = 60.0,
    ) -> None:
        """Wait for all service subprocesses to exit.

        Services self-terminate on SessionEventType.ENDED. This method
        blocks until all have exited or timeout is reached.

        Args:
            procs: List of subprocess handles from launch().
            timeout: Maximum seconds to wait per process.
        """
        for proc in procs:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Service pid=%d did not exit within timeout, killing", proc.pid
                )
                proc.kill()
                proc.wait(timeout=5)
