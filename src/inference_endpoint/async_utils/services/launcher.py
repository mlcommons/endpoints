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
import time
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
        await launcher.launch([
            ServiceConfig(
                module="inference_endpoint.async_utils.services.event_logger",
                args=["--log-dir", "/tmp/logs", "--socket-dir", socket_dir,
                      "--socket-name", "events"],
            ),
        ], timeout=30.0)

        # ... run benchmark ...

        launcher.wait_for_exit(timeout=60.0)
    """

    def __init__(self, zmq_context: ManagedZMQContext) -> None:
        self._zmq_ctx = zmq_context
        self._procs: list[subprocess.Popen] = []

    @property
    def procs(self) -> list[subprocess.Popen]:
        return self._procs

    async def launch(
        self,
        services: list[ServiceConfig],
        timeout: float | None = 30.0,
    ) -> None:
        """Spawn service subprocesses and wait for all to signal readiness.

        Each service receives ``--readiness-path`` and ``--readiness-id`` CLI
        arguments. After initialization, the service sends a ready signal via
        ``send_ready_signal()`` using the same socket_dir as the launcher.

        Launched processes are stored in ``self.procs`` for later use by
        ``wait_for_exit()`` and ``kill_all()``.

        Args:
            services: List of ServiceConfig describing each service to launch.
            timeout: Maximum total seconds to wait for all services to become ready.

        Raises:
            TimeoutError: If services don't signal readiness within timeout.
        """
        if not services:
            return

        readiness_path = f"svc_ready_{uuid.uuid4().hex[:8]}"
        receiver = ReadyCheckReceiver(
            readiness_path, self._zmq_ctx, count=len(services)
        )

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
                self._procs.append(proc)

            await receiver.wait(timeout=timeout)
            logger.info("All %d services ready", len(services))

        except Exception as e:
            # Collect all crashed subprocesses for a complete error message
            crashed = [
                (proc.pid, exit_code)
                for proc in self._procs
                if (exit_code := proc.poll()) is not None and exit_code != 0
            ]

            self.kill_all()
            receiver.close()

            if crashed:
                details = ", ".join(
                    f"pid={pid} exit={exit_code}" for pid, exit_code in crashed
                )
                raise RuntimeError(
                    f"{len(crashed)} service(s) crashed during startup: {details}"
                ) from e

            # If for some reason the reason for the exception is not a crashed subprocess,
            # re-raise the exception.
            raise

    def kill_all(self) -> None:
        """Kill all managed subprocesses."""
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()

    def wait_for_exit(self, timeout: float | None = 60.0) -> None:
        """Wait for all service subprocesses to exit.

        Services self-terminate on SessionEventType.ENDED. This method
        blocks until all have exited or the total timeout is reached.

        Args:
            timeout: Maximum total seconds to wait across all processes.
                If None, waits indefinitely.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        for proc in self._procs:
            remaining = (
                None if deadline is None else max(0, deadline - time.monotonic())
            )
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Service pid=%d did not exit within timeout, killing", proc.pid
                )
                proc.kill()
                proc.wait(timeout=5)
