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

"""Generic ZMQ-based ready-check for subprocess startup synchronization.

Uses a single PULL socket (host) with many PUSH sockets (subprocesses) for
fan-in readiness signaling. All sockets share the same IPC socket directory.

See docs/async_utils/transport/zmq/ready_check_design.md for design rationale.
"""

from __future__ import annotations

import asyncio
import logging
import time

import msgspec
import zmq

from .context import ManagedZMQContext

logger = logging.getLogger(__name__)

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(int)

_LINGER_MS = 5000  # Bounded linger to avoid hanging if receiver is gone


async def send_ready_signal(
    zmq_context: ManagedZMQContext,
    path: str,
    identity: int,
) -> None:
    """Send a single ready signal over a PUSH socket.

    Opens a PUSH socket on the given context, sends the identity, and closes.
    The subprocess's existing ZMQ context is reused — no new context created.

    Args:
        zmq_context: The subprocess's existing ManagedZMQContext.
        path: IPC socket path (relative to zmq_context.socket_dir).
        identity: Integer identity to send (e.g., worker_id or service_id).
    """
    sock = zmq_context.async_socket(zmq.PUSH)
    sock.setsockopt(zmq.LINGER, _LINGER_MS)
    zmq_context.connect(sock, path)
    await sock.send(_encoder.encode(identity))
    sock.close()
    logger.debug("Ready signal sent (identity=%d)", identity)


class ReadyCheckReceiver:
    """Host side: bind a single PULL socket, await N ready signals.

    Multiple subprocesses connect PUSH sockets to this single PULL socket
    (ZMQ fan-in). After all signals are received, the socket is closed.
    """

    def __init__(
        self,
        path: str,
        zmq_context: ManagedZMQContext,
        count: int,
    ) -> None:
        self._count = count
        self._path = path

        # Bind PULL socket for receiving ready signals
        self._sock = zmq_context.async_socket(zmq.PULL)
        zmq_context.bind(self._sock, path)

    async def wait(self, timeout: float | None = None) -> list[int]:
        """Block until ``count`` ready signals are received.

        Uses a total deadline (not per-message timeout).

        Args:
            timeout: Maximum total seconds to wait. None means wait indefinitely.

        Returns:
            List of identities received (in arrival order).

        Raises:
            TimeoutError: If not all signals arrive within timeout.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        identities: list[int] = []

        try:
            while len(identities) < self._count:
                remaining = None
                if deadline is not None:
                    remaining = max(0, deadline - time.monotonic())

                try:
                    if remaining is None:
                        raw = await self._sock.recv()
                    else:
                        raw = await asyncio.wait_for(
                            self._sock.recv(), timeout=remaining
                        )
                except TimeoutError:
                    raise TimeoutError(
                        f"Ready check failed: {len(identities)}/{self._count} "
                        f"signals received within {timeout}s"
                    ) from None

                identity = _decoder.decode(raw)
                identities.append(identity)
                logger.debug(
                    "Ready signal received (identity=%d, %d/%d)",
                    identity,
                    len(identities),
                    self._count,
                )
        except BaseException:
            # Clean up socket on any failure (timeout, cancellation, etc.)
            self.close()
            raise

        logger.debug("All %d ready signals received", self._count)
        self.close()
        return identities

    def close(self) -> None:
        """Close the PULL socket. Idempotent."""
        if self._sock is not None and not self._sock.closed:
            self._sock.close()
