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

"""Tests for ZmqWorkerPoolTransport and ReadyCheckReceiver.

Includes regression test for the 'Socket operation on non-socket' bug where
ReadyCheckReceiver.wait() closed its socket on TimeoutError, breaking the
retry loop in WorkerManager._wait_for_workers_with_liveness_check().
"""

import asyncio
import uuid

import pytest
import zmq
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqEventRecordPublisher,
)
from inference_endpoint.async_utils.transport.zmq.ready_check import (
    ReadyCheckReceiver,
)
from inference_endpoint.async_utils.transport.zmq.transport import (
    ZMQTransportConfig,
    ZmqWorkerPoolTransport,
)


@pytest.fixture(autouse=True)
def reset_zmq_singleton():
    """Ensure each test gets a fresh ManagedZMQContext singleton."""
    yield
    instance = ManagedZMQContext._instance
    if instance is not None and getattr(instance, "_initialized", False):
        instance.cleanup()
    ManagedZMQContext._instance = None


@pytest.mark.unit
@pytest.mark.asyncio
class TestReadyCheckReceiverTimeout:
    """Regression: ReadyCheckReceiver must survive timeout for retry."""

    async def test_socket_survives_timeout(self):
        """After wait() times out, the socket must still be usable for retry.

        This is the core regression test for the ENOTSOCK bug. The old code
        had `except BaseException: self.close()` which closed the socket on
        TimeoutError. The caller (_wait_for_workers_with_liveness_check)
        catches TimeoutError and retries, hitting a dead socket.
        """
        zmq_ctx = ManagedZMQContext(io_threads=1)
        dummy = zmq_ctx.socket(zmq.PUB)
        zmq_ctx.bind(dummy, "dummy_pub")

        receiver = ReadyCheckReceiver("ready_test", zmq_ctx, count=1)

        # First wait should timeout (no signals sent)
        with pytest.raises(TimeoutError):
            await receiver.wait(timeout=0.05)

        # Socket must still be usable after timeout
        assert not receiver._sock.closed, (
            "ReadyCheckReceiver closed its socket on TimeoutError — "
            "this breaks the retry loop in _wait_for_workers_with_liveness_check"
        )
        _ = receiver._sock.rcvtimeo  # Would raise ENOTSOCK if socket is dead

        # Second wait should also timeout cleanly (not ENOTSOCK)
        with pytest.raises(TimeoutError):
            await receiver.wait(timeout=0.05)

        receiver.close()
        dummy.close()
        zmq_ctx.cleanup()

    async def test_socket_closed_on_cancellation(self):
        """Socket SHOULD be closed on non-timeout exceptions (e.g. cancel)."""
        zmq_ctx = ManagedZMQContext(io_threads=1)
        dummy = zmq_ctx.socket(zmq.PUB)
        zmq_ctx.bind(dummy, "dummy_pub")

        receiver = ReadyCheckReceiver("ready_test", zmq_ctx, count=1)

        task = asyncio.create_task(receiver.wait(timeout=10.0))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert receiver._sock.closed

        dummy.close()
        zmq_ctx.cleanup()


@pytest.mark.unit
@pytest.mark.asyncio
class TestZmqPoolTransportWithPublisher:
    """Test pool transport creation with a publisher on the same context."""

    async def _create_publisher_and_pool(
        self, loop: asyncio.AbstractEventLoop, num_workers: int
    ):
        """Helper: create publisher + pool transport, test ready check socket."""
        sid = uuid.uuid4().hex[:8]
        zmq_ctx = ManagedZMQContext(io_threads=2)
        publisher = ZmqEventRecordPublisher(f"ev_pub_{sid}", zmq_ctx, loop=loop)

        pool = ZmqWorkerPoolTransport.create(
            loop, num_workers, config=ZMQTransportConfig()
        )

        rc = pool._ready_check
        assert not rc._sock.closed
        _ = rc._sock.rcvtimeo

        with pytest.raises(TimeoutError):
            await pool.wait_for_workers_ready(timeout=0.1)

        pool.cleanup()
        publisher.close()
        zmq_ctx.cleanup()

    async def test_2_workers(self):
        loop = asyncio.get_running_loop()
        await self._create_publisher_and_pool(loop, 2)

    async def test_3_workers(self):
        loop = asyncio.get_running_loop()
        await self._create_publisher_and_pool(loop, 3)

    async def test_4_workers(self):
        loop = asyncio.get_running_loop()
        await self._create_publisher_and_pool(loop, 4)

    async def test_8_workers(self):
        loop = asyncio.get_running_loop()
        await self._create_publisher_and_pool(loop, 8)


@pytest.mark.unit
@pytest.mark.asyncio
class TestZmqPoolTransportWithoutPublisher:
    """Test pool transport creation without a publisher (baseline)."""

    @pytest.mark.parametrize("num_workers", [2, 3, 4, 8])
    async def test_pool_only(self, num_workers: int):
        loop = asyncio.get_running_loop()
        zmq_ctx = ManagedZMQContext(io_threads=2)
        dummy = zmq_ctx.socket(zmq.PUB)
        zmq_ctx.bind(dummy, "dummy")

        pool = ZmqWorkerPoolTransport.create(
            loop, num_workers, config=ZMQTransportConfig()
        )

        rc = pool._ready_check
        assert not rc._sock.closed
        _ = rc._sock.rcvtimeo

        with pytest.raises(TimeoutError):
            await pool.wait_for_workers_ready(timeout=0.1)

        pool.cleanup()
        dummy.close()
        zmq_ctx.cleanup()
