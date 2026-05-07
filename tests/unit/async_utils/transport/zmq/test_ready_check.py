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

"""Tests for the generic ReadyCheck mechanism."""

import asyncio
import multiprocessing
import tempfile

import pytest
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.ready_check import (
    ReadyCheckReceiver,
    send_ready_signal,
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestReadyCheck:
    async def test_single_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with ManagedZMQContext.scoped(socket_dir=tmpdir) as ctx:
                receiver = ReadyCheckReceiver("ready_test", ctx, count=1)

                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.ensure_future(
                        send_ready_signal(ctx, "ready_test", 42)
                    )
                )

                identities = await receiver.wait(timeout=5.0)
                assert identities == [42]

    async def test_multiple_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with ManagedZMQContext.scoped(socket_dir=tmpdir) as ctx:
                receiver = ReadyCheckReceiver("ready_multi", ctx, count=3)

                async def send_all():
                    for i in range(3):
                        await send_ready_signal(ctx, "ready_multi", i)

                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.ensure_future(send_all())
                )

                identities = await receiver.wait(timeout=5.0)
                assert len(identities) == 3
                assert set(identities) == {0, 1, 2}

    async def test_timeout_is_total_deadline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with ManagedZMQContext.scoped(socket_dir=tmpdir) as ctx:
                receiver = ReadyCheckReceiver("ready_timeout", ctx, count=2)

                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.ensure_future(
                        send_ready_signal(ctx, "ready_timeout", 0)
                    )
                )

                with pytest.raises(TimeoutError, match="1/2"):
                    await receiver.wait(timeout=0.5)

    async def test_close_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with ManagedZMQContext.scoped(socket_dir=tmpdir) as ctx:
                receiver = ReadyCheckReceiver("ready_close", ctx, count=1)
                receiver.close()
                receiver.close()

    async def test_socket_survives_timeout(self):
        """Socket must NOT be closed on timeout — caller may retry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with ManagedZMQContext.scoped(socket_dir=tmpdir) as ctx:
                receiver = ReadyCheckReceiver("ready_close_timeout", ctx, count=1)
                with pytest.raises(TimeoutError):
                    await receiver.wait(timeout=0.1)
                assert not receiver._sock.closed
                receiver.close()


def _child_send_ready(socket_dir: str, path: str, identity: int) -> None:
    import uvloop

    async def _send():
        with ManagedZMQContext.scoped(socket_dir=socket_dir) as ctx:
            await send_ready_signal(ctx, path, identity)

    uvloop.run(_send())


@pytest.mark.unit
@pytest.mark.asyncio
class TestReadyCheckCrossProcess:
    async def test_cross_process_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with ManagedZMQContext.scoped(socket_dir=tmpdir) as ctx:
                receiver = ReadyCheckReceiver("ready_xproc", ctx, count=1)

                proc = multiprocessing.Process(
                    target=_child_send_ready,
                    args=(tmpdir, "ready_xproc", 99),
                )
                proc.start()

                identities = await receiver.wait(timeout=10.0)
                assert identities == [99]

                proc.join(timeout=5)

    async def test_multiple_child_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 3
            with ManagedZMQContext.scoped(socket_dir=tmpdir) as ctx:
                receiver = ReadyCheckReceiver("ready_multi_xproc", ctx, count=n)

                procs = []
                for i in range(n):
                    p = multiprocessing.Process(
                        target=_child_send_ready,
                        args=(tmpdir, "ready_multi_xproc", i),
                    )
                    p.start()
                    procs.append(p)

                identities = await receiver.wait(timeout=10.0)
                assert len(identities) == n
                assert set(identities) == set(range(n))

                for p in procs:
                    p.join(timeout=5)
