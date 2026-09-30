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
import time

import msgspec
import pytest
import zmq
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.ready_check import (
    ReadyCheckReceiver,
    send_ready_signal,
)

from tests.ready_check_helpers import wait_until_ready_signal_queued


@pytest.mark.unit
@pytest.mark.asyncio
class TestReadyCheck:
    async def test_single_signal(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_test", zmq_ctx_scope, count=1)

        asyncio.get_running_loop().call_soon(
            lambda: asyncio.ensure_future(
                send_ready_signal(zmq_ctx_scope, "ready_test", 42)
            )
        )

        identities = await receiver.wait(timeout=5.0)
        assert identities == [42]

    async def test_multiple_signals(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_multi", zmq_ctx_scope, count=3)

        async def send_all():
            for i in range(3):
                await send_ready_signal(zmq_ctx_scope, "ready_multi", i)

        asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(send_all()))

        identities = await receiver.wait(timeout=5.0)
        assert len(identities) == 3
        assert set(identities) == {0, 1, 2}

    async def test_timeout_is_total_deadline(self, zmq_ctx_scope):
        """Signals that each arrive within the timeout still hit one deadline."""
        gap_s, timeout_s = 0.3, 0.5
        # Each gap is shorter than the timeout, so a per-message
        # timeout never expires, but the third signal is sent after
        # the overall deadline.
        assert gap_s < timeout_s < 2 * gap_s
        receiver = ReadyCheckReceiver("ready_timeout", zmq_ctx_scope, count=3)

        async def send_spaced():
            for identity in range(3):
                await send_ready_signal(zmq_ctx_scope, "ready_timeout", identity)
                # Also after the last signal: it lets that signal
                # arrive before the receiver closes, so context
                # teardown doesn't wait out the sender's linger.
                await asyncio.sleep(gap_s)

        sender = asyncio.ensure_future(send_spaced())
        try:
            with pytest.raises(TimeoutError, match=r"[0-2]/3"):
                await receiver.wait(timeout=timeout_s)
        finally:
            await sender
            receiver.close()

    async def test_close_idempotent(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_close", zmq_ctx_scope, count=1)
        receiver.close()
        receiver.close()

    async def test_socket_survives_timeout(self, zmq_ctx_scope):
        """Socket must NOT be closed on timeout — caller may retry."""
        receiver = ReadyCheckReceiver("ready_close_timeout", zmq_ctx_scope, count=1)
        with pytest.raises(TimeoutError):
            await receiver.wait(timeout=0.1)
        assert not receiver._sock.closed
        receiver.close()

    async def test_signals_survive_timeout(self, zmq_ctx_scope):
        """Signals received before a timeout count toward the retried wait."""
        receiver = ReadyCheckReceiver("ready_retry", zmq_ctx_scope, count=2)

        await send_ready_signal(zmq_ctx_scope, "ready_retry", 0)
        await wait_until_ready_signal_queued(receiver)
        with pytest.raises(TimeoutError, match="1/2"):
            await receiver.wait(timeout=0.5)

        await send_ready_signal(zmq_ctx_scope, "ready_retry", 1)
        identities = await receiver.wait(timeout=5.0)
        assert identities == [0, 1]

    async def test_signal_at_deadline_is_not_lost(self, zmq_ctx_scope):
        """A signal arriving in the same loop turn as the deadline still counts."""
        timeout_s, block_s = 0.05, 0.1
        # Blocking the loop past the deadline makes the socket read and
        # the timeout land in the same loop iteration.
        assert block_s > timeout_s
        receiver = ReadyCheckReceiver("ready_deadline", zmq_ctx_scope, count=1)
        push = zmq_ctx_scope.socket(zmq.PUSH)
        zmq_ctx_scope.connect(push, "ready_deadline")
        try:
            task = asyncio.ensure_future(receiver.wait(timeout=timeout_s))
            await asyncio.sleep(0.01)
            push.send(msgspec.msgpack.encode(7))
            time.sleep(block_s)
            try:
                identities = await task
            except TimeoutError:
                identities = await receiver.wait(timeout=1.0)
            assert identities == [7]
        finally:
            push.close(linger=0)
            receiver.close()

    async def test_poll_past_deadline_does_not_block(self, zmq_ctx_scope):
        """Polls at or past the deadline return at once instead of blocking."""
        receiver = ReadyCheckReceiver("ready_late", zmq_ctx_scope, count=2)
        await send_ready_signal(zmq_ctx_scope, "ready_late", 0)
        await wait_until_ready_signal_queued(receiver)

        # With timeout=0 every poll runs at or past the deadline.
        task = asyncio.ensure_future(receiver.wait(timeout=0))
        try:
            done, _ = await asyncio.wait({task}, timeout=2.0)
            assert done, "wait() blocked after its deadline"
            with pytest.raises(TimeoutError, match="1/2 signals received so far"):
                task.result()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            receiver.close()

    async def test_timeout_reports_signals_received_so_far(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_so_far", zmq_ctx_scope, count=2)

        await send_ready_signal(zmq_ctx_scope, "ready_so_far", 0)
        await wait_until_ready_signal_queued(receiver)
        with pytest.raises(TimeoutError, match="1/2 signals received so far"):
            await receiver.wait(timeout=0.3)
        with pytest.raises(TimeoutError, match="1/2 signals received so far"):
            await receiver.wait(timeout=0.1)
        receiver.close()

    async def test_duplicate_signal_counts_once(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_dup", zmq_ctx_scope, count=2)

        await send_ready_signal(zmq_ctx_scope, "ready_dup", 0)
        await send_ready_signal(zmq_ctx_scope, "ready_dup", 0)
        await wait_until_ready_signal_queued(receiver)
        with pytest.raises(TimeoutError, match="1/2"):
            await receiver.wait(timeout=0.3)

        await send_ready_signal(zmq_ctx_scope, "ready_dup", 1)
        assert await receiver.wait(timeout=5.0) == [0, 1]

    async def test_wait_without_timeout_counts_distinct_signals(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_no_timeout", zmq_ctx_scope, count=2)

        for identity in (3, 3, 4):
            await send_ready_signal(zmq_ctx_scope, "ready_no_timeout", identity)
        assert await receiver.wait() == [3, 4]

    async def test_cancelled_wait_without_timeout_closes_socket(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_cancel", zmq_ctx_scope, count=1)

        task = asyncio.ensure_future(receiver.wait())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert receiver._sock.closed

    async def test_wait_after_success_returns_copy_of_identities(self, zmq_ctx_scope):
        receiver = ReadyCheckReceiver("ready_done", zmq_ctx_scope, count=1)

        await send_ready_signal(zmq_ctx_scope, "ready_done", 5)
        identities = await receiver.wait(timeout=5.0)
        identities.append(99)

        assert await receiver.wait(timeout=0.1) == [5]
        assert receiver._sock.closed


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
