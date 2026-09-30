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

import asyncio
import signal
import time
from collections.abc import Sequence
from multiprocessing import Process
from typing import Any, cast
from unittest.mock import MagicMock

import msgspec
import pytest
import zmq
from inference_endpoint.async_utils.transport.zmq.ready_check import (
    ReadyCheckReceiver,
    send_ready_signal,
)
from inference_endpoint.endpoint_client import worker_manager as worker_manager_module
from inference_endpoint.endpoint_client.worker_manager import WorkerManager

from tests.ready_check_helpers import wait_until_ready_signal_queued


class _FakeWorker:
    def __init__(self, alive: bool = True, pid: int = 1):
        self.alive = alive
        self.pid = pid

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    kill = terminate

    def join(self, timeout: float | None = None) -> None:
        pass


class _ReadyCheckPoolTransport:
    """Pool transport stand-in backed by a real ReadyCheckReceiver."""

    worker_connector = None

    def __init__(self, receiver: ReadyCheckReceiver):
        self.receiver = receiver
        self.identities: list[int] = []
        self.timeouts: list[float | None] = []

    async def wait_for_workers_ready(self, timeout: float | None = None) -> None:
        self.timeouts.append(timeout)
        self.identities = await self.receiver.wait(timeout=timeout)

    def cleanup(self) -> None:
        self.receiver.close()


class _WorkerDiesAtDeadlineTransport:
    """Pool transport stand-in whose ready wait times out as a worker dies."""

    worker_connector = None

    def __init__(
        self,
        worker: _FakeWorker,
        message: str = "Ready check failed: 0/1 signals received so far",
    ):
        self._worker = worker
        self._message = message
        self.cleaned_up = False

    async def wait_for_workers_ready(self, timeout: float | None = None) -> None:
        await asyncio.sleep(0.05)
        self._worker.alive = False
        raise TimeoutError(self._message)

    def cleanup(self) -> None:
        self.cleaned_up = True


class _NeverReadyTransport:
    """Pool transport stand-in whose workers never signal ready."""

    async def wait_for_workers_ready(self, timeout: float | None = None) -> None:
        await asyncio.Event().wait()


class _BareTimeoutTransport:
    """Pool transport stand-in whose ready wait times out with no message."""

    worker_connector = None

    async def wait_for_workers_ready(self, timeout: float | None = None) -> None:
        raise TimeoutError

    def cleanup(self) -> None:
        pass


class _BrokenTransport:
    """Pool transport stand-in whose ready wait fails with a non-timeout error."""

    async def wait_for_workers_ready(self, timeout: float | None = None) -> None:
        raise ConnectionError("ready socket failed")


def _assert_no_leaked_tasks() -> None:
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert pending == []


def _make_manager(
    transport: Any, *, init_timeout_s: float, num_workers: int = 0
) -> WorkerManager:
    """Build a WorkerManager on `transport` whose initialize() spawns `num_workers`."""
    http_config = MagicMock(
        num_workers=num_workers,
        cpu_affinity=None,
        worker_initialization_timeout=init_timeout_s,
        worker_graceful_shutdown_wait=0.0,
        worker_force_kill_timeout=0.0,
    )
    http_config.transport.transport_class.create.return_value = transport
    return WorkerManager(http_config, asyncio.get_running_loop())


def _make_manager_with_workers(
    transport: Any, workers: Sequence[_FakeWorker], *, init_timeout_s: float
) -> WorkerManager:
    """Build a WorkerManager whose `workers` are already running."""
    manager = _make_manager(transport, init_timeout_s=init_timeout_s)
    manager.workers = cast("list[Process]", list(workers))
    return manager


def _make_manager_for_initialize(
    transport: Any,
    workers: Sequence[_FakeWorker],
    *,
    init_timeout_s: float,
    monkeypatch: pytest.MonkeyPatch,
) -> WorkerManager:
    """Build a WorkerManager whose initialize() spawns exactly `workers`."""
    spawned = iter(workers)
    monkeypatch.setattr(
        worker_manager_module, "Process", lambda **kwargs: next(spawned)
    )
    return _make_manager(
        transport, init_timeout_s=init_timeout_s, num_workers=len(workers)
    )


@pytest.mark.unit
def test_spawned_worker_inherits_sigint_blocked(monkeypatch):
    blocked = []

    class FakeProcess:
        pid = 1

        def __init__(self, **kwargs):
            pass

        def start(self):
            current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            blocked.append(signal.SIGINT in current_mask)

    monkeypatch.setattr(worker_manager_module, "Process", FakeProcess)
    manager = object.__new__(WorkerManager)
    manager.http_config = MagicMock()
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    manager._spawn_worker(0, MagicMock())

    assert blocked == [True]
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == previous_mask


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ready_signal_at_liveness_check_is_not_lost(zmq_ctx_scope):
    """Signals that land on a liveness-check boundary still count toward init."""
    init_timeout_s, send_at_s, block_s = 3.0, 0.2, 0.2
    # WorkerManager checks liveness every 10% of the timeout. The first
    # signal is sent before the first check and the loop is blocked across
    # it, so the socket read and the check land in the same loop iteration.
    assert send_at_s < 0.1 * init_timeout_s < send_at_s + block_s
    transport = _ReadyCheckPoolTransport(
        ReadyCheckReceiver("ready_wm", zmq_ctx_scope, count=2)
    )
    push = zmq_ctx_scope.socket(zmq.PUSH)
    zmq_ctx_scope.connect(push, "ready_wm")
    manager = _make_manager_with_workers(
        transport, [_FakeWorker(), _FakeWorker()], init_timeout_s=init_timeout_s
    )

    async def start_workers():
        await asyncio.sleep(send_at_s)
        push.send(msgspec.msgpack.encode(0))
        time.sleep(block_s)
        push.send(msgspec.msgpack.encode(1))

    try:
        await asyncio.gather(
            manager._wait_for_workers_with_liveness_check(), start_workers()
        )
        assert transport.identities == [0, 1]
    finally:
        push.close(linger=0)
        transport.cleanup()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zero_init_timeout_waits_without_deadline(zmq_ctx_scope):
    transport = _ReadyCheckPoolTransport(
        ReadyCheckReceiver("ready_wm_zero", zmq_ctx_scope, count=1)
    )
    manager = _make_manager_with_workers(transport, [_FakeWorker()], init_timeout_s=0.0)

    async def start_worker():
        await asyncio.sleep(0.2)
        await send_ready_signal(zmq_ctx_scope, "ready_wm_zero", 0)

    try:
        await asyncio.gather(
            manager._wait_for_workers_with_liveness_check(), start_worker()
        )
        assert transport.timeouts == [None]
        assert transport.identities == [0]
    finally:
        transport.cleanup()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_init_timeout_error_reports_ready_count(zmq_ctx_scope, monkeypatch):
    transport = _ReadyCheckPoolTransport(
        ReadyCheckReceiver("ready_wm_timeout", zmq_ctx_scope, count=2)
    )
    workers = [_FakeWorker(), _FakeWorker()]
    manager = _make_manager_for_initialize(
        transport, workers, init_timeout_s=0.3, monkeypatch=monkeypatch
    )
    await send_ready_signal(zmq_ctx_scope, "ready_wm_timeout", 0)
    await wait_until_ready_signal_queued(transport.receiver)

    with pytest.raises(
        TimeoutError,
        match=(
            r"^Workers failed to initialize: Ready check failed: "
            r"1/2 signals received so far; timed out after 0\.3s$"
        ),
    ):
        await manager.initialize()

    # The failed init shut the spawned workers and the transport down.
    assert manager.workers == workers
    assert not any(w.is_alive() for w in workers)
    assert transport.receiver._sock.closed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_init_reports_worker_death_at_deadline_with_ready_count(monkeypatch):
    """A worker that dies as the ready wait times out is reported as dead."""
    worker = _FakeWorker(pid=7)
    transport = _WorkerDiesAtDeadlineTransport(worker)
    manager = _make_manager_for_initialize(
        transport, [worker], init_timeout_s=10.0, monkeypatch=monkeypatch
    )

    with pytest.raises(
        RuntimeError,
        match=r"PIDs \[7\]; Ready check failed: 0/1 signals received so far",
    ) as exc:
        await manager.initialize()
    assert isinstance(exc.value.__cause__, TimeoutError)
    assert transport.cleaned_up


@pytest.mark.unit
@pytest.mark.asyncio
async def test_init_timeout_names_the_timeout_when_transport_does_not(monkeypatch):
    manager = _make_manager_for_initialize(
        _BareTimeoutTransport(),
        [_FakeWorker()],
        init_timeout_s=10.0,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(
        TimeoutError,
        match=r"^Workers failed to initialize: timed out after 10\.0s$",
    ):
        await manager.initialize()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_worker_death_at_bare_timeout_names_the_timeout(monkeypatch):
    worker = _FakeWorker(pid=7)
    manager = _make_manager_for_initialize(
        _WorkerDiesAtDeadlineTransport(worker, message=""),
        [worker],
        init_timeout_s=10.0,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(
        RuntimeError,
        match=r"^Worker\(s\) died during init: PIDs \[7\]; timed out after 10\.0s$",
    ):
        await manager.initialize()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bare_timeout_without_deadline_names_no_deadline(monkeypatch):
    manager = _make_manager_for_initialize(
        _BareTimeoutTransport(),
        [_FakeWorker()],
        init_timeout_s=0.0,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(
        TimeoutError, match=r"^Workers failed to initialize: timed out$"
    ):
        await manager.initialize()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_initialize_shuts_down_workers_and_transport(
    zmq_ctx_scope, monkeypatch
):
    """Cancelling init (as SIGINT or the run watchdog does) cleans up."""
    transport = _ReadyCheckPoolTransport(
        ReadyCheckReceiver("ready_wm_init_cancel", zmq_ctx_scope, count=2)
    )
    workers = [_FakeWorker(), _FakeWorker()]
    manager = _make_manager_for_initialize(
        transport, workers, init_timeout_s=10.0, monkeypatch=monkeypatch
    )

    task = asyncio.create_task(manager.initialize())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not any(w.is_alive() for w in workers)
    assert transport.receiver._sock.closed
    _assert_no_leaked_tasks()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zero_init_timeout_still_checks_liveness():
    """With no deadline, a worker that dies mid-wait still fails init."""
    worker = _FakeWorker(pid=5)
    manager = _make_manager_with_workers(
        _NeverReadyTransport(), [worker], init_timeout_s=0.0
    )

    async def worker_crashes():
        await asyncio.sleep(0.2)
        worker.alive = False

    with pytest.raises(RuntimeError, match=r"PIDs \[5\]"):
        await asyncio.wait_for(
            asyncio.gather(
                manager._wait_for_workers_with_liveness_check(), worker_crashes()
            ),
            timeout=5.0,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ready_wait_error_propagates_unchanged():
    manager = _make_manager_with_workers(
        _BrokenTransport(), [_FakeWorker()], init_timeout_s=10.0
    )

    with pytest.raises(ConnectionError, match="ready socket failed"):
        await manager._wait_for_workers_with_liveness_check()
    _assert_no_leaked_tasks()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_worker_death_mid_wait_closes_ready_socket(zmq_ctx_scope):
    worker = _FakeWorker(pid=9)
    receiver = ReadyCheckReceiver("ready_wm_death", zmq_ctx_scope, count=1)
    manager = _make_manager_with_workers(
        _ReadyCheckPoolTransport(receiver), [worker], init_timeout_s=1.0
    )

    async def worker_crashes():
        await asyncio.sleep(0.15)
        worker.alive = False

    with pytest.raises(RuntimeError, match=r"PIDs \[9\]"):
        await asyncio.gather(
            manager._wait_for_workers_with_liveness_check(),
            worker_crashes(),
        )
    assert receiver._sock.closed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_liveness_wait_closes_ready_socket(zmq_ctx_scope):
    receiver = ReadyCheckReceiver("ready_wm_cancel", zmq_ctx_scope, count=1)
    manager = _make_manager_with_workers(
        _ReadyCheckPoolTransport(receiver), [_FakeWorker()], init_timeout_s=10.0
    )

    task = asyncio.create_task(manager._wait_for_workers_with_liveness_check())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert receiver._sock.closed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dead_worker_fails_fast_without_leaking_ready_wait():
    manager = _make_manager_with_workers(
        _NeverReadyTransport(),
        [_FakeWorker(alive=False, pid=42)],
        init_timeout_s=10.0,
    )

    with pytest.raises(RuntimeError, match=r"died during init: PIDs \[42\]"):
        await asyncio.wait_for(
            manager._wait_for_workers_with_liveness_check(), timeout=0.5
        )
    _assert_no_leaked_tasks()
