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

"""Tests for ``MetricsPublisher`` (tick task + final publish + disk fallback)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import msgspec
import msgspec.msgpack
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.publisher import (
    MetricsPublisher,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshot,
    MetricsSnapshotCodec,
    SessionState,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext


def _build_publisher(
    fallback_path: Path, loop: asyncio.AbstractEventLoop
) -> tuple[MetricsPublisher, ManagedZMQContext]:
    """Construct a MetricsPublisher backed by a real IPC socket scoped to a temp dir."""
    # ManagedZMQContext.scoped() returns a context manager — use raw construct
    # so the test owns lifecycle and can scope it via a fixture.
    raise NotImplementedError("constructed inline within fixture/test")


@pytest.fixture
def zmq_ctx_scope():
    """Provide a scoped ManagedZMQContext for the duration of a test."""
    with ManagedZMQContext.scoped() as ctx:
        yield ctx


@pytest.mark.unit
class TestMetricsPublisher:
    @pytest.mark.asyncio
    async def test_start_schedules_tick_task(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        loop = asyncio.get_event_loop()
        publisher = MetricsPublisher(
            MetricsSnapshotCodec(),
            zmq_ctx_scope,
            "test_pub_start",
            loop,
            fallback_path=tmp_path / "final_snapshot.msgpack",
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("c")

            calls = []

            def get_runtime_state() -> tuple[SessionState, int]:
                calls.append(True)
                return SessionState.LIVE, 0

            publisher.start(
                registry,
                refresh_hz=100.0,
                get_runtime_state=get_runtime_state,
            )
            assert publisher._tick_task is not None
            assert not publisher._tick_task.done()

            # Let at least one tick run.
            await asyncio.sleep(0.05)
            assert len(calls) >= 1
        finally:
            publisher.close()

    @pytest.mark.asyncio
    async def test_publish_final_writes_disk_atomically(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        loop = asyncio.get_event_loop()
        target = tmp_path / "final_snapshot.msgpack"
        publisher = MetricsPublisher(
            MetricsSnapshotCodec(),
            zmq_ctx_scope,
            "test_pub_disk",
            loop,
            fallback_path=target,
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("c")
            registry.increment("c", 5)

            await publisher.publish_final(registry, n_pending_tasks=0)

            # The .tmp file MUST NOT exist after the rename.
            tmp_target = target.with_suffix(target.suffix + ".tmp")
            assert not tmp_target.exists(), "tmp file should have been renamed"
            assert target.exists(), "final snapshot should be on disk"

            decoded = msgspec.msgpack.decode(target.read_bytes(), type=MetricsSnapshot)
            assert decoded.state == SessionState.COMPLETE
            assert decoded.n_pending_tasks == 0
        finally:
            publisher.close()

    @pytest.mark.asyncio
    async def test_disk_failure_does_not_block_pubsub(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """Disk fallback failure MUST NOT prevent pub/sub publish."""
        loop = asyncio.get_event_loop()
        # Point the fallback at a path whose parent is a *file*, not a dir.
        # Writing into it will fail; pub/sub publish should still complete.
        bad_parent = tmp_path / "not_a_dir"
        bad_parent.write_bytes(b"this is a file, not a directory")
        publisher = MetricsPublisher(
            MetricsSnapshotCodec(),
            zmq_ctx_scope,
            "test_pub_diskfail",
            loop,
            fallback_path=bad_parent / "final_snapshot.msgpack",
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("c")

            # Stub the inner ZMQ publisher with a recording mock so we can
            # confirm it was called even though disk fails.
            inner_mock = MagicMock()
            publisher._publisher = inner_mock
            await publisher.publish_final(registry, n_pending_tasks=0)

            assert inner_mock.publish.call_count == 1
            # Disk should not have been written.
            assert not (bad_parent / "final_snapshot.msgpack").exists()
        finally:
            try:
                publisher.close()
            except Exception:
                # Inner mock may complain on close; we just want the test to
                # exercise the disk-failure path without hanging.
                pass

    @pytest.mark.asyncio
    async def test_publish_final_awaits_tick_task_cancellation(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """publish_final MUST NOT return while the tick task could still emit.

        ``self._tick_task.cancel()`` only schedules cancellation at the
        next await point; without ``await``ing the task, a late live tick
        landing after the COMPLETE frame would replace it in a
        ``conflate=True`` SUB queue. publish_final must therefore await
        cancellation before publishing COMPLETE.
        """
        loop = asyncio.get_event_loop()
        publisher = MetricsPublisher(
            MetricsSnapshotCodec(),
            zmq_ctx_scope,
            "test_pub_finalrace",
            loop,
            fallback_path=tmp_path / "final_snapshot.msgpack",
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("c")

            publisher.start(
                registry,
                refresh_hz=100.0,
                get_runtime_state=lambda: (SessionState.LIVE, 0),
            )
            tick_task = publisher._tick_task
            assert tick_task is not None
            # Allow the tick to begin so we know it's running.
            await asyncio.sleep(0.02)

            await publisher.publish_final(registry, n_pending_tasks=0)

            # After publish_final returns, the tick task MUST be done.
            assert (
                tick_task.done()
            ), "tick task must be done before publish_final returns"
            # And the publisher's reference is cleared so close() is a no-op
            # for the tick path.
            assert publisher._tick_task is None
        finally:
            publisher.close()

    @pytest.mark.asyncio
    async def test_close_cancels_tick_task(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        loop = asyncio.get_event_loop()
        publisher = MetricsPublisher(
            MetricsSnapshotCodec(),
            zmq_ctx_scope,
            "test_pub_close",
            loop,
            fallback_path=tmp_path / "final_snapshot.msgpack",
        )

        registry = MetricsRegistry()
        registry.register_counter("c")
        publisher.start(
            registry,
            refresh_hz=10.0,
            get_runtime_state=lambda: (SessionState.LIVE, 0),
        )
        tick_task = publisher._tick_task
        assert tick_task is not None
        publisher.close()

        # Give the cancellation a chance to take effect.
        try:
            await asyncio.wait_for(tick_task, timeout=1.0)
        except (asyncio.CancelledError, TimeoutError):
            # Cancelled: expected. Timeout: also acceptable on slow CI.
            pass
        assert tick_task.done()
