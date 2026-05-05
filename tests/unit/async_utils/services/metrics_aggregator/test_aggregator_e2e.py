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

"""End-to-end pub/sub round-trip tests for the metrics aggregator.

The legacy E2E suite exercised the full ``EventPublisherService`` →
``MetricsAggregatorService`` → ``InMemoryKVStore`` pipeline. With the
registry/publisher refactor, the wire surface that matters at this layer
is the snapshot pub/sub channel: aggregator → ``MetricsPublisher`` →
ZMQ PUB → ``MetricsSnapshotSubscriber``.

These tests stand up a real ``MetricsPublisher`` and
``MetricsSnapshotSubscriber`` against a single ``ManagedZMQContext.scoped``
context, publish snapshots, and verify the subscriber receives them with
the expected wire shape. The full event pipeline (events → aggregator →
metrics) is covered in ``test_aggregator.py``; this module is concerned
strictly with the snapshot transport.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import zmq
from inference_endpoint.async_utils.services.metrics_aggregator.publisher import (
    MetricsPublisher,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    MetricsSnapshotCodec,
    SessionState,
)
from inference_endpoint.async_utils.services.metrics_aggregator.subscriber import (
    MetricsSnapshotSubscriber,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext

# Small but generous: ZMQ's slow-joiner means the subscriber may miss the
# first few publishes; we wait for the first delivered snapshot below
# rather than racing on a wall clock.
_WAIT_TIMEOUT = 3.0


@pytest.fixture
def zmq_ctx_scope(tmp_path: Path):
    """Provide a scoped ManagedZMQContext for the duration of a test."""
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        yield ctx


def _make_pair(
    socket_name: str,
    zmq_ctx: ManagedZMQContext,
    loop: asyncio.AbstractEventLoop,
    fallback_path: Path,
    *,
    conflate: bool = False,
) -> tuple[MetricsPublisher, MetricsSnapshotSubscriber]:
    """Bind a publisher then connect a subscriber on the same socket name.

    Order matters for IPC: the publisher binds first so the IPC file
    exists before the subscriber connects. ``conflate=False`` (default)
    keeps every received message — appropriate for these tests where we
    want to count deliveries rather than just observe the freshest.
    """
    try:
        publisher = MetricsPublisher(
            MetricsSnapshotCodec(),
            zmq_ctx,
            socket_name,
            loop,
            fallback_path=fallback_path,
        )
    except zmq.ZMQError as exc:
        pytest.skip(f"ZMQ IPC bind unavailable (sandboxed?): {exc}")
    subscriber = MetricsSnapshotSubscriber(
        socket_name, zmq_ctx, loop, conflate=conflate
    )
    subscriber.start()
    return publisher, subscriber


@pytest.mark.unit
class TestPubSubRoundtrip:
    @pytest.mark.asyncio
    async def test_publish_final_arrives_at_subscriber(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """``publish_final`` produces a COMPLETE snapshot reachable over IPC.

        This replaces the legacy single-sample pipeline assertion: the
        aggregator's ``publish_final`` is what crosses the wire, and the
        ``MetricsSnapshotSubscriber`` is what the main process uses to
        observe the run's end. The exact metric values aren't the point
        here — the round-trip + state field is.
        """
        loop = asyncio.get_event_loop()
        publisher, subscriber = _make_pair(
            "test_e2e_final",
            zmq_ctx_scope,
            loop,
            tmp_path / "final_snapshot.msgpack",
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("total_samples_completed")
            registry.increment("total_samples_completed", 7)

            # ZMQ slow-joiner: give the SUB time to attach before publishing.
            await asyncio.sleep(0.2)
            await publisher.publish_final(registry, n_pending_tasks=0)

            arrived = await subscriber.wait_for_complete(timeout=_WAIT_TIMEOUT)
            assert arrived, "subscriber must receive COMPLETE snapshot"
            assert subscriber.complete is not None
            assert subscriber.complete.state == SessionState.COMPLETE
            assert subscriber.complete.n_pending_tasks == 0
        finally:
            subscriber.close()
            publisher.close()

    @pytest.mark.asyncio
    async def test_live_tick_then_final(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """Live ticks deliver LIVE-state snapshots; final delivers COMPLETE.

        Tracks the lifecycle the main process sees: subscriber's
        ``latest`` is updated by every live tick, and ``complete`` is
        only set once. Mirrors the design v5 §1 state machine.
        """
        loop = asyncio.get_event_loop()
        publisher, subscriber = _make_pair(
            "test_e2e_live_then_final",
            zmq_ctx_scope,
            loop,
            tmp_path / "final_snapshot.msgpack",
            # conflate=True: we don't care which live tick lands, just
            # that at least one does. This is the same setting the main
            # process consumer uses.
            conflate=True,
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("c")

            # Slow-joiner grace.
            await asyncio.sleep(0.2)

            publisher.start(
                registry,
                refresh_hz=20.0,
                get_runtime_state=lambda: (SessionState.LIVE, 0),
            )

            # Wait for at least one live snapshot to arrive.
            for _ in range(50):
                await asyncio.sleep(0.05)
                if subscriber.latest is not None:
                    break
            assert subscriber.latest is not None, "expected at least one live tick"
            assert subscriber.latest.state == SessionState.LIVE
            # Complete must NOT be set yet.
            assert subscriber.complete is None

            await publisher.publish_final(registry, n_pending_tasks=0)
            arrived = await subscriber.wait_for_complete(timeout=_WAIT_TIMEOUT)
            assert arrived
            assert subscriber.complete is not None
            assert subscriber.complete.state == SessionState.COMPLETE
        finally:
            subscriber.close()
            publisher.close()

    @pytest.mark.asyncio
    async def test_multiple_metrics_round_trip(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """Counters and series both round-trip with the right payload shape.

        Counter values must be exact; series presence (count + total)
        must round-trip cleanly. Histogram bucket geometry is covered in
        ``test_registry.py`` and ``test_snapshot.py`` — here we just
        confirm the wire format survives the IPC hop.
        """
        loop = asyncio.get_event_loop()
        publisher, subscriber = _make_pair(
            "test_e2e_multimetric",
            zmq_ctx_scope,
            loop,
            tmp_path / "final_snapshot.msgpack",
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("tracked_samples_issued")
            registry.register_counter("tracked_samples_completed")
            registry.register_series(
                "sample_latency_ns",
                hdr_low=1,
                hdr_high=3_600_000_000_000,
                sig_figs=3,
                n_histogram_buckets=10,
                percentiles=(50.0, 99.0),
            )
            for _ in range(2):
                registry.increment("tracked_samples_issued")
                registry.increment("tracked_samples_completed")
            registry.record("sample_latency_ns", 1_500_000)
            registry.record("sample_latency_ns", 2_500_000)

            # Slow-joiner grace.
            await asyncio.sleep(0.2)
            await publisher.publish_final(registry, n_pending_tasks=0)

            arrived = await subscriber.wait_for_complete(timeout=_WAIT_TIMEOUT)
            assert arrived
            snap = subscriber.complete
            assert snap is not None

            # Build a name → metric lookup off the wire side.
            from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (  # noqa: E501
                CounterStat,
                SeriesStat,
            )

            counters = {
                m.name: m.value for m in snap.metrics if isinstance(m, CounterStat)
            }
            series = {m.name: m for m in snap.metrics if isinstance(m, SeriesStat)}
            assert counters["tracked_samples_issued"] == 2
            assert counters["tracked_samples_completed"] == 2
            assert "sample_latency_ns" in series
            assert series["sample_latency_ns"].count == 2
            assert series["sample_latency_ns"].total == 4_000_000
        finally:
            subscriber.close()
            publisher.close()
