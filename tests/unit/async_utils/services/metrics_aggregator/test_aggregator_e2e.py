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

"""End-to-end tests for the metrics aggregator's two delivery channels.

The aggregator publishes snapshots through two independent paths:

1. ``final_snapshot.json`` on disk — the **primary** delivery surface
   for the Report consumer. Written atomically by ``publish_final``.
2. ZMQ PUB → ``MetricsSnapshotSubscriber`` — live ticks for TUI / live
   consumers, plus a terminal-state frame at end-of-run as a
   "run is over" signal.

These tests stand up a real ``MetricsPublisher`` and
``MetricsSnapshotSubscriber`` against a single ``ManagedZMQContext.scoped``
context and verify both channels deliver the right state. The full event
pipeline (events → aggregator → metrics) is covered in
``test_aggregator.py``; this module is concerned with the publish layer.
"""

from __future__ import annotations

import asyncio
import json
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


async def _wait_for_terminal_state(
    subscriber: MetricsSnapshotSubscriber, timeout: float = _WAIT_TIMEOUT
) -> bool:
    """Poll ``subscriber.latest`` until a terminal-state frame arrives."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        latest = subscriber.latest
        if latest is not None and latest.state in (
            SessionState.COMPLETE,
            SessionState.INTERRUPTED,
        ):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.fixture
def zmq_ctx_scope(tmp_path: Path):
    """Provide a scoped ManagedZMQContext for the duration of a test."""
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        yield ctx


def _make_pair(
    socket_name: str,
    zmq_ctx: ManagedZMQContext,
    loop: asyncio.AbstractEventLoop,
    final_snapshot_path: Path,
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
            final_snapshot_path=final_snapshot_path,
        )
    except zmq.ZMQError as exc:
        pytest.skip(f"ZMQ IPC bind unavailable (sandboxed?): {exc}")
    subscriber = MetricsSnapshotSubscriber(
        socket_name, zmq_ctx, loop, conflate=conflate
    )
    subscriber.start()
    return publisher, subscriber


@pytest.mark.unit
class TestFinalSnapshotDelivery:
    @pytest.mark.asyncio
    async def test_publish_final_writes_json_and_signals_pubsub(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """``publish_final`` writes the JSON file AND fires the pub/sub signal.

        The JSON file is the primary Report source; the pub/sub frame is
        the TUI shutdown signal. Both must land on a clean shutdown.
        """
        loop = asyncio.get_event_loop()
        target = tmp_path / "final_snapshot.json"
        publisher, subscriber = _make_pair(
            "test_e2e_final",
            zmq_ctx_scope,
            loop,
            target,
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("total_samples_completed")
            registry.increment("total_samples_completed", 7)

            # ZMQ slow-joiner: give the SUB time to attach before publishing.
            await asyncio.sleep(0.2)
            await publisher.publish_final(registry, n_pending_tasks=0)

            # JSON file landed with the right terminal state.
            assert target.exists(), "publish_final must write final_snapshot.json"
            decoded = json.loads(target.read_bytes())
            assert decoded["state"] == SessionState.COMPLETE.value
            assert decoded["n_pending_tasks"] == 0

            # Pub/sub signal landed at the subscriber as the most recent frame.
            arrived = await _wait_for_terminal_state(subscriber)
            assert arrived, "subscriber must receive terminal-state frame"
            assert subscriber.latest is not None
            assert subscriber.latest.state == SessionState.COMPLETE
        finally:
            subscriber.close()
            publisher.close()

    @pytest.mark.asyncio
    async def test_live_tick_then_final(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """Live ticks deliver LIVE-state snapshots; final flips to COMPLETE.

        Tracks the lifecycle a TUI sees: subscriber's ``latest`` is
        updated by every live tick, then replaced by the terminal-state
        frame at end-of-run.
        """
        loop = asyncio.get_event_loop()
        publisher, subscriber = _make_pair(
            "test_e2e_live_then_final",
            zmq_ctx_scope,
            loop,
            tmp_path / "final_snapshot.json",
            # conflate=True mirrors the default subscriber setting — we
            # only care which state is *most recent*, not the count.
            conflate=True,
        )
        try:
            registry = MetricsRegistry()
            registry.register_counter("c")

            # Slow-joiner grace.
            await asyncio.sleep(0.2)

            publisher.start(
                registry,
                publish_interval_s=0.05,
                get_runtime_state=lambda: (SessionState.LIVE, 0),
            )

            # Wait for at least one live snapshot to arrive.
            for _ in range(50):
                await asyncio.sleep(0.05)
                if subscriber.latest is not None:
                    break
            assert subscriber.latest is not None, "expected at least one live tick"
            assert subscriber.latest.state == SessionState.LIVE

            await publisher.publish_final(registry, n_pending_tasks=0)
            arrived = await _wait_for_terminal_state(subscriber)
            assert arrived
            assert subscriber.latest is not None
            assert subscriber.latest.state == SessionState.COMPLETE
        finally:
            subscriber.close()
            publisher.close()

    @pytest.mark.asyncio
    async def test_multiple_metrics_round_trip(
        self, tmp_path: Path, zmq_ctx_scope: ManagedZMQContext
    ):
        """Counters and series both land in the JSON file with the right shape.

        Counter values must be exact; series count + total must
        round-trip. Histogram bucket geometry is covered in
        ``test_registry.py`` and ``test_snapshot.py`` — here we confirm
        the on-disk format preserves the shape end-to-end.
        """
        loop = asyncio.get_event_loop()
        target = tmp_path / "final_snapshot.json"
        publisher, subscriber = _make_pair(
            "test_e2e_multimetric",
            zmq_ctx_scope,
            loop,
            target,
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

            decoded = json.loads(target.read_bytes())
            counters = {
                m["name"]: m["value"]
                for m in decoded["metrics"]
                if m["type"] == "counter"
            }
            series = {m["name"]: m for m in decoded["metrics"] if m["type"] == "series"}
            assert counters["tracked_samples_issued"] == 2
            assert counters["tracked_samples_completed"] == 2
            assert "sample_latency_ns" in series
            assert series["sample_latency_ns"]["count"] == 2
            assert series["sample_latency_ns"]["total"] == 4_000_000
        finally:
            subscriber.close()
            publisher.close()
