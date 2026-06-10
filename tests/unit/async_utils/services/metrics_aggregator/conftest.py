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

"""Shared test doubles and factories for metrics aggregator tests.

Tests that need to inspect emitted values build them directly off a
``MetricsRegistry`` and a ``MetricsSnapshot``.

The helpers here are intentionally small — most reused-across-tests
construction lives in ``_make_aggregator`` style fixtures local to each
test file (the aggregator's wire surface is small enough that a single
shared fixture would mostly hide it).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    CounterStat,
    SeriesStat,
    SessionState,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput

# ---------------------------------------------------------------------------
# Mock BatchTokenizer — whitespace token counts; matches the BatchTokenizer
# surface the TokenBatchQueue calls (count_texts_async + message path).
# ---------------------------------------------------------------------------


class MockBatchTokenizer:
    """Mock BatchTokenizer that splits on whitespace with optional async delay."""

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    async def count_texts_async(
        self, texts: list[str], _loop: asyncio.AbstractEventLoop
    ) -> list[int]:
        if self._delay:
            await asyncio.sleep(self._delay)
        return [len(t.split()) for t in texts]

    async def count_texts_live_async(
        self, texts: list[str], loop: asyncio.AbstractEventLoop
    ) -> list[int]:
        return await self.count_texts_async(texts, loop)

    async def token_count_message_async(
        self,
        content: str,
        reasoning: str | None,
        tool_calls,
        _loop: asyncio.AbstractEventLoop,
    ) -> int:
        import msgspec

        if self._delay:
            await asyncio.sleep(self._delay)
        tool_calls_str = (
            msgspec.json.encode(list(tool_calls)).decode() if tool_calls else ""
        )
        combined = (content or "") + " " + (reasoning or "") + " " + tool_calls_str
        return len(combined.split())

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# EventRecord factories
# ---------------------------------------------------------------------------


def session_event(ev_type: SessionEventType, ts: int = 0) -> EventRecord:
    return EventRecord(event_type=ev_type, timestamp_ns=ts)


def sample_event(
    ev_type: SampleEventType, uuid: str, ts: int = 0, data=None
) -> EventRecord:
    return EventRecord(event_type=ev_type, timestamp_ns=ts, sample_uuid=uuid, data=data)


def text_output(s: str) -> TextModelOutput:
    return TextModelOutput(output=s)


def streaming_text(*chunks: str) -> TextModelOutput:
    return TextModelOutput(output=tuple(chunks))


# ---------------------------------------------------------------------------
# Registry / snapshot inspection helpers
# ---------------------------------------------------------------------------


def snapshot_counters(registry: MetricsRegistry) -> dict[str, int | float]:
    """Return all counter values from a fresh snapshot.

    State/n_pending values don't matter for counter inspection — they
    bypass the exact-vs-HDR fork. Tests that need series inspection
    should call ``snapshot_series_values`` instead.
    """
    snap = registry.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
    return {m.name: m.value for m in snap.metrics if isinstance(m, CounterStat)}


def snapshot_series_count(registry: MetricsRegistry, name: str) -> int:
    """Return ``count`` of a named series from a fresh snapshot.

    Returns 0 if the series is unregistered or has no recordings.
    """
    snap = registry.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
    for m in snap.metrics:
        if isinstance(m, SeriesStat) and m.name == name:
            return m.count
    return 0


def snapshot_series_total(registry: MetricsRegistry, name: str) -> int | float:
    """Return ``total`` of a named series from a fresh snapshot."""
    snap = registry.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
    for m in snap.metrics:
        if isinstance(m, SeriesStat) and m.name == name:
            return m.total
    return 0


# ---------------------------------------------------------------------------
# Aggregator factory
# ---------------------------------------------------------------------------


def make_aggregator(
    zmq_ctx: ManagedZMQContext,
    loop: asyncio.AbstractEventLoop,
    socket_name: str,
    *,
    tokenizer=None,
    live_flush_interval_s: float | None = None,
    streaming: bool = True,
    shutdown_event: asyncio.Event | None = None,
) -> tuple[MetricsAggregatorService, MetricsRegistry, MagicMock]:
    """Construct an aggregator wired to a real SUB socket and a mocked publisher.

    The aggregator's ``start()`` is intentionally not called: tests inject
    events directly via ``await agg.process([...])``. The publisher is a
    ``MagicMock`` so the aggregator's STARTED branch (which calls
    ``publisher.start(...)``) and ENDED branch (which calls ``publish_final``
    + ``close``) don't touch real I/O.

    Returns ``(agg, registry, publisher_mock)``.
    """
    registry = MetricsRegistry()
    # ``publish_final`` and ``aclose`` are awaited by the aggregator's
    # ENDED handler, so they must be AsyncMocks. The remaining surface
    # (``start``, ``close``) is synchronous and falls back to MagicMock's
    # default attribute behavior.
    publisher = MagicMock()
    publisher.publish_final = AsyncMock()
    publisher.aclose = AsyncMock()
    agg = MetricsAggregatorService(
        socket_name,
        zmq_ctx,
        loop,
        registry=registry,
        publisher=publisher,
        publish_interval_s=0.25,
        sig_figs=3,
        n_histogram_buckets=10,
        tokenizer=tokenizer,
        live_flush_interval_s=live_flush_interval_s,
        streaming=streaming,
        shutdown_event=shutdown_event,
    )
    return agg, registry, publisher
