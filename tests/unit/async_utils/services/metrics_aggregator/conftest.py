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

"""Shared test doubles and factories for metrics aggregator tests."""

from __future__ import annotations

import asyncio
from typing import Literal
from unittest.mock import MagicMock

from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.kv_store import (
    KVStore,
    SeriesStats,
)
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput

# ---------------------------------------------------------------------------
# In-memory KVStore for tests
# ---------------------------------------------------------------------------


class InMemoryKVStore(KVStore):
    """In-memory KVStore for unit tests. No /dev/shm files needed."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._series: dict[str, list] = {}
        self._series_dtype: dict[str, type] = {}
        self.closed: bool = False

    def create_key(
        self, key: str, key_type: Literal["series", "counter"], dtype: type = int
    ) -> None:
        if key_type == "counter" and key not in self._counters:
            self._counters[key] = 0
        elif key_type == "series" and key not in self._series:
            self._series[key] = []
            self._series_dtype[key] = dtype

    def update(self, key: str, value: int | float) -> None:
        if key in self._counters:
            self._counters[key] = int(value)
        elif key in self._series:
            self._series[key].append(value)
        else:
            raise KeyError(f"Key not created: {key}")

    def get(self, key: str) -> int | SeriesStats:
        if key in self._counters:
            return self._counters[key]
        if key in self._series:
            dtype = self._series_dtype[key]
            return SeriesStats(list(self._series[key]), dtype=dtype)
        raise KeyError(f"Key not created: {key}")

    def snapshot(self) -> dict[str, int | SeriesStats]:
        result: dict[str, int | SeriesStats] = {}
        for k, v in self._counters.items():
            result[k] = v
        for k, vals in self._series.items():
            dtype = self._series_dtype[k]
            result[k] = SeriesStats(list(vals), dtype=dtype)
        return result

    def close(self) -> None:
        self.closed = True

    # --- Test helpers ---

    def get_series_values(self, key: str) -> list:
        return list(self._series.get(key, []))

    def get_counter(self, key: str) -> int:
        return self._counters.get(key, 0)

    def get_all_series(self) -> dict[str, list[float]]:
        """All series as {metric_name: [values]}."""
        return {k: list(v) for k, v in self._series.items()}


# ---------------------------------------------------------------------------
# Mock TokenizePool
# ---------------------------------------------------------------------------


class MockTokenizePool:
    """Mock TokenizePool that splits on whitespace with artificial async delay."""

    def __init__(self, delay: float = 0.01) -> None:
        self._delay = delay

    def token_count(self, text: str) -> int:
        return len(text.split())

    async def token_count_async(
        self, text: str, _loop: asyncio.AbstractEventLoop
    ) -> int:
        await asyncio.sleep(self._delay)
        return len(text.split())

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Aggregator factories
# ---------------------------------------------------------------------------


def mock_zmq_context() -> MagicMock:
    """Create a mock ManagedZMQContext that no-ops all ZMQ operations."""
    ctx = MagicMock()
    ctx.socket.return_value = MagicMock()
    ctx.connect.return_value = "ipc:///mock/socket"
    return ctx


def make_stub_aggregator(
    kv_store: KVStore,
    tokenize_pool=None,
    streaming: bool = True,
) -> MetricsAggregatorService:
    """Create a MetricsAggregatorService with ZMQ mocked out."""
    return MetricsAggregatorService(
        "mock_path",
        mock_zmq_context(),
        MagicMock(spec=asyncio.AbstractEventLoop),
        kv_store=kv_store,
        tokenize_pool=tokenize_pool,
        streaming=streaming,
    )


def make_async_stub_aggregator(
    kv_store: KVStore,
    tokenize_pool,
    loop: asyncio.AbstractEventLoop,
    streaming: bool = True,
) -> MetricsAggregatorService:
    """Create a MetricsAggregatorService with a real loop and mock ZMQ."""
    return MetricsAggregatorService(
        "mock_path",
        mock_zmq_context(),
        loop,
        kv_store=kv_store,
        tokenize_pool=tokenize_pool,
        streaming=streaming,
    )


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
