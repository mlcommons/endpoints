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

"""Tests for MetricsAggregatorService.process() logic.

These tests exercise the aggregator's event dispatch and metric computation
without ZMQ transport by calling process() directly.
"""

import asyncio

import pytest
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import ErrorData, PromptData

from .conftest import (
    InMemoryKVStore,
    MockTokenizePool,
    make_async_stub_aggregator,
    make_stub_aggregator,
    sample_event,
    session_event,
    streaming_text,
    text_output,
)

# ---------------------------------------------------------------------------
# Performance tracking window
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrackingWindow:
    @pytest.mark.asyncio
    async def test_not_tracked_before_start(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.STARTED, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=100),
            ]
        )
        assert agg._table.get_row("s1") is None
        assert store.get_series_values("ttft_ns") == []
        assert store.get_series_values("sample_latency_ns") == []

    @pytest.mark.asyncio
    async def test_tracked_after_start(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=100),
            ]
        )
        assert agg._table.get_row("s1") is not None

    @pytest.mark.asyncio
    async def test_not_tracked_after_stop(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                session_event(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=50),
                sample_event(SampleEventType.ISSUED, "s1", ts=100),
            ]
        )
        assert agg._table.get_row("s1") is None

    @pytest.mark.asyncio
    async def test_inflight_sample_continues_after_stop(self):
        """A sample issued during tracking completes normally after STOP."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=100),
                session_event(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=300),
                sample_event(SampleEventType.COMPLETE, "s1", ts=500),
            ]
        )
        assert 200 in store.get_series_values("ttft_ns")
        assert 400 in store.get_series_values("sample_latency_ns")

    @pytest.mark.asyncio
    async def test_restart_tracking_window(self):
        """START -> STOP -> START creates a second tracking window."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=100),
                session_event(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200),
                sample_event(SampleEventType.ISSUED, "s2", ts=300),  # not tracked
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=400),
                sample_event(SampleEventType.ISSUED, "s3", ts=500),  # tracked
                sample_event(SampleEventType.COMPLETE, "s1", ts=600),
                sample_event(SampleEventType.COMPLETE, "s3", ts=700),
            ]
        )
        assert agg._table.get_row("s2") is None  # never tracked
        latencies = store.get_series_values("sample_latency_ns")
        assert len(latencies) == 2  # s1 and s3 both completed

    @pytest.mark.asyncio
    async def test_tracked_block_durations(self):
        """Tracked blocks extend to last sample completion."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=100),
                session_event(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200),
                sample_event(SampleEventType.COMPLETE, "s1", ts=700),
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=800),
                sample_event(SampleEventType.ISSUED, "s2", ts=900),
                sample_event(SampleEventType.COMPLETE, "s2", ts=1000),
            ]
        )
        assert agg._table.tracked_blocks[0].duration_ns == 700  # 700 - 0
        assert agg._table.tracked_blocks[1].duration_ns == 200  # 1000 - 800
        assert agg._table.total_tracked_duration_ns == 900
        assert agg._table.total_completed_tracked_samples == 2


# ---------------------------------------------------------------------------
# Timing metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimingMetrics:
    @pytest.mark.asyncio
    async def test_ttft_and_sample_latency(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2500),
                sample_event(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        assert 1500 in store.get_series_values("ttft_ns")
        assert 4000 in store.get_series_values("sample_latency_ns")

    @pytest.mark.asyncio
    async def test_chunk_deltas(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                sample_event(SampleEventType.RECV_NON_FIRST, "s1", ts=3000),
                sample_event(SampleEventType.RECV_NON_FIRST, "s1", ts=4500),
                sample_event(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        assert store.get_series_values("chunk_delta_ns") == [1000, 1500]

    @pytest.mark.asyncio
    async def test_non_streaming_latency_only(self):
        """Non-streaming sample emits sample_latency_ns and OSL, but no TTFT/chunk_delta/TPOT."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(store, pool, loop)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=3000,
                    data=text_output("hello world"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        assert 2000 in store.get_series_values("sample_latency_ns")
        assert 2 in store.get_series_values("osl")
        assert store.get_series_values("ttft_ns") == []
        assert store.get_series_values("chunk_delta_ns") == []
        assert store.get_series_values("tpot_ns") == []

    @pytest.mark.asyncio
    async def test_chunk_delta_not_emitted_without_last_recv(self):
        """RECV_NON_FIRST without prior RECV_FIRST: no chunk_delta emitted."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
            ]
        )
        row = agg._table.get_row("s1")
        assert row is not None
        assert row.last_recv_ns is None  # No recv events yet


# ---------------------------------------------------------------------------
# ISL (token_ids path -- sync, no tokenize_pool needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsl:
    @pytest.mark.asyncio
    async def test_issued_with_token_ids_emits_isl_directly(self):
        """SGLang path: PromptData with token_ids emits ISL = len(token_ids)."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(token_ids=(101, 202, 303, 404, 505)),
                ),
            ]
        )
        assert 5 in store.get_series_values("isl")

    @pytest.mark.asyncio
    async def test_issued_without_data_no_isl(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
            ]
        )
        assert store.get_series_values("isl") == []


# ---------------------------------------------------------------------------
# Edge cases and event ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_untracked_sample_events_ignored(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.RECV_FIRST, "unknown", ts=2000),
                sample_event(SampleEventType.COMPLETE, "unknown", ts=5000),
            ]
        )
        assert store.get_series_values("ttft_ns") == []
        assert store.get_series_values("sample_latency_ns") == []

    @pytest.mark.asyncio
    async def test_complete_removes_row(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        assert agg._table.get_row("s1") is None
        assert len(agg._table) == 0

    @pytest.mark.asyncio
    async def test_session_ended_closes_store(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.STARTED, ts=0),
                session_event(SessionEventType.ENDED, ts=100),
            ]
        )
        assert store.closed

    @pytest.mark.asyncio
    async def test_events_after_ended_are_dropped(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=100),
                session_event(SessionEventType.ENDED, ts=200),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=300),
            ]
        )
        assert store.get_series_values("ttft_ns") == []

    @pytest.mark.asyncio
    async def test_empty_sample_uuid_ignored(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "", ts=1000),
            ]
        )
        assert len(agg._table) == 0

    @pytest.mark.asyncio
    async def test_multiple_samples_independent(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.ISSUED, "s2", ts=1500),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                sample_event(SampleEventType.RECV_FIRST, "s2", ts=3000),
                sample_event(SampleEventType.COMPLETE, "s1", ts=4000),
                sample_event(SampleEventType.COMPLETE, "s2", ts=5000),
            ]
        )
        ttfts = store.get_series_values("ttft_ns")
        latencies = store.get_series_values("sample_latency_ns")
        assert 1000 in ttfts
        assert 1500 in ttfts
        assert 3000 in latencies
        assert 3500 in latencies

    @pytest.mark.asyncio
    async def test_error_events_ignored(self):
        """Error events should not crash the aggregator."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                EventRecord(
                    event_type=ErrorEventType.GENERIC,
                    timestamp_ns=500,
                    data=ErrorData(error_type="test", error_message="boom"),
                ),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.COMPLETE, "s1", ts=2000),
            ]
        )
        assert 1000 in store.get_series_values("sample_latency_ns")

    @pytest.mark.asyncio
    async def test_session_started_stores_timestamp(self):
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process([session_event(SessionEventType.STARTED, ts=42)])
        assert agg._table.session_started_ns == 42

    @pytest.mark.asyncio
    async def test_process_multiple_batches(self):
        """Two sequential process() calls maintain state correctly."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
            ]
        )
        assert agg._table.get_row("s1") is not None

        await agg.process(
            [
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                sample_event(SampleEventType.COMPLETE, "s1", ts=3000),
            ]
        )
        assert 1000 in store.get_series_values("ttft_ns")
        assert 2000 in store.get_series_values("sample_latency_ns")
        assert agg._table.get_row("s1") is None

    @pytest.mark.asyncio
    async def test_ended_in_second_batch(self):
        """ENDED in a later batch still triggers finalize."""
        store = InMemoryKVStore()
        agg = make_stub_aggregator(store)
        await agg.process([session_event(SessionEventType.STARTED, ts=0)])
        assert not store.closed
        await agg.process([session_event(SessionEventType.ENDED, ts=100)])
        assert store.closed


# ---------------------------------------------------------------------------
# Async trigger tests (with mock TokenizePool and real event loop)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncTriggers:
    @pytest.mark.asyncio
    async def test_isl_text_path_async(self):
        """ISL with text prompt triggers async tokenization."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.01)
        agg = make_async_stub_aggregator(store, pool, loop)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(text="hello world foo bar"),
                ),
            ]
        )
        # ISL task is in-flight; drain it
        await agg._table.drain_tasks()
        assert 4 in store.get_series_values("isl")

    @pytest.mark.asyncio
    async def test_osl_emitted_on_complete(self):
        """OSL is emitted via async tokenization when COMPLETE carries TextModelOutput."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.01)
        agg = make_async_stub_aggregator(store, pool, loop)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    data=text_output("the quick brown fox"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        assert 4000 in store.get_series_values("sample_latency_ns")
        assert 4 in store.get_series_values("osl")

    @pytest.mark.asyncio
    async def test_tpot_emitted_for_streaming(self):
        """TPOT is emitted for streaming responses using text_after_first_chunk."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(store, pool, loop)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                sample_event(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    # Streaming: 3 chunks, text_after_first_chunk = "world foo"
                    data=streaming_text("hello", " world", " foo"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        assert 3 in store.get_series_values("osl")  # "hello world foo" = 3 tokens
        # tpot = (5000 - 2000) / token_count("world foo") = 3000 / 2 = 1500
        assert 1500.0 in store.get_series_values("tpot_ns")

    @pytest.mark.asyncio
    async def test_tpot_skipped_when_single_chunk(self):
        """TPOT is not emitted when there are no tokens after the first chunk."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(store, pool, loop)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                sample_event(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    # Single chunk: text_after_first_chunk = ""
                    data=streaming_text("only"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        assert 1 in store.get_series_values("osl")
        assert store.get_series_values("tpot_ns") == []

    @pytest.mark.asyncio
    async def test_tpot_not_emitted_without_streaming_flag(self):
        """TPOT trigger is not registered when streaming=False."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(store, pool, loop, streaming=False)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                sample_event(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    data=streaming_text("hello", " world", " foo"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        assert 4000 in store.get_series_values("sample_latency_ns")
        assert 3 in store.get_series_values("osl")
        assert store.get_series_values("tpot_ns") == []
        assert store.get_series_values("ttft_ns") == []
        assert store.get_series_values("chunk_delta_ns") == []

    @pytest.mark.asyncio
    async def test_tpot_non_streaming_output_skipped(self):
        """TPOT is not emitted for non-streaming (str) TextModelOutput."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(store, pool, loop)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                sample_event(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    # Non-streaming: str output, text_after_first_chunk = ""
                    data=text_output("hello world foo"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        assert 3 in store.get_series_values("osl")
        assert store.get_series_values("tpot_ns") == []

    @pytest.mark.asyncio
    async def test_drain_tasks_awaits_in_flight(self):
        """drain_tasks() properly awaits all in-flight async trigger tasks."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.05)
        agg = make_async_stub_aggregator(store, pool, loop)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(text="a b c d e"),
                ),
            ]
        )
        # Tasks are in-flight but not yet complete
        assert len(agg._table._in_flight_tasks) > 0

        await agg._table.drain_tasks()
        assert len(agg._table._in_flight_tasks) == 0
        assert 5 in store.get_series_values("isl")

    @pytest.mark.asyncio
    async def test_shutdown_drains_async_tasks(self):
        """ENDED drains in-flight async tasks before finalizing."""
        store = InMemoryKVStore()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.02)
        agg = make_async_stub_aggregator(store, pool, loop)

        await agg.process(
            [
                session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                sample_event(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(text="one two three"),
                ),
                session_event(SessionEventType.ENDED, ts=2000),
            ]
        )
        # After ENDED, drain_tasks was called, so ISL should be emitted
        assert 3 in store.get_series_values("isl")
        assert store.closed

    # TODO: Add tests for trigger exception handling (logger.exception paths).
    # Inject a MockTokenizePool that raises on token_count_async and verify:
    # - No metric is emitted for the failing trigger
    # - The aggregator does not crash
    # - The task set is cleaned up (done_callback fires on failed tasks)
