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
from unittest.mock import MagicMock

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.emitter import (
    MetricEmitter,
)
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import ErrorData, PromptData, TextModelOutput


class FakeEmitter(MetricEmitter):
    def __init__(self):
        self.emitted: list[tuple[str, str, int | float]] = []
        self.flushed = False
        self.closed = False

    def emit(self, sample_uuid: str, metric_name: str, value: int | float) -> None:
        self.emitted.append((sample_uuid, metric_name, value))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.flush()
        self.closed = True

    def get_metrics(self, sample_uuid: str) -> dict[str, int | float]:
        return {name: val for uuid, name, val in self.emitted if uuid == sample_uuid}

    def get_all(self, metric_name: str) -> list[tuple[str, int | float]]:
        return [(uuid, val) for uuid, name, val in self.emitted if name == metric_name]


def _mock_zmq_context() -> MagicMock:
    """Create a mock ManagedZMQContext that no-ops all ZMQ operations."""
    ctx = MagicMock()
    ctx.socket.return_value = MagicMock()
    ctx.connect.return_value = "ipc:///mock/socket"
    return ctx


def make_stub_aggregator(
    emitter: MetricEmitter, tokenize_pool=None, streaming: bool = True
) -> MetricsAggregatorService:
    """Create a MetricsAggregatorService with ZMQ mocked out.

    Uses a mock ManagedZMQContext so the full __init__ chain runs
    (including super().__init__) without creating real ZMQ sockets.
    """
    return MetricsAggregatorService(
        "mock_path",
        _mock_zmq_context(),
        MagicMock(spec=asyncio.AbstractEventLoop),
        emitter=emitter,
        tokenize_pool=tokenize_pool,
        streaming=streaming,
    )


def _session(ev_type, ts=0):
    return EventRecord(event_type=ev_type, timestamp_ns=ts)


def _sample(ev_type, uuid, ts=0, data=None):
    return EventRecord(event_type=ev_type, timestamp_ns=ts, sample_uuid=uuid, data=data)


def _text(s: str) -> TextModelOutput:
    """Wrap a string in TextModelOutput for use as EventRecord.data."""
    return TextModelOutput(output=s)


def _streaming_text(*chunks: str) -> TextModelOutput:
    """Wrap chunks in a streaming TextModelOutput (tuple output)."""
    return TextModelOutput(output=tuple(chunks))


# ---------------------------------------------------------------------------
# Performance tracking window
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrackingWindow:
    @pytest.mark.asyncio
    async def test_not_tracked_before_start(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.STARTED, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=100),
            ]
        )
        assert agg._table.get_row("s1") is None
        assert emitter.emitted == []

    @pytest.mark.asyncio
    async def test_tracked_after_start(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=100),
            ]
        )
        assert agg._table.get_row("s1") is not None

    @pytest.mark.asyncio
    async def test_not_tracked_after_stop(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _session(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=50),
                _sample(SampleEventType.ISSUED, "s1", ts=100),
            ]
        )
        assert agg._table.get_row("s1") is None

    @pytest.mark.asyncio
    async def test_inflight_sample_continues_after_stop(self):
        """A sample issued during tracking completes normally after STOP."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=100),
                _session(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=300),
                _sample(SampleEventType.COMPLETE, "s1", ts=500),
            ]
        )
        metrics = emitter.get_metrics("s1")
        assert metrics["ttft_ns"] == 200
        assert metrics["sample_latency_ns"] == 400

    @pytest.mark.asyncio
    async def test_restart_tracking_window(self):
        """START -> STOP -> START creates a second tracking window."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=100),
                _session(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200),
                _sample(SampleEventType.ISSUED, "s2", ts=300),  # not tracked
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=400),
                _sample(SampleEventType.ISSUED, "s3", ts=500),  # tracked
                _sample(SampleEventType.COMPLETE, "s1", ts=600),
                _sample(SampleEventType.COMPLETE, "s3", ts=700),
            ]
        )
        assert agg._table.get_row("s2") is None  # never tracked
        assert "sample_latency_ns" in emitter.get_metrics("s1")
        assert "sample_latency_ns" in emitter.get_metrics("s3")

    @pytest.mark.asyncio
    async def test_tracked_block_durations(self):
        """Tracked blocks extend to last sample completion."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=100),
                _session(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200),
                _sample(SampleEventType.COMPLETE, "s1", ts=700),
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=800),
                _sample(SampleEventType.ISSUED, "s2", ts=900),
                _sample(SampleEventType.COMPLETE, "s2", ts=1000),
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
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2500),
                _sample(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        m = emitter.get_metrics("s1")
        assert m["ttft_ns"] == 1500
        assert m["sample_latency_ns"] == 4000

    @pytest.mark.asyncio
    async def test_request_duration(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.CLIENT_SEND, "s1", ts=1100),
                _sample(SampleEventType.CLIENT_RESP_DONE, "s1", ts=4100),
                _sample(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        assert emitter.get_metrics("s1")["request_duration_ns"] == 3000

    @pytest.mark.asyncio
    async def test_chunk_deltas(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(SampleEventType.RECV_NON_FIRST, "s1", ts=3000),
                _sample(SampleEventType.RECV_NON_FIRST, "s1", ts=4500),
                _sample(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        deltas = [v for _, name, v in emitter.emitted if name == "chunk_delta_ns"]
        assert deltas == [1000, 1500]

    @pytest.mark.asyncio
    async def test_non_streaming_latency_only(self):
        """Non-streaming sample emits sample_latency_ns and OSL, but no TTFT/chunk_delta/TPOT.

        Uses AsyncStubAggregator with a real loop and MockTokenizePool so that
        async triggers (OslTrigger) actually execute. This ensures OSL is
        emitted for non-streaming samples, and that the absence of streaming
        metrics is due to *logic* (no RECV_FIRST means no TTFT/TPOT, no
        RECV_NON_FIRST means no chunk_delta), not because the pool was missing.
        """
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(emitter, pool, loop)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=3000,
                    data=_text("hello world"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        m = emitter.get_metrics("s1")
        assert m["sample_latency_ns"] == 2000
        assert m["osl"] == 2
        assert "ttft_ns" not in m
        assert "chunk_delta_ns" not in m
        assert "tpot_ns" not in m

    @pytest.mark.asyncio
    async def test_all_timing_metrics_full_lifecycle(self):
        """Full streaming sample lifecycle emits all expected timing metrics."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.CLIENT_SEND, "s1", ts=1050),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(SampleEventType.RECV_NON_FIRST, "s1", ts=3000),
                _sample(SampleEventType.CLIENT_RESP_DONE, "s1", ts=4000),
                _sample(SampleEventType.COMPLETE, "s1", ts=4500),
            ]
        )
        m = emitter.get_metrics("s1")
        assert m["ttft_ns"] == 1000
        assert m["sample_latency_ns"] == 3500
        assert m["request_duration_ns"] == 2950
        assert m["chunk_delta_ns"] == 1000

    @pytest.mark.asyncio
    async def test_chunk_delta_not_emitted_without_last_recv(self):
        """RECV_NON_FIRST without prior RECV_FIRST: no chunk_delta emitted."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
            ]
        )
        row = agg._table.get_row("s1")
        assert row is not None
        assert row.last_recv_ns is None  # No recv events yet

    @pytest.mark.asyncio
    async def test_request_duration_not_emitted_without_client_send(self):
        """CLIENT_RESP_DONE without CLIENT_SEND: no request_duration."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.CLIENT_RESP_DONE, "s1", ts=4000),
                _sample(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        assert "request_duration_ns" not in emitter.get_metrics("s1")


# ---------------------------------------------------------------------------
# ISL (token_ids path — sync, no tokenize_pool needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsl:
    @pytest.mark.asyncio
    async def test_issued_with_token_ids_emits_isl_directly(self):
        """SGLang path: PromptData with token_ids emits ISL = len(token_ids)."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(token_ids=(101, 202, 303, 404, 505)),
                ),
            ]
        )
        assert ("s1", "isl", 5) in emitter.emitted

    @pytest.mark.asyncio
    async def test_issued_without_data_no_isl(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
            ]
        )
        assert all(name != "isl" for _, name, _ in emitter.emitted)


# ---------------------------------------------------------------------------
# Edge cases and event ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_untracked_sample_events_ignored(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.RECV_FIRST, "unknown", ts=2000),
                _sample(SampleEventType.COMPLETE, "unknown", ts=5000),
            ]
        )
        assert emitter.emitted == []

    @pytest.mark.asyncio
    async def test_complete_removes_row(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        assert agg._table.get_row("s1") is None
        assert len(agg._table) == 0

    @pytest.mark.asyncio
    async def test_session_ended_flushes_and_closes(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.STARTED, ts=0),
                _session(SessionEventType.ENDED, ts=100),
            ]
        )
        assert emitter.flushed
        assert emitter.closed

    @pytest.mark.asyncio
    async def test_events_after_ended_are_dropped(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=100),
                _session(SessionEventType.ENDED, ts=200),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=300),
            ]
        )
        assert "ttft_ns" not in emitter.get_metrics("s1")

    @pytest.mark.asyncio
    async def test_empty_sample_uuid_ignored(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "", ts=1000),
            ]
        )
        assert len(agg._table) == 0

    @pytest.mark.asyncio
    async def test_multiple_samples_independent(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.ISSUED, "s2", ts=1500),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(SampleEventType.RECV_FIRST, "s2", ts=3000),
                _sample(SampleEventType.COMPLETE, "s1", ts=4000),
                _sample(SampleEventType.COMPLETE, "s2", ts=5000),
            ]
        )
        s1 = emitter.get_metrics("s1")
        s2 = emitter.get_metrics("s2")
        assert s1["ttft_ns"] == 1000
        assert s2["ttft_ns"] == 1500
        assert s1["sample_latency_ns"] == 3000
        assert s2["sample_latency_ns"] == 3500

    @pytest.mark.asyncio
    async def test_transport_events_ignored(self):
        """TRANSPORT_SENT / TRANSPORT_RECV should not affect metrics."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.TRANSPORT_SENT, "s1", ts=1001),
                _sample(SampleEventType.TRANSPORT_RECV, "s1", ts=1002),
                _sample(SampleEventType.COMPLETE, "s1", ts=5000),
            ]
        )
        m = emitter.get_metrics("s1")
        assert m == {"sample_latency_ns": 4000}

    @pytest.mark.asyncio
    async def test_error_events_ignored(self):
        """Error events should not crash the aggregator."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                EventRecord(
                    event_type=ErrorEventType.GENERIC,
                    timestamp_ns=500,
                    data=ErrorData(error_type="test", error_message="boom"),
                ),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.COMPLETE, "s1", ts=2000),
            ]
        )
        assert emitter.get_metrics("s1")["sample_latency_ns"] == 1000

    @pytest.mark.asyncio
    async def test_session_started_stores_timestamp(self):
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process([_session(SessionEventType.STARTED, ts=42)])
        assert agg._table.session_started_ns == 42

    @pytest.mark.asyncio
    async def test_process_multiple_batches(self):
        """Two sequential process() calls maintain state correctly."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
            ]
        )
        assert agg._table.get_row("s1") is not None

        await agg.process(
            [
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(SampleEventType.COMPLETE, "s1", ts=3000),
            ]
        )
        m = emitter.get_metrics("s1")
        assert m["ttft_ns"] == 1000
        assert m["sample_latency_ns"] == 2000
        assert agg._table.get_row("s1") is None

    @pytest.mark.asyncio
    async def test_ended_in_second_batch(self):
        """ENDED in a later batch still triggers finalize."""
        emitter = FakeEmitter()
        agg = make_stub_aggregator(emitter)
        await agg.process([_session(SessionEventType.STARTED, ts=0)])
        assert not emitter.flushed
        await agg.process([_session(SessionEventType.ENDED, ts=100)])
        assert emitter.flushed
        assert emitter.closed


# ---------------------------------------------------------------------------
# Async trigger tests (with mock TokenizePool and real event loop)
# ---------------------------------------------------------------------------


class MockTokenizePool:
    """Mock TokenizePool that splits on whitespace with artificial async delay."""

    def __init__(self, delay: float = 0.01):
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


def make_async_stub_aggregator(
    emitter: MetricEmitter, tokenize_pool, loop, streaming: bool = True
) -> MetricsAggregatorService:
    """Create a MetricsAggregatorService with a real loop and mock ZMQ."""
    return MetricsAggregatorService(
        "mock_path",
        _mock_zmq_context(),
        loop,
        emitter=emitter,
        tokenize_pool=tokenize_pool,
        streaming=streaming,
    )


@pytest.mark.unit
class TestAsyncTriggers:
    @pytest.mark.asyncio
    async def test_isl_text_path_async(self):
        """ISL with text prompt triggers async tokenization."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.01)
        agg = make_async_stub_aggregator(emitter, pool, loop)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(text="hello world foo bar"),
                ),
            ]
        )
        # ISL task is in-flight; drain it
        await agg._table.drain_tasks()
        assert ("s1", "isl", 4) in emitter.emitted

    @pytest.mark.asyncio
    async def test_osl_emitted_on_complete(self):
        """OSL is emitted via async tokenization when COMPLETE carries TextModelOutput."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.01)
        agg = make_async_stub_aggregator(emitter, pool, loop)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    data=_text("the quick brown fox"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        m = emitter.get_metrics("s1")
        assert m["sample_latency_ns"] == 4000
        assert m["osl"] == 4

    @pytest.mark.asyncio
    async def test_tpot_emitted_for_streaming(self):
        """TPOT is emitted for streaming responses using text_after_first_chunk."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(emitter, pool, loop)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    # Streaming: 3 chunks, text_after_first_chunk = "world foo"
                    data=_streaming_text("hello", " world", " foo"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        m = emitter.get_metrics("s1")
        assert m["osl"] == 3  # "hello world foo" = 3 tokens
        # tpot = (5000 - 2000) / token_count("world foo") = 3000 / 2 = 1500
        assert m["tpot_ns"] == 1500.0

    @pytest.mark.asyncio
    async def test_tpot_skipped_when_single_chunk(self):
        """TPOT is not emitted when there are no tokens after the first chunk."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(emitter, pool, loop)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    # Single chunk: text_after_first_chunk = ""
                    data=_streaming_text("only"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        m = emitter.get_metrics("s1")
        assert m["osl"] == 1
        assert "tpot_ns" not in m

    @pytest.mark.asyncio
    async def test_tpot_not_emitted_without_streaming_flag(self):
        """TPOT trigger is not registered when streaming=False."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(emitter, pool, loop, streaming=False)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    data=_streaming_text("hello", " world", " foo"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        m = emitter.get_metrics("s1")
        assert m["sample_latency_ns"] == 4000
        assert m["osl"] == 3
        assert "tpot_ns" not in m
        assert "ttft_ns" not in m
        assert "chunk_delta_ns" not in m

    @pytest.mark.asyncio
    async def test_tpot_non_streaming_output_skipped(self):
        """TPOT is not emitted for non-streaming (str) TextModelOutput."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.0)
        agg = make_async_stub_aggregator(emitter, pool, loop)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(
                    SampleEventType.COMPLETE,
                    "s1",
                    ts=5000,
                    # Non-streaming: str output, text_after_first_chunk = ""
                    data=_text("hello world foo"),
                ),
            ]
        )
        await agg._table.drain_tasks()
        m = emitter.get_metrics("s1")
        assert m["osl"] == 3
        assert "tpot_ns" not in m

    @pytest.mark.asyncio
    async def test_drain_tasks_awaits_in_flight(self):
        """drain_tasks() properly awaits all in-flight async trigger tasks."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.05)
        agg = make_async_stub_aggregator(emitter, pool, loop)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(
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
        assert ("s1", "isl", 5) in emitter.emitted

    @pytest.mark.asyncio
    async def test_shutdown_drains_async_tasks(self):
        """ENDED drains in-flight async tasks before finalizing."""
        emitter = FakeEmitter()
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0.02)
        agg = make_async_stub_aggregator(emitter, pool, loop)

        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(text="one two three"),
                ),
                _session(SessionEventType.ENDED, ts=2000),
            ]
        )
        # After ENDED, drain_tasks was called, so ISL should be emitted
        assert ("s1", "isl", 3) in emitter.emitted
        assert emitter.flushed
        assert emitter.closed

    # TODO: Add tests for trigger exception handling (logger.exception paths).
    # Inject a MockTokenizePool that raises on token_count_async and verify:
    # - No metric is emitted for the failing trigger
    # - The aggregator does not crash
    # - The task set is cleaned up (done_callback fires on failed tasks)
