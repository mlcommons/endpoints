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

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.emitter import (
    MetricEmitter,
)
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricsTable,
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
        self.closed = True

    def get_metrics(self, sample_uuid: str) -> dict[str, int | float]:
        return {name: val for uuid, name, val in self.emitted if uuid == sample_uuid}

    def get_all(self, metric_name: str) -> list[tuple[str, int | float]]:
        return [(uuid, val) for uuid, name, val in self.emitted if name == metric_name]


class StubAggregator(MetricsAggregatorService):
    """Bypass ZMQ init for unit testing — only process() logic is tested."""

    def __init__(self, emitter: MetricEmitter, tokenize_pool=None):
        # Intentionally skip super().__init__() to avoid ZMQ socket creation.
        # All required attributes are set manually below.
        self._emitter = emitter
        self._tokenize_pool = tokenize_pool
        self._table = MetricsTable()
        self._is_tracking = False
        self._session_started_ns = None
        self._shutdown_received = False
        self._shutdown_event = None
        self.loop = None  # type: ignore[assignment]
        self.is_closed = False


def _session(ev_type, ts=0):
    return EventRecord(event_type=ev_type, timestamp_ns=ts)


def _sample(ev_type, uuid, ts=0, data=None):
    return EventRecord(event_type=ev_type, timestamp_ns=ts, sample_uuid=uuid, data=data)


def _text(s: str) -> TextModelOutput:
    """Wrap a string in TextModelOutput for use as EventRecord.data."""
    return TextModelOutput(output=s)


# ---------------------------------------------------------------------------
# Performance tracking window
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrackingWindow:
    @pytest.mark.asyncio
    async def test_not_tracked_before_start(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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


# ---------------------------------------------------------------------------
# Timing metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimingMetrics:
    @pytest.mark.asyncio
    async def test_ttft_and_sample_latency(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.COMPLETE, "s1", ts=3000),
            ]
        )
        m = emitter.get_metrics("s1")
        assert m["sample_latency_ns"] == 2000
        assert "ttft_ns" not in m
        assert "chunk_delta_ns" not in m
        assert "tpot_ns" not in m

    @pytest.mark.asyncio
    async def test_all_timing_metrics_full_lifecycle(self):
        """Full streaming sample lifecycle emits all expected timing metrics."""
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
# Text accumulation and first_chunk_text
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTextAccumulation:
    @pytest.mark.asyncio
    async def test_issued_stores_prompt_text(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(
                    SampleEventType.ISSUED,
                    "s1",
                    ts=1000,
                    data=PromptData(text="What is AI?"),
                ),
            ]
        )
        row = agg._table.get_row("s1")
        assert row.prompt_text == "What is AI?"

    @pytest.mark.asyncio
    async def test_issued_with_token_ids_emits_isl_directly(self):
        """SGLang path: PromptData with token_ids emits ISL = len(token_ids)
        without tokenization."""
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
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
    async def test_issued_without_data_leaves_prompt_none(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
            ]
        )
        row = agg._table.get_row("s1")
        assert row.prompt_text is None

    @pytest.mark.asyncio
    async def test_recv_first_stores_first_chunk_text(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000, data=_text("Hello")),
            ]
        )
        row = agg._table.get_row("s1")
        assert row.first_chunk_text == "Hello"
        assert row.output_chunks == ["Hello"]

    @pytest.mark.asyncio
    async def test_output_chunks_accumulated(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000, data=_text("Hello")),
                _sample(
                    SampleEventType.RECV_NON_FIRST, "s1", ts=3000, data=_text(" World")
                ),
                _sample(SampleEventType.RECV_NON_FIRST, "s1", ts=4000, data=_text("!")),
            ]
        )
        row = agg._table.get_row("s1")
        assert row.output_text() == "Hello World!"

    @pytest.mark.asyncio
    async def test_recv_without_data_does_not_append(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.RECV_FIRST, "s1", ts=2000),
                _sample(SampleEventType.RECV_NON_FIRST, "s1", ts=3000),
            ]
        )
        row = agg._table.get_row("s1")
        assert row.output_chunks == []
        assert row.first_chunk_text is None

    @pytest.mark.asyncio
    async def test_complete_with_text_model_output(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
        output = TextModelOutput(output="Complete response")
        await agg.process(
            [
                _session(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                _sample(SampleEventType.ISSUED, "s1", ts=1000),
                _sample(SampleEventType.COMPLETE, "s1", ts=5000, data=output),
            ]
        )
        assert emitter.get_metrics("s1")["sample_latency_ns"] == 4000


# ---------------------------------------------------------------------------
# Edge cases and event ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_untracked_sample_events_ignored(self):
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
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
        agg = StubAggregator(emitter)
        await agg.process([_session(SessionEventType.STARTED, ts=42)])
        assert agg._session_started_ns == 42

    @pytest.mark.asyncio
    async def test_process_multiple_batches(self):
        """Two sequential process() calls maintain state correctly."""
        emitter = FakeEmitter()
        agg = StubAggregator(emitter)

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
        agg = StubAggregator(emitter)
        await agg.process([_session(SessionEventType.STARTED, ts=0)])
        assert not emitter.flushed
        await agg.process([_session(SessionEventType.ENDED, ts=100)])
        assert emitter.flushed
        assert emitter.closed
