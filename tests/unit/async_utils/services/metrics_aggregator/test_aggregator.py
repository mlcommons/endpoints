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

"""Tests for ``MetricsAggregatorService.process()``.

Events are injected directly via ``await agg.process([...])``; emitted
metrics are inspected by reading the ``MetricsRegistry``'s snapshot
output. The aggregator is constructed with a real SUB socket (so the
``ZmqMessageSubscriber`` base initializes cleanly) and a mocked
``MetricsPublisher`` (so ``STARTED``/``ENDED`` paths don't touch real
I/O).
"""

from __future__ import annotations

import asyncio
import logging

import msgspec
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricCounterKey,
    MetricSeriesKey,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    SeriesStat,
    SessionState,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import (
    ErrorData,
    PhaseData,
    PhaseType,
    PromptData,
    TextModelOutput,
    TokenCountSource,
    basis_key,
)
from inference_endpoint.metrics.early_stopping import EarlyStoppingSpec
from inference_endpoint.metrics.token_count_basis import (
    ServerUsage,
    missing_usage_message,
)

from tests.test_helpers import DEFAULT_SOURCE, SERVER_USAGE_CASES

from .conftest import (
    MockBatchTokenizer,
    aggregator_scope,
    complete_event,
    make_aggregator,
    sample_event,
    session_event,
    snapshot_counters,
    snapshot_series_count,
    snapshot_series_total,
    streaming_text,
    text_output,
)

CLIENT, SERVER = TokenCountSource.CLIENT, TokenCountSource.SERVER


def _name(
    key: MetricSeriesKey, basis: TokenCountSource, *, primary: TokenCountSource
) -> str:
    """``key``'s series name on ``basis`` in a run whose primary is ``primary``."""
    return basis_key(key.value, basis, primary=primary)


# ---------------------------------------------------------------------------
# Performance tracking window
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrackingWindow:
    @pytest.mark.asyncio
    async def test_not_tracked_before_start(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_not_tracked_before", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(SessionEventType.STARTED, ts=0),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                    ]
                )
                assert agg._table.get_row("s1") is None, (
                    "Sample issued before START_PERFORMANCE_TRACKING must "
                    "not create a table row — warmup samples should be "
                    "excluded from the tracked set."
                )
                assert (
                    snapshot_series_count(registry, MetricSeriesKey.TTFT_NS.value) == 0
                )
                assert (
                    snapshot_series_count(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 0
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_tracked_after_start(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx, loop, "agg_tracked_after_start", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                    ]
                )
                assert agg._table.get_row("s1") is not None
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_not_tracked_after_stop(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx,
                loop,
                "agg_not_tracked_after_stop",
                token_count_source=DEFAULT_SOURCE,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        session_event(
                            SessionEventType.STOP_PERFORMANCE_TRACKING, ts=50
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                    ]
                )
                assert agg._table.get_row("s1") is None
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_inflight_sample_continues_after_stop(self, tmp_path):
        """A sample issued during tracking completes normally after STOP."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_inflight_after_stop", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                        session_event(
                            SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200
                        ),
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=300),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=500),
                    ]
                )
                # ttft = 300 - 100 = 200, sample_latency = 500 - 100 = 400
                assert (
                    snapshot_series_total(registry, MetricSeriesKey.TTFT_NS.value)
                    == 200
                )
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 400
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_restart_tracking_window(self, tmp_path):
        """START -> STOP -> START creates a second tracking window."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_restart_tracking", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                        session_event(
                            SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200
                        ),
                        # not tracked
                        sample_event(SampleEventType.ISSUED, "s2", ts=300),
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=400
                        ),
                        # tracked
                        sample_event(SampleEventType.ISSUED, "s3", ts=500),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=600),
                        sample_event(SampleEventType.COMPLETE, "s3", ts=700),
                    ]
                )
                # s2 was never tracked
                assert agg._table.get_row("s2") is None
                # Two completed samples (s1 and s3) emitted sample_latency_ns.
                assert (
                    snapshot_series_count(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 2
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_tracked_block_durations(self, tmp_path):
        """Tracked blocks extend to last sample completion."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx, loop, "agg_tracked_block_dur", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                        session_event(
                            SessionEventType.STOP_PERFORMANCE_TRACKING, ts=200
                        ),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=700),
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=800
                        ),
                        sample_event(SampleEventType.ISSUED, "s2", ts=900),
                        sample_event(SampleEventType.COMPLETE, "s2", ts=1000),
                    ]
                )
                assert agg._table.tracked_blocks[0].duration_ns == 700  # 700 - 0
                assert agg._table.tracked_blocks[1].duration_ns == 200  # 1000 - 800
                assert agg._table.total_tracked_duration_ns == 900
                assert agg._table.total_completed_tracked_samples == 2
            finally:
                agg.close()


# ---------------------------------------------------------------------------
# Timing metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimingMetrics:
    @pytest.mark.asyncio
    async def test_ttft_and_sample_latency(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_ttft_latency", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=2500),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=5000),
                    ]
                )
                # ttft = 2500-1000 = 1500
                # sample_latency = 5000-1000 = 4000
                assert (
                    snapshot_series_total(registry, MetricSeriesKey.TTFT_NS.value)
                    == 1500
                )
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 4000
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_chunk_deltas(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_chunk_deltas", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                        sample_event(SampleEventType.RECV_NON_FIRST, "s1", ts=3000),
                        sample_event(SampleEventType.RECV_NON_FIRST, "s1", ts=4500),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=5000),
                    ]
                )
                # chunk_delta_ns is emitted on each RECV_NON_FIRST: 3000-2000=1000 and
                # 4500-3000=1500.
                assert (
                    snapshot_series_count(
                        registry, MetricSeriesKey.CHUNK_DELTA_NS.value
                    )
                    == 2
                )
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.CHUNK_DELTA_NS.value
                    )
                    == 2500
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_non_streaming_latency_only(self, tmp_path, primary):
        """Non-streaming: emits sample_latency_ns + OSL, no TTFT/chunk_delta/TPOT."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.0)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_non_streaming",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(
                            SampleEventType.COMPLETE,
                            "s1",
                            ts=3000,
                            data=text_output("hello world"),
                        ),
                    ]
                )
                await agg._token_queue.drain_all()
                # sample_latency = 3000-1000 = 2000
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 2000
                )
                # OSL = token_count("hello world") = 2
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 2
                )
                assert (
                    snapshot_series_count(registry, MetricSeriesKey.TTFT_NS.value) == 0
                )
                assert (
                    snapshot_series_count(
                        registry, MetricSeriesKey.CHUNK_DELTA_NS.value
                    )
                    == 0
                )
                for basis in TokenCountSource:
                    tpot = _name(MetricSeriesKey.TPOT_NS, basis, primary=primary)
                    assert snapshot_series_count(registry, tpot) == 0
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_chunk_delta_not_emitted_without_last_recv(self, tmp_path):
        """RECV_NON_FIRST without prior RECV_FIRST: no chunk_delta emitted."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_chunk_delta_no_recv", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                    ]
                )
                row = agg._table.get_row("s1")
                assert row is not None
                assert row.last_recv_ns is None
            finally:
                agg.close()


# ---------------------------------------------------------------------------
# ISL (token_ids path -- sync, no tokenizer needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsl:
    @pytest.mark.asyncio
    async def test_issued_with_token_ids_emits_isl_directly(self, tmp_path, primary):
        """SGLang path: PromptData with token_ids emits ISL = len(token_ids)."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_isl_token_ids", token_count_source=primary
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(token_ids=(101, 202, 303, 404, 505)),
                        ),
                    ]
                )
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.ISL, CLIENT, primary=primary)
                    )
                    == 5
                )
                # Server ISL exists only on COMPLETE.
                server_isl = _name(MetricSeriesKey.ISL, SERVER, primary=primary)
                assert snapshot_series_count(registry, server_isl) == 0
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_issued_without_data_no_isl(self, tmp_path, primary):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_isl_no_data", token_count_source=primary
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                    ]
                )
                for basis in TokenCountSource:
                    isl = _name(MetricSeriesKey.ISL, basis, primary=primary)
                    assert snapshot_series_count(registry, isl) == 0
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_disabled_isl_skips_prompt_tokenization_keeps_osl(
        self, tmp_path, primary
    ):
        """enable_isl=False: no ISL series on either basis, no prompt enqueued;
        OSL/TPOT intact."""
        loop = asyncio.get_event_loop()
        messages = (
            {"role": "system", "content": "follow instructions"},
            {"role": "user", "content": "summarize this"},
        )
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_isl_disabled",
                token_count_source=primary,
                tokenizer=MockBatchTokenizer(),
                enable_isl=False,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(messages=messages),
                        ),
                    ]
                )
                for basis in TokenCountSource:
                    isl = _name(MetricSeriesKey.ISL, basis, primary=primary)
                    assert not registry.has_series(isl)
                assert agg.pending_tokens == 0

                await agg.process(
                    [
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                        sample_event(
                            SampleEventType.COMPLETE,
                            "s1",
                            ts=5000,
                            data=streaming_text("hello", " world", " foo"),
                        ),
                        session_event(SessionEventType.ENDED, ts=5000),
                    ]
                )
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 3
                )
                assert snapshot_series_total(
                    registry, _name(MetricSeriesKey.TPOT_NS, CLIENT, primary=primary)
                ) == pytest.approx(1500.0)
            finally:
                agg.close()


# ---------------------------------------------------------------------------
# Edge cases and event ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_untracked_sample_events_ignored(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_untracked_ignored", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.RECV_FIRST, "unknown", ts=2000),
                        sample_event(SampleEventType.COMPLETE, "unknown", ts=5000),
                    ]
                )
                assert (
                    snapshot_series_count(registry, MetricSeriesKey.TTFT_NS.value) == 0
                )
                assert (
                    snapshot_series_count(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 0
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_duplicate_started_logs_error_and_preserves_state(
        self, tmp_path, caplog
    ):
        """A duplicate ``STARTED`` event is a producer bug.

        The aggregator MUST NOT re-assign ``_session_start_ns`` on a
        second STARTED — doing so freezes ``total_duration_ns`` for the
        rest of the run (the max-of-elapsed guard never beats the new
        smaller deltas). Verify the error is logged AND the original
        start timestamp is preserved.
        """
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx, loop, "agg_dup_started", token_count_source=DEFAULT_SOURCE
            )
            try:
                with caplog.at_level("ERROR"):
                    await agg.process(
                        [
                            session_event(SessionEventType.STARTED, ts=1_000),
                            session_event(SessionEventType.STARTED, ts=5_000),
                        ]
                    )
                # Original start timestamp must be preserved.
                assert agg._session_start_ns == 1_000
                # And an error must have been logged with both timestamps.
                error_records = [
                    r for r in caplog.records if "Duplicate STARTED" in r.message
                ]
                assert (
                    len(error_records) == 1
                ), "duplicate STARTED must log exactly one error"
                assert "1000" in error_records[0].getMessage()
                assert "5000" in error_records[0].getMessage()
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_complete_removes_row(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx, loop, "agg_complete_removes", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=5000),
                    ]
                )
                assert agg._table.get_row("s1") is None
                assert len(agg._table) == 0
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_session_ended_calls_publish_final(self, tmp_path):
        """ENDED triggers ``publish_final`` and ``close`` on the publisher."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, publisher = make_aggregator(
                ctx, loop, "agg_ended_publish_final", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(SessionEventType.STARTED, ts=0),
                        session_event(SessionEventType.ENDED, ts=100),
                    ]
                )
                publisher.publish_final.assert_awaited_once()
                assert publisher.publish_final.await_args.kwargs["interrupted"] is False
                # close() is invoked twice: once explicitly in the ENDED branch,
                # then again from the aggregator's ``close`` (via _finalize).
                assert publisher.close.call_count >= 1
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_interrupted_marker_tags_final_snapshot(self, tmp_path):
        """The session's INTERRUPTED marker (published just before ENDED on an
        aborted run) must flow through to ``publish_final(interrupted=True)``
        so the ENDED-driven finalize writes state=interrupted — a ^C'd run
        must never produce a COMPLETE final snapshot."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, publisher = make_aggregator(
                ctx, loop, "agg_interrupted_marker", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(SessionEventType.STARTED, ts=0),
                        session_event(SessionEventType.INTERRUPTED, ts=50),
                        session_event(SessionEventType.ENDED, ts=100),
                    ]
                )
                publisher.publish_final.assert_awaited_once()
                assert publisher.publish_final.await_args.kwargs["interrupted"] is True
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_events_after_ended_are_dropped(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_events_after_ended", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                        session_event(SessionEventType.ENDED, ts=200),
                        # Should be dropped — the aggregator stops processing
                        # at the ENDED record.
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=300),
                    ]
                )
                assert (
                    snapshot_series_count(registry, MetricSeriesKey.TTFT_NS.value) == 0
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_empty_sample_uuid_ignored(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx, loop, "agg_empty_uuid", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "", ts=1000),
                    ]
                )
                assert len(agg._table) == 0
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_multiple_samples_independent(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_multi_independent", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.ISSUED, "s2", ts=1500),
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                        sample_event(SampleEventType.RECV_FIRST, "s2", ts=3000),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=4000),
                        sample_event(SampleEventType.COMPLETE, "s2", ts=5000),
                    ]
                )
                # ttft: s1 = 1000, s2 = 1500
                # sample_latency: s1 = 3000, s2 = 3500
                assert (
                    snapshot_series_count(registry, MetricSeriesKey.TTFT_NS.value) == 2
                )
                assert (
                    snapshot_series_total(registry, MetricSeriesKey.TTFT_NS.value)
                    == 1000 + 1500
                )
                assert (
                    snapshot_series_count(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 2
                )
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 3000 + 3500
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_error_event_increments_total_failed(self, tmp_path):
        """ERROR for an untracked event increments TOTAL_SAMPLES_FAILED only.

        Tracked-failed paths are covered by ``test_aggregator_error_handler.py``;
        here we just confirm the error doesn't crash the loop and the rest of
        the batch processes normally.
        """
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_error_total", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        EventRecord(
                            event_type=ErrorEventType.GENERIC,
                            timestamp_ns=500,
                            data=ErrorData(error_type="test", error_message="boom"),
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=2000),
                    ]
                )
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 1000
                )
                counters = snapshot_counters(registry)
                assert counters[MetricCounterKey.TOTAL_SAMPLES_FAILED.value] == 1
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_session_started_stores_timestamp(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx, loop, "agg_started_ts", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process([session_event(SessionEventType.STARTED, ts=42)])
                assert agg._table.session_started_ns == 42
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_process_multiple_batches(self, tmp_path):
        """Two sequential process() calls maintain state correctly."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_multi_batch", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
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
                assert (
                    snapshot_series_total(registry, MetricSeriesKey.TTFT_NS.value)
                    == 1000
                )
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 2000
                )
                assert agg._table.get_row("s1") is None
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_ended_in_second_batch(self, tmp_path):
        """ENDED in a later batch still triggers finalize."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, publisher = make_aggregator(
                ctx, loop, "agg_ended_second_batch", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process([session_event(SessionEventType.STARTED, ts=0)])
                publisher.publish_final.assert_not_awaited()
                await agg.process([session_event(SessionEventType.ENDED, ts=100)])
                publisher.publish_final.assert_awaited_once()
            finally:
                agg.close()


# ---------------------------------------------------------------------------
# LoadGen window aggregation (end-to-end through the event router)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadgenWindowAggregation:
    @pytest.mark.asyncio
    async def test_loadgen_window_duration_emitted(self, tmp_path):
        """The aggregator emits ``legacy_loadgen_window_duration_ns`` = first
        issue to the completion of the last-issued request.

        Sequence: STARTED, START_PERFORMANCE_TRACKING, ISSUED(s1, t=100),
        COMPLETE(s1, t=500), ISSUED(s2, t=200, last-issued), COMPLETE(s2,
        t=600), STOP_PERFORMANCE_TRACKING. Window = 600 - 100 = 500.
        """
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_loadgen_window", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(SessionEventType.STARTED, ts=0),
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=10
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=500),
                        sample_event(SampleEventType.ISSUED, "s2", ts=200),
                        sample_event(SampleEventType.COMPLETE, "s2", ts=600),
                        session_event(
                            SessionEventType.STOP_PERFORMANCE_TRACKING, ts=700
                        ),
                    ]
                )
                counters = snapshot_counters(registry)
                assert (
                    counters[MetricCounterKey.LEGACY_LOADGEN_WINDOW_DURATION_NS.value]
                    == 600 - 100
                )
            finally:
                agg.close()


# ---------------------------------------------------------------------------
# Counter accounting (issued / completed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCounterAccounting:
    @pytest.mark.asyncio
    async def test_total_vs_tracked_counters(self, tmp_path):
        """Untracked ISSUED counts toward ``total_samples_issued`` but not
        ``tracked_samples_issued``; same for COMPLETED.
        """
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_total_vs_tracked", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(SessionEventType.STARTED, ts=0),
                        # Untracked: warmup ISSUED before START_PERFORMANCE_TRACKING.
                        sample_event(SampleEventType.ISSUED, "warmup", ts=10),
                        sample_event(SampleEventType.COMPLETE, "warmup", ts=20),
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=30
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=100),
                        sample_event(SampleEventType.COMPLETE, "s1", ts=200),
                    ]
                )
                counters = snapshot_counters(registry)
                # Both samples count toward total.
                assert counters[MetricCounterKey.TOTAL_SAMPLES_ISSUED.value] == 2
                assert counters[MetricCounterKey.TOTAL_SAMPLES_COMPLETED.value] == 2
                # Only s1 was tracked.
                assert counters[MetricCounterKey.TRACKED_SAMPLES_ISSUED.value] == 1
                assert counters[MetricCounterKey.TRACKED_SAMPLES_COMPLETED.value] == 1
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_finish_reason_counters_bucket_unknown_values(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx, loop, "agg_finish_reasons", token_count_source=DEFAULT_SOURCE
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1),
                        EventRecord(
                            event_type=SampleEventType.COMPLETE,
                            timestamp_ns=2,
                            sample_uuid="s1",
                            finish_reason="stop",
                        ),
                        sample_event(SampleEventType.ISSUED, "s2", ts=3),
                        EventRecord(
                            event_type=SampleEventType.COMPLETE,
                            timestamp_ns=4,
                            sample_uuid="s2",
                            finish_reason="future_reason",
                        ),
                    ]
                )

                counters = snapshot_counters(registry)
                assert counters[MetricCounterKey.TRACKED_FINISH_REASON_STOP.value] == 1
                assert counters[MetricCounterKey.TRACKED_FINISH_REASON_OTHER.value] == 1
            finally:
                agg.close()


# ---------------------------------------------------------------------------
# Token trigger tests (with mock BatchTokenizer and real event loop)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncTriggers:
    @pytest.mark.asyncio
    async def test_isl_text_path_async(self, tmp_path, primary):
        """ISL with text prompt triggers async tokenization."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.01)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_isl_text_async",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(text="hello world foo bar"),
                        ),
                    ]
                )
                # ISL task is in-flight; drain it
                await agg._token_queue.drain_all()
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.ISL, CLIENT, primary=primary)
                    )
                    == 4
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_isl_structured_prompt_includes_reasoning_and_tools(
        self, tmp_path, primary
    ):
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer()
        messages = (
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "reasoning_content": "think carefully",
                "content": None,
                "tool_calls": ({"function": {"name": "lookup"}},),
            },
            {"role": "tool", "content": "tool result"},
        )
        tools = ({"function": {"name": "lookup"}},)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_isl_structured",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(messages=messages, tools=tools),
                        ),
                    ]
                )
                await agg._token_queue.drain_all()
                assert (
                    snapshot_series_count(
                        registry, _name(MetricSeriesKey.ISL, CLIENT, primary=primary)
                    )
                    == 1
                )
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.ISL, CLIENT, primary=primary)
                    )
                    > 5
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_osl_emitted_on_complete(self, tmp_path, primary):
        """OSL is emitted via async tokenization when COMPLETE carries text."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.01)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_osl_complete",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(
                            SampleEventType.COMPLETE,
                            "s1",
                            ts=5000,
                            data=text_output("the quick brown fox"),
                        ),
                    ]
                )
                await agg._token_queue.drain_all()
                # sample_latency_ns = 5000-1000 = 4000
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 4000
                )
                # OSL = 4 tokens
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 4
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_tpot_emitted_for_streaming(self, tmp_path, primary):
        """TPOT is emitted for streaming responses using text_after_first_chunk."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.0)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_tpot_streaming",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                        sample_event(
                            SampleEventType.COMPLETE,
                            "s1",
                            ts=5000,
                            # Streaming: text_after_first_chunk = "world foo"
                            data=streaming_text("hello", " world", " foo"),
                        ),
                    ]
                )
                await agg._token_queue.drain_all()
                # OSL = "hello world foo" = 3 tokens
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 3
                )
                # tpot = (5000 - 2000) / token_count("world foo") = 3000 / 2 = 1500
                assert snapshot_series_total(
                    registry, _name(MetricSeriesKey.TPOT_NS, CLIENT, primary=primary)
                ) == pytest.approx(1500.0)
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_tpot_skipped_when_single_chunk(self, tmp_path, primary):
        """TPOT is not emitted when there are no tokens after the first chunk."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.0)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_tpot_single_chunk",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
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
                await agg._token_queue.drain_all()
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 1
                )
                assert (
                    snapshot_series_count(
                        registry,
                        _name(MetricSeriesKey.TPOT_NS, CLIENT, primary=primary),
                    )
                    == 0
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_tpot_not_emitted_without_streaming_flag(self, tmp_path, primary):
        """When ``streaming=False``, TPOT/TTFT/chunk_delta series are NOT
        registered at all — the aggregator's snapshot has no entry for them.
        """
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.0)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_tpot_no_streaming",
                tokenizer=tokenizer,
                streaming=False,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
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
                await agg._token_queue.drain_all()
                # sample_latency / OSL still emitted in non-streaming mode.
                assert (
                    snapshot_series_total(
                        registry, MetricSeriesKey.SAMPLE_LATENCY_NS.value
                    )
                    == 4000
                )
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 3
                )
                # The streaming-only series are unregistered, on both bases.
                for basis in TokenCountSource:
                    tpot = _name(MetricSeriesKey.TPOT_NS, basis, primary=primary)
                    assert not registry.has_series(tpot)
                assert not registry.has_series(MetricSeriesKey.TTFT_NS.value)
                assert not registry.has_series(MetricSeriesKey.CHUNK_DELTA_NS.value)
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_tpot_non_streaming_output_skipped(self, tmp_path, primary):
        """TPOT is not emitted for non-streaming (str) TextModelOutput."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.0)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_tpot_str_output",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                        sample_event(
                            SampleEventType.COMPLETE,
                            "s1",
                            ts=5000,
                            # Non-streaming str output: text_after_first_chunk = ""
                            data=text_output("hello world foo"),
                        ),
                    ]
                )
                await agg._token_queue.drain_all()
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 3
                )
                assert (
                    snapshot_series_count(
                        registry,
                        _name(MetricSeriesKey.TPOT_NS, CLIENT, primary=primary),
                    )
                    == 0
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_started_arms_the_live_flush_loop(self, tmp_path):
        """STARTED starts the queue's live loop when an interval is set."""
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx,
                loop,
                "agg_live_arm",
                token_count_source=DEFAULT_SOURCE,
                tokenizer=MockBatchTokenizer(),
                live_flush_interval_s=0.01,
            )
            try:
                await agg.process([session_event(SessionEventType.STARTED, ts=0)])
                assert agg._token_queue is not None
                assert agg._token_queue._live_task is not None
                await agg.process([session_event(SessionEventType.ENDED, ts=100)])
                assert (
                    agg._token_queue._live_task is None
                ), "drain must stop the live loop"
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_flush_records_buffered_tokenizations(self, tmp_path, primary):
        """fire() buffers tokenization; flush() tokenizes the batch and records."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_flush_records",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(text="a b c d e"),
                        ),
                    ]
                )
                assert agg._token_queue is not None
                # Enqueued by fire(), not yet tokenized (no tick/drain flush).
                assert agg._token_queue.pending > 0

                await agg._token_queue.drain_all()
                assert agg._token_queue.pending == 0
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.ISL, CLIENT, primary=primary)
                    )
                    == 5
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_shutdown_flushes_buffered_tokenizations(self, tmp_path, primary):
        """ENDED flushes buffered tokenizations before finalizing."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.02)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, publisher = make_aggregator(
                ctx,
                loop,
                "agg_shutdown_drain",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(text="one two three"),
                        ),
                        session_event(SessionEventType.ENDED, ts=2000),
                    ]
                )
                # After ENDED, flush_remaining ran inside process() — ISL emitted.
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.ISL, CLIENT, primary=primary)
                    )
                    == 3
                )
                publisher.publish_final.assert_awaited_once()
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_drain_failure_reports_pending_and_finalizes(self, tmp_path):
        """A tokenizer error during the ENDED drain must not skip finalize.

        flush_remaining swallows non-timeout failures and returns the stuck
        count, so publish_final still runs with n_pending_tasks > 0 (incomplete
        drain) instead of the error escaping process() and hanging main().
        """
        loop = asyncio.get_event_loop()

        class FailingBatchTokenizer:
            async def count_batch_async(self, inputs, _loop, live=False):
                raise RuntimeError("tokenizer backend died")

        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, publisher = make_aggregator(
                ctx,
                loop,
                "agg_drain_failure",
                token_count_source=DEFAULT_SOURCE,
                tokenizer=FailingBatchTokenizer(),
            )
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(text="some text to tokenize"),
                        ),
                    ]
                )
                assert agg._token_queue is not None
                assert agg._token_queue.pending > 0
                await agg.process([session_event(SessionEventType.ENDED, ts=2000)])

                publisher.publish_final.assert_awaited_once()
                assert publisher.publish_final.await_args.kwargs["n_pending_tasks"] > 0
                publisher.aclose.assert_awaited_once()
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_drain_timeout_reports_pending_count(self, tmp_path):
        """On drain timeout, publish_final must receive n_pending_tasks > 0.

        AGENTS.md and the ``MetricsSnapshot.n_pending_tasks`` docstring
        document the consumer contract: a drain-timeout run is detected
        downstream as ``state == COMPLETE and n_pending_tasks > 0``. If
        the producer always reports 0 here, the timeout is silently
        rebadged as a clean run and the Report shows no warning.
        """
        loop = asyncio.get_event_loop()

        class BlockingBatchTokenizer:
            async def count_batch_async(self, inputs, _loop, live=False):
                await asyncio.sleep(10.0)  # exceeds drain timeout
                return [0] * len(inputs)

        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, publisher = make_aggregator(
                ctx,
                loop,
                "agg_drain_timeout",
                token_count_source=DEFAULT_SOURCE,
                tokenizer=BlockingBatchTokenizer(),
            )
            agg._drain_timeout_s = 0.05
            try:
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(
                            SampleEventType.ISSUED,
                            "s1",
                            ts=1000,
                            data=PromptData(text="some text to tokenize"),
                        ),
                    ]
                )
                assert agg._token_queue is not None
                assert (
                    agg._token_queue.pending > 0
                ), "precondition: ISL must be buffered before ENDED"
                await agg.process([session_event(SessionEventType.ENDED, ts=2000)])

                publisher.publish_final.assert_awaited_once()
                kwargs = publisher.publish_final.await_args.kwargs
                assert kwargs["n_pending_tasks"] > 0, (
                    f"drain timeout must report stuck tasks; got "
                    f"n_pending_tasks={kwargs['n_pending_tasks']}"
                )
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_tpot_osl_for_tool_call_complete(self, tmp_path, primary):
        """OSL and TPOT use message-path tokenization when COMPLETE carries tool_calls."""
        loop = asyncio.get_event_loop()
        tokenizer = MockBatchTokenizer(delay=0.0)
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, registry, _ = make_aggregator(
                ctx,
                loop,
                "agg_tpot_osl_tool_call",
                tokenizer=tokenizer,
                token_count_source=primary,
            )
            try:
                tool_call = {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
                await agg.process(
                    [
                        session_event(
                            SessionEventType.START_PERFORMANCE_TRACKING, ts=0
                        ),
                        sample_event(SampleEventType.ISSUED, "s1", ts=1000),
                        sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
                        sample_event(
                            SampleEventType.COMPLETE,
                            "s1",
                            ts=5000,
                            data=TextModelOutput(
                                output=("ok",), tool_calls=((), (tool_call,))
                            ),
                        ),
                    ]
                )
                await agg._token_queue.drain_all()
                # OSL = token_count("ok" + tool_calls_json) = 2
                assert (
                    snapshot_series_total(
                        registry, _name(MetricSeriesKey.OSL, CLIENT, primary=primary)
                    )
                    == 2
                )
                # tpot = (5000 - 2000) / token_count(tool_calls_json) = 3000 / 1 = 3000
                assert snapshot_series_total(
                    registry, _name(MetricSeriesKey.TPOT_NS, CLIENT, primary=primary)
                ) == pytest.approx(3000.0)
            finally:
                agg.close()


# ---------------------------------------------------------------------------
# Token-count bases: primary series unsuffixed, the other basis suffixed
# ---------------------------------------------------------------------------

_START = session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0)
_TOKEN_KEYS = (MetricSeriesKey.ISL, MetricSeriesKey.OSL, MetricSeriesKey.TPOT_NS)
# One streamed sample: ISSUED 1000, first chunk 2000, COMPLETE 5000. Its output
# is 7 whitespace tokens, 6 after the first chunk, beside server usage 10 / 5.
_BOTH_BASES_SAMPLE = [
    sample_event(
        SampleEventType.ISSUED, "s1", ts=1000, data=PromptData(text="w w w w")
    ),
    sample_event(SampleEventType.RECV_FIRST, "s1", ts=2000),
    complete_event(
        "s1",
        5000,
        streaming_text("a", " b c d e f g"),
        prompt_tokens=10,
        completion_tokens=5,
    ),
]
# TPOT of _BOTH_BASES_SAMPLE: 3000 ns over 6 tokenized tokens, and over
# completion_tokens - 1 = 4 server tokens.
_TPOT_BY_BASIS = {CLIENT: 500.0, SERVER: 750.0}
_CASES = [pytest.param(*case[1:], id=case[0]) for case in SERVER_USAGE_CASES]


@pytest.mark.unit
class TestTokenCountBases:
    @pytest.mark.asyncio
    async def test_series_registered_primary_unsuffixed_then_other(
        self, tmp_path, primary
    ):
        with aggregator_scope(tmp_path, "agg_order", token_count_source=primary) as (
            _,
            reg,
            _,
        ):
            snap = reg.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
        names = [m.name for m in snap.metrics if isinstance(m, SeriesStat)]
        assert names == [
            "sample_latency_ns",
            "isl",
            "osl",
            "ttft_ns",
            "chunk_delta_ns",
            "tpot_ns",
            *(_name(key, primary.other, primary=primary) for key in _TOKEN_KEYS),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [True, False])
    async def test_no_metrics_isl_drops_isl_on_both_bases(
        self, tmp_path, primary, streaming
    ):
        with aggregator_scope(
            tmp_path,
            "agg_no_isl",
            token_count_source=primary,
            tokenizer=MockBatchTokenizer(),
            enable_isl=False,
            streaming=streaming,
        ) as (agg, reg, _):
            await agg.process([_START, *_BOTH_BASES_SAMPLE])
            # Only the output was enqueued; the prompt never was.
            assert agg.pending_tokens == (2 if streaming else 1)
            for basis in TokenCountSource:
                assert not reg.has_series(
                    _name(MetricSeriesKey.ISL, basis, primary=primary)
                )
                assert reg.has_series(
                    _name(MetricSeriesKey.OSL, basis, primary=primary)
                )
                tpot = _name(MetricSeriesKey.TPOT_NS, basis, primary=primary)
                assert reg.has_series(tpot) is streaming

    @pytest.mark.asyncio
    async def test_each_basis_records_its_own_counts(self, tmp_path, primary):
        with aggregator_scope(
            tmp_path,
            "agg_bases",
            token_count_source=primary,
            tokenizer=MockBatchTokenizer(),
        ) as (agg, reg, _):
            await agg.process([_START, *_BOTH_BASES_SAMPLE])
            await agg.token_queue.drain_all()
            totals = {
                (key, basis): snapshot_series_total(
                    reg, _name(key, basis, primary=primary)
                )
                for key in _TOKEN_KEYS
                for basis in TokenCountSource
            }
        assert totals == {
            (MetricSeriesKey.ISL, CLIENT): 4,
            (MetricSeriesKey.ISL, SERVER): 10,
            (MetricSeriesKey.OSL, CLIENT): 7,
            (MetricSeriesKey.OSL, SERVER): 5,
            (MetricSeriesKey.TPOT_NS, CLIENT): _TPOT_BY_BASIS[CLIENT],
            (MetricSeriesKey.TPOT_NS, SERVER): _TPOT_BY_BASIS[SERVER],
        }

    @pytest.mark.asyncio
    async def test_server_basis_needs_no_tokenizer(self, tmp_path, primary):
        with aggregator_scope(tmp_path, "agg_srv", token_count_source=primary) as (
            agg,
            reg,
            _,
        ):
            await agg.process(
                [
                    sample_event(SampleEventType.ISSUED, "late", ts=1),  # untracked
                    _START,
                    *(
                        sample_event(SampleEventType.ISSUED, u, ts=10)
                        for u in ("s1", "s2", "err")
                    ),
                    complete_event(
                        "s1",
                        50,
                        text_output("x y"),
                        prompt_tokens=7,
                        completion_tokens=5,
                    ),
                    # A true zero: nothing is recorded on either basis.
                    complete_event(
                        "s2", 50, text_output(""), prompt_tokens=3, completion_tokens=0
                    ),
                    complete_event(
                        "err", 50, None, prompt_tokens=4, completion_tokens=4
                    ),
                    complete_event(
                        "late",
                        50,
                        text_output("q"),
                        prompt_tokens=50,
                        completion_tokens=50,
                    ),
                ]
            )
            isl = _name(MetricSeriesKey.ISL, SERVER, primary=primary)
            osl = _name(MetricSeriesKey.OSL, SERVER, primary=primary)
            assert (
                snapshot_series_count(reg, isl),
                snapshot_series_total(reg, isl),
            ) == (
                1,
                7,
            )
            assert (
                snapshot_series_count(reg, osl),
                snapshot_series_total(reg, osl),
            ) == (
                1,
                5,
            )
            assert agg.pending_tokens == 0
            assert snapshot_counters(reg)["tracked_samples_missing_usage"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("prompt", "completion", "output", "expected"), _CASES)
    async def test_missing_usage_counted_for_both_primaries(
        self, tmp_path, primary, prompt, completion, output, expected
    ):
        with aggregator_scope(tmp_path, "agg_miss", token_count_source=primary) as (
            agg,
            reg,
            _,
        ):
            await agg.process(
                [
                    _START,
                    sample_event(SampleEventType.ISSUED, "s1", ts=10),
                    sample_event(SampleEventType.ISSUED, "err", ts=10),
                    complete_event(
                        "s1",
                        20,
                        output,
                        prompt_tokens=prompt,
                        completion_tokens=completion,
                    ),
                    # An error completes without data or usage: never missing.
                    complete_event("err", 20, None),
                ]
            )
            c = snapshot_counters(reg)
            recorded = {
                key: snapshot_series_count(reg, _name(key, SERVER, primary=primary))
                for key in (MetricSeriesKey.ISL, MetricSeriesKey.OSL)
            }
        assert c["tracked_samples_missing_usage"] == (expected is ServerUsage.MISSING)
        assert c["tracked_samples_missing_usage_empty_output"] == 0
        # Server ISL and OSL cover exactly the usable samples.
        usable = int(expected is ServerUsage.USABLE)
        assert recorded == {MetricSeriesKey.ISL: usable, MetricSeriesKey.OSL: usable}

    @pytest.mark.asyncio
    async def test_first_miss_logged_once_at_the_primary_level(
        self, tmp_path, primary, caplog
    ):
        events = [sample_event(SampleEventType.ISSUED, "untracked", ts=1), _START]
        events += [
            sample_event(SampleEventType.ISSUED, u, ts=10)
            for u in ("ok", "a", "b", "c", "err")
        ]
        events += [
            complete_event("untracked", 20, text_output("x")),
            complete_event(
                "ok", 20, text_output("w"), prompt_tokens=1, completion_tokens=1
            ),
            complete_event("a", 20, text_output("x y")),
            complete_event("b", 20, text_output("")),  # empty output
            complete_event("c", 20, text_output("z"), prompt_tokens=3),  # one UNSET
            complete_event("err", 20, None),  # error: excluded
        ]
        with (
            caplog.at_level(logging.INFO),
            aggregator_scope(tmp_path, "agg_miss_log", token_count_source=primary) as (
                agg,
                reg,
                _,
            ),
        ):
            await agg.process(events)
            c = snapshot_counters(reg)
        assert c["tracked_samples_missing_usage"] == 3
        assert c["tracked_samples_missing_usage_empty_output"] == 1
        # "a" is the first miss and the second successful tracked sample.
        message = missing_usage_message(1, 0, 2, primary=primary)
        level = logging.WARNING if primary is SERVER else logging.INFO
        logged = [
            r.levelno
            for r in caplog.records
            if "returned no server token usage" in r.getMessage()
        ]
        assert logged == [level]
        assert any(message in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_paired_osl_counts_only_usable_samples(self, tmp_path, primary):
        events = [_START]
        for case_id, prompt, completion, output, _ in SERVER_USAGE_CASES:
            events += [
                sample_event(SampleEventType.ISSUED, case_id, ts=10),
                complete_event(
                    case_id,
                    20,
                    output,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                ),
            ]
        usable = [case for case in SERVER_USAGE_CASES if case[4] is ServerUsage.USABLE]
        with aggregator_scope(
            tmp_path,
            "agg_pair",
            token_count_source=primary,
            tokenizer=MockBatchTokenizer(),
        ) as (agg, reg, _):
            await agg.process(events)
            # The client OSL recorder runs at the token-queue flush.
            assert snapshot_counters(reg)["osl_paired_samples"] == 0
            await agg.token_queue.drain_all()
            c = snapshot_counters(reg)
        assert (
            c["osl_paired_samples"],
            c["osl_paired_client_tokens"],
            c["osl_paired_server_tokens"],
        ) == (
            len(usable),
            sum(len(str(case[3]).split()) for case in usable),
            sum(case[2] for case in usable),
        )

    @pytest.mark.asyncio
    async def test_early_stopping_and_steady_state_attach_to_primary_tpot_only(
        self, tmp_path, primary
    ):
        reg = MetricsRegistry(early_stopping=EarlyStoppingSpec())
        phase = EventRecord(
            event_type=SessionEventType.PHASE_START,
            timestamp_ns=0,
            data=PhaseData(
                phase_type=PhaseType.PERFORMANCE, drain_after=True, num_turns=1
            ),
        )
        with aggregator_scope(
            tmp_path,
            "agg_primary_tpot",
            token_count_source=primary,
            tokenizer=MockBatchTokenizer(),
            steady_state_profile="concurrency",
            registry=reg,
        ) as (agg, _, _):
            await agg.process([phase, _START, *_BOTH_BASES_SAMPLE])
            await agg.token_queue.drain_all()
            collected = list(agg._collector.series()[0].tpot_ns)
            snap = reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        with_es = {
            m.name
            for m in snap.metrics
            if isinstance(m, SeriesStat) and m.early_stopping_percentiles is not None
        }
        assert with_es == {"sample_latency_ns", "ttft_ns", "tpot_ns"}
        assert collected == [_TPOT_BY_BASIS[primary]]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "completion_tokens, recv_first_ts, expected",
        [
            (4, 2000, 1000.0),  # (5000 - 2000) / (4 - 1)
            (1, 2000, None),  # <= 1: skipped (LoadGen would log a negative)
            (msgspec.UNSET, 2000, None),
            (4, None, None),  # streaming only
            (4, 5000, None),  # delta == 0: skipped
            (4, 6000, None),  # RECV_FIRST stamped after COMPLETE: skipped
        ],
    )
    async def test_server_tpot_uses_loadgen_rule(
        self, tmp_path, primary, completion_tokens, recv_first_ts, expected
    ):
        events = [_START, sample_event(SampleEventType.ISSUED, "s1", ts=1000)]
        if recv_first_ts is not None:
            events.append(
                sample_event(SampleEventType.RECV_FIRST, "s1", ts=recv_first_ts)
            )
        events.append(
            complete_event(
                "s1",
                5000,
                streaming_text("a", " b"),
                prompt_tokens=2,
                completion_tokens=completion_tokens,
            )
        )
        tpot = _name(MetricSeriesKey.TPOT_NS, SERVER, primary=primary)
        with aggregator_scope(tmp_path, "agg_tpot", token_count_source=primary) as (
            agg,
            reg,
            _,
        ):
            await agg.process(events)
            assert snapshot_series_count(reg, tpot) == (expected is not None)
            if expected is not None:
                assert snapshot_series_total(reg, tpot) == pytest.approx(expected)


@pytest.mark.unit
def test_aggregator_requires_token_count_source():
    with pytest.raises(TypeError, match="token_count_source"):
        MetricsAggregatorService(
            "unused",
            None,
            None,
            registry=MetricsRegistry(),
            publisher=None,
            publish_interval_s=0.25,
            sig_figs=3,
            n_histogram_buckets=10,
        )
