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

"""Tests for ``MetricsTable``, ``SampleRow``, and ``TrackedBlock``.

The table is registry-agnostic for most flows — these tests pass a
fresh ``MetricsRegistry`` per test and do not register any triggers,
so the registry is only used to satisfy the constructor signature.
"""

from __future__ import annotations

import asyncio

import msgspec
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricsTable,
    SampleRow,
    TrackedBlock,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)


def _new_table() -> MetricsTable:
    """A MetricsTable backed by a fresh, empty MetricsRegistry."""
    return MetricsTable(MetricsRegistry())


@pytest.mark.unit
class TestSampleRow:
    def test_initial_timestamps_are_none(self):
        row = SampleRow("s1")
        assert row.issued_ns is None
        assert row.complete_ns is None
        assert row.recv_first_ns is None
        assert row.last_recv_ns is None
        assert row.tracked_block_idx == -1

    def test_is_msgspec_struct(self):
        row = SampleRow("s1")
        assert isinstance(row, msgspec.Struct)


@pytest.mark.unit
class TestTrackedBlock:
    def test_duration_ns(self):
        block = TrackedBlock(start_ns=100, last_complete_ns=500)
        assert block.duration_ns == 400

    def test_empty_block_duration_zero(self):
        block = TrackedBlock(start_ns=100, last_complete_ns=100)
        assert block.duration_ns == 0
        assert block.completed_samples == 0

    def test_completed_samples_increment(self):
        block = TrackedBlock(start_ns=0, last_complete_ns=0)
        block.completed_samples += 1
        block.last_complete_ns = 500
        assert block.duration_ns == 500
        assert block.completed_samples == 1


@pytest.mark.unit
class TestMetricsTable:
    def test_create_and_get_row(self):
        table = _new_table()
        table.is_tracking = True
        table.tracked_blocks.append(TrackedBlock(start_ns=0, last_complete_ns=0))
        ev = EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=100, sample_uuid="s1"
        )
        table.set_field("s1", "issued_ns", 100, ev)
        assert table.get_row("s1") is not None
        assert len(table) == 1

    def test_complete_removes_row(self):
        table = _new_table()
        table.is_tracking = True
        table.tracked_blocks.append(TrackedBlock(start_ns=0, last_complete_ns=0))
        issued = EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=100, sample_uuid="s1"
        )
        table.set_field("s1", "issued_ns", 100, issued)
        complete = EventRecord(
            event_type=SampleEventType.COMPLETE, timestamp_ns=500, sample_uuid="s1"
        )
        table.set_field("s1", "complete_ns", 500, complete)
        assert table.get_row("s1") is None
        assert len(table) == 0

    def test_set_field_noop_for_untracked(self):
        table = _new_table()
        ev = EventRecord(
            event_type=SampleEventType.RECV_FIRST,
            timestamp_ns=200,
            sample_uuid="unknown",
        )
        table.set_field("unknown", "recv_first_ns", 200, ev)
        assert table.get_row("unknown") is None

    def test_issued_noop_when_not_tracking(self):
        table = _new_table()
        ev = EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=100, sample_uuid="s1"
        )
        table.set_field("s1", "issued_ns", 100, ev)
        assert table.get_row("s1") is None

    def test_duplicate_issued_returns_existing(self):
        table = _new_table()
        table.is_tracking = True
        table.tracked_blocks.append(TrackedBlock(start_ns=0, last_complete_ns=0))
        ev1 = EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=100, sample_uuid="s1"
        )
        table.set_field("s1", "issued_ns", 100, ev1)
        row1 = table.get_row("s1")
        ev2 = EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=200, sample_uuid="s1"
        )
        table.set_field("s1", "issued_ns", 200, ev2)
        assert table.get_row("s1") is row1
        assert len(table) == 1

    def test_multiple_rows(self):
        table = _new_table()
        table.is_tracking = True
        table.tracked_blocks.append(TrackedBlock(start_ns=0, last_complete_ns=0))
        for uuid in ("s1", "s2", "s3"):
            ev = EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=100,
                sample_uuid=uuid,
            )
            table.set_field(uuid, "issued_ns", 100, ev)
        assert len(table) == 3

    def test_handle_session_started(self):
        table = _new_table()
        ev = EventRecord(event_type=SessionEventType.STARTED, timestamp_ns=42)
        table.handle_session_event(ev)
        assert table.session_started_ns == 42

    def test_handle_start_stop_tracking(self):
        table = _new_table()
        assert not table.is_tracking

        start = EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=100
        )
        table.handle_session_event(start)
        assert table.is_tracking
        assert len(table.tracked_blocks) == 1
        assert table.tracked_blocks[0].start_ns == 100

        stop = EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=200
        )
        table.handle_session_event(stop)
        assert not table.is_tracking

    def test_duplicate_start_is_noop(self):
        table = _new_table()
        start1 = EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=100
        )
        start2 = EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=200
        )
        table.handle_session_event(start1)
        table.handle_session_event(start2)
        assert len(table.tracked_blocks) == 1

    def test_tracked_block_updated_on_complete(self):
        table = _new_table()
        start = EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
        )
        table.handle_session_event(start)
        issued = EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=100, sample_uuid="s1"
        )
        table.set_field("s1", "issued_ns", 100, issued)
        complete = EventRecord(
            event_type=SampleEventType.COMPLETE, timestamp_ns=500, sample_uuid="s1"
        )
        table.set_field("s1", "complete_ns", 500, complete)

        assert table.tracked_blocks[0].last_complete_ns == 500
        assert table.tracked_blocks[0].completed_samples == 1
        assert table.total_tracked_duration_ns == 500
        assert table.total_completed_tracked_samples == 1

    def test_multiple_tracking_windows(self):
        table = _new_table()

        # Block 0
        table.handle_session_event(
            EventRecord(
                event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0
            )
        )
        table.set_field(
            "s1",
            "issued_ns",
            100,
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=100,
                sample_uuid="s1",
            ),
        )
        table.handle_session_event(
            EventRecord(
                event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=200
            )
        )
        # s1 completes after STOP — still extends block 0
        table.set_field(
            "s1",
            "complete_ns",
            600,
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                timestamp_ns=600,
                sample_uuid="s1",
            ),
        )

        # Block 1
        table.handle_session_event(
            EventRecord(
                event_type=SessionEventType.START_PERFORMANCE_TRACKING,
                timestamp_ns=800,
            )
        )
        table.set_field(
            "s2",
            "issued_ns",
            900,
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=900,
                sample_uuid="s2",
            ),
        )
        table.set_field(
            "s2",
            "complete_ns",
            1000,
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                timestamp_ns=1000,
                sample_uuid="s2",
            ),
        )

        assert table.tracked_blocks[0].duration_ns == 600  # 600 - 0
        assert table.tracked_blocks[1].duration_ns == 200  # 1000 - 800
        assert table.total_tracked_duration_ns == 800
        assert table.total_completed_tracked_samples == 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestOslTriggerToolCalls:
    """OslTrigger routes to message path when tool_calls are present."""

    async def test_osl_with_tool_calls_uses_message_path(self):
        """OslTrigger stores combined content+tool_calls word count."""
        from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
            OslTrigger,
            SampleRow,
        )
        from inference_endpoint.core.types import TextModelOutput

        from .conftest import MockTokenizePool, snapshot_series_count

        registry = MetricsRegistry()
        registry.register_series("osl", hdr_low=1, hdr_high=100_000)
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0)
        trigger = OslTrigger(registry, pool, loop)

        tool_calls = (
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            },
        )
        tmo = TextModelOutput(output="hello world", tool_calls=tool_calls)
        ev = EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=1000,
            sample_uuid="s1",
            data=tmo,
        )
        row = SampleRow(sample_uuid="s1")
        task = trigger.fire(ev, row, {})
        assert task is not None
        await task

        assert snapshot_series_count(registry, "osl") == 1

    async def test_osl_without_tool_calls_uses_text_path(self):
        """OslTrigger uses text path for output with no tool_calls (regression guard)."""
        from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
            OslTrigger,
            SampleRow,
        )
        from inference_endpoint.core.types import TextModelOutput

        from .conftest import MockTokenizePool, snapshot_series_count

        registry = MetricsRegistry()
        registry.register_series("osl", hdr_low=1, hdr_high=100_000)
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0)
        trigger = OslTrigger(registry, pool, loop)

        tmo = TextModelOutput(output="hello world")
        ev = EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=1000,
            sample_uuid="s1",
            data=tmo,
        )
        row = SampleRow(sample_uuid="s1")
        task = trigger.fire(ev, row, {})
        assert task is not None
        await task

        assert snapshot_series_count(registry, "osl") == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestTpotTriggerToolCalls:
    """TpotTrigger routes to message path when tool_calls are present."""

    async def test_tpot_tool_calls_only_response(self):
        """TpotTrigger includes tool_calls in TPOT denominator for agentic responses."""
        from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
            SampleField,
            SampleRow,
            TpotTrigger,
        )
        from inference_endpoint.core.types import TextModelOutput

        from .conftest import MockTokenizePool, snapshot_series_count

        registry = MetricsRegistry()
        registry.register_series(
            "tpot_ns", hdr_low=1, hdr_high=100_000_000_000, dtype=float
        )
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0)
        trigger = TpotTrigger(registry, pool, loop)

        tool_calls = (
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            },
        )
        tmo = TextModelOutput(output=[], tool_calls=tool_calls)
        ev = EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=2000,
            sample_uuid="s1",
            data=tmo,
        )
        row = SampleRow(sample_uuid="s1")
        # RECV_FIRST_NS was set at t=1000
        pre_change = {SampleField.RECV_FIRST_NS: 1000}
        task = trigger.fire(ev, row, pre_change)
        assert task is not None
        await task

        assert snapshot_series_count(registry, "tpot_ns") == 1

    async def test_tpot_uses_tool_call_deltas_after_first_chunk(self):
        from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
            SampleField,
            SampleRow,
            TpotTrigger,
        )
        from inference_endpoint.core.types import TextModelOutput

        from .conftest import MockTokenizePool, snapshot_series_total

        registry = MetricsRegistry()
        registry.register_series(
            "tpot_ns", hdr_low=1, hdr_high=100_000_000_000, dtype=float
        )
        loop = asyncio.get_running_loop()
        pool = MockTokenizePool(delay=0)
        trigger = TpotTrigger(registry, pool, loop)

        tool_call_chunks = (
            (
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "first chunk "},
                },
            ),
            ({"index": 0, "function": {"arguments": "after chunk"}},),
        )
        tmo = TextModelOutput(
            output=[],
            tool_calls=tool_call_chunks,
        )
        ev = EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=5000,
            sample_uuid="s1",
            data=tmo,
        )
        row = SampleRow(sample_uuid="s1")
        pre_change = {SampleField.RECV_FIRST_NS: 1000}
        task = trigger.fire(ev, row, pre_change)
        assert task is not None
        await task

        assert snapshot_series_total(registry, "tpot_ns") == pytest.approx(2000.0)


def _start_tracking(table: MetricsTable, start_ns: int = 0) -> None:
    """Open a real tracking block at ``start_ns`` (STARTED + START)."""
    table.handle_session_event(
        EventRecord(event_type=SessionEventType.STARTED, timestamp_ns=start_ns)
    )
    table.handle_session_event(
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING,
            timestamp_ns=start_ns,
        )
    )


def _issue(table: MetricsTable, uuid: str, ts: int) -> None:
    table.set_field(
        uuid,
        "issued_ns",
        ts,
        EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=ts, sample_uuid=uuid
        ),
    )


def _complete(table: MetricsTable, uuid: str, ts: int) -> None:
    table.set_field(
        uuid,
        "complete_ns",
        ts,
        EventRecord(
            event_type=SampleEventType.COMPLETE, timestamp_ns=ts, sample_uuid=uuid
        ),
    )


@pytest.mark.unit
class TestLoadgenWindow:
    """``MetricsTable.total_loadgen_window_ns`` — the legacy MLPerf LoadGen Server
    ``final_query_all_samples_done_time`` analog: from the first issued
    tracked request to the completion of the last-issued request that
    completed.
    """

    def test_first_issue_to_last_issued_completion(self):
        """s1 issued early (t=100) completes late (t=1000); s2 issued late
        (t=200) completes early (t=300). s2 has the largest issued_ns, so the
        window ends at s2's completion (300) and starts at the first issue
        (100): window = 300 - 100.
        """
        table = _new_table()
        _start_tracking(table, start_ns=50)
        _issue(table, "s1", 100)
        _issue(table, "s2", 200)
        _complete(table, "s2", 300)
        _complete(table, "s1", 1000)

        # Start at first issue (100), not block start (50); end at the
        # last-issued (s2) completion (300).
        assert table.total_loadgen_window_ns == 300 - 100

    def test_no_completions_yields_zero(self):
        """If no tracked request ever COMPLETEs, the window end is never set →
        total_loadgen_window_ns == 0.
        """
        table = _new_table()
        _start_tracking(table, start_ns=0)
        _issue(table, "s1", 100)
        # s1 is issued but never completes.

        assert table.total_loadgen_window_ns == 0
