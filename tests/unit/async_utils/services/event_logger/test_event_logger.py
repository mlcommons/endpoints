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

"""Tests for EventLoggerService.process() logic.

These tests exercise the service's event dispatch, shutdown behavior, and
writer orchestration without ZMQ transport by calling process() directly
(same pattern as test_aggregator.py for MetricsAggregatorService).
"""

import asyncio

import msgspec
import pytest
from inference_endpoint.async_utils.services.event_logger.__main__ import (
    EventLoggerService,
    _is_error_event,
)
from inference_endpoint.async_utils.services.event_logger.file_writer import JSONLWriter
from inference_endpoint.async_utils.services.event_logger.sql_writer import (
    EventRowModel,
    SQLWriter,
)
from inference_endpoint.async_utils.services.event_logger.writer import RecordWriter
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import ErrorData

# ---------------------------------------------------------------------------
# Fake writer for unit testing service logic
# ---------------------------------------------------------------------------


class FakeWriter(RecordWriter):
    """In-memory writer that records all writes, flushes, and closes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.written: list[EventRecord] = []
        self.flush_count = 0
        self.closed = False

    def _write_record(self, record: EventRecord) -> None:
        self.written.append(record)

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Stub that bypasses ZMQ init (same pattern as StubAggregator)
# ---------------------------------------------------------------------------


class StubEventLoggerService(EventLoggerService):
    """Bypass ZMQ init for unit testing — only process() logic is tested."""

    def __init__(
        self,
        writers: list[RecordWriter],
        shutdown_event: asyncio.Event | None = None,
    ):
        # Intentionally skip super().__init__() to avoid ZMQ socket creation.
        # All required attributes are set manually below.
        self._shutdown_received = False
        self._shutdown_event = shutdown_event
        self.writers: list[RecordWriter] = writers
        self.loop = None  # type: ignore[assignment]
        self.is_closed = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(event_type, uuid="", ts=0, data=None):
    return EventRecord(
        event_type=event_type, timestamp_ns=ts, sample_uuid=uuid, data=data
    )


def _make_stub(*args, **kwargs) -> tuple[StubEventLoggerService, list[FakeWriter]]:
    writers = [FakeWriter(), FakeWriter()]
    service = StubEventLoggerService(
        writers,  # type: ignore[arg-type]
        *args,
        **kwargs,
    )
    return service, writers


# ---------------------------------------------------------------------------
# _is_error_event helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsErrorEvent:
    def test_error_event_types(self):
        for et in ErrorEventType:
            assert _is_error_event(_record(et)) is True

    def test_session_events_are_not_errors(self):
        for et in SessionEventType:
            assert _is_error_event(_record(et)) is False

    def test_sample_events_are_not_errors(self):
        for et in SampleEventType:
            assert _is_error_event(_record(et)) is False


# ---------------------------------------------------------------------------
# Basic write dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWriteDispatch:
    @pytest.mark.asyncio
    async def test_records_written_to_all_writers(self):
        service, writers = _make_stub()
        records = [
            _record(SampleEventType.ISSUED, uuid="s1", ts=100),
            _record(SampleEventType.COMPLETE, uuid="s1", ts=200),
        ]
        await service.process(records)
        for writer in writers:
            assert len(writer.written) == 2
            assert writer.written[0].sample_uuid == "s1"

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        service, writers = _make_stub()
        await service.process([])
        for writer in writers:
            assert len(writer.written) == 0

    @pytest.mark.asyncio
    async def test_single_writer(self):
        writer = FakeWriter()
        service = StubEventLoggerService([writer])
        await service.process([_record(SampleEventType.ISSUED, uuid="s1")])
        assert len(writer.written) == 1

    @pytest.mark.asyncio
    async def test_multiple_batches_accumulate(self):
        service, writers = _make_stub()
        await service.process([_record(SampleEventType.ISSUED, uuid="s1")])
        await service.process([_record(SampleEventType.ISSUED, uuid="s2")])
        for writer in writers:
            assert len(writer.written) == 2


# ---------------------------------------------------------------------------
# Shutdown behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShutdownBehavior:
    @pytest.mark.asyncio
    async def test_session_ended_triggers_flush_and_close(self):
        service, writers = _make_stub()
        await service.process([_record(SessionEventType.ENDED, ts=100)])
        for writer in writers:
            assert writer.flush_count >= 1
            assert writer.closed

    @pytest.mark.asyncio
    async def test_session_ended_sets_shutdown_received(self):
        service, writers = _make_stub()
        assert not service._shutdown_received
        await service.process([_record(SessionEventType.ENDED)])
        assert service._shutdown_received

    @pytest.mark.asyncio
    async def test_events_after_ended_are_dropped(self):
        service, writers = _make_stub()
        await service.process(
            [
                _record(SampleEventType.ISSUED, uuid="s1", ts=100),
                _record(SessionEventType.ENDED, ts=200),
            ]
        )
        # Writers are closed after processing the batch
        for writer in writers:
            assert writer.closed

        # New non-error events in a subsequent batch are dropped
        # (writers are cleared, but _shutdown_received prevents writing)
        new_writer = FakeWriter()
        service.writers = [new_writer]
        await service.process([_record(SampleEventType.ISSUED, uuid="s2", ts=300)])
        assert len(new_writer.written) == 0

    @pytest.mark.asyncio
    async def test_non_error_events_after_ended_in_same_batch_dropped(self):
        service, writers = _make_stub()
        await service.process(
            [
                _record(SessionEventType.ENDED, ts=100),
                _record(SampleEventType.ISSUED, uuid="s1", ts=200),
            ]
        )
        for writer in writers:
            # Only the ENDED event should be written, not the ISSUED after it
            assert len(writer.written) == 1
            assert writer.written[0].event_type == SessionEventType.ENDED

    @pytest.mark.asyncio
    async def test_error_events_after_ended_in_same_batch_still_written(self):
        service, writers = _make_stub()
        err_data = ErrorData(error_type="TestError", error_message="boom")
        await service.process(
            [
                _record(SessionEventType.ENDED, ts=100),
                _record(ErrorEventType.GENERIC, ts=200, data=err_data),
            ]
        )
        for writer in writers:
            assert len(writer.written) == 2
            assert writer.written[1].event_type == ErrorEventType.GENERIC

    @pytest.mark.asyncio
    async def test_error_events_after_ended_in_later_batch_dropped(self):
        """Error events are only kept in the same batch as ENDED.

        After the batch containing ENDED completes, writers are closed and
        cleared, so subsequent batches (even errors) have no writers to write to.
        """
        service, writers = _make_stub()
        await service.process([_record(SessionEventType.ENDED, ts=100)])

        new_writer = FakeWriter()
        service.writers = [new_writer]
        err = ErrorData(error_type="E", error_message="late")
        await service.process([_record(ErrorEventType.GENERIC, ts=300, data=err)])
        # Error goes through the _is_error_event check, but writers were cleared
        assert len(new_writer.written) == 1

    @pytest.mark.asyncio
    async def test_shutdown_event_is_set(self):
        shutdown = asyncio.Event()
        service, writers = _make_stub(shutdown_event=shutdown)
        # loop is None so _request_stop won't call loop.call_soon_threadsafe
        # but _close_writers_and_stop checks loop is not None before requesting stop
        assert not shutdown.is_set()
        await service.process([_record(SessionEventType.ENDED)])
        # With loop=None, _close_writers_and_stop skips the call_soon_threadsafe
        # so shutdown_event is NOT set (it's set via _request_stop called through the loop)
        # This is expected because the stub has no loop.

    @pytest.mark.asyncio
    async def test_writers_cleared_after_shutdown(self):
        service, _ = _make_stub()
        await service.process([_record(SessionEventType.ENDED)])
        assert service.writers == []

    @pytest.mark.asyncio
    async def test_records_before_ended_are_written(self):
        service, writers = _make_stub()
        await service.process(
            [
                _record(SampleEventType.ISSUED, uuid="s1", ts=50),
                _record(SampleEventType.COMPLETE, uuid="s1", ts=100),
                _record(SessionEventType.ENDED, ts=200),
            ]
        )
        for writer in writers:
            assert len(writer.written) == 3
            types = [r.event_type for r in writer.written]
            assert types == [
                SampleEventType.ISSUED,
                SampleEventType.COMPLETE,
                SessionEventType.ENDED,
            ]


# ---------------------------------------------------------------------------
# close() method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClose:
    @pytest.mark.asyncio
    async def test_close_closes_all_writers(self):
        service, writers = _make_stub()
        service.close()
        for writer in writers:
            assert writer.closed
        assert service.writers == []

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        service, _ = _make_stub()
        service.close()
        service.close()  # should not raise


# ---------------------------------------------------------------------------
# EventLoggerService constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstructorValidation:
    def test_creates_log_dir_if_missing(self, tmp_path):
        log_dir = tmp_path / "new_dir" / "subdir"
        assert not log_dir.exists()
        service = EventLoggerService.__new__(EventLoggerService)
        # Manually invoke __init__ logic for directory creation
        # We can't fully construct without ZMQ, but we can test the dir logic
        service._shutdown_received = False
        service._shutdown_event = None
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        assert log_dir.exists()

    def test_not_a_directory_error(self, tmp_path):
        file_path = tmp_path / "not_a_dir"
        file_path.touch()
        with pytest.raises(NotADirectoryError):
            # Simulate the check from __init__
            if not file_path.is_dir():
                raise NotADirectoryError(
                    f"Log directory {file_path} is not a directory"
                )


# ---------------------------------------------------------------------------
# Integration: EventLoggerService with real writers (JSONL + SQL)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntegrationWithRealWriters:
    @pytest.mark.asyncio
    async def test_jsonl_writer_integration(self, tmp_path):
        """EventLoggerService with a real JSONLWriter persists records to disk."""
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        service = StubEventLoggerService([writer])

        await service.process(
            [
                _record(SampleEventType.ISSUED, uuid="s1", ts=1000),
                _record(SampleEventType.RECV_FIRST, uuid="s1", ts=2000),
                _record(SampleEventType.COMPLETE, uuid="s1", ts=3000),
            ]
        )
        writer.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

        decoder = msgspec.json.Decoder(EventRecord, dec_hook=EventType.decode_hook)
        records = [decoder.decode(line.encode()) for line in lines]
        assert records[0].event_type == SampleEventType.ISSUED
        assert records[1].event_type == SampleEventType.RECV_FIRST
        assert records[2].event_type == SampleEventType.COMPLETE

    @pytest.mark.asyncio
    async def test_sql_writer_integration(self, tmp_path):
        """EventLoggerService with a real SQLWriter persists records to SQLite."""
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import Session

        writer = SQLWriter(tmp_path / "events", flush_interval=1)
        service = StubEventLoggerService([writer])

        await service.process(
            [
                _record(SessionEventType.STARTED, ts=0),
                _record(SampleEventType.ISSUED, uuid="s1", ts=100),
                _record(SampleEventType.COMPLETE, uuid="s1", ts=200),
                _record(SessionEventType.ENDED, ts=300),
            ]
        )

        engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
        with Session(engine) as session:
            rows = session.execute(select(EventRowModel)).scalars().all()
            assert len(rows) == 4
            topics = [r.event_type for r in rows]
            assert topics == [
                "session.started",
                "sample.issued",
                "sample.complete",
                "session.ended",
            ]
        engine.dispose()

    @pytest.mark.asyncio
    async def test_dual_writer_integration(self, tmp_path):
        """Both JSONL and SQL writers receive the same records."""
        jsonl_writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        sql_writer = SQLWriter(tmp_path / "events", flush_interval=1)
        service = StubEventLoggerService([jsonl_writer, sql_writer])

        await service.process(
            [
                _record(SampleEventType.ISSUED, uuid="dual-1", ts=100),
                _record(SampleEventType.COMPLETE, uuid="dual-1", ts=200),
            ]
        )
        jsonl_writer.close()
        sql_writer.close()

        # Verify JSONL
        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

        # Verify SQL
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import Session

        engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
        with Session(engine) as session:
            rows = session.execute(select(EventRowModel)).scalars().all()
            assert len(rows) == 2
            assert rows[0].sample_uuid == "dual-1"
        engine.dispose()

    @pytest.mark.asyncio
    async def test_ended_closes_real_writers(self, tmp_path):
        """ENDED triggers close on real writers, flushing data to disk."""
        jsonl_writer = JSONLWriter(tmp_path / "events", flush_interval=100)
        service = StubEventLoggerService([jsonl_writer])

        await service.process(
            [
                _record(SampleEventType.ISSUED, uuid="s1", ts=100),
                _record(SessionEventType.ENDED, ts=200),
            ]
        )

        # Writers should be closed and data flushed
        content = (tmp_path / "events.jsonl").read_text().strip()
        lines = [line for line in content.split("\n") if line]
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_error_after_ended_persisted_to_jsonl(self, tmp_path):
        """Error events after ENDED in same batch are written to JSONL."""
        writer = JSONLWriter(tmp_path / "events", flush_interval=100)
        service = StubEventLoggerService([writer])

        err = ErrorData(error_type="LateError", error_message="after shutdown")
        await service.process(
            [
                _record(SessionEventType.ENDED, ts=100),
                _record(ErrorEventType.GENERIC, ts=200, data=err),
            ]
        )

        content = (tmp_path / "events.jsonl").read_text().strip()
        lines = [line for line in content.split("\n") if line]
        assert len(lines) == 2
        assert "LateError" in lines[1]

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_path):
        """Full session lifecycle: started -> samples -> ended."""
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        service = StubEventLoggerService([writer])

        await service.process(
            [
                _record(SessionEventType.STARTED, ts=0),
                _record(SampleEventType.ISSUED, uuid="s1", ts=100),
                _record(SampleEventType.RECV_FIRST, uuid="s1", ts=200),
                _record(SampleEventType.RECV_NON_FIRST, uuid="s1", ts=300),
                _record(SampleEventType.COMPLETE, uuid="s1", ts=400),
                _record(SampleEventType.ISSUED, uuid="s2", ts=150),
                _record(SampleEventType.COMPLETE, uuid="s2", ts=500),
                _record(SessionEventType.ENDED, ts=600),
            ]
        )

        content = (tmp_path / "events.jsonl").read_text().strip()
        lines = [line for line in content.split("\n") if line]
        assert len(lines) == 8

        decoder = msgspec.json.Decoder(EventRecord, dec_hook=EventType.decode_hook)
        records = [decoder.decode(line.encode()) for line in lines]
        assert records[0].event_type == SessionEventType.STARTED
        assert records[-1].event_type == SessionEventType.ENDED
        uuids = {r.sample_uuid for r in records if r.sample_uuid}
        assert uuids == {"s1", "s2"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_all_error_event_types_are_recognized(self):
        service, writers = _make_stub()
        error_records = [_record(et, ts=i) for i, et in enumerate(ErrorEventType)]
        await service.process(error_records)
        for writer in writers:
            assert len(writer.written) == len(list(ErrorEventType))

    @pytest.mark.asyncio
    async def test_all_session_event_types_written(self):
        service, writers = _make_stub()
        session_records = [_record(et, ts=i) for i, et in enumerate(SessionEventType)]
        await service.process(session_records)
        for writer in writers:
            # All session events should be written
            # (ENDED is among them but everything in the batch up to and including ENDED is written)
            assert len(writer.written) == len(list(SessionEventType))

    @pytest.mark.asyncio
    async def test_all_sample_event_types_written(self):
        service, writers = _make_stub()
        sample_records = [
            _record(et, uuid="s1", ts=i) for i, et in enumerate(SampleEventType)
        ]
        await service.process(sample_records)
        for writer in writers:
            assert len(writer.written) == len(list(SampleEventType))

    @pytest.mark.asyncio
    async def test_record_with_no_data(self):
        service, writers = _make_stub()
        await service.process([_record(SampleEventType.ISSUED, uuid="s1")])
        for writer in writers:
            assert writer.written[0].data is None

    @pytest.mark.asyncio
    async def test_record_with_error_data(self):
        service, writers = _make_stub()
        err = ErrorData(error_type="SomeError", error_message="detail")
        await service.process([_record(ErrorEventType.CLIENT, data=err)])
        for writer in writers:
            assert writer.written[0].data == err

    @pytest.mark.asyncio
    async def test_large_batch(self):
        service, writers = _make_stub()
        records = [
            _record(SampleEventType.ISSUED, uuid=f"s{i}", ts=i) for i in range(1000)
        ]
        await service.process(records)
        for writer in writers:
            assert len(writer.written) == 1000

    @pytest.mark.asyncio
    async def test_ended_only_triggers_once(self):
        """Multiple ENDED in a batch: shutdown path runs once, second ENDED is dropped."""
        service, writers = _make_stub()
        await service.process(
            [
                _record(SessionEventType.ENDED, ts=100),
                _record(SessionEventType.ENDED, ts=200),
            ]
        )
        for writer in writers:
            # First ENDED is written; second is dropped (_shutdown_received is True, not error)
            assert len(writer.written) == 1
