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
# Basic write dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWriteDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "case_desc, records",
        [
            (
                "sample events",
                [
                    _record(SampleEventType.ISSUED, uuid="s1", ts=100),
                    _record(SampleEventType.COMPLETE, uuid="s1", ts=200),
                ],
            ),
            ("no data", [_record(SampleEventType.ISSUED, uuid="s1")]),
            (
                "error data",
                [
                    _record(
                        ErrorEventType.CLIENT,
                        data=ErrorData(error_type="SomeError", error_message="detail"),
                    ),
                ],
            ),
        ],
    )
    async def test_records_written_to_all_writers(self, case_desc, records):
        service, writers = _make_stub()
        await service.process(records)
        for writer in writers:
            assert writer.written == records

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        service, writers = _make_stub()
        await service.process([])
        for writer in writers:
            assert len(writer.written) == 0

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
            assert writer.flush_count == 1
            assert writer.closed

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "case_desc, trailing_record",
        [
            ("sample event", _record(SampleEventType.ISSUED, uuid="s1", ts=200)),
            (
                "error event",
                _record(
                    ErrorEventType.GENERIC,
                    ts=200,
                    data=ErrorData(error_type="E", error_message="boom"),
                ),
            ),
        ],
    )
    async def test_events_after_ended_same_batch(self, case_desc, trailing_record):
        """All event types after ENDED in the same batch are dropped."""
        service, writers = _make_stub()
        await service.process(
            [_record(SessionEventType.ENDED, ts=100), trailing_record]
        )
        for writer in writers:
            assert len(writer.written) == 1
            assert writer.written[0].event_type == SessionEventType.ENDED

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
    async def test_events_after_ended_not_persisted_to_jsonl(self, tmp_path):
        """All events after ENDED (including errors) are dropped from JSONL."""
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
        assert len(lines) == 1
        assert "LateError" not in lines[0]

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
    @pytest.mark.parametrize(
        "case_desc, event_enum, make_record",
        [
            ("error types", ErrorEventType, lambda et, i: _record(et, ts=i)),
            ("session types", SessionEventType, lambda et, i: _record(et, ts=i)),
            (
                "sample types",
                SampleEventType,
                lambda et, i: _record(et, uuid="s1", ts=i),
            ),
        ],
    )
    async def test_all_event_types_written(self, case_desc, event_enum, make_record):
        """Every member of each EventType enum is written when no ENDED precedes it."""
        service, writers = _make_stub()
        records = [make_record(et, i) for i, et in enumerate(event_enum)]
        await service.process(records)
        for writer in writers:
            assert len(writer.written) == len(list(event_enum))

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
