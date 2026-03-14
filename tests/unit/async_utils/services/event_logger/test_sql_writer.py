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

"""Tests for SQLWriter and EventRowModel."""

import msgspec
import pytest
from inference_endpoint.async_utils.services.event_logger.sql_writer import (
    Base,
    EventRowModel,
    SQLWriter,
    _record_to_row,
)
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import ErrorData
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


def _record(event_type, uuid="", ts=0, data=None):
    return EventRecord(
        event_type=event_type, timestamp_ns=ts, sample_uuid=uuid, data=data
    )


# ---------------------------------------------------------------------------
# _record_to_row conversion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecordToRow:
    def test_sample_event_topic(self):
        row = _record_to_row(_record(SampleEventType.ISSUED, uuid="s1", ts=1000))
        assert row.event_type == "sample.issued"
        assert row.sample_uuid == "s1"
        assert row.timestamp_ns == 1000

    def test_session_event_topic(self):
        row = _record_to_row(_record(SessionEventType.ENDED, ts=42))
        assert row.event_type == "session.ended"
        assert row.sample_uuid == ""
        assert row.timestamp_ns == 42

    def test_error_event_topic(self):
        row = _record_to_row(_record(ErrorEventType.GENERIC, ts=99))
        assert row.event_type == "error.generic"

    def test_data_is_json_encoded(self):
        err = ErrorData(error_type="TestError", error_message="boom")
        row = _record_to_row(_record(SampleEventType.COMPLETE, data=err))
        decoded = msgspec.json.decode(row.data)
        assert "TestError" in str(decoded)

    def test_none_data_encodes_to_null(self):
        row = _record_to_row(_record(SampleEventType.ISSUED))
        decoded = msgspec.json.decode(row.data)
        assert decoded is None


# ---------------------------------------------------------------------------
# EventRowModel schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventRowModel:
    def test_table_name(self):
        assert EventRowModel.__tablename__ == "events"

    def test_schema_creates_in_memory_db(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(
                EventRowModel(
                    sample_uuid="s1",
                    event_type="sample.issued",
                    timestamp_ns=100,
                    data=b"null",
                )
            )
            session.commit()
            rows = session.execute(select(EventRowModel)).scalars().all()
            assert len(rows) == 1
            assert rows[0].sample_uuid == "s1"
        engine.dispose()

    def test_autoincrement_id(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            for i in range(3):
                session.add(
                    EventRowModel(
                        sample_uuid=f"s{i}",
                        event_type="sample.issued",
                        timestamp_ns=i,
                        data=b"null",
                    )
                )
            session.commit()
            rows = session.execute(select(EventRowModel)).scalars().all()
            ids = [r.id for r in rows]
            assert ids == [1, 2, 3]
        engine.dispose()


# ---------------------------------------------------------------------------
# SQLWriter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSQLWriter:
    def test_creates_db_file(self, tmp_path):
        writer = SQLWriter(tmp_path / "events")
        try:
            assert (tmp_path / "events.db").exists()
        finally:
            writer.close()

    def test_write_and_flush_persists(self, tmp_path):
        writer = SQLWriter(tmp_path / "events", flush_interval=1)
        try:
            writer.write(_record(SampleEventType.ISSUED, uuid="s1", ts=1000))
        finally:
            writer.close()

        engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
        with Session(engine) as session:
            rows = session.execute(select(EventRowModel)).scalars().all()
            assert len(rows) == 1
            assert rows[0].sample_uuid == "s1"
            assert rows[0].event_type == "sample.issued"
            assert rows[0].timestamp_ns == 1000
        engine.dispose()

    def test_multiple_records(self, tmp_path):
        writer = SQLWriter(tmp_path / "events", flush_interval=1)
        try:
            for i in range(5):
                writer.write(_record(SampleEventType.ISSUED, uuid=f"s{i}", ts=i * 100))
        finally:
            writer.close()

        engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
        with Session(engine) as session:
            rows = session.execute(select(EventRowModel)).scalars().all()
            assert len(rows) == 5
            uuids = {r.sample_uuid for r in rows}
            assert uuids == {f"s{i}" for i in range(5)}
        engine.dispose()

    def test_close_commits_pending(self, tmp_path):
        """Records written without reaching flush_interval are committed on close."""
        writer = SQLWriter(tmp_path / "events", flush_interval=100)
        for i in range(3):
            writer.write(_record(SampleEventType.ISSUED, uuid=f"s{i}", ts=i))
        writer.close()

        engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
        with Session(engine) as session:
            rows = session.execute(select(EventRowModel)).scalars().all()
            assert len(rows) == 3
        engine.dispose()

    def test_close_idempotent(self, tmp_path):
        writer = SQLWriter(tmp_path / "events")
        writer.close()
        writer.close()  # should not raise

    def test_write_after_close_is_noop(self, tmp_path):
        writer = SQLWriter(tmp_path / "events")
        writer.close()
        # _write_record guards on session is not None
        writer._write_record(_record(SampleEventType.ISSUED))

    def test_custom_url(self, tmp_path):
        db_path = tmp_path / "custom.db"
        writer = SQLWriter(tmp_path / "ignored", url=f"sqlite:///{db_path}")
        try:
            writer.write(_record(SampleEventType.ISSUED, uuid="url-test", ts=1))
            writer.flush()
        finally:
            writer.close()

        engine = create_engine(f"sqlite:///{db_path}")
        with Session(engine) as session:
            rows = session.execute(select(EventRowModel)).scalars().all()
            assert len(rows) == 1
            assert rows[0].sample_uuid == "url-test"
        engine.dispose()

    def test_flush_interval(self, tmp_path):
        writer = SQLWriter(tmp_path / "events", flush_interval=3)
        try:
            for i in range(2):
                writer.write(_record(SampleEventType.ISSUED, uuid=f"s{i}", ts=i))

            # Before flush interval, records are not yet committed to a separate reader
            engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
            with Session(engine) as session:
                rows = session.execute(select(EventRowModel)).scalars().all()
                assert len(rows) == 0
            engine.dispose()

            writer.write(_record(SampleEventType.ISSUED, uuid="s2", ts=2))

            # Now at flush_interval=3, should be committed
            engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
            with Session(engine) as session:
                rows = session.execute(select(EventRowModel)).scalars().all()
                assert len(rows) == 3
            engine.dispose()
        finally:
            writer.close()

    def test_mixed_event_types(self, tmp_path):
        writer = SQLWriter(tmp_path / "events", flush_interval=1)
        try:
            writer.write(_record(SessionEventType.STARTED, ts=0))
            writer.write(_record(SampleEventType.ISSUED, uuid="s1", ts=100))
            writer.write(
                _record(
                    ErrorEventType.GENERIC,
                    ts=200,
                    data=ErrorData(error_type="E", error_message="msg"),
                )
            )
            writer.write(_record(SessionEventType.ENDED, ts=300))
        finally:
            writer.close()

        engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
        with Session(engine) as session:
            rows = session.execute(select(EventRowModel)).scalars().all()
            topics = [r.event_type for r in rows]
            assert topics == [
                "session.started",
                "sample.issued",
                "error.generic",
                "session.ended",
            ]
        engine.dispose()
