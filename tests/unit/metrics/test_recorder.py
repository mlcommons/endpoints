# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import multiprocessing
import sqlite3
import uuid
from collections import namedtuple
from unittest.mock import patch

import orjson
import pytest
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
from inference_endpoint.metrics.recorder import (
    EventRecorder,
    EventRecorderSingletonViolation,
    EventRow,
    sqlite3_cursor,
)


def test_event_row_to_table_query():
    """Test that EventRow.to_table_query() generates correct SQL CREATE TABLE statement."""
    query = EventRow.to_table_query()

    # Verify it's a CREATE TABLE statement
    assert query.startswith("CREATE TABLE IF NOT EXISTS events")

    # Verify all expected fields are present with correct types
    assert "sample_uuid TEXT" in query
    assert "event_type TEXT" in query
    assert "timestamp_ns INTEGER" in query
    assert "data BLOB" in query

    # Verify the query is valid SQL by executing it
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()

    # Verify the table was created with correct schema
    cursor.execute("PRAGMA table_info(events)")
    columns = cursor.fetchall()

    # Extract column names and types
    column_info = {col[1]: col[2] for col in columns}  # col[1] is name, col[2] is type

    assert "sample_uuid" in column_info
    assert "event_type" in column_info
    assert "timestamp_ns" in column_info
    assert "data" in column_info
    assert column_info["timestamp_ns"] == "INTEGER"
    assert column_info["data"] == "BLOB"

    cursor.close()
    conn.close()


def test_event_row_insert_query():
    """Test that EventRow.insert_query() generates correct SQL INSERT statement."""
    query = EventRow.insert_query()

    # Verify it's an INSERT statement with correct structure
    assert query.startswith("INSERT INTO events")
    assert "VALUES" in query

    # Verify all fields are included
    assert "sample_uuid" in query
    assert "event_type" in query
    assert "timestamp_ns" in query
    assert "data" in query

    # Count placeholders - should be 4 (one for each field)
    placeholder_count = query.count("?")
    assert placeholder_count == 4

    # Verify the query is valid SQL by creating a table and inserting
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(EventRow.to_table_query())

    # Try inserting a row using the generated query
    test_data = ("test_uuid", "TEST_EVENT", 12345, b"test_data")
    cursor.execute(query, test_data)
    conn.commit()

    # Verify the data was inserted
    cursor.execute("SELECT * FROM events")
    rows = cursor.fetchall()
    assert len(rows) == 1
    assert rows[0] == test_data

    cursor.close()
    conn.close()


def test_event_row_to_insert_params(sample_uuids):
    """Test that EventRow.to_insert_params() returns correct tuple for SQL insertion."""
    uuid1 = sample_uuids(1)
    test_data = {"key": "value", "number": 42}

    event_row = EventRow(
        sample_uuid=uuid1,
        event_type=SampleEvent.FIRST_CHUNK,
        timestamp_ns=10000,
        data=orjson.dumps(test_data),
    )

    params = event_row.to_insert_params()

    # Verify the tuple has correct structure
    assert isinstance(params, tuple)
    assert len(params) == 4

    # Verify each field
    assert params[0] == uuid1
    assert params[1] == SampleEvent.FIRST_CHUNK.value
    assert params[2] == 10000
    assert params[3] == orjson.dumps(test_data)


def test_event_row_to_insert_params_empty_data(sample_uuids):
    """Test EventRow.to_insert_params() with empty data field."""
    uuid1 = sample_uuids(1)

    event_row = EventRow(
        sample_uuid=uuid1,
        event_type=SessionEvent.LOADGEN_ISSUE_CALLED,
        timestamp_ns=5000,
        data=b"",
    )

    params = event_row.to_insert_params()

    assert params[0] == uuid1
    assert params[1] == SessionEvent.LOADGEN_ISSUE_CALLED.value
    assert params[2] == 5000
    assert params[3] == b""


def test_event_row_integration_with_sqlite(sample_uuids):
    """Integration test: Create table, insert EventRow, and verify data roundtrip."""
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create table using EventRow.to_table_query()
    cursor.execute(EventRow.to_table_query())
    conn.commit()

    # Create test events with various data types
    events = [
        EventRow(
            sample_uuid=uuid1,
            event_type=SessionEvent.LOADGEN_ISSUE_CALLED,
            timestamp_ns=10000,
            data=b"",
        ),
        EventRow(
            sample_uuid=uuid2,
            event_type=SampleEvent.FIRST_CHUNK,
            timestamp_ns=10100,
            data=orjson.dumps({"chunk": "Hello"}),
        ),
        EventRow(
            sample_uuid=uuid2,
            event_type=SampleEvent.COMPLETE,
            timestamp_ns=10200,
            data=orjson.dumps({"output": ["Hello", " World"]}),
        ),
    ]

    # Insert using EventRow.insert_query()
    insert_query = EventRow.insert_query()
    for event in events:
        cursor.execute(insert_query, event.to_insert_params())
    conn.commit()

    # Verify data was inserted correctly
    cursor.execute("SELECT * FROM events")
    rows = cursor.fetchall()

    assert len(rows) == 3

    # Verify first row (empty data)
    assert rows[0][0] == uuid1
    assert rows[0][1] == SessionEvent.LOADGEN_ISSUE_CALLED.value
    assert rows[0][2] == 10000
    assert rows[0][3] == b""

    # Verify second row (with JSON data)
    assert rows[1][0] == uuid2
    assert rows[1][1] == SampleEvent.FIRST_CHUNK.value
    assert rows[1][2] == 10100
    assert orjson.loads(rows[1][3]) == {"chunk": "Hello"}

    # Verify third row (with complex JSON data)
    assert rows[2][0] == uuid2
    assert rows[2][1] == SampleEvent.COMPLETE.value
    assert rows[2][2] == 10200
    assert orjson.loads(rows[2][3]) == {"output": ["Hello", " World"]}

    cursor.close()
    conn.close()


def get_EventRecorder(*args, **kwargs):
    # Set requirement to 128MB for testing
    return EventRecorder(*args, min_memory_req_bytes=128 * 1024 * 1024, **kwargs)


def test_event_recorder_singleton_violation_create_multiple():
    with get_EventRecorder():
        with pytest.raises(EventRecorderSingletonViolation):
            with get_EventRecorder():
                pass


def test_event_recorder_singleton_violation_close_non_active():
    with get_EventRecorder():
        other_rec = get_EventRecorder()
        with pytest.raises(EventRecorderSingletonViolation):
            other_rec.close()


def test_event_recorder_singleton_violation_record_event_non_active(sample_uuids):
    assert (
        EventRecorder.LIVE is None
    ), "Cannot run test - EventRecorder is active from previous test"
    with pytest.raises(EventRecorderSingletonViolation):
        EventRecorder.record_event(
            SessionEvent.LOADGEN_ISSUE_CALLED, 10000, sample_uuid=sample_uuids(1)
        )
    assert (
        EventRecorder.record_event(
            SessionEvent.LOADGEN_ISSUE_CALLED,
            10000,
            sample_uuid=sample_uuids(1),
            assert_active=False,
        )
        is False
    )


def test_record_event(sample_uuids):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)
    uuid3 = sample_uuids(3)

    with get_EventRecorder() as rec:
        rec.record_event(SessionEvent.LOADGEN_ISSUE_CALLED, 10000, sample_uuid=uuid1)
        rec.record_event(SessionEvent.LOADGEN_ISSUE_CALLED, 10003, sample_uuid=uuid2)
        rec.record_event(SampleEvent.FIRST_CHUNK, 10010, sample_uuid=uuid1)
        rec.record_event(SampleEvent.FIRST_CHUNK, 10190, sample_uuid=uuid2)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10201, sample_uuid=uuid1)
        rec.record_event(SessionEvent.LOADGEN_ISSUE_CALLED, 10202, sample_uuid=uuid3)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10203, sample_uuid=uuid1)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10210, sample_uuid=uuid2)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10211, sample_uuid=uuid1)
        rec.record_event(SampleEvent.COMPLETE, 10211, sample_uuid=uuid1)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10214, sample_uuid=uuid2)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10217, sample_uuid=uuid2)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10219, sample_uuid=uuid2)
        rec.record_event(SampleEvent.COMPLETE, 10219, sample_uuid=uuid2)

        # Wait for writer thread to process all events
        rec.wait_for_writes()

        # Read from the database directly
        with sqlite3_cursor(rec.connection_name) as (cursor, _):
            actual_rows = cursor.execute("SELECT * FROM events").fetchall()

    expected_rows = [
        (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
        (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
        (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
        (uuid2, SampleEvent.FIRST_CHUNK.value, 10190, b""),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10201, b""),
        (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10202, b""),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10203, b""),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10210, b""),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10211, b""),
        (uuid1, SampleEvent.COMPLETE.value, 10211, b""),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10214, b""),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10217, b""),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10219, b""),
        (uuid2, SampleEvent.COMPLETE.value, 10219, b""),
    ]

    assert expected_rows == actual_rows
    assert len(actual_rows) == 14


def worker_proc_read_entries(sess_id, events_created_ev, uuid1, uuid2):
    events_created_ev.wait()
    expected_rows = [
        (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000, b""),
        (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003, b""),
        (uuid1, SampleEvent.FIRST_CHUNK.value, 10010, b""),
        (uuid2, SampleEvent.FIRST_CHUNK.value, 10190, b""),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10201, b""),
    ]
    with sqlite3_cursor(EventRecorder.db_path(sess_id)) as (cursor, _):
        actual_rows = cursor.execute("SELECT * FROM events").fetchall()
    assert expected_rows == actual_rows


def test_shm_usage(sample_uuids):
    uuid1 = sample_uuids(1)
    uuid2 = sample_uuids(2)

    # Set mp start method
    ctx = multiprocessing.get_context("spawn")
    events_created_ev = ctx.Event()
    sess_id = uuid.uuid4().hex

    worker_proc = ctx.Process(
        target=worker_proc_read_entries, args=(sess_id, events_created_ev, uuid1, uuid2)
    )
    worker_proc.start()

    with get_EventRecorder(session_id=sess_id) as rec:
        rec.record_event(SessionEvent.LOADGEN_ISSUE_CALLED, 10000, sample_uuid=uuid1)
        rec.record_event(SessionEvent.LOADGEN_ISSUE_CALLED, 10003, sample_uuid=uuid2)
        rec.record_event(SampleEvent.FIRST_CHUNK, 10010, sample_uuid=uuid1)
        rec.record_event(SampleEvent.FIRST_CHUNK, 10190, sample_uuid=uuid2)
        rec.record_event(SampleEvent.NON_FIRST_CHUNK, 10201, sample_uuid=uuid1)
        # Wait for writer thread to process all events
        rec.wait_for_writes()
    events_created_ev.set()

    worker_proc.join(timeout=10)
    if worker_proc.is_alive():
        worker_proc.terminate()
        worker_proc.join(timeout=1)
        assert (
            not worker_proc.is_alive()
        ), "Worker process could not be terminated after cleanup"
        raise AssertionError("Worker process failed to complete in a reasonable time")
    assert worker_proc.exitcode == 0


MemStat = namedtuple("MemStat", ["total", "used", "free"])


@patch("inference_endpoint.metrics.recorder.shutil.disk_usage")
def test_shm_too_small(mock_run):
    mock_run.return_value = MemStat(
        total=64 * 1024 * 1024, used=0, free=64 * 1024 * 1024
    )
    with pytest.raises(MemoryError) as err:
        EventRecorder(
            min_memory_req_bytes=512 * 1024 * 1024
        )  # Instantiate will not init the connection
    assert "total space" in err.value.args[0]


@patch("inference_endpoint.metrics.recorder.shutil.disk_usage")
def test_shm_not_enough_space(mock_run):
    mock_run.return_value = MemStat(
        total=1024 * 1024 * 1024, used=64 * 1024 * 1024, free=960 * 1024 * 1024
    )
    with pytest.raises(MemoryError) as err:
        EventRecorder(
            min_memory_req_bytes=1024 * 1024 * 1024
        )  # Instantiate will not init the connection
    assert "free space" in err.value.args[0]
    assert "960MB" in err.value.args[0]
