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
import uuid
from collections import namedtuple
from unittest.mock import patch

import pytest
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
from inference_endpoint.metrics.recorder import (
    EventRecorder,
    EventRecorderSingletonViolation,
    sqlite3_cursor,
)
from inference_endpoint.metrics.reporter import MetricsReporter


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


def test_record_event(events_db, sample_uuids):
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
        (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000),
        (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003),
        (uuid1, SampleEvent.FIRST_CHUNK.value, 10010),
        (uuid2, SampleEvent.FIRST_CHUNK.value, 10190),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10201),
        (uuid3, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10202),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10203),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10210),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10211),
        (uuid1, SampleEvent.COMPLETE.value, 10211),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10214),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10217),
        (uuid2, SampleEvent.NON_FIRST_CHUNK.value, 10219),
        (uuid2, SampleEvent.COMPLETE.value, 10219),
    ]

    assert expected_rows == actual_rows
    assert len(actual_rows) == 14


def worker_proc_read_entries(sess_id, events_created_ev, uuid1, uuid2):
    events_created_ev.wait()
    expected_rows = [
        (uuid1, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10000),
        (uuid2, SessionEvent.LOADGEN_ISSUE_CALLED.value, 10003),
        (uuid1, SampleEvent.FIRST_CHUNK.value, 10010),
        (uuid2, SampleEvent.FIRST_CHUNK.value, 10190),
        (uuid1, SampleEvent.NON_FIRST_CHUNK.value, 10201),
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

    worker_proc.join(timeout=5)
    if worker_proc.is_alive():
        worker_proc.terminate()
        worker_proc.join(timeout=1)
        assert (
            not worker_proc.is_alive()
        ), "Worker process could not be terminated after cleanup"
        raise AssertionError("Worker process failed to complete in a reasonable time")
    assert worker_proc.exitcode == 0


def test_sample_counting(events_db):
    with MetricsReporter(events_db) as reporter:
        stats = reporter.get_sample_statuses()
        assert stats["completed"] == 2
        assert stats["in_flight"] == 1


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
