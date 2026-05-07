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

"""Tests for RecordWriter base class flush-interval logic."""

import pytest
from inference_endpoint.async_utils.services.event_logger.writer import RecordWriter
from inference_endpoint.core.record import EventRecord, SampleEventType


class ConcreteWriter(RecordWriter):
    """Minimal concrete writer for testing the base class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.written: list[EventRecord] = []
        self.flush_count = 0
        self.closed = False

    def _write_record(self, record: EventRecord) -> None:
        self.written.append(record)

    def close(self) -> None:
        self.closed = True

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _make_record(uuid: str = "s1", ts: int = 0) -> EventRecord:
    return EventRecord(
        event_type=SampleEventType.ISSUED, timestamp_ns=ts, sample_uuid=uuid
    )


@pytest.mark.unit
class TestRecordWriterFlushInterval:
    def test_no_auto_flush_when_interval_is_none(self):
        writer = ConcreteWriter(flush_interval=None)
        for _ in range(200):
            writer.write(_make_record())
        assert writer.flush_count == 0
        assert len(writer.written) == 200

    def test_auto_flush_at_interval(self):
        writer = ConcreteWriter(flush_interval=5)
        for _ in range(5):
            writer.write(_make_record())
        assert writer.flush_count == 1
        assert len(writer.written) == 5

    def test_auto_flush_repeats(self):
        writer = ConcreteWriter(flush_interval=3)
        for _ in range(9):
            writer.write(_make_record())
        assert writer.flush_count == 3

    def test_no_flush_before_interval_reached(self):
        writer = ConcreteWriter(flush_interval=10)
        for _ in range(9):
            writer.write(_make_record())
        assert writer.flush_count == 0

    def test_manual_flush_resets_counter(self):
        writer = ConcreteWriter(flush_interval=5)
        for _ in range(3):
            writer.write(_make_record())
        assert writer.flush_count == 0

        writer.flush()
        assert writer.flush_count == 1

        # After manual flush, counter is reset — need 5 more writes
        for _ in range(4):
            writer.write(_make_record())
        assert writer.flush_count == 1  # still 1, not yet 5 since last flush

        writer.write(_make_record())
        assert writer.flush_count == 2

    def test_flush_interval_of_one(self):
        writer = ConcreteWriter(flush_interval=1)
        for _ in range(5):
            writer.write(_make_record())
        assert writer.flush_count == 5

    def test_write_delegates_to_subclass(self):
        writer = ConcreteWriter(flush_interval=None)
        record = _make_record(uuid="test-uuid", ts=42)
        writer.write(record)
        assert writer.written == [record]

    def test_close_is_callable(self):
        writer = ConcreteWriter(flush_interval=None)
        writer.close()
        assert writer.closed
