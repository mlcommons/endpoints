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

"""Tests for JSONLWriter."""

import json

import msgspec
import pytest
from inference_endpoint.async_utils.services.event_logger.file_writer import (
    JSONLWriter,
)
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import ErrorData


def _record(event_type, uuid="", ts=0, data=None):
    return EventRecord(
        event_type=event_type, timestamp_ns=ts, sample_uuid=uuid, data=data
    )


# ---------------------------------------------------------------------------
# JSONLWriter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONLWriter:
    def test_creates_file_with_jsonl_extension(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events")
        try:
            assert writer.file_path.suffix == ".jsonl"
            assert writer.file_path.exists()
        finally:
            writer.close()

    def test_writes_valid_jsonl(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        try:
            writer.write(_record(SampleEventType.ISSUED, uuid="s1", ts=1000))
            writer.write(_record(SampleEventType.COMPLETE, uuid="s1", ts=2000))
        finally:
            writer.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_record_roundtrip_fields(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        try:
            writer.write(_record(SampleEventType.ISSUED, uuid="abc-123", ts=999))
        finally:
            writer.close()

        line = (tmp_path / "events.jsonl").read_text().strip()
        json.loads(line)  # Verify it's valid JSON

        # EventRecord is array_like, so decode with msgspec to verify fields
        decoder = msgspec.json.Decoder(EventRecord, dec_hook=EventType.decode_hook)
        decoded = decoder.decode(line.encode("utf-8"))
        assert decoded.sample_uuid == "abc-123"
        assert decoded.timestamp_ns == 999

    def test_session_event_encoded_as_topic(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        try:
            writer.write(_record(SessionEventType.ENDED, ts=42))
        finally:
            writer.close()

        line = (tmp_path / "events.jsonl").read_text().strip()
        # The encode_hook converts EventType to its topic string
        assert "session.ended" in line

    def test_error_data_encoded(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        err = ErrorData(error_type="TestError", error_message="something broke")
        try:
            writer.write(
                _record(SampleEventType.COMPLETE, uuid="err-1", ts=100, data=err)
            )
        finally:
            writer.close()

        line = (tmp_path / "events.jsonl").read_text().strip()
        assert "TestError" in line
        assert "something broke" in line

    def test_flush_interval_with_jsonl_writer(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events", flush_interval=3)
        try:
            for i in range(3):
                writer.write(_record(SampleEventType.ISSUED, uuid=f"s{i}", ts=i))
            # After flush_interval writes the file should be flushed
            content = (tmp_path / "events.jsonl").read_text()
            lines = [line for line in content.strip().split("\n") if line]
            assert len(lines) == 3
        finally:
            writer.close()

    def test_close_flushes_remaining(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events", flush_interval=100)
        writer.write(_record(SampleEventType.ISSUED, uuid="s1", ts=1))
        writer.close()

        content = (tmp_path / "events.jsonl").read_text()
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) == 1

    def test_multiple_writes_preserve_order(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events", flush_interval=1)
        try:
            for i in range(10):
                writer.write(_record(SampleEventType.ISSUED, uuid=f"s{i}", ts=i * 100))
        finally:
            writer.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 10

        decoder = msgspec.json.Decoder(EventRecord, dec_hook=EventType.decode_hook)
        timestamps = [decoder.decode(line.encode()).timestamp_ns for line in lines]
        assert timestamps == list(range(0, 1000, 100))

    def test_close_idempotent(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events")
        writer.close()
        writer.close()  # should not raise

    def test_write_after_close_is_noop(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events")
        writer.close()
        # _write_record guards on file_obj is not None
        writer._write_record(_record(SampleEventType.ISSUED))

    def test_flush_after_close_is_noop(self, tmp_path):
        writer = JSONLWriter(tmp_path / "events")
        writer.close()
        writer.flush()  # should not raise
