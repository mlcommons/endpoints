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

"""Unit tests for EventRecord and related types (serialization / deserialization)."""

import time

import pytest
from inference_endpoint.core.record import (
    TOPIC_FRAME_SIZE,
    ErrorEventType,
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
    decode_event_record,
    encode_event_record,
)
from inference_endpoint.core.types import ErrorData, TextModelOutput


class TestEventType:
    def test_category_base_raises_subclasses_return_expected(self):
        with pytest.raises(AttributeError):
            EventType.category()
        assert SessionEventType.category() == "session"
        assert ErrorEventType.category() == "error"
        assert SampleEventType.category() == "sample"

    def test_topic_returns_category_dot_value(self):
        assert SessionEventType.STARTED.topic == "session.started"
        assert SessionEventType.STARTED.value == "started"

        assert SampleEventType.COMPLETE.topic == "sample.complete"
        assert SampleEventType.COMPLETE.value == "complete"

        assert ErrorEventType.GENERIC.topic == "error.generic"
        assert ErrorEventType.GENERIC.value == "generic"

    def test_members_are_instance_of_event_type_and_behave_as_strings(self):
        assert isinstance(SessionEventType.STARTED, EventType)
        assert isinstance(ErrorEventType.GENERIC, EventType)
        assert isinstance(SampleEventType.COMPLETE, EventType)
        assert SessionEventType.STARTED.value == "started"
        assert SampleEventType.ISSUED.value == "issued"


class TestEventRecordConstruction:
    def test_construction_with_only_event_type_uses_defaults(self):
        before = time.monotonic_ns()
        record = EventRecord(event_type=SessionEventType.STARTED)
        after = time.monotonic_ns()
        assert before <= record.timestamp_ns <= after
        assert record.sample_uuid == ""
        assert record.data is None


class TestEncodeEventRecord:
    def test_returns_tuple_of_topic_bytes_padded_and_payload_bytes_with_valid_msgpack(
        self,
    ):
        """encode_event_record returns (topic_bytes_padded, payload) for single-frame ZMQ."""
        record = EventRecord(
            event_type=SampleEventType.ISSUED,
            sample_uuid="test-uuid",
            data="test-output",
        )
        topic_bytes, payload = encode_event_record(record)
        assert isinstance(topic_bytes, bytes)
        assert len(topic_bytes) == TOPIC_FRAME_SIZE
        assert topic_bytes.rstrip(b"\x00") == b"sample.issued"
        assert isinstance(payload, bytes)
        decoded = decode_event_record(payload)
        assert decoded.sample_uuid == "test-uuid"
        assert decoded.data == "test-output"

    def test_topic_bytes_padded_matches_event_type_for_session_sample_error(self):
        """Topic is null-padded to TOPIC_FRAME_SIZE for single-frame ZMQ sends."""
        for ev, expected_prefix in [
            (SessionEventType.STARTED, "session.started"),
            (SessionEventType.ENDED, "session.ended"),
            (SampleEventType.COMPLETE, "sample.complete"),
            (ErrorEventType.GENERIC, "error.generic"),
        ]:
            topic_bytes, _ = encode_event_record(EventRecord(event_type=ev))
            assert len(topic_bytes) == TOPIC_FRAME_SIZE
            assert topic_bytes.rstrip(b"\x00") == expected_prefix.encode("utf-8")


class TestEventRecordRoundTrip:
    def test_session_event_round_trips_with_all_fields(self):
        record = EventRecord(
            event_type=SessionEventType.STARTED,
            sample_uuid="sess-1",
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == SessionEventType.STARTED.topic
        assert decoded.sample_uuid == "sess-1"
        assert decoded.data is None
        assert isinstance(decoded.timestamp_ns, int)
        assert decoded.timestamp_ns == record.timestamp_ns

    def test_sample_event_round_trips_with_output(self):
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="sample-42",
            data="output text",
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == SampleEventType.COMPLETE.topic
        assert decoded.sample_uuid == "sample-42"
        assert decoded.data == "output text"

    def test_sample_event_round_trips_with_text_model_output(self):
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="sample-42",
            data=TextModelOutput(output="out", reasoning="reason"),
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == SampleEventType.COMPLETE.topic
        assert decoded.sample_uuid == "sample-42"
        assert isinstance(decoded.data, TextModelOutput)
        assert decoded.data.output == "out"
        assert decoded.data.reasoning == "reason"

    def test_error_event_round_trips_with_error_data(self):
        record = EventRecord(
            event_type=ErrorEventType.LOADGEN,
            data=ErrorData(
                error_type="LoadgenError",
                error_message="error details",
            ),
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == ErrorEventType.LOADGEN.topic
        assert isinstance(decoded.data, ErrorData)
        assert decoded.data.error_type == "LoadgenError"
        assert decoded.data.error_message == "error details"
        assert decoded.sample_uuid == ""

    def test_record_with_only_event_type_round_trips_with_defaults(self):
        record = EventRecord(event_type=SessionEventType.ENDED)
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == SessionEventType.ENDED.topic
        assert decoded.sample_uuid == ""
        assert decoded.data is None
        assert decoded.timestamp_ns > 0

    def test_explicit_timestamp_ns_preserved_round_trip(self):
        ts = 1234567890
        record = EventRecord(
            event_type=SampleEventType.ISSUED,
            timestamp_ns=ts,
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.timestamp_ns == ts
