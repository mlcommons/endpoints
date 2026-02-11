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

import msgspec
import pytest
from inference_endpoint.async_utils.transport.record import (
    ErrorEventType,
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
    decode_event_record,
    encode_event_record,
)


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
        assert record.data == {}
        assert isinstance(record.data, dict)


class TestEncodeEventRecord:
    def test_returns_tuple_of_topic_str_and_payload_bytes_with_valid_msgpack(self):
        record = EventRecord(
            event_type=SampleEventType.ISSUED,
            sample_uuid="test-uuid",
            data={"key": "value"},
        )
        topic, payload = encode_event_record(record)
        assert isinstance(topic, str)
        assert topic == "sample.issued"
        assert isinstance(payload, bytes)
        decoded = msgspec.msgpack.decode(payload)
        assert isinstance(decoded, dict)
        assert decoded.get("sample_uuid") == "test-uuid"
        assert decoded.get("data") == {"key": "value"}

    def test_topic_matches_event_type_for_session_sample_error(self):
        for ev, expected_prefix in [
            (SessionEventType.STARTED, "session.started"),
            (SessionEventType.ENDED, "session.ended"),
            (SampleEventType.COMPLETE, "sample.complete"),
            (ErrorEventType.GENERIC, "error.generic"),
        ]:
            topic, _ = encode_event_record(EventRecord(event_type=ev))
            assert topic == expected_prefix


class TestEventRecordRoundTrip:
    def test_session_event_round_trips_with_all_fields(self):
        record = EventRecord(
            event_type=SessionEventType.STARTED,
            sample_uuid="sess-1",
            data={"session_id": "abc"},
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == SessionEventType.STARTED.topic
        assert decoded.sample_uuid == "sess-1"
        assert decoded.data == {"session_id": "abc"}
        assert isinstance(decoded.timestamp_ns, int)
        assert decoded.timestamp_ns == record.timestamp_ns

    def test_sample_event_round_trips(self):
        record = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid="sample-42",
            data={"latency_ns": 1000},
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == SampleEventType.COMPLETE.topic
        assert decoded.sample_uuid == "sample-42"
        assert decoded.data == {"latency_ns": 1000}

    def test_error_event_round_trips_with_defaults(self):
        record = EventRecord(
            event_type=ErrorEventType.LOADGEN,
            data={"message": "error details"},
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == ErrorEventType.LOADGEN.topic
        assert decoded.data == {"message": "error details"}
        assert decoded.sample_uuid == ""

    def test_record_with_only_event_type_round_trips_with_defaults(self):
        record = EventRecord(event_type=SessionEventType.ENDED)
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.event_type.topic == SessionEventType.ENDED.topic
        assert decoded.sample_uuid == ""
        assert decoded.data == {}
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

    def test_nested_and_list_data_round_trips(self):
        record = EventRecord(
            event_type=SampleEventType.TRANSPORT_RECV,
            data={"nested": {"a": 1}, "list": [1, 2, 3]},
        )
        _, payload = encode_event_record(record)
        decoded = decode_event_record(payload)
        assert decoded.data == {"nested": {"a": 1}, "list": [1, 2, 3]}
