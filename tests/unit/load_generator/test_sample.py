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

import time
from unittest.mock import patch

import pytest
from inference_endpoint.core.types import ErrorData, QueryResult, StreamChunk
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.load_generator.sample import Sample, SampleEventHandler


def test_sample_uniqueness():
    sample_uuids = [Sample(None).uuid for _ in range(1000)]
    assert len(set(sample_uuids)) == len(sample_uuids), "Sample UUIDs should be unique"


def test_sample_lazy_data_loading():
    sample = Sample(None)
    sample.data = "test_data"
    assert sample.data == "test_data"

    with pytest.raises(AttributeError):
        sample.data = "test_data2"


def test_sample_eager_data_loading():
    sample = Sample("my data")

    with pytest.raises(AttributeError):
        sample.data = "test_data2"

    assert sample.data == "my data"


@patch("inference_endpoint.load_generator.sample.EventRecorder.record_event")
def test_sample_callback_times(record_event_mock):
    events = []

    sample = Sample(None)
    first_chunk = StreamChunk(id=sample.uuid, metadata={"first_chunk": True})
    non_first_chunk = StreamChunk(id=sample.uuid, metadata={"first_chunk": False})
    complete_result = QueryResult(id=sample.uuid)

    def fake_record_event(
        ev_type: SampleEvent, timestamp_ns: int, sample_uuid: str, **kwargs
    ):
        assert sample_uuid == sample.uuid
        events.append((ev_type, timestamp_ns))

    record_event_mock.side_effect = fake_record_event

    sleep_time_sec = 0.01

    SampleEventHandler.stream_chunk_complete(first_chunk)
    time.sleep(sleep_time_sec)
    SampleEventHandler.stream_chunk_complete(non_first_chunk)
    time.sleep(sleep_time_sec)
    SampleEventHandler.query_result_complete(complete_result)

    assert len(events) == 3
    assert record_event_mock.call_count == 3

    assert events[0][0] == SampleEvent.FIRST_CHUNK
    assert events[1][0] == SampleEvent.NON_FIRST_CHUNK
    assert events[2][0] == SampleEvent.COMPLETE
    assert events[0][1] < events[1][1]
    assert events[1][1] < events[2][1]

    # Times are in nanoseconds - convert to seconds to compare with sleep time
    tpot_1_sec = (events[1][1] - events[0][1]) / 1e9
    tpot_2_sec = (events[2][1] - events[1][1]) / 1e9

    # Resolution of time.sleep is very coarse, so simply check that duration is
    # greater than the sleep time
    assert tpot_1_sec > sleep_time_sec
    assert tpot_2_sec > sleep_time_sec


@patch("inference_endpoint.load_generator.sample.EventRecorder.record_event")
def test_sample_invalid_type_errors(record_event_mock):
    record_event_mock.return_value = None

    chunk = StreamChunk(id="123", metadata={"first_chunk": True})
    result = QueryResult(id="123")

    with pytest.raises(AssertionError, match="Invalid chunk type"):
        SampleEventHandler.stream_chunk_complete(result)

    with pytest.raises(AssertionError, match="Invalid result type"):
        SampleEventHandler.query_result_complete(chunk)


@patch("inference_endpoint.load_generator.sample.EventRecorder.record_event")
def test_sample_event_handler_register_hook(record_event_mock):
    record_event_mock.return_value = None

    progress_counter = [0, 0]

    def progress_hook(_):
        progress_counter[1] += 1

    def non_first_chunk_hook(_):
        progress_counter[0] += 1

    SampleEventHandler.register_hook(SampleEvent.COMPLETE, progress_hook)
    SampleEventHandler.register_hook(SampleEvent.NON_FIRST_CHUNK, non_first_chunk_hook)

    SampleEventHandler.stream_chunk_complete(
        StreamChunk(id="123", metadata={"first_chunk": True})
    )
    assert progress_counter == [0, 0]

    SampleEventHandler.query_result_complete(QueryResult(id="123"))
    assert progress_counter == [0, 1]

    SampleEventHandler.stream_chunk_complete(
        StreamChunk(id="123", metadata={"first_chunk": True})
    )
    assert progress_counter == [0, 1]

    SampleEventHandler.stream_chunk_complete(
        StreamChunk(id="123", metadata={"first_chunk": False})
    )
    assert progress_counter == [1, 1]


@patch("inference_endpoint.load_generator.sample.record_exception")
@patch("inference_endpoint.load_generator.sample.EventRecorder.record_event")
def test_query_result_complete_does_not_advance_conversation_on_error(
    record_event_mock, record_exception_mock, clean_sample_event_hooks
):
    record_event_mock.return_value = None
    record_exception_mock.return_value = None

    manager = ConversationManager()
    state = manager.get_or_create("conv_001", None)
    manager.mark_turn_issued("conv_001", 1, "Hello")
    SampleEventHandler.set_conversation_manager(manager)

    result = QueryResult(
        id="123",
        metadata={"conversation_id": "conv_001", "turn_number": 1},
        error=ErrorData(error_type="TimeoutError", error_message="request failed"),
    )

    SampleEventHandler.query_result_complete(result)

    # mark_turn_failed advances the conversation and adds error placeholder
    assert state.current_turn == 2  # pending_user_turn (1) + 1
    assert state.pending_user_turn is None  # Cleared after failure
    assert state.message_history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "[ERROR: Turn failed or timed out]"},
    ]
    record_exception_mock.assert_called_once()
