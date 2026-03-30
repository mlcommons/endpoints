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

import pytest
from inference_endpoint.core.types import QueryResult, StreamChunk
from inference_endpoint.load_generator.sample import (
    Sample,
    SampleEvent,
    SampleEventHandler,
)


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


def test_sample_callback_times():
    """Test that hooks are invoked in the correct order for stream chunks and completion."""
    events = []

    sample = Sample(None)
    first_chunk = StreamChunk(id=sample.uuid, metadata={"first_chunk": True})
    non_first_chunk = StreamChunk(id=sample.uuid, metadata={"first_chunk": False})
    complete_result = QueryResult(id=sample.uuid)

    def record_first_chunk(chunk):
        events.append((SampleEvent.FIRST_CHUNK, time.monotonic_ns()))

    def record_non_first_chunk(chunk):
        events.append((SampleEvent.NON_FIRST_CHUNK, time.monotonic_ns()))

    def record_complete(result):
        events.append((SampleEvent.COMPLETE, time.monotonic_ns()))

    SampleEventHandler.register_hook(SampleEvent.FIRST_CHUNK, record_first_chunk)
    SampleEventHandler.register_hook(
        SampleEvent.NON_FIRST_CHUNK, record_non_first_chunk
    )
    SampleEventHandler.register_hook(SampleEvent.COMPLETE, record_complete)

    sleep_time_sec = 0.01

    SampleEventHandler.stream_chunk_complete(first_chunk)
    time.sleep(sleep_time_sec)
    SampleEventHandler.stream_chunk_complete(non_first_chunk)
    time.sleep(sleep_time_sec)
    SampleEventHandler.query_result_complete(complete_result)

    assert len(events) == 3

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

    SampleEventHandler.clear_hooks()


def test_sample_invalid_type_errors():
    chunk = StreamChunk(id="123", metadata={"first_chunk": True})
    result = QueryResult(id="123")

    with pytest.raises(AssertionError, match="Invalid chunk type"):
        SampleEventHandler.stream_chunk_complete(result)

    with pytest.raises(AssertionError, match="Invalid result type"):
        SampleEventHandler.query_result_complete(chunk)


def test_sample_event_handler_register_hook():
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
