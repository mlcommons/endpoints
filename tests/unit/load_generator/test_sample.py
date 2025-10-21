import time
from unittest.mock import patch

from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.load_generator.sample import Sample


def test_sample_uniqueness():
    sample_uuids = [Sample(None).uuid for _ in range(1000)]
    assert len(set(sample_uuids)) == len(sample_uuids), "Sample UUIDs should be unique"


@patch("inference_endpoint.load_generator.sample.EventRecorder.record_event")
def test_sample_callback_times(record_event_mock):
    events = []

    sample = Sample(None)

    def fake_record_event(ev_type: SampleEvent, timestamp_ns: int, sample_uuid: str):
        assert sample_uuid == sample.uuid
        events.append((ev_type, timestamp_ns))

    record_event_mock.side_effect = fake_record_event

    sleep_time_sec = 0.01

    sample.on_first_chunk(None)
    time.sleep(sleep_time_sec)
    sample.on_non_first_chunk(None)
    time.sleep(sleep_time_sec)
    sample.on_complete(None)

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
