from enum import Enum


class Event(Enum):
    pass


class SessionEvent(Event):
    TEST_STARTED = "test_started"
    TEST_ENDED = "test_ended"


class SampleEvent(Event):
    COMPLETE = "complete"
    FIRST_CHUNK = "first_chunk_received"
    NON_FIRST_CHUNK = "non_first_chunk_received"
    REQUEST_SENT = "request_sent"
