import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..metrics.recorder import EventRecorder
from .events import SampleEvent


class Sample:
    """Represents a sample for the SampleIssuer to send to the inference endpoint."""

    __slots__ = ["uuid", "data", "completed"]

    def __init__(self, data: Any):
        # 128-bit UUID might be a little overkill for our use case, we can investigate slimming down memory usage
        self.uuid = uuid.uuid4().hex

        self.data = data
        self.completed = False

    def on_first_chunk(self, chunk: Any):
        EventRecorder.record_event(
            SampleEvent.FIRST_CHUNK,
            time.monotonic_ns(),
            sample_uuid=self.uuid,
        )

        # Avoid unused argument linter error. Implement logging the output in the future.
        _ = chunk

    def on_non_first_chunk(self, chunk: Any):
        EventRecorder.record_event(
            SampleEvent.NON_FIRST_CHUNK,
            time.monotonic_ns(),
            sample_uuid=self.uuid,
        )

        # Avoid unused argument linter error. Implement logging the output in the future.
        _ = chunk

    def on_complete(self, chunks: list[Any]):
        if self.completed:
            raise RuntimeError(f"Sample {self.uuid} has already been completed")

        EventRecorder.record_event(
            SampleEvent.COMPLETE,
            time.monotonic_ns(),
            sample_uuid=self.uuid,
        )
        self.completed = True

        # Avoid unused argument linter error. Implement logging the output in the future.
        _ = chunks


@dataclass
class IssuedSample:
    """Contains data about a sample that has been issued to the inference endpoint.

    SampleIssuer is not allowed to know the actual sample index of the data to prevent cheating
    and response caching. This class contains metadata about the sample for bookkeeping by the
    LoadGenerator and BenchmarkSession.
    """

    sample: Sample
    index: int
    issue_timestamp_ns: int
