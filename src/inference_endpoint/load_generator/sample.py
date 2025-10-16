import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from ..dataset_manager.dataloader import DataLoader
from ..metrics.recorder import EventRecorder
from .events import SampleEvent


@dataclass(frozen=True)
class Sample:
    uuid: str
    callbacks: dict[SampleEvent, Callable]
    get_bytes: Callable


class SampleCallback:
    """Represents a callback function for a sample, which is called when the corresponding
    event occurs.

    A SampleCallback should take a single value when called, corresponding to the event type.
    The type of this value depends on the SampleIssuer used that is handling the data, requests,
    and invoking the callbacks. It is up to the implementer to ensure compatibility between the
    SampleFactory and SampleIssuer being used during a BenchmarkSession.

    - SampleEvent.REQUEST_SENT: The value is ignored, and can be anything when passed in.
    - SampleEvent.FIRST_CHUNK: The value represents the chunk of data received by the SUT endpoint.
    - SampleEvent.NON_FIRST_CHUNK: The value represents the chunk of data received by the SUT endpoint.
    - SampleEvent.COMPLETE: An Iterable containing all the chunks received by the SUT endpoint, in the
      order they were received.
    """

    def __call__(self, value: Any):
        pass


class EventRecordCallback(SampleCallback):
    """Callback that records an event to the event recorder."""

    def __init__(
        self, event_recorder: EventRecorder, sample_uuid: str, ev_type: SampleEvent
    ):
        self.event_recorder = event_recorder
        self.sample_uuid = sample_uuid
        self.ev_type = ev_type

    def __call__(self, value: Any):
        self.event_recorder.record_event(
            ev_type=self.ev_type,
            sample_uuid=self.sample_uuid,
            timestamp_ns=time.monotonic_ns(),
        )


class SampleFactory:
    def __init__(self, dataloader: DataLoader, event_recorder: EventRecorder):
        self.dataloader = dataloader
        self.event_recorder = event_recorder

    @staticmethod
    def sample_get_bytes(dataloader: DataLoader, sample_index: int):
        return dataloader.load_sample(sample_index)

    def get_sample_callbacks(
        self, sample_index: int, sample_uuid: str
    ) -> dict[SampleEvent, Callable]:
        """Gets the callbacks for the given sample ID.

        Args:
            sample_index (int): Index of the sample.
            sample_uuid (str): Unique identifier for the sample.

        Returns:
            dict[SampleEvent, Callable]: A dictionary mapping SampleEvents to callbacks for the given sample ID.
        """
        _ = sample_index  # Explicitly ignore unused argument to avoid linter warnings.
        return {
            ev: EventRecordCallback(
                event_recorder=self.event_recorder, sample_uuid=sample_uuid, ev_type=ev
            )
            for ev in SampleEvent
        }

    def __call__(self, sample_index: int) -> Sample:
        sample_uuid = uuid.uuid4().hex
        return Sample(
            uuid=sample_uuid,
            callbacks=self.get_sample_callbacks(sample_index, sample_uuid),
            get_bytes=partial(
                self.__class__.sample_get_bytes, self.dataloader, sample_index
            ),
        )
