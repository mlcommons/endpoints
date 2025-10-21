import time
from abc import ABC, abstractmethod
from typing import Any

from ..dataset_manager.dataloader import DataLoader
from ..metrics.recorder import EventRecorder
from ..utils import sleep_ns
from .events import SessionEvent
from .sample import IssuedSample, Sample
from .scheduler import Scheduler


class SampleIssuer(ABC):
    """Abstract base class for SampleIssuers. SampleIssuers are responsible for ingesting samples,
    building requests from those samples, and sending those requests to the SUT endpoint.
    """

    def start(self):  # noqa: B027
        """Optional setup method to be called once the instance is created to set up any dependency
        components.
        """
        pass

    @abstractmethod
    def issue(self, sample: Sample):
        """Issue a sample to the SUT endpoint."""
        raise NotImplementedError

    def shutdown(self):  # noqa: B027
        """Optional teardown method to be called when the instance is no longer needed."""
        pass


class LoadGenerator(ABC):
    def __init__(
        self,
        sample_issuer: SampleIssuer,
        sample_class: type[Sample],
        dataloader: DataLoader,
    ):
        self.sample_issuer = sample_issuer
        self.sample_class = sample_class
        self.dataloader = dataloader

    @abstractmethod
    def __next__(self) -> tuple[Sample, int]:
        """Issues the next sample according to the Load Generator strategy.
        Note that this method should only return once the sample has been issued,
        and any blocking / delay mechanism should be done here.

        Returns:
            Sample: The sample issued
            int: The timestamp that the sample was issued at
        """
        raise NotImplementedError

    def __iter__(self):
        return self

    def load_sample_data(self, sample_index: int, sample_uuid: str) -> Any:
        sample_data = self.dataloader.load_sample(sample_index)
        EventRecorder.record_event(
            SessionEvent.LOADGEN_DATA_LOAD,
            time.monotonic_ns(),
            sample_uuid=sample_uuid,
        )
        return sample_data

    def issue_sample(self, sample: Sample) -> int:
        """Invoke the SampleIssuer, recording the issue call timestamp"""
        timestamp_ns = time.monotonic_ns()

        # Currently, EventRecorder will raise an Exception if the in-flight sample
        # counter is negative. This happens if the SampleIssuer somehow invokes a
        # SampleEvent.COMPLETE event before the record_event call for LOADGEN_ISSUE_CALLED
        # goes off.
        # This can be solved by just recording the issue() call right before actually
        # invoking it. If this timing mechanism is a problem, we can remove the
        # negative check in EventRecorder, since the order of insertions doesn't matter
        # as much if the timestamps are correct.
        EventRecorder.record_event(
            SessionEvent.LOADGEN_ISSUE_CALLED,
            timestamp_ns,
            sample_uuid=sample.uuid,
        )
        self.sample_issuer.issue(sample)
        return timestamp_ns


class SchedulerBasedLoadGenerator(LoadGenerator):
    def __init__(
        self,
        sample_issuer: SampleIssuer,
        sample_class: type[Sample],
        dataloader: DataLoader,
        scheduler: Scheduler,
    ):
        super().__init__(sample_issuer, sample_class, dataloader)

        self.scheduler = scheduler
        self._iterator = None
        self.last_issue_timestamp_ns = 0

    def __next__(self) -> IssuedSample:
        # Let raised StopIteration be propagated up the stack
        s_idx, delay_ns = next(self._iterator)

        # Data loading is not timed for Time-to-Token metrics. It is assumed that the
        # hypothetical user would have put the data into memory available for a network
        # request beforehand.
        sample = self.sample_class(None)  # Create sample object first to generate uuid
        sample.data = self.load_sample_data(s_idx, sample.uuid)

        scheduled_issue_timestamp_ns = self.last_issue_timestamp_ns + delay_ns
        while (now := time.monotonic_ns()) < scheduled_issue_timestamp_ns:
            sleep_ns(scheduled_issue_timestamp_ns - now)
        self.last_issue_timestamp_ns = self.issue_sample(sample)
        return IssuedSample(sample, s_idx, self.last_issue_timestamp_ns)

    def __iter__(self):
        if self._iterator is not None:
            raise RuntimeError(
                "SchedulerBasedLoadGenerator can only be iterated over once"
            )
        self._iterator = iter(self.scheduler)
        return self
