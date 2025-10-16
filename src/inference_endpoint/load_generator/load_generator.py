import time
from abc import ABC, abstractmethod

from ..utils import sleep_ns
from .sample import Sample, SampleFactory
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
        sample_factory: SampleFactory,
    ):
        self.sample_issuer = sample_issuer
        self.sample_factory = sample_factory

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


class SchedulerBasedLoadGenerator(LoadGenerator):
    def __init__(
        self,
        sample_issuer: SampleIssuer,
        sample_factory: SampleFactory,
        scheduler: Scheduler,
    ):
        super().__init__(sample_issuer, sample_factory)

        self.scheduler = scheduler
        self._iterator = None
        self.last_issue_timestamp_ns = 0

    def __next__(self) -> tuple[Sample, int]:
        # Let raised StopIteration be propagated up the stack
        s_idx, delay_ns = next(self._iterator)
        sample = self.sample_factory(s_idx)

        scheduled_issue_timestamp_ns = self.last_issue_timestamp_ns + delay_ns
        while (now := time.monotonic_ns()) < scheduled_issue_timestamp_ns:
            sleep_ns(scheduled_issue_timestamp_ns - now)
        self.last_issue_timestamp_ns = (
            time.monotonic_ns()
        )  # Timestamp when issue is called
        self.sample_issuer.issue(sample)
        return sample, self.last_issue_timestamp_ns

    def __iter__(self):
        if self._iterator is not None:
            raise RuntimeError(
                "SchedulerBasedLoadGenerator can only be iterated over once"
            )
        self._iterator = iter(self.scheduler)
        return self
