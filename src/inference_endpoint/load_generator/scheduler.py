import math
import random
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from functools import partial

from .. import metrics
from ..config.ruleset import RuntimeSettings
from ..dataset_manager.dataloader import DataLoader


class SampleEvent(Enum):
    COMPLETE = "complete"
    FIRST_CHUNK = "first_chunk_received"
    NON_FIRST_CHUNK = "non_first_chunk_received"
    REQUEST_SENT = "request_sent"


@dataclass(frozen=True)
class Sample:
    uuid: int
    callbacks: dict[SampleEvent, Callable]
    get_bytes: Callable


class SampleFactory:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    @staticmethod
    def sample_complete_callback(output, sid=None):
        """Scheduler-specific callback for sample completion.

        Args:
            output: The output of the sample. This is typically the raw bytes of the output.
            sid: The sample ID.
        """
        pass

    @staticmethod
    def sample_get_bytes(dataloader: DataLoader, sample_index: int):
        return dataloader.load_sample(sample_index)

    def get_sample_callbacks(self, sample_index: int) -> dict[SampleEvent, Callable]:
        """Gets the callbacks for the given sample ID."""
        return {
            SampleEvent.COMPLETE: partial(
                self.__class__.sample_complete_callback, sid=sample_index
            ),
        }

    def __call__(self, sample_index: int) -> Sample:
        return Sample(
            uuid=uuid.uuid4().int,
            callbacks=self.get_sample_callbacks(sample_index),
            get_bytes=partial(
                self.__class__.sample_get_bytes, self.dataloader, sample_index
            ),
        )


class SampleOrder(ABC):
    def __init__(
        self, total_samples_to_issue: int, n_samples_in_dataset: int, rng=random
    ):
        """
        Args:
            total_samples_to_issue (int): The total number of samples to issue.
            max_sample_index (int): The maximum sample index.
        """
        self.total_samples_to_issue = total_samples_to_issue
        self.n_samples_in_dataset = n_samples_in_dataset
        self.rng = rng

        self._issued_samples = 0

    def __iter__(self) -> Iterator[int]:
        while self._issued_samples < self.total_samples_to_issue:
            yield self.next_sample_index()
            self._issued_samples += 1

    @abstractmethod
    def next_sample_index(self) -> int:
        raise NotImplementedError


class WithoutReplacementSampleOrder(SampleOrder):
    """Sample order where a sample index cannot repeat until all samples in a dataset have been issued."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_order = list(range(self.n_samples_in_dataset))
        self._curr_idx = (
            self.n_samples_in_dataset + 1
        )  # Ensure we start at an invalid index to force shuffle

    def _reset(self):
        self.rng.shuffle(self.index_order)
        self._curr_idx = 0

    def next_sample_index(self) -> int:
        if self._curr_idx >= len(self.index_order):
            self._reset()
        retval = self.index_order[self._curr_idx]
        self._curr_idx += 1
        return retval


class WithReplacementSampleOrder(SampleOrder):
    """Sample order where a sample index can repeat, even if the dataset has not been exhausted, i.e. sampling with replacement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_sample_index(self) -> int:
        return self.rng.randint(0, self.n_samples_in_dataset - 1)


def uniform_delay_fn(max_delay_ns: int = 0, rng: random.Random = random):
    def _fn():
        if max_delay_ns == 0:
            return 0
        return rng.uniform(0, max_delay_ns)

    return _fn


def poisson_delay_fn(expected_queries_per_second: float, rng: random.Random = random):
    queries_per_ns = expected_queries_per_second / 1e9

    def _fn():
        if queries_per_ns == 0:
            return 0
        return rng.expovariate(lambd=queries_per_ns)  # lambd=1/mean, where mean=latency

    return _fn


class Scheduler:
    """Schedulers are responsible for building queries and determining when they should be issued to the SUT."""

    def __init__(
        self,
        runtime_settings: RuntimeSettings,
        dataloader: DataLoader,
        sample_factory_cls: type[SampleFactory],
        sample_order_cls: type[SampleOrder],
    ):
        self.runtime_settings = runtime_settings
        self.dataloader = dataloader
        self.sample_factory = sample_factory_cls(dataloader)

        self.total_samples_to_issue = (
            runtime_settings.n_samples_to_issue
            if runtime_settings.n_samples_to_issue
            else self.calc_total_samples_to_issue()
        )
        self.n_unique_samples = runtime_settings.n_samples_from_dataset
        self.sample_order = iter(
            sample_order_cls(
                self.total_samples_to_issue,
                self.n_unique_samples,
                rng=self.runtime_settings.rng_sample_index,
            )
        )
        self.delay_fn = None  # Subclasses must set this

    def calc_total_samples_to_issue(self) -> int:
        """Calculate the total number of samples to issue to the SUT throughout the course of the test run.

        If `runtime_settings.n_samples_to_issue` is set, then this method is not called.
        If it is not set, then it is calculated based on the scenario-based Scheduler implementation.

        Returns:
            int: The total number of samples to issue to the SUT throughout the course of the test run.
        """
        metric_target = self.runtime_settings.metric_target
        if isinstance(metric_target, metrics.Throughput):
            expected_sps = metric_target.target
            expected_samples = expected_sps * (
                self.runtime_settings.min_duration_ms / 1000
            )
        elif isinstance(metric_target, metrics.QueryLatency):
            expected_samples = (
                self.runtime_settings.min_duration_ms / metric_target.target
            )
        else:
            raise NotImplementedError(
                f"Scheduler does not support metric target type: {type(metric_target)}"
            )
        return math.ceil(expected_samples * (1.1))  # 10% padding for variance

    def __iter__(self):
        for s_idx in self.sample_order:
            sample = self.sample_factory(s_idx)
            delay_ns = self.delay_fn()
            yield sample, delay_ns
            # Load generator should track last_issue_timestamp, sleep the difference, and then issue.


class MaxThroughputScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_fn = uniform_delay_fn(rng=self.runtime_settings.rng_sched)


class NetworkActivitySimulationScheduler(Scheduler):
    """Simulate client-server network usage behavior by using a Poisson process to issue queries.
    The delay between each sample is sampled from an exponential distribution, centered around the expected latency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_fn = poisson_delay_fn(
            expected_queries_per_second=self.runtime_settings.metric_target.target,
            rng=self.runtime_settings.rng_sched,
        )
