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

import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from ..config.runtime_settings import RuntimeSettings
from ..config.schema import LoadPatternType
from .sample import SampleEvent, SampleEventHandler


class SampleOrder(ABC):
    """Abstract base class for sample ordering strategies.

    SampleOrder determines which dataset sample to use next when issuing queries.
    Different strategies enable different testing scenarios:

    The SampleOrder is an iterator that yields sample indices from the dataset.
    It handles wrapping around when total_samples_to_issue > dataset size.

    Attributes:
        total_samples_to_issue: Total number of samples to issue during benchmark.
        n_samples_in_dataset: Number of unique samples available in dataset.
        rng: Random number generator for reproducible randomness.
        _issued_samples: Counter of samples issued so far.
    """

    def __init__(
        self, total_samples_to_issue: int, n_samples_in_dataset: int, rng=random
    ):
        """Initialize sample ordering strategy.

        Args:
            total_samples_to_issue: The total number of samples to issue.
                                   May be larger than n_samples_in_dataset.
            n_samples_in_dataset: The number of unique samples in the dataset.
            rng: Random number generator (for reproducibility via seeding).
        """
        self.total_samples_to_issue = total_samples_to_issue
        self.n_samples_in_dataset = n_samples_in_dataset
        self.rng = rng

        self._issued_samples = 0

    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices to issue.

        Yields sample indices until total_samples_to_issue is reached.

        Yields:
            Sample index (0 to n_samples_in_dataset-1).
        """
        while self._issued_samples < self.total_samples_to_issue:
            yield self.next_sample_index()
            self._issued_samples += 1

    @abstractmethod
    def next_sample_index(self) -> int:
        """Get the next sample index to issue.

        Returns:
            Sample index (0 to n_samples_in_dataset-1).
        """
        raise NotImplementedError


class WithoutReplacementSampleOrder(SampleOrder):
    """Sample ordering without replacement - shuffle dataset, use all samples before repeating.

    This strategy ensures balanced coverage of the dataset:
    1. Shuffles all dataset indices randomly
    2. Issues them one by one until exhausted
    3. Reshuffles and repeats if more samples needed

    Use this for:
    - Fair benchmarking (all samples used equally)
    - Avoiding bias from repeated samples
    - Deterministic results with seed control

    Example with 3-sample dataset, 7 samples to issue:
    - Shuffle: [2, 0, 1]
    - Issue: 2, 0, 1 (first pass)
    - Reshuffle: [1, 2, 0]
    - Issue: 1, 2, 0, 1 (second pass, partial)

    Attributes:
        index_order: Current shuffled order of indices.
        _curr_idx: Position in current shuffle (resets after each complete pass).
    """

    def __init__(self, *args, **kwargs):
        """Initialize without-replacement sample ordering.

        Args:
            *args: Forwarded to SampleOrder.__init__.
            **kwargs: Forwarded to SampleOrder.__init__.
        """
        super().__init__(*args, **kwargs)
        self.index_order = list(range(self.n_samples_in_dataset))
        self._curr_idx = (
            self.n_samples_in_dataset + 1
        )  # Ensure we start at an invalid index to force shuffle

    def _reset(self):
        """Shuffle indices and reset position for next pass."""
        self.rng.shuffle(self.index_order)
        self._curr_idx = 0

    def next_sample_index(self) -> int:
        """Get next sample index from current shuffle, reshuffling if needed.

        Returns:
            Sample index from dataset.
        """
        if self._curr_idx >= len(self.index_order):
            self._reset()
        retval = self.index_order[self._curr_idx]
        self._curr_idx += 1
        return retval


class WithReplacementSampleOrder(SampleOrder):
    """Sample ordering with replacement - truly random sampling from dataset.

    Each sample is chosen uniformly at random from the entire dataset,
    independent of previous choices. The same sample can (and will) appear
    multiple times, even consecutively.

    Use this for:
    - Stress testing with realistic randomness
    - Simulating unpredictable user behavior
    - When dataset coverage balance is not important

    Example with 3-sample dataset, 7 samples to issue:
    - Might produce: [1, 1, 0, 2, 1, 0, 0]
    - Note repeated samples even without exhausting dataset
    """

    def __init__(self, *args, **kwargs):
        """Initialize with-replacement sample ordering.

        Args:
            *args: Forwarded to SampleOrder.__init__.
            **kwargs: Forwarded to SampleOrder.__init__.
        """
        super().__init__(*args, **kwargs)

    def next_sample_index(self) -> int:
        """Get random sample index from dataset.

        Returns:
            Random sample index (uniform distribution over dataset).
        """
        return self.rng.randint(0, self.n_samples_in_dataset - 1)


def uniform_delay_fn(
    max_delay_ns: int = 0, rng: random.Random | None = None
) -> Callable[[], float]:
    """Create a uniform delay function for schedulers.

    Returns a function that generates delays uniformly distributed between
    0 and max_delay_ns. Used for max throughput (max_delay_ns=0) or uniform
    load distribution.

    Args:
        max_delay_ns: Maximum delay in nanoseconds. If 0, always returns 0 (no delay).
        rng: Random number generator for reproducibility.

    Returns:
        Function that returns delay in nanoseconds (float).
    """
    rng = rng or random.Random()

    def _fn():
        if max_delay_ns == 0:
            return 0
        return rng.uniform(0, max_delay_ns)

    return _fn


def poisson_delay_fn(
    expected_queries_per_second: float, rng: random.Random | None = None
) -> Callable[[], float]:
    """Create a Poisson-distributed delay function for realistic online benchmarking.

    Returns a function that generates delays following an exponential distribution
    (inter-arrival times of a Poisson process). This models realistic user/client
    behavior where requests arrive independently at a target rate.

    The exponential distribution has the property that:
    - Mean inter-arrival time = 1 / expected_qps
    - Variance = mean^2 (high variability, realistic for network traffic)

    Args:
        expected_queries_per_second: Target QPS (queries per second).
        rng: Random number generator for reproducibility.

    Returns:
        Function that returns delay in nanoseconds (float).
    """
    rng = rng or random.Random()
    queries_per_ns = expected_queries_per_second / 1e9

    def _fn():
        if queries_per_ns == 0:
            return 0
        return rng.expovariate(lambd=queries_per_ns)  # lambd=1/mean, where mean=latency

    return _fn


class Scheduler:
    """Base class for query scheduling strategies that control benchmark load patterns.

    Schedulers determine:
    1. Sample ordering (which sample to use next)
    2. Timing delays (when to issue the next query)

    They combine a SampleOrder (what to issue) with a delay function (when to issue)
    to produce a stream of (sample_index, delay_ns) pairs.

    Scheduler implementations auto-register via __init_subclass__ by specifying
    the load_pattern parameter. This enables runtime selection of schedulers:

        scheduler_cls = Scheduler.get_implementation(LoadPatternType.POISSON)
        scheduler = scheduler_cls(runtime_settings, sample_order_cls)

    Built-in schedulers:
    - MaxThroughputScheduler: Issues all queries immediately (offline mode)
    - PoissonDistributionScheduler: Poisson-distributed delays (online mode)
    - ConcurrencyScheduler: Fixed concurrency level (online mode)

    Attributes:
        _IMPL_MAP: Class-level registry mapping LoadPatternType to Scheduler classes.
        runtime_settings: Runtime configuration (QPS, duration, seeds, etc.).
        total_samples_to_issue: Total queries to issue during benchmark.
        n_unique_samples: Number of unique samples in dataset.
        sample_order: Iterator over sample indices to use.
        delay_fn: Function returning delay before next query (nanoseconds).
    """

    # Registry for scheduler implementations (populated via __init_subclass__)
    _IMPL_MAP: dict[LoadPatternType, type["Scheduler"]] = {}

    def __init__(
        self,
        runtime_settings: RuntimeSettings,
        sample_order_cls: type[SampleOrder],
    ):
        """Initialize scheduler with runtime settings and sample ordering strategy.

        Args:
            runtime_settings: Runtime configuration containing QPS, duration, seeds.
            sample_order_cls: SampleOrder class to use for sample selection.
        """
        self.runtime_settings = runtime_settings

        self.total_samples_to_issue = runtime_settings.total_samples_to_issue()
        self.n_unique_samples = runtime_settings.n_samples_from_dataset
        self.sample_order = iter(
            sample_order_cls(
                self.total_samples_to_issue,
                self.n_unique_samples,
                rng=self.runtime_settings.rng_sample_index,
            )
        )
        self.delay_fn = None  # Subclasses must set this

    def __iter__(self):
        """Iterate over (sample_index, delay_ns) pairs.

        Yields:
            Tuple of (sample_index, delay_ns):
            - sample_index: Index of sample to issue next
            - delay_ns: Nanoseconds to wait before issuing
        """
        for s_idx in self.sample_order:
            yield s_idx, self.delay_fn()

    def __init_subclass__(cls, load_pattern: LoadPatternType | None = None, **kwargs):
        """Auto-register scheduler implementations.

        Args:
            load_pattern: LoadPatternType to bind this scheduler to

        Raises:
            ValueError: If load_pattern already registered
        """
        super().__init_subclass__(**kwargs)

        if load_pattern is not None:
            if load_pattern in Scheduler._IMPL_MAP:
                raise ValueError(
                    f"Cannot bind {cls.__name__} to {load_pattern} - "
                    f"Already bound to {Scheduler._IMPL_MAP[load_pattern].__name__}"
                )
            Scheduler._IMPL_MAP[load_pattern] = cls

    @classmethod
    def get_implementation(cls, load_pattern: LoadPatternType) -> type["Scheduler"]:
        """Get scheduler implementation for load pattern.

        Args:
            load_pattern: LoadPatternType enum

        Returns:
            Scheduler subclass

        Raises:
            NotImplementedError: If no implementation registered
            KeyError: If load_pattern invalid
        """
        if load_pattern not in cls._IMPL_MAP:
            available_str = ", ".join(p.value for p in cls._IMPL_MAP.keys())
            raise KeyError(
                f"No scheduler registered for '{load_pattern.value}'. "
                f"Available: {available_str}"
            )
        return cls._IMPL_MAP[load_pattern]


class MaxThroughputScheduler(Scheduler, load_pattern=LoadPatternType.MAX_THROUGHPUT):
    """Offline max throughput scheduler (all queries at t=0).

    Auto-registers for LoadPatternType.MAX_THROUGHPUT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_fn = uniform_delay_fn(rng=self.runtime_settings.rng_sched)


class PoissonDistributionScheduler(Scheduler, load_pattern=LoadPatternType.POISSON):
    """Poisson-distributed query scheduler for online benchmarking.

    Simulates realistic client-server network usage by using a Poisson process
    to issue queries. The delay between each sample is sampled from an exponential
    distribution, centered around the expected latency based on target QPS.

    Use this scheduler for online latency testing with sustained QPS.

    Auto-registers for LoadPatternType.POISSON.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_fn = poisson_delay_fn(
            expected_queries_per_second=self.runtime_settings.metric_target.target,
            rng=self.runtime_settings.rng_sched,
        )


class ConcurrencyScheduler(Scheduler, load_pattern=LoadPatternType.CONCURRENCY):
    """Concurrency-based scheduler that maintains fixed concurrent requests.

    Issues queries based on COMPLETION events rather than time delays.
    Maintains target concurrency level (e.g., always 32 requests in-flight).

    Auto-registers for LoadPatternType.CONCURRENCY.
    """

    def __init__(self, runtime_settings: RuntimeSettings, sample_order_cls):
        super().__init__(runtime_settings, sample_order_cls)
        assert runtime_settings.load_pattern is not None
        target_concurrency = runtime_settings.load_pattern.target_concurrency
        if target_concurrency is None or target_concurrency <= 0:
            raise ValueError(
                f"target_concurrency must be > 0 for CONCURRENCY load pattern, got {target_concurrency}"
            )

        # Use threading.Condition for concurrency control with explicit counter
        self._condition = threading.Condition()
        self._inflight = 0
        self._target_concurrency = target_concurrency

        # Register completion hook - free up slot when query completes
        SampleEventHandler.register_hook(SampleEvent.COMPLETE, self._release_slot)

        # Unused (required by Scheduler interface)
        self.delay_fn = lambda: None

    def _release_slot(self, result=None):
        """Release a concurrency slot and notify waiting threads.

        Args:
            result: QueryResult from completed query (unused, required by hook signature)
        """
        with self._condition:
            self._inflight -= 1
            self._condition.notify()

    def __iter__(self):
        """
        Iterate over sample indices to issue.
        Yields sample indices until total_samples_to_issue is reached.

        Waits for available concurrency slot before yielding each sample index.
        """
        for s_idx in self.sample_order:
            with self._condition:
                while self._inflight >= self._target_concurrency:
                    self._condition.wait()
                self._inflight += 1
            yield s_idx, 0
