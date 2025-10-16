"""Defined base class for Ruleset. Different benchmarking competitions can define their own
Ruleset classes and any other necessary requirements for their benchmark.

Such requirements benchmarks may or may not care about are:
- The backend model(s) being run
- The specific datasets being used
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .. import metrics


@dataclass(frozen=True)
class RuntimeSettings:
    """Internal class for runtime settings derived from a UserConfig and Ruleset. This should *never* be
    instantiated by the user, and only by `UserConfig.for_ruleset`."""

    metric_target: metrics.Metric
    reported_metrics: list[metrics.Metric]
    min_duration_ms: int
    max_duration_ms: int
    n_samples_from_dataset: int
    n_samples_to_issue: int
    rng_sched: random.Random
    rng_sample_index: random.Random

    def total_samples_to_issue(self, padding_factor: float = 1.1) -> int:
        """Calculate the total number of samples to issue to the SUT throughout the course of the test run.

        If `n_samples_to_issue` is set, then it is returned.
        If it is not set, then it is calculated based on the metric target and minimum test duration.

        Args:
            padding_factor (float): Factor to multiply the expected number of samples by to account for variance.
                                    Use 1.0 for no padding. (Default: 1.1)

        Returns:
            int: The total number of samples to issue to the SUT throughout the course of the test run.
        """
        if self.n_samples_to_issue:
            return self.n_samples_to_issue

        if isinstance(self.metric_target, metrics.Throughput):
            expected_sps = self.metric_target.target
            expected_samples = expected_sps * (self.min_duration_ms / 1000)
        elif isinstance(self.metric_target, metrics.QueryLatency):
            expected_samples = self.min_duration_ms / self.metric_target.target
        else:
            raise NotImplementedError(
                f"Cannot infer n_samples_to_issue from metric target type: {type(self.metric_target)}"
            )
        return math.ceil(expected_samples * (padding_factor))


@dataclass(frozen=True)
class BenchmarkSuiteRuleset(ABC):
    """Base class for rulesets for benchmarking competitions."""

    version: str
    """Version number of this ruleset for the benchmark suite"""

    scheduler_rng_seed: int | None
    """Random seed for the scheduler. Set to None for unseeded randomization."""

    sample_index_rng_seed: int | None
    """Random seed for the sample index. Set to None for unseeded randomization."""

    @abstractmethod
    def apply_user_config(self, *args, **kwargs) -> RuntimeSettings:
        """Apply a UserConfig to this ruleset to obtain runtime settings. Each benchmark suite may
        define and implement its own subclass of RuntimeSettings for bookkeeping purposes.
        """
        raise NotImplementedError
