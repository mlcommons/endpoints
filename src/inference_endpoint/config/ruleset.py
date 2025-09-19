"""Defined base class for Ruleset. Different benchmarking competitions can define their own
Ruleset classes and any other necessary requirements for their benchmark.

Such requirements benchmarks may or may not care about are:
- The backend model(s) being run
- The specific datasets being used
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass

import random

from .. import metrics


@dataclass(frozen=True)
class RuntimeSettings:
    """Internal class for runtime settings derived from a UserConfig and Ruleset. This should *never* be instantiated by the user, and only by `UserConfig.for_ruleset`.
    """
    metric_target: metrics.Metric
    reported_metrics: list[metrics.Metric]
    min_duration_ms: int
    max_duration_ms: int
    n_samples_from_dataset: int
    n_samples_to_issue: int
    rng_sched: random.Random
    rng_sample_index: random.Random


@dataclass(frozen=True)
class BenchmarkSuiteRuleset(ABC):
    """Base class for rulesets for benchmarking competitions.
    """
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
