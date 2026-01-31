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

"""This module contains a code representation of the rules for the current round of MLPerf Inference.

These values are derived directly from the MLPerf Inference Policies document:
https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
"""

import random
from dataclasses import dataclass, field
from enum import Enum

from endpoints.src.inference_endpoint.config.schema import SystemDefaults

from .... import metrics
from ...ruleset_base import BenchmarkSuiteRuleset
from ...runtime_settings import RuntimeSettings
from ...user_config import UserConfig
from . import models


@dataclass(frozen=True)
class PerModelRuleset:
    # max_samples_memory_capacity: int = None       # Formerly 'performance_sample_count'.
    # Maximum number of samples that can fit in memory. If None, the size of the dataset is used..
    # See notes below for more details on why this is commented out.
    min_duration_ms_valid: int = (
        10 * 60 * 1000
    )  # Minimum duration in milliseconds required for a valid run
    max_duration_ms_valid: int | None = (
        None  # Maximum duration in milliseconds. Used as a timeout / kill for a benchmark run. Set to None for no timeout.
    )
    min_sample_count_valid: int | None = (
        None  # Minimum number of samples required to be sent to the SUT for a valid run, if None, no minimum is enforced
    )
    metric: type[metrics.Metric] | None = (
        None  # any subclass of Metric. Used as metric to evaluate the performance of the benchmark.
    )
    reported_metrics: list[type[metrics.Metric]] = field(
        default_factory=list
    )  # List of metrics that are reported to the submission checker.


@dataclass(frozen=True)
class PerQueryRuleset(PerModelRuleset):
    min_sample_count_valid: int = 270336
    metric: type[metrics.Metric] = metrics.Throughput
    target_latency_percentile: float = (
        99.0  # Percentile of per-query latencies to use for metric comparison
    )
    max_latency_threshold_ms: int | None = (
        None  # Maximum latency threshold in milliseconds for the specified percentile latency allowed for a valid run.
    )
    reported_metrics: list[type[metrics.Metric]] = field(
        default_factory=lambda: [metrics.Throughput]
    )


@dataclass(frozen=True)
class TokenBasedRuleset(PerModelRuleset):
    min_sample_count_valid: int = 270336
    metric: type[metrics.Metric] = metrics.Throughput
    max_ttft_latency_ms: int | None = (
        None  # Maximum TTFT latency in milliseconds allowed for a valid run
    )
    max_tpot_latency_ms: int | None = (
        None  # Maximum TPoT latency in milliseconds allowed for a valid run
    )
    reported_metrics: list[type[metrics.Metric]] = field(
        default_factory=lambda: [metrics.Throughput, metrics.TTFT, metrics.TPOT]
    )


# Notes:
# The following fields' values are taken from submission checker
#  - min_sample_count_valid (from min-queries key)
#  - max_samples_memory_capacity (this is performance_sample_count)
#        Also I think this is completely bizarre... I searched all of github.com/mlcommons/inference for references to performance_sample_count:
#        https://github.com/search?q=repo%3Amlcommons%2Finference%20performance_sample_count&type=code
#        There is not a ***SINGLE*** instance of performance_sample_count being used by *any* component of Loadgen or reference implementation.
#        The only mentions of this value are in documentation, logging the value, bindings, and setting the value to pass into said bindings.
#        As such I'm just deleting it... I think the intention was that it's there for the user to more efficiently load samples to RAM, but in practice
#        because it's a user-provided value, any implementation that *would* use this feature does it on it's own anyway.
#
# The following are taken from mlperf.conf
#  - max_ttft_latency_ms
#  - max_tpot_latency_ms
#  - target_latency_percentile
#  - scheduler_rng_seed
#  - sample_index_rng_seed


class OptimizationPriority(Enum):
    # TODO: Name subject to change
    THROUGHPUT = "Moderate 99% percentile latency, max throughput"
    LOW_LATENCY_INTERACTIVE = (
        "Strict latency constraint for Interactive sessions, TPOT > TTFT > throughput"
    )


@dataclass(frozen=True)
class _RuntimeSettings(RuntimeSettings):
    """MLCommons-specific runtime settings extending base RuntimeSettings.

    This class adds MLCommons-specific fields (model, optimization priority, rules)
    to the base RuntimeSettings. It should never be instantiated by users directly,
    only by RoundRuleset.apply_user_config().

    Extends RuntimeSettings via inheritance to include all base fields plus
    MLCommons-specific configuration.
    """

    model: models._Model
    """The model being benchmarked (e.g., Llama3_1_8b)"""

    optimization_priority: OptimizationPriority
    """Optimization priority (THROUGHPUT or LOW_LATENCY_INTERACTIVE)"""

    rules: PerModelRuleset
    """The specific per-model rules being applied"""


@dataclass(frozen=True)
class RoundRuleset(BenchmarkSuiteRuleset):
    benchmark_rulesets: dict[models._Model, dict[OptimizationPriority, PerModelRuleset]]

    def apply_user_config(
        self,
        model: models._Model,
        user_config: UserConfig,
        opt_prio: OptimizationPriority = OptimizationPriority.THROUGHPUT,
    ) -> _RuntimeSettings:
        if model not in self.benchmark_rulesets:
            raise ValueError(
                f"Model {model.name} not found in rules for round {self.version}"
            )

        # Check if model supports the requested optimization priority
        if opt_prio not in self.benchmark_rulesets[model]:
            raise ValueError(
                f"Model {model.name} does not support optimization priority {opt_prio}"
            )

        ruleset = self.benchmark_rulesets[model][opt_prio]

        metric_target: metrics.Metric | None = None
        if user_config.user_metric_target and ruleset.metric:
            metric_target = ruleset.metric(user_config.user_metric_target)

        reported_metrics: list[metrics.Metric] = []
        for mtype in ruleset.reported_metrics:
            if mtype == ruleset.metric and metric_target:
                reported_metrics.append(metric_target)
            elif mtype == metrics.TTFT:
                assert isinstance(ruleset, TokenBasedRuleset)
                reported_metrics.append(metrics.TTFT(ruleset.max_ttft_latency_ms))
            elif mtype == metrics.TPOT:
                assert isinstance(ruleset, TokenBasedRuleset)
                reported_metrics.append(metrics.TPOT(ruleset.max_tpot_latency_ms))
            elif mtype == metrics.QueryLatency and ruleset.metric == metrics.Throughput:
                # If we specify throughput and want to also report per query latency, infer latency from inverting qps.
                reported_metrics.append(
                    metrics.QueryLatency(target_qps=user_config.user_metric_target)
                )
            elif mtype == metrics.Throughput and ruleset.metric == metrics.QueryLatency:
                assert user_config.user_metric_target is not None
                # If we specify per query latency, infer qps by inverting
                target_qps = 1000 / user_config.user_metric_target
                reported_metrics.append(metrics.Throughput(target_qps=target_qps))
            else:
                raise ValueError(
                    f"Invalid metric type: {mtype} for ruleset type {ruleset.__class__.__name__}"
                )

        min_duration_ms = ruleset.min_duration_ms_valid
        if user_config.min_duration_ms is not None:
            min_duration_ms = user_config.min_duration_ms

        max_duration_ms = ruleset.max_duration_ms_valid
        if user_config.max_duration_ms is not None:
            max_duration_ms = user_config.max_duration_ms
        assert (
            max_duration_ms is not None and max_duration_ms >= min_duration_ms
        ), "Max duration must be greater than or equal to min duration"

        n_samples_from_dataset = model.dataset.size
        if user_config.ds_subset_size:
            n_samples_from_dataset = user_config.ds_subset_size

        total_sample_count = None
        if user_config.total_sample_count:
            total_sample_count = user_config.total_sample_count

        min_sample_count = ruleset.min_sample_count_valid
        if user_config.min_sample_count is not None:
            min_sample_count = user_config.min_sample_count

        return _RuntimeSettings(
            metric_target=metric_target
            if metric_target is not None
            else SystemDefaults.DEFAULT_METRIC,
            reported_metrics=reported_metrics,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            n_samples_from_dataset=n_samples_from_dataset,
            n_samples_to_issue=total_sample_count,
            min_sample_count=min_sample_count if min_sample_count is not None else 1,
            rng_sched=random.Random(self.scheduler_rng_seed),
            rng_sample_index=random.Random(self.sample_index_rng_seed),
            load_pattern=None,  # not part user config
            optimization_priority=opt_prio,
            model=model,
            rules=ruleset,
        )


_v5_1 = RoundRuleset(
    version="v5.1",
    scheduler_rng_seed=18209322760996052031,
    sample_index_rng_seed=14771362308971278857,
    benchmark_rulesets={
        models.DeepSeek_R1: {
            OptimizationPriority.THROUGHPUT: TokenBasedRuleset(
                max_ttft_latency_ms=2000, max_tpot_latency_ms=80
            )
        },
        models.Llama3_1_8b: {
            OptimizationPriority.THROUGHPUT: TokenBasedRuleset(
                max_ttft_latency_ms=2000, max_tpot_latency_ms=100
            ),
            OptimizationPriority.LOW_LATENCY_INTERACTIVE: TokenBasedRuleset(
                max_ttft_latency_ms=500, max_tpot_latency_ms=30
            ),
        },
        models.Llama2_70b: {
            OptimizationPriority.THROUGHPUT: TokenBasedRuleset(
                max_ttft_latency_ms=2000, max_tpot_latency_ms=200
            ),
            OptimizationPriority.LOW_LATENCY_INTERACTIVE: TokenBasedRuleset(
                max_ttft_latency_ms=450, max_tpot_latency_ms=40
            ),
        },
        models.Llama3_1_405b: {
            OptimizationPriority.THROUGHPUT: TokenBasedRuleset(
                max_ttft_latency_ms=6000, max_tpot_latency_ms=175
            ),
            OptimizationPriority.LOW_LATENCY_INTERACTIVE: TokenBasedRuleset(
                max_ttft_latency_ms=4500, max_tpot_latency_ms=80
            ),
        },
        models.Mixtral8x7B: {
            OptimizationPriority.THROUGHPUT: TokenBasedRuleset(
                max_ttft_latency_ms=2000, max_tpot_latency_ms=200
            ),
        },
    },
)


CURRENT = _v5_1
