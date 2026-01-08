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

"""
TODO: PoC only, subject to change!

Runtime settings for benchmark execution.

This module contains the canonical RuntimeSettings dataclass that represents
the immutable configuration derived from user YAML configs and rulesets.
This is the single source of truth for runtime configuration.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import metrics

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .ruleset_base import BenchmarkSuiteRuleset
    from .schema import BenchmarkConfig, LoadPattern


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Immutable runtime settings for benchmark execution.

    This class represents the final configuration derived from user YAML configs
    and ruleset constraints. It should never be instantiated directly by users,
    but rather created through:
    - Ruleset.apply_user_config() for ruleset-constrained configs
    - RuntimeSettings.from_config() factory method (to be added in Phase 3)

    All fields are immutable (frozen dataclass) to prevent accidental modification
    during benchmark execution.
    """

    metric_target: metrics.Metric
    """Primary metric to target (e.g., Throughput(100) for 100 QPS)"""

    reported_metrics: list[metrics.Metric]
    """List of metrics to collect and report"""

    min_duration_ms: int
    """Minimum benchmark duration in milliseconds"""

    max_duration_ms: int
    """Maximum benchmark duration in milliseconds (timeout)"""

    n_samples_from_dataset: int
    """Number of samples to load from dataset"""

    n_samples_to_issue: int | None
    """Total number of samples to issue to SUT (None = calculate automatically)"""

    min_sample_count: int
    """Minimum number of samples required for valid run"""

    rng_sched: random.Random
    """Random number generator for scheduler"""

    rng_sample_index: random.Random
    """Random number generator for sample indexing"""

    load_pattern: LoadPattern | None
    """Load pattern configuration"""

    @classmethod
    def from_config(
        cls,
        config: BenchmarkConfig,
        dataloader_num_samples: int,
        ruleset: BenchmarkSuiteRuleset | None = None,
        **overrides,
    ) -> RuntimeSettings:
        """Create RuntimeSettings from BenchmarkConfig.

        This is the primary factory method for creating RuntimeSettings from
        a validated BenchmarkConfig (Pydantic model).

        Args:
            config: Validated BenchmarkConfig from yaml_loader
            dataloader_num_samples: Number of samples loaded from dataset
            ruleset: Optional ruleset to apply constraints (delegates to ruleset's apply_user_config)
            **overrides: Additional fields to override (e.g., for testing)

        Returns:
            Immutable RuntimeSettings instance

        Note: If a ruleset is provided, it would handle the conversion with competition-specific logic.
        For now, we use default conversion. Full ruleset integration is deferred to Phase 4.
        """
        if ruleset is not None:
            # Ruleset handles conversion with competition-specific logic
            # This would need UserConfig which we don't have in the current CLI flow
            # For now, we use default conversion even if ruleset is provided
            # Full ruleset integration is deferred to Phase 4
            pass

        return cls._from_config_default(config, dataloader_num_samples, **overrides)

    @classmethod
    def _from_config_default(
        cls, config: BenchmarkConfig, dataloader_num_samples: int, **overrides
    ) -> RuntimeSettings:
        """Default conversion from BenchmarkConfig to RuntimeSettings.

        This method extracts values from the BenchmarkConfig Pydantic model
        and builds an immutable RuntimeSettings dataclass.

        Args:
            config: Validated BenchmarkConfig
            dataloader_num_samples: Number of samples from loaded dataset
            **overrides: Additional field overrides

        Returns:
            Immutable RuntimeSettings dataclass
        """
        # Extract settings from immutable Pydantic models
        runtime_cfg = config.settings.runtime
        load_pattern_cfg = config.settings.load_pattern

        # TODO: The default target_qps should be None in Offline mode, but we use 10.0 for now.
        # This is a temporary solution to avoid breaking changes.
        effective_qps = (
            load_pattern_cfg.target_qps
            if load_pattern_cfg.target_qps is not None
            else 10.0
        )

        # Build kwargs from Pydantic models
        kwargs = {
            "metric_target": metrics.Throughput(effective_qps),
            "reported_metrics": [metrics.Throughput(effective_qps)],
            "min_duration_ms": runtime_cfg.min_duration_ms,
            "max_duration_ms": runtime_cfg.max_duration_ms,
            "n_samples_from_dataset": dataloader_num_samples,
            "n_samples_to_issue": runtime_cfg.n_samples_to_issue,  # From config (CLI --num-samples or YAML)
            "min_sample_count": 1,
            "rng_sched": random.Random(runtime_cfg.scheduler_random_seed),
            "rng_sample_index": random.Random(runtime_cfg.dataloader_random_seed),
            "load_pattern": load_pattern_cfg,
        }

        # Apply overrides
        kwargs.update(overrides)

        return cls(**kwargs)

    def total_samples_to_issue(
        self, padding_factor: float = 1.1, align_to_dataset_size: bool = True
    ) -> int:
        """Calculate the total number of samples to issue to the SUT throughout the course of the test run.

        Priority:
        1. If `n_samples_to_issue` is set, return it (explicit override)
        2. If min_duration_ms=0, return all dataset samples (new CLI default)
        3. Otherwise, calculate from metric target * duration

        Args:
            padding_factor (float): Factor to multiply the expected number of samples by to account for variance.
                                    Use 1.0 for no padding. (Default: 1.1)
            align_to_dataset_size (bool): Whether to pad the total number of samples up to the nearest multiple of
                                          dataset size. (Default: True)

        Returns:
            int: The total number of samples to issue to the SUT throughout the course of the test run.
        """
        # min_sample is not in effect here (CLI dominated), it will be used in the ruleset.
        if self.n_samples_to_issue is not None:
            logger.debug(
                f"Sample count: {self.n_samples_to_issue} (explicit override via --num-samples or YAML n_samples_to_issue)"
            )
            return self.n_samples_to_issue

        # If min_duration is 0, use all dataset samples (new CLI default behavior)
        if self.min_duration_ms == 0:
            result = max(self.min_sample_count, self.n_samples_from_dataset)
            logger.debug(
                f"Sample count: {result} (using all dataset samples, duration=0)"
            )
            return result

        # Calculate from duration and metric target
        if isinstance(self.metric_target, metrics.Throughput):
            expected_sps = self.metric_target.target
            expected_samples = expected_sps * (self.min_duration_ms / 1000)
        elif isinstance(self.metric_target, metrics.QueryLatency):
            expected_samples = self.min_duration_ms / self.metric_target.target
        else:
            raise NotImplementedError(
                f"Cannot infer n_samples_to_issue from metric target type: {type(self.metric_target)}"
            )

        result = max(
            self.min_sample_count, math.ceil(expected_samples * padding_factor)
        )
        logger.debug(
            f"Sample count: {result} (calculated from duration={self.min_duration_ms}ms × target_qps={self.metric_target.target} × padding={padding_factor})"
        )

        # Pad to multiples of dataset size
        if (
            align_to_dataset_size
            and self.n_samples_from_dataset > 0
            and (rem := result % self.n_samples_from_dataset) != 0
        ):
            result += self.n_samples_from_dataset - rem
            logger.debug(f"Padded sample count: {result}")
        return result
