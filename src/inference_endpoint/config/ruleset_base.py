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

"""Base class for benchmark rulesets.

This module defines the abstract interface that all benchmark competition rulesets
must implement. Different benchmarking competitions (e.g., MLCommons, custom competitions)
can define their own Ruleset classes with specific requirements and constraints.

Rulesets are responsible for:
- Defining competition-specific constraints (min duration, sample counts, etc.)
- Validating user configurations against competition rules
- Transforming user configurations into RuntimeSettings for execution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime_settings import RuntimeSettings


@dataclass(frozen=True)
class BenchmarkSuiteRuleset(ABC):
    """Base class for rulesets for benchmarking competitions.

    Each competition (e.g., MLCommons MLPerf Inference) can define its own ruleset
    by inheriting from this class and implementing the abstract methods.

    Rulesets define:
    - Version information for the competition rules
    - Random seeds for reproducibility
    - Validation logic for user configurations
    - Transformation logic from user config to RuntimeSettings
    """

    version: str
    """Version number of this ruleset for the benchmark suite"""

    scheduler_rng_seed: int | None
    """Random seed for the scheduler. Set to None for unseeded randomization."""

    sample_index_rng_seed: int | None
    """Random seed for the sample index. Set to None for unseeded randomization."""

    @abstractmethod
    def apply_user_config(self, *args, **kwargs) -> RuntimeSettings:
        """Apply a UserConfig to this ruleset to obtain runtime settings.

        Each benchmark suite may define and implement its own subclass of RuntimeSettings
        for bookkeeping purposes. This method should:
        1. Validate the user configuration against ruleset constraints
        2. Apply ruleset-specific defaults and requirements
        3. Create and return an immutable RuntimeSettings instance

        Args:
            *args, **kwargs: Arguments specific to each ruleset implementation

        Returns:
            RuntimeSettings: Immutable runtime configuration ready for execution

        Raises:
            ValueError: If configuration violates ruleset constraints
        """
        raise NotImplementedError
