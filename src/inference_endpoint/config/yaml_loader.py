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

YAML configuration loading and merging with CLI arguments."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import (
    BenchmarkConfig,
    LoadPatternType,
    TestType,
)

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration error."""

    pass


class ConfigLoader:
    """Load and validate YAML configuration files."""

    @staticmethod
    def load_yaml(path: Path) -> BenchmarkConfig:
        """Load and validate YAML config file.

        Args:
            path: Path to YAML config file

        Returns:
            Validated BenchmarkConfig

        Raises:
            ConfigError: If file not found or validation fails

        Note: Delegates to BenchmarkConfig.from_yaml_file() to avoid duplication.
        """
        try:
            config = BenchmarkConfig.from_yaml_file(path)
            logger.info(f"Loaded config: {config.name} (type: {config.type})")
            return config
        except FileNotFoundError as e:
            raise ConfigError(str(e)) from e
        except (yaml.YAMLError, ValidationError) as e:
            raise ConfigError(f"Config validation failed: {e}") from e

    @staticmethod
    def validate_config(config: BenchmarkConfig, benchmark_mode=None) -> None:
        """Validate configuration consistency.

        This method validates the BenchmarkConfig but does NOT modify it.
        Immutable configs should not be changed - any issues should raise errors.

        Args:
            config: BenchmarkConfig to validate
            benchmark_mode: BenchmarkMode enum (OFFLINE or ONLINE), or string, or None

        Raises:
            ConfigError: If configuration is invalid

        Note: Uses BenchmarkConfig.validate_all() for comprehensive validation.
        This method adds additional logging and warning messages.
        """
        # Convert string to enum if needed
        if isinstance(benchmark_mode, str):
            benchmark_mode = TestType(benchmark_mode)

        # Use BenchmarkConfig's comprehensive validation
        try:
            config.validate_all(benchmark_mode)
        except ValueError as e:
            raise ConfigError(str(e)) from e

        # Additional warnings (not errors)
        load_pattern_type = config.settings.load_pattern.type
        if (
            benchmark_mode == TestType.ONLINE
            and load_pattern_type == LoadPatternType.MAX_THROUGHPUT
        ):
            logger.warning(
                "Online benchmark with 'max_throughput' pattern - consider using 'poisson' for sustained QPS"
            )
        elif load_pattern_type == LoadPatternType.CONCURRENCY:
            logger.info(
                "Concurrency-based pattern selected (will maintain fixed concurrent requests)"
            )
