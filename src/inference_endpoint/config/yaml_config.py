# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""YAML configuration loading and merging with CLI arguments."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import BenchmarkConfig, TestType

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration error."""

    pass


class ConfigLoader:
    """Load and merge YAML configs with CLI arguments."""

    @staticmethod
    def load_yaml(path: Path) -> BenchmarkConfig:
        """Load and validate YAML config file.

        Args:
            path: Path to YAML config file

        Returns:
            Validated BenchmarkConfig

        Raises:
            ConfigError: If file not found or validation fails
        """
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            config = BenchmarkConfig(**data)
            logger.info(f"Loaded config: {config.name} (type: {config.type})")
            return config

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}") from e
        except ValidationError as e:
            raise ConfigError(f"Config validation failed: {e}") from e

    @staticmethod
    def merge_with_cli_args(
        config: BenchmarkConfig, cli_args: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge YAML config with CLI arguments.

        Priority: baseline (locked) < YAML settings < CLI args

        Args:
            config: Loaded YAML config
            cli_args: CLI arguments to merge

        Returns:
            Merged configuration dict

            TODO: Should return frozen dataclass instead of mutable dict.
            See architecture-refactoring-plan.md #8 for type-safe approach.

        Raises:
            ConfigError: If CLI tries to override locked baseline
        """
        # Start with baseline settings
        merged = {}

        # Check for locked baseline violations
        if config.is_locked():
            violations = []
            # Locked fields: model, ruleset, datasets, load_pattern
            if cli_args.get("model") is not None:
                violations.append("model")
            if cli_args.get("ruleset") is not None:
                violations.append("ruleset")
            if cli_args.get("datasets") is not None:
                violations.append("datasets")
            # Note: qps is allowed (runtime optimization)
            # load_pattern type is locked, but qps can be overridden

            if violations:
                raise ConfigError(
                    f"Cannot override locked fields in submission config: {', '.join(violations)}. "
                    f"Set baseline.locked=false in YAML to allow overrides."
                )

        # Merge in order: baseline -> YAML settings -> CLI args
        if config.baseline:
            merged["model"] = config.baseline.model
            merged["ruleset"] = config.baseline.ruleset

        # Model params from YAML
        merged["model_params"] = config.model_params.model_dump()

        # Settings from YAML
        merged["runtime"] = config.settings.runtime.model_dump()
        merged["load_pattern"] = config.settings.load_pattern.model_dump()
        merged["client"] = config.settings.client.model_dump()

        # Datasets from YAML
        merged["datasets"] = [d.model_dump() for d in config.datasets]

        # Endpoint config from YAML
        merged["endpoint"] = config.endpoint_config.endpoint
        merged["api_key"] = config.endpoint_config.api_key

        # Override with CLI args (only non-None values)
        for key, value in cli_args.items():
            if value is not None:
                if key == "endpoint":
                    merged["endpoint"] = value
                elif key == "api_key":
                    merged["api_key"] = value
                elif key == "model":
                    if "baseline" in merged:
                        merged["baseline"]["model"] = value
                    # Also store separately for non-baseline configs
                    merged["model"] = value
                elif key == "qps":
                    merged["load_pattern"]["qps"] = value
                elif key == "workers":
                    merged["client"]["workers"] = value
                elif key == "concurrency":
                    merged["client"]["max_concurrency"] = value
                elif key == "duration":
                    merged["runtime"]["min_duration_ms"] = value * 1000
                elif key == "min_tokens":
                    if "osl_distribution" not in merged["model_params"]:
                        merged["model_params"]["osl_distribution"] = {}
                    merged["model_params"]["osl_distribution"]["min"] = value
                elif key == "max_tokens":
                    merged["model_params"]["max_new_tokens"] = value
                    if "osl_distribution" not in merged["model_params"]:
                        merged["model_params"]["osl_distribution"] = {}
                    merged["model_params"]["osl_distribution"]["max"] = value

        return merged

    @staticmethod
    def validate_config(config: dict, benchmark_mode=None) -> None:
        """Validate configuration consistency.

        Args:
            config: Merged configuration dict
            benchmark_mode: BenchmarkMode enum (OFFLINE or ONLINE), or string, or None

        Raises:
            ConfigError: If configuration is invalid

        QPS Usage Notes:
        - Offline: QPS used to calculate total queries (not rate-limiting)
        - Online (Poisson): QPS sets scheduler rate (rate-limited)
        - Online (Concurrency): QPS not used, concurrency dominates (TODO)
        """
        # Convert string to enum if needed
        if isinstance(benchmark_mode, str):
            benchmark_mode = TestType(benchmark_mode)

        load_pattern_type = config.get("load_pattern", {}).get("type")

        # Rule: Offline benchmarks must use max_throughput load pattern
        if benchmark_mode == TestType.OFFLINE:
            if load_pattern_type and load_pattern_type != "max_throughput":
                raise ConfigError(
                    f"Offline benchmarks must use 'max_throughput' load pattern, got '{load_pattern_type}'. "
                    f"Poisson and concurrency patterns only apply to online scenarios."
                )
            # Force max_throughput for offline
            config["load_pattern"]["type"] = "max_throughput"

        # Rule: Online benchmarks should use poisson or concurrency (when implemented)
        elif benchmark_mode == TestType.ONLINE:
            if load_pattern_type == "max_throughput":
                logger.warning(
                    "Online benchmark with 'max_throughput' pattern - consider using 'poisson' for sustained QPS"
                )
            elif load_pattern_type == "concurrency":
                # Future: concurrency-based mode
                logger.info(
                    "Concurrency-based pattern selected (will maintain fixed concurrent requests)"
                )

        # Rule: Ensure max_concurrency can handle target_concurrency
        target_concurrency = config.get("load_pattern", {}).get("target_concurrency")
        max_concurrency = config.get("client", {}).get("max_concurrency")

        if target_concurrency is not None and max_concurrency is not None:
            if max_concurrency < target_concurrency:
                # Auto-adjust: max_concurrency should be at least target_concurrency
                logger.warning(
                    f"max_concurrency ({max_concurrency}) < target_concurrency ({target_concurrency})"
                )
                logger.info(f"Auto-adjusting max_concurrency to {target_concurrency}")
                config["client"]["max_concurrency"] = target_concurrency
            elif max_concurrency > target_concurrency:
                logger.info(
                    f"max_concurrency ({max_concurrency}) > target_concurrency ({target_concurrency}) - OK"
                )

    @staticmethod
    def apply_cli_overrides_to_dict(
        config_dict: dict[str, Any], cli_args: dict[str, Any]
    ) -> None:
        """Apply CLI overrides directly to a config dict (for quick benchmarks).

        This is a simpler version of merge_with_cli_args for cases where
        there's no YAML config and no locking concerns.

        Args:
            config_dict: Configuration dict to update
            cli_args: CLI arguments to apply
        """
        for key, value in cli_args.items():
            if value is None:
                continue

            if key == "endpoint":
                config_dict["endpoint"] = value
            elif key == "api_key":
                config_dict["api_key"] = value
            elif key == "model":
                config_dict["model"] = value
            elif key == "qps":
                config_dict["load_pattern"]["qps"] = value
            elif key == "workers":
                config_dict["client"]["workers"] = value
            elif key == "concurrency":
                config_dict["client"]["max_concurrency"] = value
            elif key == "duration":
                config_dict["runtime"]["min_duration_ms"] = value * 1000
            elif key == "min_tokens":
                if "osl_distribution" not in config_dict["model_params"]:
                    config_dict["model_params"]["osl_distribution"] = {}
                config_dict["model_params"]["osl_distribution"]["min"] = value
            elif key == "max_tokens":
                config_dict["model_params"]["max_new_tokens"] = value
                if "osl_distribution" not in config_dict["model_params"]:
                    config_dict["model_params"]["osl_distribution"] = {}
                config_dict["model_params"]["osl_distribution"]["max"] = value

    @staticmethod
    def create_default_config(test_type: str) -> dict[str, Any]:
        """Create default config for quick benchmark modes.

        Args:
            test_type: "offline" or "online"

        Returns:
            Default configuration dict
        """
        if test_type == "offline":
            return {
                "load_pattern": {"type": "max_throughput", "qps": 10.0},
                "runtime": {
                    "min_duration_ms": 600000,  # 10 minutes
                    "max_duration_ms": 1800000,  # 30 minutes
                    "random_seed": 42,
                },
                "client": {
                    "workers": 4,
                    "max_concurrency": 32,
                },
                "model_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 1024,
                },
                "mode": "perf",
            }
        elif test_type == "online":
            return {
                "load_pattern": {"type": "poisson", "qps": 10},
                "runtime": {
                    "min_duration_ms": 600000,  # 10 minutes
                    "max_duration_ms": 1800000,  # 30 minutes
                    "random_seed": 42,
                },
                "client": {
                    "workers": 4,
                    "max_concurrency": 32,
                },
                "model_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 1024,
                },
                "mode": "perf",
            }
        else:
            raise ConfigError(f"Unknown test type: {test_type}")
