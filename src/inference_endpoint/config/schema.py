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

"""
TODO: PoC only, subject to change!

Configuration schema definitions for YAML-based benchmark configs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .. import metrics
from .ruleset_base import BenchmarkSuiteRuleset


class APIType(str, Enum):
    OPENAI = "openai"
    SGLANG = "sglang"

    def default_route(self) -> str:
        match self:
            case APIType.OPENAI:
                return "/v1/chat/completions"
            case APIType.SGLANG:
                return "/generate"
            case _:
                raise ValueError(f"Invalid API type: {self}")


class LoadPatternType(str, Enum):
    """Load pattern types."""

    MAX_THROUGHPUT = "max_throughput"  # Offline: all queries at t=0
    POISSON = "poisson"  # Online: fixed QPS with Poisson distribution
    CONCURRENCY = "concurrency"  # Online: fixed concurrent requests
    BURST = "burst"  # Burst pattern (TODO)
    STEP = "step"  # Step pattern (TODO)


class OSLDistributionType(str, Enum):
    """Output Sequence Length distribution types."""

    ORIGINAL = "original"  # Use original distribution from dataset (default)
    FIXED = "fixed"  # Fixed length for all outputs
    UNIFORM = "uniform"  # Uniform distribution between min and max
    NORMAL = "normal"  # Normal/Gaussian distribution


class DatasetType(str, Enum):
    """Dataset purpose type."""

    PERFORMANCE = "performance"
    ACCURACY = "accuracy"


class EvalMethod(str, Enum):
    """Evaluation methods for accuracy testing."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    JUDGE = "judge"


class TestMode(str, Enum):
    """Test mode determining what to collect.

    - PERF: Performance metrics only (no response storage)
    - ACC: Accuracy metrics (collect and evaluate responses)
    - BOTH: Both performance and accuracy (selective collection by dataset type)
    """

    PERF = "perf"
    ACC = "acc"
    BOTH = "both"


class StreamingMode(str, Enum):
    """Streaming mode for response handling.

    - AUTO: Automatically enable for online mode, disable for offline mode
    - ON: Force streaming enabled (for TTFT metrics)
    - OFF: Force streaming disabled
    """

    AUTO = "auto"
    ON = "on"
    OFF = "off"


class TestType(str, Enum):
    """Test type for both config classification and execution mode.

    - OFFLINE: Max throughput benchmark (all queries at t=0)
    - ONLINE: Sustained QPS benchmark (Poisson or concurrency-based)
    - EVAL: Accuracy evaluation
    - SUBMISSION: Official submission (may include both perf and accuracy)
    """

    OFFLINE = "offline"
    ONLINE = "online"
    EVAL = "eval"
    SUBMISSION = "submission"


# Mapping from template type strings to TestType enums
# Single source of truth for template type conversion
TEMPLATE_TYPE_MAP = {
    "offline": TestType.OFFLINE,
    "online": TestType.ONLINE,
    "eval": TestType.EVAL,
    "submission": TestType.SUBMISSION,
}


class OSLDistribution(BaseModel):
    """Output Sequence Length distribution configuration.

    Distribution types:
    - ORIGINAL: Use the natural distribution from the dataset (default)
    - FIXED: All outputs have the same length (uses mean value)
    - UNIFORM: Uniformly distributed between min and max
    - NORMAL: Normal/Gaussian distribution with mean and std
    """

    type: OSLDistributionType = OSLDistributionType.ORIGINAL
    mean: int | None = None  # Required for FIXED and NORMAL
    std: int | None = None  # Required for NORMAL
    min: int = 1  # Required for UNIFORM, bounds for all types
    max: int = 2048  # Required for UNIFORM, bounds for all types


class ModelParams(BaseModel):
    """Model generation parameters."""

    name: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int = 1024
    osl_distribution: OSLDistribution | None = None
    streaming: StreamingMode = StreamingMode.AUTO


class SubmissionReference(BaseModel):
    """Reference configuration for official benchmark submissions.

    Links a submission to a specific model and ruleset (competition rules).
    The ruleset defines constraints like min duration, sample counts, and
    performance targets that must be met for a valid submission.

    Example:
        submission_ref:
          model: "llama-2-70b"
          ruleset: "mlperf-inference-v5.1"
    """

    model: str  # Model identifier (e.g., "llama-2-70b")
    ruleset: str  # Ruleset name/version (e.g., "mlperf-inference-v5.1")

    def get_ruleset_instance(self) -> BenchmarkSuiteRuleset:
        """Get the actual ruleset instance from registry.

        Returns:
            BenchmarkSuiteRuleset instance

        Raises:
            KeyError: If ruleset not found in registry
        """
        from .ruleset_registry import get_ruleset

        return get_ruleset(self.ruleset)


class Dataset(BaseModel):
    """Dataset configuration."""

    name: str
    type: DatasetType
    path: str
    format: str | None = None
    samples: int | None = None
    eval_method: EvalMethod | None = None
    parser: dict | None = None


class RuntimeConfig(BaseModel):
    """Runtime configuration from YAML (user-facing).

    Note: This is the YAML schema for runtime configuration.
    The actual execution uses config.runtime_settings.RuntimeSettings
    (a frozen dataclass with more fields derived from this + ruleset).

    This class represents user inputs, while RuntimeSettings represents
    the fully-resolved execution configuration.

    Sample count priority (in RuntimeSettings.total_samples_to_issue()):
    1. n_samples_to_issue (if specified) - explicit override
    2. All dataset samples (if min_duration_ms=0) - default CLI behavior
    3. Calculated from QPS * duration (if min_duration_ms>0) - duration-based
    """

    min_duration_ms: int = 600000  # 10 minutes
    max_duration_ms: int = 1800000  # 30 minutes
    n_samples_to_issue: int | None = (
        None  # Explicit sample count override (None = auto-calculate)
    )
    scheduler_random_seed: int = 42  # For Poisson/distribution sampling
    dataloader_random_seed: int = 42  # For dataset shuffling


class LoadPattern(BaseModel):
    """Load pattern configuration.

    Different patterns use target_qps differently:
    - max_throughput: target_qps used for calculating total queries (offline, optional with default)
    - poisson: target_qps sets scheduler rate (online, required - validated)
    - concurrency: issue at fixed target_concurrency (online, required - validated)
    """

    type: LoadPatternType = LoadPatternType.MAX_THROUGHPUT
    target_qps: float | None = (
        None  # Target QPS - required for poisson pattern, optional otherwise
    )
    target_concurrency: int | None = None  # For concurrency mode, ignored otherwise


class ClientSettings(BaseModel):
    """HTTP client configuration.

    Only workers are required to configure the client.
    Timeout is handled by the HTTP client internally.

    """

    workers: int = 4
    record_worker_events: bool = False
    log_level: str = "INFO"


class Settings(BaseModel):
    """Test settings (can be overridden by CLI)."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    load_pattern: LoadPattern = Field(default_factory=LoadPattern)
    client: ClientSettings = Field(default_factory=ClientSettings)


def _default_metrics() -> list[str]:
    """
    TODO: PoC only, subject to change!
    Default metrics to collect."""
    return ["throughput", "latency", "ttft", "tpot"]


class Metrics(BaseModel):
    """Metrics collection configuration.

    Note: Currently uses string-based metric names for YAML simplicity.
    Use get_metric_types() to convert to actual Metric type classes.
    """

    collect: list[str] = Field(default_factory=_default_metrics)

    def get_metric_types(self) -> list[type[metrics.Metric]]:
        """Convert string metric names to Metric type classes.

        Returns:
            List of Metric type classes corresponding to collect list

        Raises:
            ValueError: If metric name is not recognized
        """
        metric_map = {
            "throughput": metrics.Throughput,
            "latency": metrics.QueryLatency,
            "ttft": metrics.TTFT,
            "tpot": metrics.TPOT,
        }

        result = []
        for name in self.collect:
            if name not in metric_map:
                raise ValueError(
                    f"Unknown metric name: {name}. Available: {list(metric_map.keys())}"
                )
            result.append(metric_map[name])

        return result


class EndpointConfig(BaseModel):
    """Endpoint connection configuration.

    Contains endpoint URL and authentication settings.
    API type refers to the API implementation used on the endpoint based on industry standards.
    The Default API type is APIType.OPENAI, which refers to the the /v1/chat/completions route.
    """

    endpoint: str | None = None
    api_key: str | None = None
    api_type: APIType = APIType.OPENAI


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration from YAML.

    This is the root configuration model. It's immutable (frozen) to prevent
    accidental modifications during benchmark execution.
    """

    model_config = {"frozen": True}  # Pydantic v2 frozen config

    name: str
    version: str = "1.0"
    type: TestType
    submission_ref: SubmissionReference | None = None  # For SUBMISSION type configs
    # For SUBMISSION: specify offline or online
    benchmark_mode: TestType | None = None
    model_params: ModelParams = Field(default_factory=ModelParams)
    datasets: list[Dataset]
    settings: Settings = Field(default_factory=Settings)
    metrics: Metrics = Field(default_factory=Metrics)
    endpoint_config: EndpointConfig = Field(default_factory=EndpointConfig)
    report_dir: Path | None = None
    timeout: int | None = None
    verbose: bool = False

    @classmethod
    def from_yaml_file(cls, path: Path) -> BenchmarkConfig:
        """Load BenchmarkConfig from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            BenchmarkConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't match schema
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml_file(self, path: Path, exclude_none: bool = True) -> None:
        """Save BenchmarkConfig to YAML file.

        Args:
            path: Path to save YAML file
            exclude_none: Whether to exclude None values (default: True)
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(exclude_none=exclude_none, mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def get_benchmark_mode(self) -> TestType | None:
        """Get the benchmark execution mode.

        For OFFLINE/ONLINE types, returns the type itself.
        For SUBMISSION, returns the explicitly set benchmark_mode.
        For EVAL, returns None (no benchmark execution).
        """
        if self.type in [TestType.OFFLINE, TestType.ONLINE]:
            return self.type
        elif self.type == TestType.SUBMISSION:
            return self.benchmark_mode  # Must be set for submissions
        else:
            return None

    def get_single_dataset(self) -> Dataset | None:
        """Get single dataset for benchmark execution.

        CURRENT LIMITATION: Only single dataset execution is supported.
        This method selects one dataset from the config:
        - Prefers first performance dataset
        - Falls back to first dataset of any type

        Returns:
            Single dataset to use, or None if no datasets configured

        TODO: Multi-dataset support
        Future enhancement should:
        1. Support parallel dataset loading and indexing
        2. Support dataset mixing strategies (e.g. random, sequential, weighted)
        3. Support dataset-specific metrics (in the post processing eval)
        """
        if not self.datasets:
            return None

        # TODO: When multi-dataset is supported, this logic should move to DatasetSelector
        # For now, just pick the first performance dataset
        perf_datasets = [d for d in self.datasets if d.type == DatasetType.PERFORMANCE]
        if perf_datasets:
            return perf_datasets[0]

        return self.datasets[0]

    def validate_required_fields(self) -> None:
        """Validate that required fields are populated.

        Raises:
            ValueError: If required fields are missing
        """
        if not self.endpoint_config.endpoint:
            raise ValueError(
                "Endpoint required: specify --endpoint URL or set in YAML config"
            )

        # Model is required for production benchmarks
        # For submissions, baseline.model is checked
        # For others, it could be in various places (TODO: unify model handling)
        # For now, we'll warn but not enforce (gradual migration)

    def validate_load_pattern(self, benchmark_mode: TestType) -> None:
        """Validate load pattern is appropriate for benchmark mode.

        Args:
            benchmark_mode: The benchmark execution mode

        Raises:
            ValueError: If load pattern doesn't match benchmark mode or required parameters are missing
        """
        load_pattern_type = self.settings.load_pattern.type
        target_qps = self.settings.load_pattern.target_qps
        target_concurrency = self.settings.load_pattern.target_concurrency

        if benchmark_mode == TestType.OFFLINE:
            if load_pattern_type != LoadPatternType.MAX_THROUGHPUT:
                raise ValueError(
                    f"Offline benchmarks must use 'max_throughput' load pattern, got '{load_pattern_type}'"
                )

        elif benchmark_mode == TestType.ONLINE:
            # Online mode validation
            if load_pattern_type == LoadPatternType.POISSON:
                # Poisson pattern requires target_qps to be specified
                if target_qps is None or target_qps <= 0:
                    raise ValueError(
                        "Online mode with poisson pattern requires positive target_qps. "
                        "Specify target queries per second (e.g., target_qps: 100 in YAML or --target-qps 100 in CLI)"
                    )
            elif load_pattern_type == LoadPatternType.CONCURRENCY:
                # Concurrency pattern requires target_concurrency > 0
                if not target_concurrency or target_concurrency <= 0:
                    raise ValueError(
                        "Concurrency load pattern requires target_concurrency > 0. "
                        "Specify number of concurrent requests (e.g., target_concurrency: 10 under load_pattern in YAML or --concurrency 10 in CLI)"
                    )

    def validate_client_settings(self) -> None:
        """Validate client settings are reasonable.

        Raises:
            ValueError: If settings are invalid
        """
        if self.settings.client.workers < 1:
            raise ValueError(
                f"workers must be >= 1, got {self.settings.client.workers}"
            )

    def validate_runtime_settings(self) -> None:
        """Validate runtime settings are reasonable.

        Raises:
            ValueError: If settings are invalid
        """
        if (
            self.settings.runtime.max_duration_ms
            < self.settings.runtime.min_duration_ms
        ):
            raise ValueError(
                f"max_duration_ms ({self.settings.runtime.max_duration_ms}) must be >= "
                f"min_duration_ms ({self.settings.runtime.min_duration_ms})"
            )

        if self.settings.runtime.min_duration_ms < 0:
            raise ValueError(
                f"min_duration_ms must be >= 0, got {self.settings.runtime.min_duration_ms}"
            )

    def validate_datasets(self) -> None:
        """Validate dataset configuration.

        Raises:
            ValueError: If dataset configuration is invalid
        """
        if not self.datasets:
            # Empty datasets is OK for CLI-based benchmarks
            return

        # Check for duplicate dataset names
        names = [d.name for d in self.datasets]
        duplicates = [name for name in set(names) if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate dataset names: {duplicates}")

    def validate_all(self, benchmark_mode: TestType | None = None) -> None:
        """Run all validation checks.

        Args:
            benchmark_mode: Optional benchmark mode for load pattern validation

        Raises:
            ValueError: If any validation fails
        """
        self.validate_required_fields()
        self.validate_client_settings()
        self.validate_runtime_settings()
        self.validate_datasets()

        if benchmark_mode:
            self.validate_load_pattern(benchmark_mode)

    @classmethod
    def create_default_config(cls, test_type: TestType) -> BenchmarkConfig:
        """Create default BenchmarkConfig for a given test type.

        This is the source of truth for default configurations. Used by:
        - Template generation (init command)
        - Testing and examples
        - CLI fallbacks

        Args:
            test_type: TestType enum (OFFLINE, ONLINE, EVAL, or SUBMISSION)

        Returns:
            Default BenchmarkConfig instance

        Raises:
            NotImplementedError: If test_type is EVAL or SUBMISSION (not yet implemented)
            ValueError: If test_type is invalid
        """
        if test_type == TestType.OFFLINE:
            return cls(
                name="default_offline",
                version="1.0",
                type=TestType.OFFLINE,
                datasets=[],
                settings=Settings(
                    load_pattern=LoadPattern(
                        type=LoadPatternType.MAX_THROUGHPUT, target_qps=None
                    ),
                    runtime=RuntimeConfig(
                        min_duration_ms=600000,
                        max_duration_ms=1800000,
                        scheduler_random_seed=42,
                        dataloader_random_seed=42,
                    ),
                    client=ClientSettings(workers=4),
                ),
                model_params=ModelParams(temperature=0.7, max_new_tokens=1024),
                metrics=Metrics(),
                endpoint_config=EndpointConfig(),
            )
        elif test_type == TestType.ONLINE:
            return cls(
                name="default_online",
                version="1.0",
                type=TestType.ONLINE,
                datasets=[],
                settings=Settings(
                    load_pattern=LoadPattern(
                        type=LoadPatternType.POISSON, target_qps=10.0
                    ),
                    runtime=RuntimeConfig(
                        min_duration_ms=600000,
                        max_duration_ms=1800000,
                        scheduler_random_seed=42,
                        dataloader_random_seed=42,
                    ),
                    client=ClientSettings(workers=4),
                ),
                model_params=ModelParams(temperature=0.7, max_new_tokens=1024),
                metrics=Metrics(),
                endpoint_config=EndpointConfig(),
            )
        elif test_type == TestType.EVAL:
            raise NotImplementedError(
                "Default EVAL config not yet implemented. "
                "EVAL templates will be added in future release."
            )
        elif test_type == TestType.SUBMISSION:
            raise NotImplementedError(
                "Default SUBMISSION config not yet implemented. "
                "SUBMISSION templates will be added in future release."
            )
        else:
            raise ValueError(f"Unknown test type: {test_type}")
