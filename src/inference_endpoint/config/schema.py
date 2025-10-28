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

"""Configuration schema definitions for YAML-based benchmark configs."""

from enum import Enum

from pydantic import BaseModel, Field


class LoadPatternType(str, Enum):
    """Load pattern types."""

    MAX_THROUGHPUT = "max_throughput"  # Offline: all queries at t=0
    POISSON = "poisson"  # Online: fixed QPS with Poisson distribution
    CONCURRENCY = "concurrency"  # Online: fixed concurrent requests (TODO)
    BURST = "burst"  # Burst pattern (TODO)
    STEP = "step"  # Step pattern (TODO)


class OSLDistributionType(str, Enum):
    """Output Sequence Length distribution types."""

    FIXED = "fixed"
    UNIFORM = "uniform"
    NORMAL = "normal"


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


class OSLDistribution(BaseModel):
    """Output Sequence Length distribution configuration."""

    type: OSLDistributionType = OSLDistributionType.FIXED
    mean: int | None = None
    std: int | None = None
    min: int = 1
    max: int = 2048


class ModelParams(BaseModel):
    """Model generation parameters."""

    temperature: float = 0.7
    top_k: int | None = None
    top_p: float | None = None
    max_new_tokens: int = 1024
    osl_distribution: OSLDistribution | None = None


class Baseline(BaseModel):
    """Locked baseline configuration for official submissions.

    TODO: This overlaps with BenchmarkSuiteRuleset concept.
    Should integrate with actual ruleset classes instead of string references.
    See architecture-refactoring-plan.md for integration plan.
    """

    locked: bool = False
    model: str  # Model identifier (e.g., "llama-2-70b")
    ruleset: str  # Ruleset version (e.g., "mlperf-inference-v6.0")
    # TODO: Change to: ruleset: BenchmarkSuiteRuleset | str | None


class Dataset(BaseModel):
    """Dataset configuration."""

    name: str
    type: DatasetType
    path: str
    format: str = "pkl"
    samples: int | None = None
    eval_method: EvalMethod | None = None


class RuntimeSettings(BaseModel):
    """Runtime configuration settings.

    TODO: This duplicates config/ruleset.py RuntimeSettings.
    See architecture-refactoring-plan.md for unification plan.
    Frontend (YAML) vs Backend (execution) - should be unified.
    """

    min_duration_ms: int = 600000  # 10 minutes
    max_duration_ms: int = 1800000  # 30 minutes
    random_seed: int = 42


class LoadPattern(BaseModel):
    """Load pattern configuration.

    Different patterns use QPS differently:
    - max_throughput: QPS used for calculating total queries (offline)
    - poisson: QPS sets scheduler rate (online, rate-limited)
    - concurrency: QPS not used, concurrency limit dominates (TODO)
    """

    type: LoadPatternType = LoadPatternType.MAX_THROUGHPUT
    qps: float = (
        10.0  # Default QPS - queries per second (usage depends on pattern type)
    )
    target_concurrency: int | None = None  # For concurrency mode (TODO)


class ClientSettings(BaseModel):
    """HTTP client configuration.

    Only workers and max_concurrency are required to configure the client.
    Timeout is handled by the HTTP client internally.
    """

    workers: int = 4
    max_concurrency: int = 32


class Settings(BaseModel):
    """Test settings (can be overridden by CLI)."""

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    load_pattern: LoadPattern = Field(default_factory=LoadPattern)
    client: ClientSettings = Field(default_factory=ClientSettings)


def _default_metrics() -> list[str]:
    """Default metrics to collect."""
    return ["throughput", "latency", "ttft", "tpot"]


class Metrics(BaseModel):
    """Metrics collection configuration.

    TODO: This uses string metrics while ruleset.py uses metrics.Metric types.
    Should unify to use metrics.Metric throughout for type safety.
    """

    collect: list[str] = Field(default_factory=_default_metrics)


class EndpointConfig(BaseModel):
    """Endpoint connection configuration (lowest priority in config merging).

    Contains endpoint URL and authentication settings.
    """

    endpoint: str | None = None
    api_key: str | None = None


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration from YAML."""

    name: str
    version: str = "1.0"
    type: TestType
    baseline: Baseline | None = None
    benchmark_mode: TestType | None = None  # For SUBMISSION: specify offline or online
    model_params: ModelParams = Field(default_factory=ModelParams)
    datasets: list[Dataset]
    settings: Settings = Field(default_factory=Settings)
    metrics: Metrics = Field(default_factory=Metrics)
    endpoint_config: EndpointConfig = Field(default_factory=EndpointConfig)

    def is_locked(self) -> bool:
        """Check if baseline is locked."""
        return self.baseline is not None and self.baseline.locked

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
