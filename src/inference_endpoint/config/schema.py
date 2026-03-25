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

"""Configuration schema — single source of truth for YAML and CLI.

All Pydantic models here define both the YAML config structure and the CLI interface.
cyclopts auto-generates CLI flags from fields. Use cyclopts.Parameter(alias=...)
on Annotated fields to declare shorthand aliases alongside dotted paths.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, Union

import cyclopts
import yaml
from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .. import metrics
from ..core.types import APIType
from ..endpoint_client.config import HTTPClientConfig
from ..utils import WithUpdatesMixin
from .ruleset_base import BenchmarkSuiteRuleset
from .utils import parse_dataset_string, resolve_env_vars


class SystemDefaults(BaseModel):
    DEFAULT_TIMEOUT: ClassVar[float] = 300.0
    DEFAULT_METRIC: ClassVar[metrics.Metric] = metrics.Throughput(0.0)


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

    model_config = {"extra": "forbid"}

    type: OSLDistributionType = Field(
        OSLDistributionType.ORIGINAL, description="Distribution type"
    )
    mean: int | None = Field(None, description="Mean length (FIXED/NORMAL)")
    std: int | None = Field(None, description="Std deviation (NORMAL)")
    min: Annotated[
        int,
        cyclopts.Parameter(alias="--min-output-tokens", help="Minimum output length"),
    ] = 1
    max: int = Field(2048, description="Maximum output length")


class ModelParams(BaseModel):
    """Model generation parameters."""

    model_config = {"extra": "forbid"}

    name: Annotated[
        str,
        cyclopts.Parameter(alias="--model", help="Model name", required=True),
    ] = ""
    temperature: float | None = Field(None, description="Sampling temperature")
    top_k: int | None = Field(None, description="Top-K sampling")
    top_p: float | None = Field(None, description="Top-P (nucleus) sampling")
    repetition_penalty: float | None = Field(None, description="Repetition penalty")
    max_new_tokens: Annotated[
        int, cyclopts.Parameter(alias="--max-output-tokens", help="Max output tokens")
    ] = 1024
    osl_distribution: OSLDistribution | None = Field(
        None, description="Output sequence length distribution"
    )
    streaming: Annotated[
        StreamingMode,
        cyclopts.Parameter(alias="--streaming", help="Streaming mode: auto/on/off"),
    ] = StreamingMode.AUTO


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

    model_config = {"extra": "forbid"}

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
    """Dataset configuration.

    Name and type have smart defaults: name is auto-derived from path,
    type defaults to PERFORMANCE.

    Accepts CLI strings via BeforeValidator on BenchmarkConfig.datasets:
    ``[perf|acc:]<path>[,key=value...]``
    """

    model_config = {"extra": "forbid"}

    name: str = Field("", description="Dataset name (auto-derived from path if empty)")
    type: DatasetType = Field(
        DatasetType.PERFORMANCE, description="Dataset purpose: performance or accuracy"
    )
    path: Annotated[
        str | None, cyclopts.Parameter(alias="--dataset", help="Dataset file path")
    ] = None
    format: str | None = Field(None, description="Dataset format (auto-detected)")
    samples: int | None = Field(None, description="Number of samples to use")
    eval_method: EvalMethod | None = Field(
        None, description="Accuracy evaluation method"
    )
    parser: dict[str, str] | None = Field(None, description="Column remapping")
    accuracy_config: AccuracyConfig | None = Field(
        None, description="Accuracy evaluation settings"
    )

    @model_validator(mode="after")
    def _auto_derive_name(self) -> Self:
        """Derive name from path stem if not explicitly provided."""
        if not self.name and self.path:
            object.__setattr__(self, "name", Path(self.path).stem)
        return self


class AccuracyConfig(BaseModel):
    """Accuracy configuration.

    The eval_method is the method to use to evaluate the accuracy of the model.
    Currently only "pass_at_1" is supported.
    The ground_truth is the column in the dataset that contains the ground truth.
    Defaults to "ground_truth" if not specified.
    The extractor is the extractor to use to extract the ground truth from the output.
    Currently "boxed_math_extractor" and "abcd_extractor" are supported.
    The num_repeats is the number of times to repeat the dataset for evaluation.
    Defaults to 1 if not specified.

    Example:
        accuracy_config:
          eval_method: "pass_at_1"
          ground_truth: "answer"
          extractor: "boxed_math_extractor"
          num_repeats: 5
    """

    model_config = {"extra": "forbid"}

    eval_method: str | None = None
    ground_truth: str | None = None
    extractor: str | None = None
    num_repeats: int = 1


class RuntimeConfig(BaseModel):
    """Runtime configuration.

    Sample count priority (in RuntimeSettings.total_samples_to_issue()):
    1. n_samples_to_issue (if specified) — explicit override
    2. Calculated from QPS * duration — duration-based (default: 600000ms)
    3. All dataset samples — fallback when duration is 0
    """

    model_config = {"extra": "forbid"}

    min_duration_ms: Annotated[
        int,
        cyclopts.Parameter(
            alias="--duration", help="Min duration (ms, or with suffix: 600s, 10m)"
        ),
    ] = Field(600000, ge=0)
    max_duration_ms: int = Field(
        1800000, ge=0, description="Maximum test duration in ms"
    )

    @field_validator("min_duration_ms", "max_duration_ms", mode="before")
    @classmethod
    def _parse_duration_suffix(cls, v: object) -> object:
        """Accept duration with unit suffix: 600s, 10m, 600000ms, or plain int (ms)."""
        if isinstance(v, str):
            v = v.strip()
            if v.endswith("ms"):
                return int(v[:-2])
            if v.endswith("m"):
                return int(float(v[:-1]) * 60_000)
            if v.endswith("s"):
                return int(float(v[:-1]) * 1000)
        return v

    n_samples_to_issue: Annotated[
        int | None,
        cyclopts.Parameter(alias="--num-samples", help="Sample count override"),
    ] = None
    scheduler_random_seed: int = Field(42, description="Scheduler RNG seed")
    dataloader_random_seed: int = Field(42, description="Dataloader RNG seed")

    @model_validator(mode="after")
    def _validate_durations(self) -> Self:
        if self.max_duration_ms < self.min_duration_ms:
            raise ValueError(
                f"max_duration_ms ({self.max_duration_ms}) must be >= "
                f"min_duration_ms ({self.min_duration_ms})"
            )
        return self


class LoadPattern(BaseModel):
    """Load pattern configuration.

    Different patterns use target_qps differently:
    - max_throughput: target_qps used for calculating total queries (offline, optional with default)
    - poisson: target_qps sets scheduler rate (online, required - validated)
    - concurrency: issue at fixed target_concurrency (online, required - validated)
    """

    model_config = {"extra": "forbid"}

    type: Annotated[
        LoadPatternType,
        cyclopts.Parameter(alias="--load-pattern", help="Load pattern type"),
    ] = LoadPatternType.MAX_THROUGHPUT
    target_qps: Annotated[
        float | None, cyclopts.Parameter(alias="--target-qps", help="Target QPS")
    ] = None
    target_concurrency: Annotated[
        int | None,
        cyclopts.Parameter(alias="--concurrency", help="Concurrent requests"),
    ] = None

    @model_validator(mode="after")
    def _validate_completeness(self) -> Self:
        if self.type == LoadPatternType.POISSON and (
            self.target_qps is None or self.target_qps <= 0
        ):
            raise ValueError("Poisson requires --target-qps (e.g., --target-qps 100)")
        if self.type == LoadPatternType.CONCURRENCY and (
            not self.target_concurrency or self.target_concurrency <= 0
        ):
            raise ValueError(
                "Concurrency requires --concurrency (e.g., --concurrency 10)"
            )
        return self


@cyclopts.Parameter(name="*")
class Settings(BaseModel):
    """Test settings."""

    model_config = {"extra": "forbid"}

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    load_pattern: LoadPattern = Field(default_factory=LoadPattern)
    client: HTTPClientConfig = Field(default_factory=HTTPClientConfig)


class OfflineSettings(Settings):
    """Offline mode default settings."""

    load_pattern: Annotated[LoadPattern, cyclopts.Parameter(show=False)] = Field(
        default_factory=lambda: LoadPattern(type=LoadPatternType.MAX_THROUGHPUT)
    )


class OnlineSettings(Settings):
    """Online mode default settings."""

    pass


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

    model_config = {"extra": "forbid"}

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

    model_config = {"extra": "forbid"}

    endpoints: Annotated[
        list[str],
        cyclopts.Parameter(alias="--endpoints", help="Endpoint URL(s)", negative=""),
    ] = Field(min_length=1)
    api_key: Annotated[
        str | None, cyclopts.Parameter(alias="--api-key", help="API key")
    ] = None
    api_type: Annotated[
        APIType,
        cyclopts.Parameter(alias="--api-type", help="API type: openai or sglang"),
    ] = APIType.OPENAI


class BenchmarkConfig(WithUpdatesMixin, BaseModel):
    """Benchmark configuration — single source of truth for YAML and CLI.

    Immutable (frozen) to prevent accidental modifications during execution.
    cyclopts auto-generates CLI flags from fields. Use cyclopts.Parameter(name=...)
    on Annotated fields to declare flat shorthand aliases.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: Annotated[str, cyclopts.Parameter(show=False)] = Field(
        "", description="Benchmark name (auto-derived from type if empty)"
    )
    version: Annotated[str, cyclopts.Parameter(show=False)] = Field(
        "1.0", description="Config version"
    )
    type: Annotated[TestType, cyclopts.Parameter(show=False)] = Field(
        description="Test type: offline, online, eval, submission"
    )
    submission_ref: Annotated[
        SubmissionReference | None, cyclopts.Parameter(show=False)
    ] = None
    benchmark_mode: Annotated[
        Literal[TestType.OFFLINE, TestType.ONLINE] | None,
        cyclopts.Parameter(show=False),
    ] = None
    model_params: ModelParams = Field(default_factory=ModelParams)
    datasets: Annotated[list[Dataset], cyclopts.Parameter(show=False)] = Field(
        default_factory=list, description="Dataset configs"
    )
    settings: Settings = Field(default_factory=Settings)
    metrics: Annotated[Metrics, cyclopts.Parameter(show=False)] = Field(
        default_factory=Metrics
    )
    endpoint_config: EndpointConfig
    report_dir: Annotated[
        Path | None,
        cyclopts.Parameter(alias="--report-dir", help="Report output directory"),
    ] = None
    timeout: Annotated[
        float | None,
        cyclopts.Parameter(alias="--timeout", help="Global timeout in seconds"),
    ] = None
    # verbose is handled by cyclopts meta app (-v flag), not here
    verbose: Annotated[bool, cyclopts.Parameter(show=False)] = False
    enable_cpu_affinity: Annotated[
        bool,
        cyclopts.Parameter(
            negative="--no-cpu-affinity",
            help="NUMA-aware CPU pinning",
        ),
    ] = True

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_dataset_strings(cls, v: object) -> object:
        """Accept CLI dataset strings alongside Dataset dicts/objects.

        Grammar: ``[perf|acc:]<path>[,key=value...]``
        """
        if isinstance(v, list):
            return [parse_dataset_string(x) if isinstance(x, str) else x for x in v]
        return v

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> Self:
        """Resolve defaults and validate on frozen model after construction.

        Defaults:
        - Derive name from type if empty
        - Resolve AUTO streaming (offline=OFF, online=ON)
        - Resolve model name from submission_ref

        Validation:
        - Workers must be -1 (auto) or >= 1
        - max_duration_ms >= min_duration_ms >= 0
        - No duplicate dataset (name, type) pairs
        - Load pattern must match test type
        """
        # --- Resolve defaults ---
        mp_updates: dict[str, object] = {}

        if not self.name:
            object.__setattr__(self, "name", f"{self.type.value}_benchmark")

        effective_mode = (
            self.benchmark_mode if self.type == TestType.SUBMISSION else self.type
        )

        if self.model_params.streaming == StreamingMode.AUTO:
            mp_updates["streaming"] = (
                StreamingMode.OFF
                if effective_mode in (TestType.OFFLINE,)
                else StreamingMode.ON
            )

        if not self.model_params.name and self.submission_ref:
            mp_updates["name"] = self.submission_ref.model

        if mp_updates:
            object.__setattr__(
                self,
                "model_params",
                self.model_params.model_copy(update=mp_updates),
            )

        if not self.model_params.name:
            raise ValueError("Required: --model-params.name [--model]")

        # --- Validate (cross-model checks only; sub-models self-validate) ---
        if self.type == TestType.SUBMISSION and not self.benchmark_mode:
            raise ValueError(
                "SUBMISSION configs must specify benchmark_mode (offline or online)"
            )

        # Duplicate datasets — same (name, type) would collide in results.json
        if self.datasets:
            pairs = [(d.name, d.type) for d in self.datasets]
            dupes = [
                f"{n} ({t.value})" for (n, t), cnt in Counter(pairs).items() if cnt > 1
            ]
            if dupes:
                raise ValueError(f"Duplicate dataset names: {dupes}")

        # Load pattern type vs test type (sub-model validates completeness)
        lp = self.settings.load_pattern
        if effective_mode == TestType.OFFLINE:
            if lp.type != LoadPatternType.MAX_THROUGHPUT:
                raise ValueError(
                    f"Offline benchmarks must use 'max_throughput', got '{lp.type}'"
                )
        elif effective_mode == TestType.ONLINE:
            if lp.type not in (LoadPatternType.POISSON, LoadPatternType.CONCURRENCY):
                raise ValueError(
                    "Online mode requires --load-pattern (poisson or concurrency)"
                )

        return self

    @classmethod
    def from_yaml_file(cls, path: Path) -> BenchmarkConfig:
        """Load BenchmarkConfig from YAML file.

        Auto-selects OfflineBenchmarkConfig/OnlineBenchmarkConfig based on
        the ``type`` field so YAML gets the same defaults as CLI.

        Args:
            path: Path to YAML file

        Returns:
            BenchmarkConfig (or subclass) instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or doesn't match schema
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = path.read_text()
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping, got {type(data).__name__}")
        resolve_env_vars(data)

        return _config_adapter.validate_python(data)

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

    @staticmethod
    def create_default_config(test_type: TestType) -> BenchmarkConfig:
        """Create default BenchmarkConfig for a given test type.

        Delegates to the appropriate subclass so field defaults are the
        single source of truth.  Only placeholder values (endpoints, model
        name, dataset path) are set explicitly.

        Args:
            test_type: TestType enum (OFFLINE, ONLINE, EVAL, or SUBMISSION)

        Returns:
            BenchmarkConfig (or subclass) instance

        Raises:
            NotImplementedError: If test_type is EVAL or SUBMISSION
            ValueError: If test_type is invalid
        """
        _common = {
            "model_params": ModelParams(name="<MODEL_NAME>"),
            "datasets": [Dataset(path="<DATASET_PATH>")],
            "endpoint_config": EndpointConfig(endpoints=["<ENDPOINT_URL>"]),
        }
        if test_type == TestType.OFFLINE:
            return OfflineBenchmarkConfig(**_common)
        if test_type == TestType.ONLINE:
            return OnlineBenchmarkConfig(
                **_common,
                settings=OnlineSettings(
                    load_pattern=LoadPattern(
                        type=LoadPatternType.POISSON, target_qps=10.0
                    ),
                ),
            )
        if test_type == TestType.EVAL:
            raise NotImplementedError(
                "Default EVAL config not yet implemented. "
                "EVAL templates will be added in future release."
            )
        if test_type == TestType.SUBMISSION:
            raise NotImplementedError(
                "Default SUBMISSION config not yet implemented. "
                "SUBMISSION templates will be added in future release."
            )
        raise ValueError(f"Unknown test type: {test_type}")


@cyclopts.Parameter(name="*")
class OfflineBenchmarkConfig(BenchmarkConfig):
    """Offline benchmark config — type locked, load pattern hidden."""

    type: Annotated[Literal[TestType.OFFLINE], cyclopts.Parameter(show=False)] = (
        TestType.OFFLINE
    )  # type: ignore[assignment]
    settings: OfflineSettings = Field(default_factory=OfflineSettings)


@cyclopts.Parameter(name="*")
class OnlineBenchmarkConfig(BenchmarkConfig):
    """Online benchmark config — type locked."""

    type: Annotated[Literal[TestType.ONLINE], cyclopts.Parameter(show=False)] = (
        TestType.ONLINE
    )  # type: ignore[assignment]
    settings: OnlineSettings = Field(default_factory=OnlineSettings)


def _config_discriminator(v: Any) -> str:
    t = v.get("type", "") if isinstance(v, dict) else str(getattr(v, "type", ""))
    return str(t) if str(t) in ("offline", "online") else "base"


_ConfigUnion = Union[  # noqa: UP007 — runtime Union needed for TypeAdapter + __future__.annotations
    Annotated[OfflineBenchmarkConfig, Tag("offline")],
    Annotated[OnlineBenchmarkConfig, Tag("online")],
    Annotated[BenchmarkConfig, Tag("base")],
]
_config_adapter: TypeAdapter[BenchmarkConfig] = TypeAdapter(
    Annotated[_ConfigUnion, Discriminator(_config_discriminator)]
)
