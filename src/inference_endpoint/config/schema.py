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

import logging
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, Union

import cyclopts
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SerializerFunctionWrapHandler,
    Tag,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from .. import metrics
from ..core.types import APIType
from ..endpoint_client.config import HTTPClientConfig
from ..exceptions import CLIError
from ..utils import WithUpdatesMixin
from .ruleset_base import BenchmarkSuiteRuleset
from .utils import parse_dataset_string, resolve_env_vars

logger = logging.getLogger(__name__)


class SystemDefaults(BaseModel):
    DEFAULT_TIMEOUT: ClassVar[float] = 300.0
    DEFAULT_METRIC: ClassVar[metrics.Metric] = metrics.Throughput(0.0)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return the result.

    For overlapping keys whose values are both dicts, recurse; otherwise the
    override value wins. Mutates a *copy* — callers can safely pass model_dump()
    output. Used by ``Dataset.effective_generation_config`` so a sparse nested
    override (e.g. ``{osl_distribution: {max: 512}}``) preserves siblings.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ModelParams fields that drive the single global tokenizer / MetricsAggregator
# (launched once from top-level model_params), so a per-dataset override would
# desync ISL/OSL/TTFT/TPOT accounting without changing what is measured. Rejected
# as generation_config_override keys — they are per-run/identity, not per-dataset.
_METRICS_DECOUPLED_OVERRIDE_KEYS = frozenset({"name", "streaming", "tokenizer_name"})


def _non_default_completion_controls(mp: ModelParams) -> list[str]:
    """Completion-only ModelParams controls set to a non-default value.

    ``min_new_tokens``/``skip_special_tokens`` are only honored by the
    ``openai_completions`` adapter; ``BenchmarkConfig`` rejects them for other
    ``api_type``s. Shared by the top-level and per-dataset-override checks so
    both config surfaces validate identically.
    """
    checks = {
        "min_new_tokens": mp.min_new_tokens != 1,
        "skip_special_tokens": not mp.skip_special_tokens,
    }
    return [name for name, non_default in checks.items() if non_default]


class LoadPatternType(str, Enum):
    """Load pattern types."""

    MAX_THROUGHPUT = "max_throughput"  # Offline: all queries at t=0
    POISSON = "poisson"  # Online: fixed QPS with Poisson distribution
    CONCURRENCY = "concurrency"  # Online: fixed concurrent requests
    AGENTIC_INFERENCE = (
        "agentic_inference"  # Agentic inference conversations with turn sequencing
    )
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


class ScorerMethod(str, Enum):
    """Registered scorer methods for accuracy evaluation."""

    PASS_AT_1 = "pass_at_1"
    STRING_MATCH = "string_match"
    ROUGE = "rouge"
    CODE_BENCH = "code_bench_scorer"
    SHOPIFY_CATEGORY_F1 = "shopify_category_f1"
    AGENTIC_INFERENCE_INLINE = "agentic_inference_inline"
    VBENCH = "vbench"
    BFCL_V4 = "bfcl_v4"
    LEGACY_MLPERF_DEEPSEEK_R1 = "legacy_mlperf_deepseek_r1"


class AuditTestId(str, Enum):
    """Registered compliance audit test identifiers."""

    # Output-caching audit — MLPerf TEST04 (duplicate-query caching detection).
    OUTPUT_CACHING_TEST = "output_caching_test"


class OutputCachingTestConfig(BaseModel):
    """Configuration for the output-caching audit (MLPerf TEST04).

    The output-caching test runs two back-to-back phases — a reference run of
    distinct samples and an audit run that repeats one fixed sample — then
    checks that the audit QPS does not exceed the reference QPS by more than
    ``threshold``. A large speedup indicates the SUT is caching responses.

    samples: reference-phase query count (required — an explicit count keeps
        the per-phase completion check meaningful; a duration-driven phase has
        no independent target to validate completion against)
    audit_samples: audit-phase query count (None → equals samples)
    sample_index: which dataset row is repeated (MLCommons performance_issue_same_index)
    threshold: tolerance shared by both pass checks — each phase must complete
        ≥ requested * (1 - threshold), and audit_qps must stay < ref_qps * (1 + threshold)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    test: Literal[AuditTestId.OUTPUT_CACHING_TEST]
    only: bool = Field(
        False,
        description="Run only the audit — skip the main benchmark (upstream-style standalone TEST04)",
    )
    samples: int = Field(..., ge=1, description="Reference phase query count")
    audit_samples: int | None = Field(
        None, ge=1, description="Audit phase query count (default: equals samples)"
    )
    sample_index: int = Field(
        0, ge=0, description="Dataset row index repeated in the audit phase"
    )
    threshold: float = Field(
        0.10,
        gt=0,
        lt=1,
        description=(
            "Tolerance for both checks: each phase must complete "
            "≥ requested * (1 - threshold), and audit_qps must stay "
            "< ref_qps * (1 + threshold)"
        ),
    )


# Single member today; becomes
# Annotated[OutputCachingTestConfig | ..., Field(discriminator="test")]
# when additional audit tests are added.
AuditConfig = OutputCachingTestConfig


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

    model_config = ConfigDict(extra="forbid", frozen=True)

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

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: Annotated[
        str,
        cyclopts.Parameter(alias="--model", help="Model name", required=True),
    ] = ""
    temperature: float | None = Field(None, description="Sampling temperature")
    seed: Annotated[
        int | None,
        cyclopts.Parameter(
            alias="--seed", help="Random seed for reproducible sampling"
        ),
    ] = Field(None, description="Random seed for reproducible sampling")
    top_k: int | None = Field(None, description="Top-K sampling")
    top_p: float | None = Field(None, description="Top-P (nucleus) sampling")
    repetition_penalty: float | None = Field(None, description="Repetition penalty")
    presence_penalty: float | None = Field(None, description="Presence penalty")
    frequency_penalty: float | None = Field(None, description="Frequency penalty")
    chat_template_kwargs: dict[str, Any] | None = Field(
        None,
        description="Per-request chat-template kwargs forwarded to compatible servers.",
    )
    max_new_tokens: Annotated[
        int, cyclopts.Parameter(alias="--max-output-tokens", help="Max output tokens")
    ] = 1024
    min_new_tokens: int = Field(
        1,
        ge=0,
        description="Minimum output tokens for OpenAI text-completions servers",
    )
    skip_special_tokens: bool = Field(
        True,
        description=(
            "Whether OpenAI text-completions servers omit special tokens from decoded output"
        ),
    )
    osl_distribution: OSLDistribution | None = Field(
        None, description="Output sequence length distribution"
    )
    streaming: Annotated[
        StreamingMode,
        cyclopts.Parameter(alias="--streaming", help="Streaming mode: auto/on/off"),
    ] = StreamingMode.AUTO
    tokenizer_name: Annotated[
        str | None,
        cyclopts.Parameter(
            alias="--tokenizer",
            help="HF repo ID or local path for the tokenizer. Overrides model name for client-side token metrics (ISL/OSL/TPOT).",
        ),
    ] = None

    @model_validator(mode="after")
    def _validate_generation_lengths(self) -> Self:
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                "min_new_tokens must be less than or equal to max_new_tokens"
            )
        return self


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

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

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


class AgenticInferenceConfig(BaseModel):
    """Agentic inference conversation configuration.

    Configuration for benchmarking conversational AI workloads with turn sequencing.
    Enables testing agentic inference conversations where each turn depends on previous responses.
    Presence of this block in the dataset config enables agentic inference mode.

    Attributes:
        turn_timeout_s: Deadline between issuing a turn and receiving its
            response. A timeout aborts that turn and all remaining client
            turns of the same conversation because subsequent turns depend
            on the timed-out response.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_timeout_s: float = Field(
        default=86400.0,
        gt=0,
        description=(
            "Per-turn timeout in seconds. A timeout aborts that turn and all "
            "remaining turns in the same conversation."
        ),
    )
    enable_salt: bool = Field(
        False,
        description=(
            "Add deterministic salt markers before and after the system prompt "
            "to prevent KV cache reuse across trajectories in agentic inference setting."
        ),
    )
    inject_tool_delay: bool = Field(
        False,
        description=(
            "Pause for a predefined duration between turns. Duration is defined "
            "in dataset."
        ),
    )
    num_trajectories_to_issue: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Number of conversation trajectories to start. Defaults to one pass "
            "over the dataset; values above the dataset size repeat trajectories "
            "with unique logical conversation ids."
        ),
    )
    stop_issuing_on_first_user_complete: bool = Field(
        False,
        description=(
            "When performance tracking stops because the first concurrency slot "
            "has no next trajectory left to assign, also stop issuing future "
            "turns. If false, replay continues outside the performance window "
            "for accuracy/log coverage."
        ),
    )


class Dataset(BaseModel):
    """Dataset configuration.

    Name and type have smart defaults: name is auto-derived from path,
    type defaults to PERFORMANCE.

    Accepts CLI strings via BeforeValidator on BenchmarkConfig.datasets:
    ``[perf|acc:]<path>[,key=value...]``
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field("", description="Dataset name (auto-derived from path if empty)")
    type: DatasetType = Field(
        DatasetType.PERFORMANCE, description="Dataset purpose: performance or accuracy"
    )
    path: Annotated[
        str | None, cyclopts.Parameter(alias="--dataset", help="Dataset file path")
    ] = None
    format: str | None = Field(None, description="Dataset format (auto-detected)")
    samples: int | None = Field(None, gt=0, description="Number of samples to use")
    eval_method: EvalMethod | None = Field(
        None, description="Accuracy evaluation method"
    )
    parser: dict[str, str] | None = Field(
        None, description="Column remapping: {prompt: <col>, system: <col>}"
    )
    generate_params: dict[str, Any] | None = Field(
        None, description="Dataset-specific parameters passed to the generate() method"
    )
    accuracy_config: AccuracyConfig | None = Field(
        None, description="Accuracy evaluation settings"
    )
    agentic_inference: AgenticInferenceConfig | None = Field(
        None, description="Agentic inference conversation configuration"
    )
    # Per-dataset generation config is a first-class capability: different
    # accuracy datasets legitimately want different generation settings (e.g.
    # per-dataset max OSL or top_p, as seen in DS-V4), and dataset-scoping also
    # enables per-dataset dynamic OSL distributions. Only generation knobs are
    # overridable — per-run/identity fields (`_METRICS_DECOUPLED_OVERRIDE_KEYS`:
    # name / streaming / tokenizer_name) drive the single global tokenizer and
    # MetricsAggregator, so overriding them per-dataset would desync ISL/OSL/
    # TTFT/TPOT accounting; they are rejected at validation.
    #
    # TODO(post-mortem): split ModelParams into a per-run ModelIdentity and a
    # GenerationConfig, so the override surface is exactly the generation fields
    # and identity fields cannot be named here at all. Field/method names use
    # "generation_config" to keep that migration mechanical.
    #
    # Nested dicts (`osl_distribution`, `chat_template_kwargs`) are deep-merged
    # so sparse overrides preserve sibling defaults.
    generation_config_override: dict[str, Any] | None = Field(
        None,
        description=(
            "Per-dataset overrides for the top-level model_params (sparse — "
            "only the fields you want to override). Merged on top of "
            "BenchmarkConfig.model_params at dataset-load time. Useful for "
            "MLPerf-style runs where accuracy and performance use different "
            "output budgets in the same fleet, e.g. "
            "generation_config_override: {max_new_tokens: 32768, "
            "temperature: 0.0}. NOTE: per-run/identity keys (`name`, "
            "`streaming`, `tokenizer_name`) are rejected here — set them on "
            "top-level model_params."
        ),
    )

    @model_validator(mode="after")
    def _auto_derive_name(self) -> Self:
        """Derive name from path stem if not explicitly provided."""
        if not self.name and self.path:
            object.__setattr__(self, "name", Path(self.path).stem)
        return self

    @model_validator(mode="after")
    def _validate_generation_config_override(self) -> Self:
        """Fail fast on unknown keys and on per-run/identity keys the single
        global tokenizer / MetricsAggregator would ignore. Override *values*
        are validated at merge time (see ``effective_generation_config``)
        because cross-field validation needs the base ``ModelParams`` from
        ``BenchmarkConfig``.
        """
        if self.generation_config_override:
            keys = set(self.generation_config_override)
            valid = set(ModelParams.model_fields)
            bad = sorted(keys - valid)
            if bad:
                raise ValueError(
                    f"Dataset '{self.name}': unknown keys in "
                    f"generation_config_override: {bad}. "
                    f"Valid keys: {sorted(valid)}"
                )
            decoupled = sorted(keys & _METRICS_DECOUPLED_OVERRIDE_KEYS)
            if decoupled:
                raise ValueError(
                    f"Dataset '{self.name}': generation_config_override keys "
                    f"{decoupled} are not honored per-dataset — the single "
                    "global tokenizer / metrics aggregator is launched from "
                    "top-level model_params, so a per-dataset value would "
                    "desync ISL/OSL/TTFT/TPOT accounting. Set them on "
                    "top-level model_params instead."
                )
        return self

    def effective_generation_config(self, base: ModelParams) -> ModelParams:
        """Return base merged with this dataset's generation-config overrides.

        Nested dicts are deep-merged so a sparse nested override preserves
        sibling defaults (e.g. ``{osl_distribution: {max: 512}}`` keeps the
        base ``type/mean/std/min``). The merged dict is re-validated through
        ``ModelParams.model_validate`` so type-invalid scalar overrides (e.g.
        ``temperature: 'hot'``) are rejected. Note that this only catches
        scalar invalidity — a sparse nested override whose merged result
        passes default-validation will not raise (callers that need stricter
        nested validation should set ``base`` to an explicit instance).
        """
        if not self.generation_config_override:
            return base
        merged = _deep_merge(base.model_dump(), self.generation_config_override)
        return ModelParams.model_validate(merged)


class AccuracyConfig(BaseModel):
    """Accuracy configuration.

    eval_method: Scorer to use (see ScorerMethod enum for options).
    ground_truth: Column in the dataset containing ground truth. Defaults to "ground_truth".
    extractor: Post-processor to extract answers from model output
        (abcd_extractor, boxed_math_extractor, identity_extractor, python_code_extractor).
        Optional for scorers that declare REQUIRES_EXTRACTOR = False (e.g. vbench).
    num_repeats: Number of times to repeat the dataset for evaluation. Defaults to 1.
    extras: Free-form keyword args forwarded to the scorer's ``__init__`` —
        used for scorer-specific knobs that don't warrant a top-level field
        (e.g. ``vbench_project_path``, ``subprocess_timeout_s`` for VBench).

    Example:
        accuracy_config:
          eval_method: "pass_at_1"
          ground_truth: "answer"
          extractor: "boxed_math_extractor"
          num_repeats: 5
          extras:
            vbench_project_path: "/path/to/accuracy"
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    eval_method: ScorerMethod | None = Field(None, description="Scorer method")
    ground_truth: str | None = Field(None, description="Ground truth column name")
    extractor: str | None = Field(
        None,
        description="Answer extractor (abcd_extractor, boxed_math_extractor, identity_extractor, python_code_extractor)",
    )
    num_repeats: int = Field(
        1, ge=1, description="Repeat dataset N times for evaluation"
    )
    extras: dict[str, Any] | None = Field(
        None,
        description="Free-form scorer kwargs (e.g. vbench_project_path, subprocess_timeout_s)",
    )


class RuntimeConfig(BaseModel):
    """Runtime configuration.

    Sample count priority (in RuntimeSettings.total_samples_to_issue()):
    1. n_samples_to_issue (if specified) — explicit override
    2. Calculated from QPS * duration — duration-based (default: 600000ms)
    3. All dataset samples — fallback when duration is 0
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_duration_ms: Annotated[
        int,
        cyclopts.Parameter(
            alias="--duration", help="Min duration (ms, or with suffix: 600s, 10m)"
        ),
    ] = Field(600000, ge=0)
    max_duration_ms: int = Field(
        0,
        ge=0,
        description="Maximum test duration in ms (0 for no limit)",
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
    ] = Field(None, gt=0)
    scheduler_random_seed: int = Field(42, description="Scheduler RNG seed")
    dataloader_random_seed: int = Field(42, description="Dataloader RNG seed")

    @model_validator(mode="after")
    def _validate_durations(self) -> Self:
        if self.max_duration_ms != 0 and self.max_duration_ms < self.min_duration_ms:
            raise ValueError(
                f"max_duration_ms ({self.max_duration_ms}) must be >= "
                f"min_duration_ms ({self.min_duration_ms})"
            )
        return self


@cyclopts.Parameter(name="*")
class LoadPattern(BaseModel):
    """Load pattern configuration.

    Different patterns use target_qps differently:
    - max_throughput: target_qps used for calculating total queries (offline, optional with default)
    - poisson: target_qps sets scheduler rate (online, required - validated)
    - concurrency: issue at fixed target_concurrency (online, required - validated)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        LoadPatternType,
        cyclopts.Parameter(name="--load-pattern", help="Load pattern type"),
    ] = LoadPatternType.MAX_THROUGHPUT
    target_qps: Annotated[
        float | None, cyclopts.Parameter(alias="--target-qps", help="Target QPS")
    ] = Field(None, gt=0)
    target_concurrency: Annotated[
        int | None,
        cyclopts.Parameter(alias="--concurrency", help="Concurrent requests"),
    ] = Field(None, gt=0)

    # TODO(vir): remove once the formal tail-cutting mechanism lands.
    use_legacy_loadgen_qps_metrics: Annotated[
        bool,
        cyclopts.Parameter(
            negative="--no-use-legacy-loadgen-qps-metrics",
            help=(
                "Only applies to the poisson load pattern. Report QPS/TPS using "
                "the legacy MLPerf LoadGen Server 'completed' definition — (completed-1)/T "
                "and tokens/T, T = first issued request to completion of the "
                "last-issued request (see mlcommons/inference loadgen/results.cc). "
                "--no-... uses endpoints-native completed/duration. Ignored for "
                "non-poisson patterns."
            ),
        ),
    ] = True

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # use_legacy_loadgen_qps_metrics only applies to poisson; drop it from
        # the serialized form (and thus YAML templates) for other patterns.
        data = handler(self)
        if self.type != LoadPatternType.POISSON:
            data.pop("use_legacy_loadgen_qps_metrics", None)
        return data

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
        if self.type == LoadPatternType.AGENTIC_INFERENCE and (
            not self.target_concurrency or self.target_concurrency <= 0
        ):
            raise ValueError(
                "Agentic inference requires --concurrency (e.g., --concurrency 96)"
            )
        return self


@cyclopts.Parameter(name="*")
class WarmupConfig(BaseModel):
    """Warmup phase configuration. Runs before the performance phase; results are not recorded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Annotated[
        bool,
        cyclopts.Parameter(
            alias="--warmup", help="Enable warmup phase before performance run"
        ),
    ] = Field(False, description="Enable warmup phase before performance run")
    n_requests: Annotated[
        int | None,
        cyclopts.Parameter(
            alias="--warmup-requests",
            help="Warmup request count (None = full dataset once)",
        ),
    ] = Field(None, gt=0, description="Warmup request count (None = full dataset once)")
    salt: Annotated[
        bool,
        cyclopts.Parameter(
            alias="--warmup-salt",
            help="Prepend a unique random hex salt to each warmup prompt",
        ),
    ] = Field(
        True, description="Prepend a unique random hex salt to each warmup prompt"
    )
    drain: Annotated[
        bool,
        cyclopts.Parameter(
            alias="--warmup-drain",
            help="Drain in-flight warmup requests before starting the performance phase",
        ),
    ] = Field(
        False,
        description="Drain in-flight warmup requests before starting the performance phase",
    )
    warmup_random_seed: Annotated[
        int,
        cyclopts.Parameter(
            alias="--warmup-seed",
            help="RNG seed for warmup scheduling and sample ordering",
        ),
    ] = Field(42, description="RNG seed for warmup scheduling and sample ordering")


class DrainConfig(BaseModel):
    """Per-phase in-flight response drain timeout configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    warmup_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--warmup-drain-timeout",
            help="Warmup drain timeout in seconds (None = wait indefinitely)",
        ),
    ] = Field(
        240.0,
        gt=0,
        description="Warmup drain timeout in seconds (None = wait indefinitely)",
    )
    performance_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--performance-drain-timeout",
            help="Performance drain timeout in seconds (None = wait indefinitely)",
        ),
    ] = Field(
        240.0,
        gt=0,
        description="Performance drain timeout in seconds (None = wait indefinitely)",
    )
    accuracy_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--accuracy-drain-timeout",
            help="Accuracy drain timeout in seconds (None = wait indefinitely)",
        ),
    ] = Field(
        None,
        gt=0,
        description="Accuracy drain timeout in seconds (None = wait indefinitely)",
    )
    metrics_drain_timeout_s: Annotated[
        float,
        cyclopts.Parameter(
            alias="--metrics-drain-timeout",
            help=(
                "Wall-clock budget (seconds) for the metrics aggregator to finish "
                "tokenizing buffered samples after the run ends. Set to 0 to wait "
                "indefinitely. Increase for very large datasets where the end-of-run "
                "tokenize batch is big."
            ),
        ),
    ] = Field(
        0.0,
        ge=0,
        description=(
            "Wall-clock budget (seconds) to finish tokenizing buffered samples "
            "after ENDED (default: 0 = unlimited). An incomplete drain is "
            "surfaced via n_pending_tasks > 0, never silently dropped."
        ),
    )
    metrics_tokenizer_workers: Annotated[
        int,
        cyclopts.Parameter(
            alias="--metrics-tokenizer-workers",
            help=(
                "In-process tokenizer threads for live (mid-run) ISL/OSL/TPOT in "
                "the metrics aggregator. 0 defers all tokenization to the "
                "end-of-run drain, which always uses the auto-sized sharded pool."
            ),
        ),
    ] = Field(
        2,
        ge=0,
        description=(
            "In-process tokenizer threads for live (mid-run) ISL/OSL/TPOT "
            "(default: 2; 0 = defer everything to the end-of-run drain)."
        ),
    )


class ProfilerEngine(str, Enum):
    """Inference engine whose profiling protocol the client should drive.

    Selects the HTTP path layout used to derive start/stop URLs from
    ``endpoint_config.endpoints``. Each value corresponds to one server-side
    profiling protocol; add a new variant + ``_PROFILE_PATHS`` row to support
    another engine.
    """

    VLLM = "vllm"


@cyclopts.Parameter(name="*")
class ProfilingConfig(BaseModel):
    """Client-side trigger for the server's profiler.

    When ``engine`` is set, fires POST ``<start_path>`` at performance-phase
    begin and POST ``<stop_path>`` at performance-phase end. URLs are derived
    using the engine-specific protocol from ``urls`` when set, otherwise
    from ``endpoint_config.endpoints``.
    Server must be launched with profiling enabled (e.g. vLLM's
    ``--profiler-config.profiler=cuda|torch``); the schedule
    (``delay_iterations``, ``max_iterations``) is set there, not here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: Annotated[
        ProfilerEngine | None,
        cyclopts.Parameter(
            alias="--profile",
            help="Profile the named inference engine around the performance phase",
        ),
    ] = Field(
        None,
        description="Profile the named inference engine around the performance phase",
    )
    urls: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            alias="--profile-urls",
            help="Override URL(s) for profiler triggers; "
            "defaults to endpoint_config.endpoints",
            negative="",
        ),
    ] = Field(
        None,
        description="URL(s) the profiler start/stop triggers are derived from. "
        "When None, derived from endpoint_config.endpoints instead. Use when "
        "the profiler admin endpoint differs from the inference endpoint.",
    )

    @field_validator("urls", mode="after")
    @classmethod
    def _validate_url_scheme(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for url in v:
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"Profiling endpoint URL must include scheme "
                    f"(http:// or https://), got: {url!r}"
                )
        return v


@cyclopts.Parameter(name="*")
class Settings(BaseModel):
    """Test settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    load_pattern: LoadPattern = Field(default_factory=LoadPattern)
    client: HTTPClientConfig = Field(default_factory=HTTPClientConfig)
    drain: DrainConfig = Field(
        default_factory=DrainConfig,
        description="Per-phase in-flight response drain timeout configuration",
    )
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    service_ready_timeout_s: Annotated[
        float,
        cyclopts.Parameter(
            alias="--service-ready-timeout",
            help="Seconds to wait for metrics/event-logger services to start",
        ),
    ] = Field(
        default=30.0,
        ge=0,
        description="Seconds to wait for metrics-aggregator/event-logger services to become ready.",
    )


class OfflineSettings(Settings):
    """Offline mode default settings."""

    load_pattern: Annotated[LoadPattern, cyclopts.Parameter(show=False)] = Field(
        default_factory=lambda: LoadPattern(type=LoadPatternType.MAX_THROUGHPUT)
    )


class OnlineSettings(Settings):
    """Online mode default settings."""

    pass


class EndpointConfig(BaseModel):
    """Endpoint connection configuration.

    Contains endpoint URL and authentication settings.
    API type refers to the API implementation used on the endpoint based on industry standards.
    The Default API type is APIType.OPENAI, which refers to the the /v1/chat/completions route.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    endpoints: Annotated[
        list[str],
        cyclopts.Parameter(alias="--endpoints", help="Endpoint URL(s)", negative=""),
    ] = Field(
        min_length=1,
        description="Endpoint URL(s). Must include scheme, e.g. 'http://host:port'.",
    )
    api_key: Annotated[
        str | None, cyclopts.Parameter(alias="--api-key", help="API key")
    ] = None
    api_type: Annotated[
        APIType,
        cyclopts.Parameter(
            alias="--api-type", help="API type: openai, sglang, or videogen"
        ),
    ] = APIType.OPENAI

    @field_validator("endpoints", mode="after")
    @classmethod
    def _validate_endpoint_scheme(cls, v: list[str]) -> list[str]:
        for url in v:
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"Endpoint URL must include scheme (http:// or https://), got: {url!r}"
                )
        return v


class BenchmarkConfig(WithUpdatesMixin, BaseModel):
    """Benchmark configuration — single source of truth for YAML and CLI.

    Immutable (frozen) to prevent accidental modifications during execution.
    cyclopts auto-generates CLI flags from fields. Use cyclopts.Parameter(name=...)
    on Annotated fields to declare flat shorthand aliases.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

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
    verbose: Annotated[bool, cyclopts.Parameter(show=False)] = Field(
        False, description="Enable verbose logging"
    )
    enable_cpu_affinity: Annotated[
        bool,
        cyclopts.Parameter(
            negative="--no-cpu-affinity",
            help="NUMA-aware CPU pinning",
        ),
    ] = True
    audit: Annotated[AuditConfig | None, cyclopts.Parameter(show=False)] = Field(
        None,
        description="Compliance audit config (YAML only). When set, runs the audit after the main benchmark.",
    )

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

        # TODO(vir): Move API-type-specific validation out of this generic
        # cross-model validator and into the selected adapter. Requires a larger refactor.
        #
        # Completion-only controls must be gated by api_type for BOTH the
        # top-level model_params AND every per-dataset generation_config_override,
        # so the two config surfaces validate identically. Merge each dataset's
        # effective params once here (parse time) — this also surfaces
        # value-invalid overrides before setup produces side effects — and reuse
        # the result for the agentic-inference check below.
        effective_by_dataset: dict[int, ModelParams] = {
            id(dataset): dataset.effective_generation_config(self.model_params)
            for dataset in self.datasets
            if dataset.generation_config_override
        }
        completion_control_surfaces: list[tuple[str, ModelParams]] = [
            ("model_params", self.model_params)
        ]
        for dataset in self.datasets:
            effective = effective_by_dataset.get(id(dataset))
            if effective is not None:
                completion_control_surfaces.append(
                    (
                        f"datasets['{dataset.name}'].generation_config_override",
                        effective,
                    )
                )
        for prefix, mp in completion_control_surfaces:
            controls = _non_default_completion_controls(mp)
            if controls and self.endpoint_config.api_type != APIType.OPENAI_COMPLETIONS:
                names = " and ".join(f"{prefix}.{name}" for name in controls)
                verb = "requires" if len(controls) == 1 else "require"
                raise ValueError(
                    f"{names} {verb} endpoint_config.api_type=openai_completions"
                )
        for dataset in self.datasets:
            if dataset.agentic_inference is None:
                continue
            effective = effective_by_dataset.get(id(dataset), self.model_params)
            if _non_default_completion_controls(effective):
                raise ValueError(
                    "OpenAI text-completion generation controls are not supported "
                    "for agentic inference datasets"
                )

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
            if lp.type not in (
                LoadPatternType.POISSON,
                LoadPatternType.CONCURRENCY,
                LoadPatternType.AGENTIC_INFERENCE,
            ):
                raise ValueError(
                    "Online mode requires --load-pattern (poisson, concurrency, or agentic_inference)"
                )

        # Cross-validate load_pattern.type=agentic_inference against the
        # performance dataset agentic_inference config.
        has_agentic_inference_perf_dataset = any(
            d.agentic_inference is not None
            for d in (self.datasets or [])
            if d.type == DatasetType.PERFORMANCE
        )
        has_agentic_inference_non_perf_dataset = any(
            d.agentic_inference is not None
            for d in (self.datasets or [])
            if d.type != DatasetType.PERFORMANCE
        )
        if has_agentic_inference_non_perf_dataset:
            raise ValueError(
                "agentic_inference config is only supported on performance datasets; "
                "accuracy datasets with agentic_inference are not supported"
            )
        if (
            lp.type == LoadPatternType.AGENTIC_INFERENCE
            and not has_agentic_inference_perf_dataset
        ):
            raise ValueError(
                "load_pattern.type=agentic_inference requires the performance "
                "dataset to have agentic_inference config"
            )
        if (
            lp.type == LoadPatternType.AGENTIC_INFERENCE
            and self.settings.runtime.n_samples_to_issue is not None
        ):
            raise ValueError(
                "runtime.n_samples_to_issue is not supported for agentic inference runs; "
                "use datasets[].agentic_inference.num_trajectories_to_issue instead"
            )
        if (
            has_agentic_inference_perf_dataset
            and lp.type != LoadPatternType.AGENTIC_INFERENCE
        ):
            raise ValueError(
                "Performance dataset with agentic_inference config requires "
                "load_pattern.type=agentic_inference, "
                f"got '{lp.type}'"
            )

        # Pin RNG seeds from the submission ruleset. Done last so the values
        # are baked into the config before any consumer reads them — the config
        # dump to the report dir, RuntimeSettings.from_config, and the report
        # seeds block all see the pinned values.
        self._apply_ruleset_seed_overrides()

        return self

    def _apply_ruleset_seed_overrides(self) -> None:
        """Override runtime + warmup RNG seeds from the selected submission ruleset.

        MLPerf rounds pin the RNG seeds; this mirrors LoadGen locking the core
        seeds from ``user.conf`` (a submitter cannot substitute their own).
        If ``submission_ref`` is unset, the config is left unchanged. If it
        names an unregistered ruleset, a ``type=SUBMISSION`` config errors (a
        submission cannot silently fall back to default seeds), while any other
        type is left unchanged so non-submission/placeholder configs still work.

        The warmup phase is reseeded from the sample-index (dataloader) seed so
        its sample order derives from the same pinned seed as the perf phase.
        Only the seed *value* is propagated — each phase builds its own
        ``random.Random`` downstream, so the RNG object is never shared.
        """
        if self.submission_ref is None:
            return
        try:
            ruleset = self.submission_ref.get_ruleset_instance()
        except KeyError as e:
            if self.type == TestType.SUBMISSION:
                raise ValueError(
                    f"submission_ref.ruleset {self.submission_ref.ruleset!r} is not "
                    "registered; a submission must pin official RNG seeds and cannot "
                    "fall back to defaults."
                ) from e
            logger.warning(
                "submission_ref.ruleset %r is not registered; skipping ruleset "
                "seed overrides.",
                self.submission_ref.ruleset,
            )
            return

        # A ruleset used as a submission_ref must pin both seeds. ``None`` means
        # "unseeded" in the general ruleset contract (ruleset_base.py), but an
        # unseeded submission is incoherent and would silently diverge from the
        # random.Random(None) path in RoundRuleset.apply_user_config. Reject it.
        if ruleset.scheduler_rng_seed is None or ruleset.sample_index_rng_seed is None:
            raise ValueError(
                f"submission_ref.ruleset {self.submission_ref.ruleset!r} leaves an "
                "RNG seed unset; a pinned ruleset must define both the scheduler "
                "and sample-index seeds."
            )

        # Rebuild through model_validate (not model_copy(update=)): with
        # extra='forbid' this validates seed *values* and rejects renamed/unknown
        # fields. model_copy(update=) writes straight into __dict__, so a
        # wrong-typed (e.g. str) or renamed seed would slip through unchecked.
        runtime = self.settings.runtime
        new_runtime = type(runtime).model_validate(
            {
                **runtime.model_dump(),
                "scheduler_random_seed": ruleset.scheduler_rng_seed,
                "dataloader_random_seed": ruleset.sample_index_rng_seed,
            }
        )
        warmup = self.settings.warmup
        new_warmup = type(warmup).model_validate(
            {
                **warmup.model_dump(),
                "warmup_random_seed": ruleset.sample_index_rng_seed,
            }
        )
        object.__setattr__(
            self,
            "settings",
            self.settings.model_copy(
                update={"runtime": new_runtime, "warmup": new_warmup}
            ),
        )
        logger.debug(
            "Pinned RNG seeds from ruleset %r: scheduler=%s sample_index=%s "
            "(warmup reseeded from sample_index)",
            self.submission_ref.ruleset,
            ruleset.scheduler_rng_seed,
            ruleset.sample_index_rng_seed,
        )

    @model_validator(mode="after")
    def _propagate_client_api_type(self) -> Self:
        """Sync client.api_type from endpoint_config.api_type at construction.

        ``endpoint_config.api_type`` is the user-facing source of truth.
        ``HTTPClientConfig.api_type`` is internal and only exists so the
        adapter/accumulator can be resolved by ``_resolve_defaults``. Without
        this propagation, a YAML/CLI that selects SGLang on ``endpoint_config``
        would leave the client with the OpenAI adapter until ``execute.py``
        patched it at runtime.
        """
        target = self.endpoint_config.api_type
        if self.settings.client.api_type != target:
            new_client = self.settings.client.with_updates(
                api_type=target,
                adapter=None,
                accumulator=None,
            )
            object.__setattr__(self.settings, "client", new_client)
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
            CLIError: If test_type is EVAL or SUBMISSION (not yet implemented)
            ValueError: If test_type is invalid
        """
        _common = {
            "model_params": ModelParams(name="<MODEL_NAME>"),
            "datasets": [Dataset(path="<DATASET_PATH>")],
            "endpoint_config": EndpointConfig(endpoints=["http://localhost:8000"]),
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
            raise CLIError(
                "Default EVAL config not yet implemented. "
                "Track progress at: https://github.com/mlcommons/endpoints/issues/4"
            )
        if test_type == TestType.SUBMISSION:
            raise CLIError(
                "Default SUBMISSION config not yet implemented. "
                "Track progress at: https://github.com/mlcommons/endpoints/issues/5"
            )
        raise ValueError(f"Unknown test type: {test_type}")


@cyclopts.Parameter(name="*")
class OfflineBenchmarkConfig(BenchmarkConfig):
    """Offline benchmark config — type locked, load pattern hidden."""

    type: Annotated[Literal[TestType.OFFLINE], cyclopts.Parameter(show=False)] = (
        TestType.OFFLINE
    )  # type: ignore[assignment]
    settings: OfflineSettings = Field(default_factory=OfflineSettings)  # type: ignore[reportIncompatibleVariableOverride]


@cyclopts.Parameter(name="*")
class OnlineBenchmarkConfig(BenchmarkConfig):
    """Online benchmark config — type locked."""

    type: Annotated[Literal[TestType.ONLINE], cyclopts.Parameter(show=False)] = (
        TestType.ONLINE
    )  # type: ignore[assignment]
    settings: OnlineSettings = Field(default_factory=OnlineSettings)  # type: ignore[reportIncompatibleVariableOverride]


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
