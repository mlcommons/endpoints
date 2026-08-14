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

All Pydantic models define both the YAML config structure and the CLI
interface. cyclopts auto-generates CLI flags from fields. Use
cyclopts.Parameter(alias=...) on Annotated fields to declare shorthand
aliases alongside dotted paths.

Split criterion: one module per config domain (audit / model_params /
datasets / settings / timeouts), with every name — model, enum, helper —
living beside its owner; this module owns only the root aggregate
(``BenchmarkConfig``, its cross-field validation, and the root-level
``TestType``/``TestMode`` enums) plus the explicit re-export hub, so every
existing ``config.schema`` import site keeps working.
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cyclopts
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ..core.types import APIType
from ..exceptions import CLIError
from ..utils import WithUpdatesMixin
from .audit import AuditConfig, AuditTestId, OutputCachingTestConfig
from .datasets import (
    AccuracyConfig,
    AgenticInferenceConfig,
    Dataset,
    DatasetType,
    EvalMethod,
    ScorerMethod,
)
from .model_params import (
    ModelParams,
    OSLDistribution,
    OSLDistributionType,
    StreamingMode,
    SubmissionReference,
    _non_default_completion_controls,
)
from .settings import (
    EarlyStoppingConfig,
    LoadPattern,
    LoadPatternType,
    OfflineSettings,
    OnlineSettings,
    ProfilerEngine,
    ProfilingConfig,
    RuntimeConfig,
    Settings,
    WarmupConfig,
)
from .timeouts import Timeouts
from .utils import parse_dataset_string, resolve_env_vars

__all__ = [
    "APIType",
    "AccuracyConfig",
    "AgenticInferenceConfig",
    "AuditConfig",
    "AuditTestId",
    "BenchmarkConfig",
    "Dataset",
    "DatasetType",
    "EarlyStoppingConfig",
    "EndpointConfig",
    "EvalMethod",
    "LoadPattern",
    "LoadPatternType",
    "ModelParams",
    "OSLDistribution",
    "OSLDistributionType",
    "OfflineBenchmarkConfig",
    "OfflineSettings",
    "OnlineBenchmarkConfig",
    "OnlineSettings",
    "OutputCachingTestConfig",
    "ProfilerEngine",
    "ProfilingConfig",
    "RuntimeConfig",
    "ScorerMethod",
    "Settings",
    "StreamingMode",
    "SubmissionReference",
    "TestMode",
    "TestType",
    "Timeouts",
    "WarmupConfig",
]

logger = logging.getLogger(__name__)


class TestMode(str, Enum):
    """Test mode controlling performance issuance and response collection.

    - PERF: Run performance and ordinary configured scoring without in-process
      collection; skip scorers that own an external evaluation run
    - ACC: Skip performance and collect responses for configured scoring
    - BOTH: Run performance and configured scoring with response collection
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

        uses_swe_bench = any(
            dataset.accuracy_config is not None
            and dataset.accuracy_config.eval_method == ScorerMethod.SWE_BENCH
            for dataset in self.datasets
        )
        if uses_swe_bench and len(self.endpoint_config.endpoints) != 1:
            raise ValueError(
                "SWE-bench service mode supports exactly one endpoint URL; "
                f"got {len(self.endpoint_config.endpoints)}."
            )

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

        # Duplicate datasets — same (name, type) would collide in the accuracy report
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

        # Forward target_concurrency as SWE-bench workers when unset.
        concurrency = (
            lp.target_concurrency
            if lp.type
            in (LoadPatternType.CONCURRENCY, LoadPatternType.AGENTIC_INFERENCE)
            and lp.target_concurrency
            else None
        )
        if concurrency is not None and self.datasets:
            updated_datasets = []
            changed = False
            for ds in self.datasets:
                acc = ds.accuracy_config
                if (
                    acc is not None
                    and acc.eval_method == ScorerMethod.SWE_BENCH
                    and (acc.extras is None or acc.extras.get("workers") is None)
                ):
                    new_extras = {**(acc.extras or {}), "workers": concurrency}
                    new_acc = acc.model_copy(update={"extras": new_extras})
                    ds = ds.model_copy(update={"accuracy_config": new_acc})
                    changed = True
                updated_datasets.append(ds)
            if changed:
                object.__setattr__(self, "datasets", updated_datasets)

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

    @staticmethod
    def _is_secret_field_name(key: Any) -> bool:
        normalized = str(key).strip().lower().replace("-", "_")
        return (
            normalized
            in {
                "api_key",
                "access_token",
                "authorization",
                "auth_token",
                "password",
                "token",
            }
            or normalized.endswith(("_key", "_token", "_password"))
            or "secret" in normalized
        )

    @staticmethod
    def _redact_url_secrets(value: str) -> str:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return value
        if parsed.scheme.lower() not in {"http", "https", "ws", "wss"} or not (
            parsed.netloc
        ):
            return value

        changed = False
        netloc = parsed.netloc
        if "@" in netloc:
            netloc = f"<redacted>@{netloc.rsplit('@', 1)[1]}"
            changed = True

        query = parse_qsl(parsed.query, keep_blank_values=True)
        redacted_query: list[tuple[str, str]] = []
        for key, item in query:
            if BenchmarkConfig._is_secret_field_name(key):
                item = "<redacted>"
                changed = True
            redacted_query.append((key, item))

        if not changed:
            return value
        return urlunsplit(
            parsed._replace(netloc=netloc, query=urlencode(redacted_query, doseq=True))
        )

    @staticmethod
    def _redact_secret_fields(value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if BenchmarkConfig._is_secret_field_name(key):
                    redacted[key] = "<redacted>"
                else:
                    redacted[key] = BenchmarkConfig._redact_secret_fields(item)
            return redacted
        if isinstance(value, list):
            return [BenchmarkConfig._redact_secret_fields(item) for item in value]
        if isinstance(value, str):
            return BenchmarkConfig._redact_url_secrets(value)
        return value

    def to_yaml_file(
        self,
        path: Path,
        exclude_none: bool = True,
        *,
        redact_secrets: bool = False,
    ) -> None:
        """Save BenchmarkConfig to YAML file.

        Args:
            path: Path to save YAML file
            exclude_none: Whether to exclude None values (default: True)
            redact_secrets: Replace secret-like values before persistence.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(exclude_none=exclude_none, mode="json")
        if redact_secrets:
            payload = self._redact_secret_fields(payload)

        with open(path, "w") as f:
            yaml.dump(
                payload,
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
