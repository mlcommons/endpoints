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

"""Test settings models (the ``settings:`` block).

Split criterion: one module per config domain; the runtime/load-pattern/
warmup/profiling settings and the ``Settings`` aggregate live here.
``config/schema.py`` re-exports the public surface.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

import cyclopts
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ..endpoint_client.config import HTTPClientConfig
from .enums import LoadPatternType, ProfilerEngine
from .timeouts import Timeouts


class RuntimeConfig(BaseModel):
    """Runtime configuration.

    Sample count priority (in RuntimeSettings.total_samples_to_issue()):
    1. n_samples_to_issue (if specified) — explicit override
    2. All dataset samples — issue the dataset once

    ``max_duration_ms`` is a workload duration (part of the benchmark
    definition), not a give-up deadline — those live in ``settings.timeouts``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_duration_ms: int | None = Field(
        None,
        gt=0,
        description="Maximum test duration in ms (None for no limit)",
    )

    @field_validator("max_duration_ms", mode="before")
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

    def __str__(self) -> str:
        """Human-readable "type (param=value)" form for logging, e.g.
        ``concurrency (target_concurrency=7)`` / ``poisson (target_qps=10.0)``.
        Patterns without a driving parameter render as just the type name.
        """
        if self.type in (
            LoadPatternType.CONCURRENCY,
            LoadPatternType.AGENTIC_INFERENCE,
        ):
            return f"{self.type.value} (target_concurrency={self.target_concurrency})"
        if self.type == LoadPatternType.POISSON:
            return f"{self.type.value} (target_qps={self.target_qps})"
        return self.type.value


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


class EarlyStoppingConfig(BaseModel):
    """MLPerf-style early-stopping percentile estimates (on by default).

    Adds conservative, confidence-backed estimates of the tail percentiles to the
    TTFT / TPOT / latency metrics in ``result_summary.json``. Computed once at run
    COMPLETE from data the aggregator already keeps (hot path untouched), and the
    output field is additive — so it is on by default; ``enabled: false`` is the
    single opt-out (e.g. for consumers that strictly validate the summary schema).
    Percentile targets, confidence (0.99), and tolerance (0.0) are LoadGen-parity
    constants in ``metrics/early_stopping.py``, not knobs. Estimate-only: no
    target-latency pass/fail and no dynamic mid-run halt. See ``docs/early_stopping.md``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Annotated[
        bool,
        cyclopts.Parameter(
            alias="--early-stopping",  # --no-early-stopping is the meaningful opt-out
            help="Report MLPerf early-stopping percentile estimates for TTFT/TPOT/latency",
        ),
    ] = Field(True, description="Early-stopping percentile estimates (default on)")


@cyclopts.Parameter(name="*")
class Settings(BaseModel):
    """Test settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    load_pattern: LoadPattern = Field(default_factory=LoadPattern)
    client: HTTPClientConfig = Field(default_factory=HTTPClientConfig)
    timeouts: Timeouts = Field(
        default_factory=Timeouts,
        description="All global waits and deadlines (see config/timeouts.py)",
    )
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
        description="MLPerf early-stopping percentile estimates (on by default; enabled: false opts out)",
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
        4,
        ge=0,
        description=(
            "In-process tokenizer threads for live (mid-run) ISL/OSL/TPOT "
            "(default: 4; 0 = defer everything to the end-of-run drain)."
        ),
    )


class OfflineSettings(Settings):
    """Offline mode default settings."""

    load_pattern: Annotated[LoadPattern, cyclopts.Parameter(show=False)] = Field(
        default_factory=lambda: LoadPattern(type=LoadPatternType.MAX_THROUGHPUT)
    )


class OnlineSettings(Settings):
    """Online mode default settings."""

    pass
