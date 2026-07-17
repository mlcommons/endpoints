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

"""Benchmark execution — phased architecture.

Phases:
    1. setup_benchmark()        — load tokenizer, dataset, config (no IO)
    2. run_benchmark_async()    — HTTP client + async BenchmarkSession
    3. finalize_benchmark()     — accuracy scoring, results JSON

Cohesive sub-concerns live in sibling modules: profiler triggers (``profiling``),
accuracy scoring (``accuracy``), and the ZMQ/metrics/event-logger service lifecycle
(``pipeline``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import signal
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import msgspec.json
import msgspec.structs
from huggingface_hub import model_info
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.commands.benchmark.accuracy import (
    AccuracyConfiguration,
    _score_accuracy,
    write_accuracy_results,
)
from inference_endpoint.commands.benchmark.pipeline import MetricsPipeline
from inference_endpoint.commands.benchmark.profiling import (
    ProfileController,
    _write_profiling_section,
)
from inference_endpoint.compliance import AuditRunSpec
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    BenchmarkConfig,
    DatasetType,
    LoadPattern,
    LoadPatternType,
    StreamingMode,
    TestMode,
    TestType,
)
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.agentic_inference_dataset import (
    AgenticInferenceDataset,
)
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.endpoint_client.cpu_affinity import AffinityPlan, pin_loadgen
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.evaluation import Extractor
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.exceptions import (
    ExecutionError,
    InputValidationError,
    SetupError,
)
from inference_endpoint.load_generator.agentic_inference_strategy import (
    AgenticInferenceStrategy,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    PhaseConfig,
    PhaseType,
    SessionResult,
)
from inference_endpoint.metrics.report import Report

transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def _default_report_path() -> Path:
    """Default report path with timestamp."""
    return Path(
        f"{tempfile.gettempdir()}/reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def resolve_report_dir(config: BenchmarkConfig) -> Path:
    """Resolve the run's report directory, defaulting to a timestamped path.

    Exposed so callers that need the report dir before invoking
    ``setup_benchmark`` (e.g. to share one directory tree across multiple
    runs against the same config) resolve it identically rather than
    duplicating the default-path logic.
    """
    return Path(config.report_dir) if config.report_dir else _default_report_path()


class ResponseCollector:
    """Collects query responses and errors for accuracy evaluation."""

    def __init__(self, collect_responses: bool = False, pbar: tqdm | None = None):
        self.collect_responses = collect_responses
        self.responses: dict[str, str] = {}
        self.errors: list[str] = []
        self.count = 0
        self.pbar = pbar

    def on_complete_hook(self, result: QueryResult) -> None:
        """Handle query completion (called once per query via QueryResult)."""
        self.count += 1
        if result.error:
            self.errors.append(f"Sample {result.id}: {result.error}")
            if self.pbar:
                self.pbar.set_postfix(refresh=True, errors=len(self.errors))
        elif self.collect_responses:
            self.responses[result.id] = result.get_response_output_string()
        if self.pbar:
            self.pbar.update(1)


@dataclass
class BenchmarkResult:
    """Output of run_benchmark_async — all data needed for finalization."""

    session: SessionResult
    collector: ResponseCollector
    report: Report | None
    tmpfs_dir: Path
    # Profile trigger payload {engine: str, starts: [...], stops: [...]} when
    # settings.profiling.engine is set; None otherwise. Rendered into
    # report.txt and a sibling profiling.json by finalize_benchmark.
    profiling: dict[str, Any] | None = None


@dataclass
class BenchmarkContext:
    """All state needed to run a benchmark, created by setup_benchmark.

    Derived values are computed as properties from config, not stored redundantly.
    """

    config: BenchmarkConfig
    test_mode: TestMode
    report_dir: Path
    tokenizer_name: str | None
    dataloader: Dataset | None
    rt_settings: RuntimeSettings | None
    total_samples: int
    accuracy_datasets: list[Dataset] = field(default_factory=list)
    eval_configs: list[AccuracyConfiguration] = field(default_factory=list)
    affinity_plan: AffinityPlan | None = None

    @property
    def collect_responses(self) -> bool:
        return self.test_mode in (TestMode.ACC, TestMode.BOTH)

    @property
    def accuracy_only(self) -> bool:
        """TestMode.ACC is the single source of truth for accuracy-only runs."""
        return self.test_mode == TestMode.ACC

    @property
    def benchmark_mode(self) -> TestType | None:
        return self.config.get_benchmark_mode()

    @property
    def enable_streaming(self) -> bool:
        return self.config.model_params.streaming == StreamingMode.ON


def _check_tokenizer_exists(model_name: str) -> bool:
    """Check if a tokenizer exists for the model (local dir or HF repo, no download).

    Returns True if a tokenizer is available, False otherwise. This function is
    a probe — it never loads or downloads the tokenizer itself. Downstream
    consumers that need tokenization (e.g. the MetricsAggregator subprocess
    for ISL/OSL/TPOT, Harmony transforms for prompt preprocessing, and any
    future plugin with its own tokenization need) each load their own instance
    as required.

    ``model_name`` may be a local checkpoint directory (e.g. an NVFP4 snapshot
    cached under ``/root/.cache/huggingface/hub/...``) or an HF repo ID. Local
    directories are probed directly; otherwise we ask the HF Hub for the file
    listing.
    """
    try:
        local_path = Path(model_name)
        if local_path.is_dir():
            siblings = {p.name for p in local_path.iterdir() if p.is_file()}
        else:
            info = model_info(model_name)
            siblings = {s.rfilename for s in (info.siblings or [])}

        has_tokenizer = (
            "tokenizer_config.json" in siblings or "tokenizer.json" in siblings
        )

        if has_tokenizer:
            logger.info(f"Tokenizer available for model: {model_name}")
        else:
            logger.warning(f"Model {model_name} found but has no tokenizer files")
        return has_tokenizer

    except ImportError:
        # huggingface_hub not installed — fall back to assuming it works
        logger.info(
            f"huggingface_hub not installed, assuming tokenizer exists for {model_name}"
        )
        return True
    except Exception as e:
        logger.warning(f"Could not verify tokenizer for {model_name}: {e}")
        logger.warning(
            "Continuing without tokenizer (ISL/OSL/TPOT metrics will be unavailable)"
        )
        return False


def _resolve_accuracy_components(
    dataset_name: str, accuracy_config: Any | None
) -> tuple[type[Scorer], type[Extractor] | None]:
    """Validate scorer/extractor config and return resolved classes."""
    if accuracy_config is None or accuracy_config.eval_method is None:
        raise InputValidationError(
            f"Dataset '{dataset_name}' requires accuracy_config with eval_method"
        )

    try:
        scorer_cls = Scorer.get(accuracy_config.eval_method)
    except KeyError as exc:
        raise InputValidationError(str(exc)) from exc
    extractor_name = accuracy_config.extractor
    if extractor_name is None:
        if scorer_cls.REQUIRES_EXTRACTOR:
            raise InputValidationError(
                f"Dataset '{dataset_name}' uses scorer "
                f"'{accuracy_config.eval_method}' which requires an extractor"
            )
        extractor_cls: type[Extractor] | None = None
    else:
        try:
            extractor_cls = Extractor.get(extractor_name)
        except KeyError as exc:
            raise InputValidationError(str(exc)) from exc
    return scorer_cls, extractor_cls


def _load_datasets(
    config: BenchmarkConfig,
    report_dir: Path,
    test_mode: TestMode,
) -> tuple[Dataset | None, list[Dataset], list[AccuracyConfiguration]]:
    """Load performance and accuracy datasets. Returns (perf_loader, acc_datasets, eval_configs)."""
    accuracy_only = test_mode == TestMode.ACC
    accuracy_cfgs = [ds for ds in config.datasets if ds.type == DatasetType.ACCURACY]
    performance_cfgs = [
        ds for ds in config.datasets if ds.type == DatasetType.PERFORMANCE
    ]

    if accuracy_only:
        if not accuracy_cfgs:
            raise InputValidationError(
                "--accuracy-only requires at least one accuracy dataset"
            )
    elif not performance_cfgs:
        raise InputValidationError("At least one performance dataset required")

    accuracy_datasets: list[Dataset] = []
    eval_configs: list[AccuracyConfiguration] = []

    # Pack the evaluation parameters for each accuracy dataset
    for acc_cfg in accuracy_cfgs:
        scorer_cls, extractor_cls = _resolve_accuracy_components(
            acc_cfg.name, acc_cfg.accuracy_config
        )
        assert acc_cfg.accuracy_config is not None

        ds = DataLoaderFactory.create_loader(
            acc_cfg, num_repeats=acc_cfg.accuracy_config.num_repeats
        )
        accuracy_datasets.append(ds)
        # TODO add tests and defaults
        eval_configs.append(
            AccuracyConfiguration(
                scorer_cls,
                extractor_cls,
                acc_cfg.name,
                ds,
                report_dir,
                acc_cfg.accuracy_config.ground_truth,
                acc_cfg.accuracy_config.num_repeats,
                acc_cfg.accuracy_config.extras or {},
                dataset_type=DatasetType.ACCURACY,
            )
        )
        # Value/api-type validity of the override is already enforced at config
        # construction (BenchmarkConfig validates each dataset's effective params),
        # so this cannot raise for a validated config.
        ds_model_params = acc_cfg.effective_generation_config(config.model_params)
        ds.load(api_type=config.endpoint_config.api_type, model_params=ds_model_params)
        logger.info(f"Loaded {ds} - {ds.num_samples()} samples")

    if not accuracy_cfgs:
        logger.info("No separate accuracy datasets provided")

    dataloader: Dataset | None = None
    # --accuracy-only skips the performance dataset entirely (including its inline
    # accuracy scorer), so a single config carrying both a performance and an
    # accuracy dataset can run accuracy on its own.
    if performance_cfgs and not accuracy_only:
        if len(performance_cfgs) > 1:
            raise InputValidationError("Multiple performance datasets not supported")
        perf_cfg = performance_cfgs[0]
        # Override validity is enforced at config construction (see accuracy loop).
        perf_model_params = perf_cfg.effective_generation_config(config.model_params)
        try:
            dataloader = DataLoaderFactory.create_loader(perf_cfg)
            dataloader.load(
                api_type=config.endpoint_config.api_type,
                model_params=perf_model_params,
            )
            logger.info(f"Loaded {dataloader.num_samples()} samples")
        except FileNotFoundError as e:
            raise InputValidationError(
                f"Dataset file not found: {perf_cfg.path}"
            ) from e
        except Exception as e:
            raise SetupError(f"Failed to load dataset: {e}") from e

        if perf_cfg.accuracy_config is not None:
            accuracy_config = perf_cfg.accuracy_config
            if accuracy_config.num_repeats != 1:
                raise InputValidationError(
                    f"Dataset '{perf_cfg.name}' is a performance dataset; "
                    "accuracy_config.num_repeats must be 1 because scoring runs on "
                    "already-issued performance outputs"
                )
            scorer_cls, extractor_cls = _resolve_accuracy_components(
                perf_cfg.name, accuracy_config
            )

            eval_configs.append(
                AccuracyConfiguration(
                    scorer_cls,
                    extractor_cls,
                    "performance",
                    dataloader,
                    report_dir,
                    accuracy_config.ground_truth,
                    accuracy_config.num_repeats,
                    accuracy_config.extras or {},
                    dataset_type=DatasetType.PERFORMANCE,
                )
            )

    return dataloader, accuracy_datasets, eval_configs


def setup_benchmark(
    config: BenchmarkConfig,
    test_mode: TestMode,
    audit_run_spec: AuditRunSpec | None = None,
) -> BenchmarkContext:
    """Load tokenizer, dataset, create scheduler, setup report dir.

    ``audit_run_spec``, when set, overrides the issue count and sample order
    for a compliance-audit phase (see ``commands/audit.py:run_audit``).
    """
    # Accuracy-only runs force single-stream (1 worker / 1 connection) for
    # deterministic sample ordering. Bake it into the config here — before CPU
    # affinity, report_dir/config.yaml persistence, and RuntimeSettings — so the
    # written config.yaml matches what actually runs. The compliance gate reads
    # config.yaml and asserts single_stream; without this it would fail a valid
    # accuracy-only run whose source config declared multiple workers.
    if test_mode == TestMode.ACC:
        settings_update: dict[str, Any] = {
            "client": config.settings.client.with_updates(
                num_workers=1, max_connections=1
            )
        }
        # The compliance single_stream gate also reads
        # load_pattern.target_concurrency when it is set. Normalize it to 1 so a
        # combined config that declares concurrency > 1 does not fail single_stream
        # on an accuracy-only run whose client is already forced to one connection.
        load_pattern = config.settings.load_pattern
        if (
            load_pattern.target_concurrency is not None
            and load_pattern.target_concurrency != 1
        ):
            settings_update["load_pattern"] = load_pattern.model_copy(
                update={"target_concurrency": 1}
            )
        config = config.with_updates(
            settings=config.settings.model_copy(update=settings_update)
        )

    # CPU affinity
    affinity_plan = (
        pin_loadgen(config.settings.client.num_workers)
        if config.enable_cpu_affinity
        else None
    )

    # Report directory
    report_dir = resolve_report_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(report_dir / "config.yaml")

    # Tokenizer check (light API call, no download)
    model_name = config.model_params.name
    tokenizer_override = config.model_params.tokenizer_name
    tokenizer_name: str | None
    if tokenizer_override:
        if not _check_tokenizer_exists(tokenizer_override):
            raise SetupError(
                f"Tokenizer override '{tokenizer_override}' could not be verified. "
                "Check that the HF repo ID or local path is correct, accessible, and contains tokenizer files. "
                "See logs above for details."
            )
        tokenizer_name = tokenizer_override
    else:
        tokenizer_name = model_name if _check_tokenizer_exists(model_name) else None

    # Streaming
    logger.info(
        f"Streaming: {'enabled' if config.model_params.streaming == StreamingMode.ON else 'disabled'}"
        f" ({config.model_params.streaming.value})"
    )

    # Datasets
    dataloader, accuracy_datasets, eval_configs = _load_datasets(
        config, report_dir, test_mode
    )

    rt_settings: RuntimeSettings | None = None
    total_samples = 0
    if dataloader is not None:
        rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
        if audit_run_spec is not None:
            rt_settings = dataclass_replace(
                rt_settings,
                n_samples_to_issue=audit_run_spec.n_samples,
                sample_order=audit_run_spec.sample_order,
            )
        total_samples = rt_settings.total_samples_to_issue()

    if accuracy_datasets:
        total_samples += sum(ds.num_samples() * ds.repeats for ds in accuracy_datasets)

    collect_responses = test_mode in (TestMode.ACC, TestMode.BOTH)
    logger.info(
        f"Mode: {test_mode}, Target QPS: {config.settings.load_pattern.target_qps}, Responses: {collect_responses}"
    )
    if rt_settings is not None:
        logger.info(
            f"Min Duration: {rt_settings.min_duration_ms / 1000:.1f}s, Expected samples: {total_samples}"
        )
    else:
        logger.info(f"Accuracy-only mode, Expected samples: {total_samples}")

    return BenchmarkContext(
        config=config,
        test_mode=test_mode,
        report_dir=report_dir,
        tokenizer_name=tokenizer_name,
        dataloader=dataloader,
        rt_settings=rt_settings,
        total_samples=total_samples,
        accuracy_datasets=accuracy_datasets,
        eval_configs=eval_configs,
        affinity_plan=affinity_plan,
    )


def _build_phases(
    ctx: BenchmarkContext,
    perf_strategy: AgenticInferenceStrategy | None = None,
) -> list[PhaseConfig]:
    """Build the phase list from BenchmarkContext."""
    phases: list[PhaseConfig] = []
    drain_cfg = ctx.config.settings.drain

    if ctx.dataloader is not None and ctx.rt_settings is not None:
        warmup_cfg = ctx.config.settings.warmup
        if warmup_cfg.enabled:
            warmup_dataset: Dataset = (
                ctx.dataloader.with_salt(
                    random.Random(warmup_cfg.warmup_random_seed + 2)
                )
                if warmup_cfg.salt
                else ctx.dataloader
            )
            warmup_rt = dataclass_replace(
                ctx.rt_settings,
                min_duration_ms=0,
                max_duration_ms=None,
                n_samples_from_dataset=ctx.dataloader.num_samples(),
                n_samples_to_issue=warmup_cfg.n_requests,
                min_sample_count=1,
                rng_sched=random.Random(warmup_cfg.warmup_random_seed),
                rng_sample_index=random.Random(warmup_cfg.warmup_random_seed + 1),
                load_pattern=ctx.rt_settings.load_pattern,
            )
            phases.append(
                PhaseConfig(
                    "warmup",
                    warmup_rt,
                    warmup_dataset,
                    PhaseType.WARMUP,
                    drain_after=warmup_cfg.drain,
                    drain_timeout=drain_cfg.warmup_timeout_s,
                )
            )

        phases.append(
            PhaseConfig(
                "performance",
                ctx.rt_settings,
                ctx.dataloader,
                PhaseType.PERFORMANCE,
                strategy=perf_strategy,
                drain_timeout=drain_cfg.performance_timeout_s,
            )
        )

    # Accuracy mirrors the perf load pattern so evaluation exercises the
    # endpoint the same way it was benchmarked. AGENTIC_INFERENCE can't drive
    # the (non-agentic) accuracy datasets — create_load_strategy rejects it —
    # so it (and a missing perf pattern) falls back to MAX_THROUGHPUT.
    perf_lp = ctx.rt_settings.load_pattern if ctx.rt_settings is not None else None
    if perf_lp is None or perf_lp.type == LoadPatternType.AGENTIC_INFERENCE:
        acc_load_pattern = LoadPattern(type=LoadPatternType.MAX_THROUGHPUT)
    else:
        acc_load_pattern = perf_lp

    # Accuracy phases — use eval_cfg.dataset_name as phase name so it matches
    # what Scorer._load_sample_index_map() looks up in sample_idx_map.json
    for eval_cfg in ctx.eval_configs:
        if eval_cfg.dataset_type == DatasetType.PERFORMANCE:
            continue
        acc_ds = eval_cfg.dataset
        if isinstance(acc_ds, AgenticInferenceDataset):
            raise InputValidationError(
                f"Accuracy dataset '{eval_cfg.dataset_name}' is an "
                "AgenticInferenceDataset, which is not yet supported for "
                "accuracy evaluation."
            )
        logger.info(
            "Accuracy issuer '%s' load mode: %s",
            eval_cfg.dataset_name,
            acc_load_pattern,
        )
        rng_settings = ctx.rt_settings or RuntimeSettings.from_config(
            ctx.config, acc_ds.num_samples()
        )
        acc_settings = RuntimeSettings(
            metric_target=rng_settings.metric_target,
            reported_metrics=rng_settings.reported_metrics,
            min_duration_ms=0,
            max_duration_ms=None,
            n_samples_from_dataset=acc_ds.num_samples(),
            n_samples_to_issue=acc_ds.num_samples() * acc_ds.repeats,
            min_sample_count=acc_ds.num_samples() * acc_ds.repeats,
            rng_sched=rng_settings.rng_sched,
            rng_sample_index=rng_settings.rng_sample_index,
            load_pattern=acc_load_pattern,
        )
        phases.append(
            PhaseConfig(
                eval_cfg.dataset_name,
                acc_settings,
                acc_ds,
                PhaseType.ACCURACY,
                drain_timeout=drain_cfg.accuracy_timeout_s,
            )
        )

    return phases


class _PerfPhaseTimeout:
    """Session-stop timer that bounds the PERFORMANCE phase only.

    ``max_duration_ms`` is a safety cap on the performance phase. The timer is
    armed when the performance phase starts and cancelled as soon as any later
    phase starts, so it can never truncate a subsequent accuracy phase: a
    combined perf+accuracy run must let accuracy finish regardless of how long
    perf ran.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        max_duration_ms: int | None,
        on_timeout: Callable[[], None],
    ) -> None:
        self._loop = loop
        self._max_duration_ms = max_duration_ms
        self._on_timeout = on_timeout
        self._handle: asyncio.TimerHandle | None = None

    def on_phase_start(self, phase_type: PhaseType) -> None:
        self.cancel()
        if phase_type == PhaseType.PERFORMANCE and self._max_duration_ms is not None:
            self._handle = self._loop.call_later(
                self._max_duration_ms / 1000.0, self._on_timeout
            )

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


async def _create_issuer(
    ctx: BenchmarkContext, loop: asyncio.AbstractEventLoop
) -> tuple[HttpClientSampleIssuer, HTTPEndpointClient]:
    """Create the HTTP endpoint client + sample issuer, or raise SetupError."""
    config = ctx.config
    endpoints = config.endpoint_config.endpoints
    logger.info(f"Connecting: {endpoints}")
    try:
        api_type: APIType = config.endpoint_config.api_type
        # client.api_type is propagated from endpoint_config.api_type by
        # BenchmarkConfig._propagate_client_api_type — no override needed here.
        client_overrides: dict = {
            "endpoint_urls": [
                urljoin(e.rstrip("/") + "/", api_type.default_route())
                for e in endpoints
            ],
            "api_key": config.endpoint_config.api_key,
            "event_logs_dir": ctx.report_dir,
            "cpu_affinity": ctx.affinity_plan,
        }
        if ctx.accuracy_only:
            # Single-stream (num_workers=1, max_connections=1) is baked into
            # config in setup_benchmark so it is persisted to config.yaml;
            # no runtime override needed here.
            logger.info(
                "Accuracy-only: single-stream (1 worker, 1 connection) for "
                "deterministic ordering"
            )
        http_config = config.settings.client.with_updates(**client_overrides)
        http_client = await HTTPEndpointClient.create(http_config, loop)
        issuer = HttpClientSampleIssuer(http_client)
        return issuer, http_client
    except Exception as e:
        raise SetupError(f"Failed to connect to endpoint: {e}") from e


def _build_agentic_strategy(
    ctx: BenchmarkContext,
) -> AgenticInferenceStrategy | None:
    """Build the agentic inference strategy when the perf dataset uses it."""
    if not isinstance(ctx.dataloader, AgenticInferenceDataset):
        return None
    agentic_cfg = None
    if ctx.config.datasets:
        perf_ds_cfg = next(
            (d for d in ctx.config.datasets if d.type == DatasetType.PERFORMANCE),
            None,
        )
        if perf_ds_cfg is not None:
            agentic_cfg = perf_ds_cfg.agentic_inference
    assert ctx.dataloader.conversation_metadata is not None
    return AgenticInferenceStrategy(
        conversation_manager=ConversationManager(),
        dataset_metadata=ctx.dataloader.conversation_metadata,
        agentic_inference_config=agentic_cfg,
        target_concurrency=ctx.config.settings.load_pattern.target_concurrency,
    )


def _wire_on_sample_complete(
    collector: ResponseCollector,
    agentic_inference_strategy: AgenticInferenceStrategy | None,
    publisher: Any,
) -> Callable[[QueryResult], None]:
    """Compose the per-sample completion callback (agentic strategy + collector)."""
    if agentic_inference_strategy is None:
        return collector.on_complete_hook

    def _on_sample_complete(result: QueryResult) -> None:
        try:
            agentic_inference_strategy.on_sample_complete(result)
        except Exception:
            logger.exception(
                "agentic_inference_strategy.on_sample_complete failed (result=%s)",
                result.id,
            )
        try:
            collector.on_complete_hook(result)
        except Exception:
            logger.exception("collector.on_complete_hook failed (result=%s)", result.id)

    agentic_inference_strategy._session_on_sample_complete = _on_sample_complete
    agentic_inference_strategy._session_publisher = publisher
    return _on_sample_complete


async def _run_benchmark_async(
    ctx: BenchmarkContext,
    loop: asyncio.AbstractEventLoop,
) -> BenchmarkResult:
    """Run async benchmark session."""
    config = ctx.config
    session_id = f"cli_benchmark_{uuid.uuid4().hex[:8]}"

    # Progress bar + response collector
    pbar = tqdm(
        desc=f"{config.model_params.name} (Streaming: {ctx.enable_streaming})",
        total=ctx.total_samples,
        smoothing=0,
    )
    collector = ResponseCollector(collect_responses=ctx.collect_responses, pbar=pbar)

    # Tmpfs for high-frequency writes (event log); execute owns its lifecycle
    # (salvage + rmtree). metrics_output_dir lives on disk under the report dir so
    # the final snapshot is preserved with the rest of the run artifacts — it is
    # NOT tmpfs and is never removed here. Paths are computed here (no mkdir) so the
    # cleanup `except` can always reference tmpfs_dir; the mkdirs run inside the try
    # so a mkdir failure that already created tmpfs_dir is still salvaged/removed.
    shm = Path("/dev/shm")
    tmpfs_base = shm if shm.exists() else Path(tempfile.gettempdir())
    tmpfs_dir = tmpfs_base / f"benchmark_{session_id}"
    event_log_dir = tmpfs_dir / "events"
    metrics_output_dir = ctx.report_dir / "metrics"

    pipe = MetricsPipeline(
        config,
        tokenizer_name=ctx.tokenizer_name,
        enable_streaming=ctx.enable_streaming,
        event_log_dir=event_log_dir,
        metrics_output_dir=metrics_output_dir,
        loop=loop,
    )
    report: Report | None = None
    profiler: ProfileController

    try:
        try:
            tmpfs_dir.mkdir(parents=True, exist_ok=True)
            event_log_dir.mkdir(parents=True, exist_ok=True)
            metrics_output_dir.mkdir(parents=True, exist_ok=True)
            # Pre-derive profile URLs (inside the run scope so a bad config —
            # engine set but no endpoints — fails before the run yet still triggers
            # the tmpfs/pbar cleanup in the except/finally below).
            profiler = ProfileController(
                config.settings.profiling.engine,
                config.endpoint_config.endpoints,
                config.settings.profiling.urls,
            )
            await pipe.start()
            # start() guarantees the publisher exists; narrow it for the type checker.
            publisher = pipe.publisher
            assert publisher is not None

            try:
                issuer, http_client = await _create_issuer(ctx, loop)
            except SetupError:
                await pipe.abort()
                raise

            agentic_inference_strategy = _build_agentic_strategy(ctx)
            on_sample_complete = _wire_on_sample_complete(
                collector, agentic_inference_strategy, publisher
            )

            session = BenchmarkSession(
                issuer=issuer,
                event_publisher=publisher,
                loop=loop,
                on_sample_complete=on_sample_complete,
                session_id=session_id,
            )
            phases = _build_phases(ctx, perf_strategy=agentic_inference_strategy)

            max_duration_ms = (
                ctx.rt_settings.max_duration_ms if ctx.rt_settings is not None else None
            )
            _timeout_done = False
            session_completed_normally = False

            def _on_global_timeout() -> None:
                if not _timeout_done:
                    logger.warning(
                        "Performance phase max_duration reached (%d ms); "
                        "ending performance phase.",
                        max_duration_ms,
                    )
                    # Stop only the perf phase, not the whole session, so a combined
                    # perf+accuracy run still runs accuracy after the perf cap.
                    session.stop_current_phase()

            perf_timeout = _PerfPhaseTimeout(loop, max_duration_ms, _on_global_timeout)

            def _on_phase_start(phase: PhaseConfig) -> None:
                # _PerfPhaseTimeout arms the perf cap on PERFORMANCE and cancels it
                # when any later phase starts, so a combined perf+accuracy run can
                # never have its accuracy phase truncated by the perf cap.
                perf_timeout.on_phase_start(phase.phase_type)
                if phase.phase_type != PhaseType.PERFORMANCE:
                    return
                # Fire /start_profile sequentially before any perf request is
                # issued, so the server is armed when traffic begins.
                profiler.start()

            loop.add_signal_handler(signal.SIGINT, session.stop)
            try:
                result = await session.run(phases, on_phase_start=_on_phase_start)
                session_completed_normally = True
            except Exception as e:
                raise ExecutionError(f"Benchmark execution failed: {e}") from e
            finally:
                _timeout_done = True
                perf_timeout.cancel()
                loop.remove_signal_handler(signal.SIGINT)
                # Fire /stop_profile for URLs whose /start_profile succeeded.
                # Unifies the clean phase-end path and the abort path — both reach
                # this block.
                profiler.stop(session_completed_normally)
                logger.info("Cleaning up...")
                try:
                    if http_client:
                        await http_client.shutdown_async()
                except Exception as e:
                    logger.warning(f"Client cleanup error: {e}")
                # Graceful drain runs on both the clean-finish and session-failure
                # paths (BenchmarkSession.run publishes ENDED in its own finally, so
                # a failed run still has a terminal snapshot worth draining).
                report = await pipe.drain_and_build_report()
        finally:
            # Cleanup runs on every path. pipe.close() (ZMQ scope exit) must run
            # even if pbar.close() raises, and a failure here propagates to the
            # outer except → tmpfs salvage — matching the monolith's guarantees
            # (pbar.close was inside the ZMQ `with`, whose __exit__ always ran).
            try:
                pbar.close()
            finally:
                pipe.close()
    except BaseException:
        if tmpfs_dir.exists():
            _salvage_tmpfs(ctx.report_dir, tmpfs_dir)
            shutil.rmtree(tmpfs_dir, ignore_errors=True)
        raise

    return BenchmarkResult(
        session=result,
        collector=collector,
        report=report,
        tmpfs_dir=tmpfs_dir,
        profiling=profiler.payload(),
    )


def run_benchmark_async(ctx: BenchmarkContext) -> BenchmarkResult:
    """Run async benchmark. Sync entry point — drives the event loop."""
    loop = LoopManager().default_loop
    return loop.run_until_complete(_run_benchmark_async(ctx, loop))


def _write_scoring_artifacts(
    ctx: BenchmarkContext,
    result: SessionResult,
    tmpfs_dir: Path,
) -> None:
    """Write sample_idx_map.json and copy events.jsonl for Scorer consumption.

    events.jsonl is written by EventLoggerService to tmpfs during the benchmark.
    We copy it to report_dir (typically on disk) during finalization.
    """

    # sample_idx_map.json — {dataset_name: {uuid: sample_index}}
    sample_idx_map: dict[str, dict[str, int]] = {}
    for phase_result in result.phase_results:
        sample_idx_map[phase_result.name] = phase_result.uuid_to_index

    map_path = ctx.report_dir / "sample_idx_map.json"
    with map_path.open("wb") as f:
        f.write(msgspec.json.format(msgspec.json.encode(sample_idx_map), indent=2))
    logger.debug(f"Wrote {map_path}")

    # Copy events.jsonl from tmpfs to report_dir.
    # Tmpfs cleanup is handled by run_benchmark()'s finally block.
    _salvage_tmpfs(ctx.report_dir, tmpfs_dir)


def _salvage_tmpfs(report_dir: Path, tmpfs_dir: Path) -> None:
    """Copy all salvageable artifacts from tmpfs to report_dir.

    Called during normal finalization and on interrupt/crash to preserve logs.
    Safe to call multiple times (skips if already copied or tmpfs is gone).
    """
    if not tmpfs_dir.exists():
        return

    # events.jsonl (from EventLoggerService)
    src_events = tmpfs_dir / "events" / "events.jsonl"
    if src_events.exists():
        dst_events = report_dir / "events.jsonl"
        shutil.copy2(src_events, dst_events)
        logger.debug(f"Copied {src_events} -> {dst_events}")


def _write_report_artifacts(
    ctx: BenchmarkContext, report: Report, profiling: dict[str, Any] | None
) -> None:
    """Display the report and write result_summary.json + report.txt.

    result_summary.json is the self-complete machine-readable report (carries
    qps/tps/seeds/accuracy via Report.to_json); report.txt is the full
    human-readable dump; the console log shows the summary.
    """
    report.display(fn=lambda s: logger.info(s), summary_only=True)
    performance_dir = ctx.report_dir / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    report.to_json(save_to=performance_dir / "result_summary.json")

    report_txt = ctx.report_dir / "report.txt"
    with report_txt.open("w") as f:
        report.display(fn=lambda s: print(s, file=f))
        if profiling is not None:
            _write_profiling_section(f, profiling)
    logger.info("Report written to %s", report_txt)


def _summarize_and_log_metrics(
    ctx: BenchmarkContext,
    report: Report | None,
    result: SessionResult,
    collector: ResponseCollector,
) -> None:
    """Log the run's headline metrics, preferring Report over SessionResult."""
    if report is not None and report.duration_ns is not None:
        perf_elapsed = report.duration_ns / 1e9
        total_issued = report.n_samples_issued
        n_errors = report.n_samples_failed
        qps = report.qps or 0.0
    else:
        perf = result.perf_results[0] if result.perf_results else None
        if perf:
            perf_elapsed = (perf.end_time_ns - perf.start_time_ns) / 1e9
            total_issued = perf.issued_count
        else:
            perf_elapsed = (result.end_time_ns - result.start_time_ns) / 1e9
            total_issued = 0
        n_errors = len(collector.errors)
        qps = total_issued / perf_elapsed if perf_elapsed > 0 else 0.0

    logger.info(f"Completed in {perf_elapsed:.1f}s")
    if ctx.accuracy_only:
        acc_total = sum(ds.num_samples() * ds.repeats for ds in ctx.accuracy_datasets)
        logger.info(f"Accuracy-only: {acc_total} samples evaluated")
    else:
        logger.info(
            f"Results: {max(0, total_issued - n_errors)}/{total_issued} successful"
        )
        if qps > 0:
            logger.info(f"Estimated QPS: {qps:.1f}")

    if collector.errors:
        logger.warning(f"Errors: {len(collector.errors)}")
        for err in collector.errors[:3]:
            logger.debug(f"  {err}")
        if len(collector.errors) > 3:
            logger.debug(f"  ... +{len(collector.errors) - 3} more")


def finalize_benchmark(ctx: BenchmarkContext, bench: BenchmarkResult) -> None:
    """Score accuracy, aggregate results, write JSON."""
    result = bench.session
    collector = bench.collector
    report = bench.report

    # Sibling profiling.json — kept separate so Report stays a pure
    # snapshot-derived struct.
    if bench.profiling is not None:
        (ctx.report_dir / "profiling.json").write_text(
            json.dumps(bench.profiling, indent=2)
        )

    # Write scoring artifacts + copy event log from tmpfs to disk (scorers read
    # sample_idx_map.json + events.jsonl from here).
    _write_scoring_artifacts(ctx, result, bench.tmpfs_dir)

    # Accuracy scoring (one entry per accuracy dataset). Scoring runs before the
    # report is written so the accuracy headline can be attached, but the report
    # is written in the `finally` below so a scoring failure (e.g. lcb-service
    # unreachable, missing eval subproject, bad extras) still leaves the perf
    # run's result_summary.json / report.txt on disk instead of discarding them —
    # then the exception propagates as before.
    accuracy_scores: list[dict[str, Any]] = []
    try:
        accuracy_scores = _score_accuracy(ctx, result)
    finally:
        # Attach the per-dataset accuracy list so result_summary.json, the
        # console summary, and report.txt all carry it (stays [] on a scoring
        # failure).
        if report is not None:
            report = msgspec.structs.replace(report, accuracy=accuracy_scores)
        # Display the report + write result_summary.json / report.txt.
        if report is not None:
            _write_report_artifacts(ctx, report, bench.profiling)

    _summarize_and_log_metrics(ctx, report, result, collector)

    # Emit the accuracy results as a focused artifact under accuracy/. Written
    # after the report artifacts so a write failure here can't discard them.
    write_accuracy_results(ctx.report_dir, accuracy_scores)


def run_benchmark(
    config: BenchmarkConfig,
    test_mode: TestMode,
) -> Path:
    """Orchestrate setup → execute → finalize for the main run.

    ``test_mode`` is the single source of truth for what runs: ``ACC`` is an
    accuracy-only run (no performance phase), ``PERF`` performance-only, and
    ``BOTH`` runs performance then accuracy. The CLI ``--accuracy-only`` flag is
    a convenience alias that resolves to ``TestMode.ACC``.

    Returns the run's ``report_dir`` so the caller can locate artifacts (and, for
    a config with an ``audit:`` block, point ``run_audit`` at ``<report_dir>/audit``).
    The compliance audit is dispatched by the caller (``cli._run``), not here, so
    this module does not depend on ``commands.audit``.
    """
    logger.debug(
        "BenchmarkConfig (%s):\n%s",
        type(config).__name__,
        config.model_dump_json(indent=2, exclude_none=True),
    )
    ctx = setup_benchmark(config, test_mode)
    bench: BenchmarkResult | None = None
    try:
        bench = run_benchmark_async(ctx)
        finalize_benchmark(ctx, bench)
    except KeyboardInterrupt:
        # Salvage results (finally), then propagate to main.py -> exit 130.
        logger.warning("Benchmark interrupted by user")
        raise
    finally:
        if bench:
            if bench.tmpfs_dir.exists():
                _salvage_tmpfs(ctx.report_dir, bench.tmpfs_dir)
                shutil.rmtree(bench.tmpfs_dir, ignore_errors=True)
            logger.info(f"Partial results saved to {ctx.report_dir}")

    return ctx.report_dir
