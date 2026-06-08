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
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import signal
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import msgspec
import msgspec.json
from huggingface_hub import model_info
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from inference_endpoint.async_utils.event_publisher import EventPublisherService
from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.services.launcher import (
    ServiceConfig,
    ServiceLauncher,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    snapshot_to_dict,
)
from inference_endpoint.async_utils.services.metrics_aggregator.subscriber import (
    MetricsSnapshotSubscriber,
)
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    _normalize_tool_calls_for_template,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
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
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset
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
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.multi_turn_strategy import MultiTurnStrategy
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    PhaseConfig,
    PhaseResult,
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


@dataclass
class AccuracyConfiguration:
    scorer: type[Scorer]
    extractor: type[Extractor] | None
    dataset_name: str
    dataset: Dataset
    report_dir: Path
    ground_truth_column: str | None
    num_repeats: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkContext:
    """All state needed to run a benchmark, created by setup_benchmark.

    Derived values are computed as properties from config, not stored redundantly.
    """

    config: BenchmarkConfig
    test_mode: TestMode
    report_dir: Path
    tokenizer_name: str | None
    dataloader: Dataset
    rt_settings: RuntimeSettings
    total_samples: int
    accuracy_datasets: list[Dataset] = field(default_factory=list)
    eval_configs: list[AccuracyConfiguration] = field(default_factory=list)
    affinity_plan: AffinityPlan | None = None

    @property
    def collect_responses(self) -> bool:
        return self.test_mode in (TestMode.ACC, TestMode.BOTH)

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


def _resolve_tokenizer_name(model_name: str) -> str | None:
    """Resolve a tokenizer path/repo for ISL/OSL/TPOT metrics.

    ``model_params.name`` may be a container-only path (e.g. ``/models/...``) that
    does not exist on the host running the benchmark client. Prefer
    ``TOKENIZER_MODEL_PATH`` when set, then probe ``model_name`` and common fallbacks.
    """
    candidates: list[str] = []
    env_path = os.environ.get("TOKENIZER_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(model_name)
    if model_name.startswith("/models/"):
        hf_id = model_name.removeprefix("/models/").lstrip("/")
        if hf_id:
            candidates.append(hf_id)
        default_host = Path(f"/data/workloads-inference/models/{hf_id}")
        if default_host.is_dir():
            candidates.append(str(default_host))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _check_tokenizer_exists(candidate):
            return candidate
    return None


def _load_datasets(
    config: BenchmarkConfig, report_dir: Path
) -> tuple[Dataset, list[Dataset], list[AccuracyConfiguration]]:
    """Load performance and accuracy datasets. Returns (perf_loader, acc_datasets, eval_configs)."""
    # Get dataset - from CLI or from config
    # TODO: Dataset Logic is not yet fully implemented
    accuracy_cfgs = [ds for ds in config.datasets if ds.type == DatasetType.ACCURACY]
    performance_cfgs = [
        ds for ds in config.datasets if ds.type == DatasetType.PERFORMANCE
    ]

    if not performance_cfgs:
        raise InputValidationError("At least one performance dataset required")

    accuracy_datasets: list[Dataset] = []
    eval_configs: list[AccuracyConfiguration] = []

    # Pack the evaluation parameters for each accuracy dataset
    for acc_cfg in accuracy_cfgs:
        if (
            acc_cfg.accuracy_config is None
            or acc_cfg.accuracy_config.eval_method is None
        ):
            raise InputValidationError(
                f"Dataset '{acc_cfg.name}' requires accuracy_config with eval_method"
            )

        scorer_cls = Scorer.get(acc_cfg.accuracy_config.eval_method)
        extractor_name = acc_cfg.accuracy_config.extractor
        if extractor_name is None:
            if scorer_cls.REQUIRES_EXTRACTOR:
                raise InputValidationError(
                    f"Dataset '{acc_cfg.name}' uses scorer "
                    f"'{acc_cfg.accuracy_config.eval_method}' which requires an extractor"
                )
            extractor_cls: type[Extractor] | None = None
        else:
            extractor_cls = Extractor.get(extractor_name)

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
            )
        )
        ds.load(
            api_type=config.endpoint_config.api_type, model_params=config.model_params
        )
        logger.info(f"Loaded {ds} - {ds.num_samples()} samples")

    if not accuracy_cfgs:
        logger.info("No accuracy datasets provided")
    if len(performance_cfgs) > 1:
        raise InputValidationError("Multiple performance datasets not supported")

    try:
        dataloader = DataLoaderFactory.create_loader(performance_cfgs[0])
        dataloader.load(
            api_type=config.endpoint_config.api_type, model_params=config.model_params
        )
        logger.info(f"Loaded {dataloader.num_samples()} samples")
    except FileNotFoundError as e:
        raise InputValidationError(
            f"Dataset file not found: {performance_cfgs[0].path}"
        ) from e
    except Exception as e:
        raise SetupError(f"Failed to load dataset: {e}") from e

    return dataloader, accuracy_datasets, eval_configs


def _precompute_isl_for_multi_turn(
    dataloader: MultiTurnDataset, tokenizer_name: str
) -> None:
    """Tokenize pre-built message lists and store token counts in each sample.

    Runs apply_chat_template once per client turn so the hot-path IslTrigger
    sync path (len(token_ids)) is used instead of on-the-fly text tokenization.
    Only affects dataset-history turns; live-history turns override 'messages'
    at runtime so the stored input_tokens are stale (acceptable approximation).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        logger.exception(
            "ISL pre-computation: failed to load tokenizer %s; "
            "falling back to text-tokenization at runtime",
            tokenizer_name,
        )
        return
    skipped = 0
    first_failure_logged = False
    for sample in dataloader.data or []:
        messages = sample.get("messages")
        if not messages:
            continue
        try:
            normalized_messages = []
            for msg in messages:
                if msg.get("tool_calls"):
                    msg = {
                        **msg,
                        "tool_calls": _normalize_tool_calls_for_template(
                            msg["tool_calls"]
                        ),
                    }
                normalized_messages.append(msg)
            tools = sample.get("tools")
            raw = tokenizer.apply_chat_template(
                normalized_messages,
                tools=tools if tools else None,
                tokenize=True,
                add_generation_prompt=True,
            )
            # Some tokenizers (e.g. Qwen3 fast tokenizer) return BatchEncoding
            # instead of a plain list; extract .input_ids in that case.
            token_ids: list[int] = raw.input_ids if hasattr(raw, "input_ids") else raw
            sample["input_tokens"] = token_ids
        except Exception:
            if not first_failure_logged:
                logger.exception(
                    "ISL pre-computation: apply_chat_template failed (first failure shown)"
                )
                first_failure_logged = True
            skipped += 1
    if skipped:
        logger.warning(
            "ISL pre-computation: %d turn(s) skipped (apply_chat_template failed)",
            skipped,
        )
    total_with_messages = len([s for s in (dataloader.data or []) if s.get("messages")])
    if total_with_messages > 0 and skipped == total_with_messages:
        logger.warning(
            "ISL precomputation: all %d turn(s) failed apply_chat_template; "
            "ISL metrics will use text-tokenization fallback. "
            "Check tokenizer/template compatibility.",
            total_with_messages,
        )


def setup_benchmark(config: BenchmarkConfig, test_mode: TestMode) -> BenchmarkContext:
    """Load tokenizer, dataset, create scheduler, setup report dir."""
    # CPU affinity
    affinity_plan = (
        pin_loadgen(config.settings.client.num_workers)
        if config.enable_cpu_affinity
        else None
    )

    # Report directory
    report_dir = (
        Path(config.report_dir) if config.report_dir else _default_report_path()
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(report_dir / "config.yaml")

    # Tokenizer for metrics aggregator (ISL/OSL/TPOT). API model name may differ.
    model_name = config.model_params.name
    tokenizer_name = _resolve_tokenizer_name(model_name)

    # Streaming
    logger.info(
        f"Streaming: {'enabled' if config.model_params.streaming == StreamingMode.ON else 'disabled'}"
        f" ({config.model_params.streaming.value})"
    )

    # Datasets
    dataloader, accuracy_datasets, eval_configs = _load_datasets(config, report_dir)

    if isinstance(dataloader, MultiTurnDataset) and tokenizer_name is not None:
        logger.info("Pre-computing ISL token counts for multi-turn dataset…")
        _precompute_isl_for_multi_turn(dataloader, tokenizer_name)

    # Setup runtime settings using factory method
    rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())

    # Calculate and display expected sample count
    total_samples = rt_settings.total_samples_to_issue()
    if accuracy_datasets:
        total_samples += sum(ds.num_samples() * ds.repeats for ds in accuracy_datasets)

    collect_responses = test_mode in (TestMode.ACC, TestMode.BOTH)
    logger.info(
        f"Mode: {test_mode}, Target QPS: {config.settings.load_pattern.target_qps}, Responses: {collect_responses}"
    )
    logger.info(
        f"Min Duration: {rt_settings.min_duration_ms / 1000:.1f}s, Expected samples: {total_samples}"
    )

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
    perf_strategy: MultiTurnStrategy | None = None,
) -> list[PhaseConfig]:
    """Build the phase list from BenchmarkContext."""
    phases: list[PhaseConfig] = []

    # Warmup phase (optional, before performance)
    warmup_cfg = ctx.config.settings.warmup
    if warmup_cfg.enabled:
        warmup_dataset: Dataset = (
            ctx.dataloader.with_salt(random.Random(warmup_cfg.warmup_random_seed + 2))
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
            )
        )

    # Performance phase
    phases.append(
        PhaseConfig(
            "performance",
            ctx.rt_settings,
            ctx.dataloader,
            PhaseType.PERFORMANCE,
            strategy=perf_strategy,
        )
    )

    # Accuracy phases — use eval_cfg.dataset_name as phase name so it matches
    # what Scorer._load_sample_index_map() looks up in sample_idx_map.json
    for eval_cfg in ctx.eval_configs:
        acc_ds = eval_cfg.dataset
        if isinstance(acc_ds, MultiTurnDataset):
            raise InputValidationError(
                f"Accuracy dataset '{eval_cfg.dataset_name}' is a MultiTurnDataset, "
                "which is not yet supported for accuracy evaluation."
            )
        # Accuracy phases use bounded concurrency (not MAX_THROUGHPUT burst) so long
        # runs do not queue thousands of requests on the inference server at once.
        acc_concurrency = max(1, ctx.config.settings.client.num_workers)
        acc_load_pattern: LoadPattern | None = LoadPattern(
            type=LoadPatternType.CONCURRENCY,
            target_concurrency=acc_concurrency,
        )
        acc_settings = RuntimeSettings(
            metric_target=ctx.rt_settings.metric_target,
            reported_metrics=ctx.rt_settings.reported_metrics,
            min_duration_ms=0,
            max_duration_ms=None,
            n_samples_from_dataset=acc_ds.num_samples(),
            n_samples_to_issue=acc_ds.num_samples() * acc_ds.repeats,
            min_sample_count=acc_ds.num_samples() * acc_ds.repeats,
            rng_sched=ctx.rt_settings.rng_sched,
            rng_sample_index=ctx.rt_settings.rng_sample_index,
            load_pattern=acc_load_pattern,
        )
        phases.append(
            PhaseConfig(eval_cfg.dataset_name, acc_settings, acc_ds, PhaseType.ACCURACY)
        )

    return phases


def _load_final_snapshot_from_disk(path: Path) -> dict[str, Any] | None:
    """Read the persisted ``final_snapshot.json`` written by the aggregator.

    Returns the snapshot in its dict form — the same shape produced by
    ``snapshot_to_dict`` and consumed by ``Report.from_snapshot``. No
    intermediate Struct decode (see ``Report.from_snapshot`` docstring
    for why the dict shape is the consumer contract).

    Returns ``None`` if the file is missing (the aggregator was killed
    by an uncatchable signal — SIGKILL, OOM-kill — before its handler
    could write) or unreadable.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except Exception as e:  # noqa: BLE001 — best-effort.
        logger.warning("Failed to read final snapshot %s: %s", path, e)
        return None


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

    # ZMQ context for event publishing + service launcher
    with ManagedZMQContext.scoped(io_threads=2) as zmq_ctx:
        # Event publisher
        publisher = EventPublisherService(zmq_ctx)
        pub_socket_name = publisher.socket_name

        # Tmpfs for high-frequency writes (event log).
        shm = Path("/dev/shm")
        use_shm = shm.exists()
        tmpfs_base = shm if use_shm else Path(tempfile.gettempdir())
        tmpfs_dir = tmpfs_base / f"benchmark_{session_id}"
        tmpfs_dir.mkdir(parents=True, exist_ok=True)

        event_log_dir = tmpfs_dir / "events"
        event_log_dir.mkdir(parents=True, exist_ok=True)

        # Metrics-snapshot output (disk fallback for the final snapshot).
        # Lives under the report dir so it's preserved with the rest of
        # the run artifacts.
        metrics_output_dir = ctx.report_dir / "metrics"
        metrics_output_dir.mkdir(parents=True, exist_ok=True)

        metrics_socket_name = f"metrics_pub_{uuid.uuid4().hex[:8]}"

        # Connect the metrics-snapshot subscriber BEFORE launching the
        # aggregator subprocess that binds the matching PUB socket. ZMQ
        # tolerates connect-before-bind on IPC (the connect resolves once
        # the binder appears), and starting the SUB reader early gives
        # the subscription handshake time to complete during the
        # ~1-2 second subprocess-launch window. This eliminates the
        # slow-joiner risk of dropping early live ticks (or the worst
        # case: missing COMPLETE if the SUB handshake never warms up).
        if zmq_ctx.socket_dir is None:
            raise RuntimeError("ZMQ socket_dir must be set after publisher bind")
        metrics_subscriber = MetricsSnapshotSubscriber(
            metrics_socket_name, zmq_ctx, loop
        )
        metrics_subscriber.start()

        # Launch service subprocesses
        launcher = ServiceLauncher(zmq_ctx)
        aggregator_args: list[str] = [
            "--socket-dir",
            zmq_ctx.socket_dir,
            "--socket-name",
            pub_socket_name,
            "--metrics-socket",
            metrics_socket_name,
            "--metrics-output-dir",
            str(metrics_output_dir),
        ]
        if ctx.enable_streaming:
            aggregator_args.append("--streaming")
        if ctx.tokenizer_name is not None:
            aggregator_args.extend(["--tokenizer", ctx.tokenizer_name])

        # EventLoggerService writes events.jsonl to tmpfs (high-frequency writes)
        event_logger_args: list[str] = [
            "--log-dir",
            str(event_log_dir),
            "--socket-dir",
            zmq_ctx.socket_dir,
            "--socket-name",
            pub_socket_name,
            "--writers",
            "jsonl",
        ]

        await launcher.launch(
            [
                ServiceConfig(
                    module="inference_endpoint.async_utils.services.metrics_aggregator",
                    args=aggregator_args,
                ),
                ServiceConfig(
                    module="inference_endpoint.async_utils.services.event_logger",
                    args=event_logger_args,
                ),
            ],
            timeout=30.0,
        )

        # Create endpoint client on the shared loop
        endpoints = config.endpoint_config.endpoints
        logger.info(f"Connecting: {endpoints}")
        http_client: HTTPEndpointClient | None = None
        try:
            api_type: APIType = config.endpoint_config.api_type
            # client.api_type is propagated from endpoint_config.api_type by
            # BenchmarkConfig._propagate_client_api_type — no override needed here.
            http_config = config.settings.client.with_updates(
                endpoint_urls=[
                    urljoin(e.rstrip("/") + "/", api_type.default_route())
                    for e in endpoints
                ],
                api_key=config.endpoint_config.api_key,
                event_logs_dir=ctx.report_dir,
                cpu_affinity=ctx.affinity_plan,
            )
            http_client = await HTTPEndpointClient.create(http_config, loop)
            issuer = HttpClientSampleIssuer(http_client)
        except Exception as e:
            pbar.close()
            publisher.close()
            launcher.kill_all()
            raise SetupError(f"Failed to connect to endpoint: {e}") from e

        # Build multi-turn strategy if the performance dataset is a MultiTurnDataset.
        multi_turn_strategy: MultiTurnStrategy | None = None
        if isinstance(ctx.dataloader, MultiTurnDataset):
            mt_cfg = None
            if ctx.config.datasets:
                perf_ds_cfg = next(
                    (
                        d
                        for d in ctx.config.datasets
                        if d.type == DatasetType.PERFORMANCE
                    ),
                    None,
                )
                if perf_ds_cfg is not None:
                    mt_cfg = perf_ds_cfg.multi_turn
            assert ctx.dataloader.conversation_metadata is not None
            multi_turn_strategy = MultiTurnStrategy(
                conversation_manager=ConversationManager(),
                dataset_metadata=ctx.dataloader.conversation_metadata,
                multi_turn_config=mt_cfg,
                target_concurrency=ctx.config.settings.load_pattern.target_concurrency,
            )

        _on_sample_complete: Callable[[QueryResult], None]
        if multi_turn_strategy is not None:

            def _on_sample_complete(result: QueryResult) -> None:
                try:
                    multi_turn_strategy.on_sample_complete(result)
                except Exception:
                    logger.exception(
                        "multi_turn_strategy.on_sample_complete failed (result=%s)",
                        result.id,
                    )
                try:
                    collector.on_complete_hook(result)
                except Exception:
                    logger.exception(
                        "collector.on_complete_hook failed (result=%s)", result.id
                    )

            multi_turn_strategy._session_on_sample_complete = _on_sample_complete
            multi_turn_strategy._session_publisher = publisher

        else:
            _on_sample_complete = collector.on_complete_hook

        # Create session
        session = BenchmarkSession(
            issuer=issuer,
            event_publisher=publisher,
            loop=loop,
            on_sample_complete=_on_sample_complete,
            session_id=session_id,
        )

        phases = _build_phases(ctx, perf_strategy=multi_turn_strategy)
        report: Report | None = None

        # Timer starts when the performance phase begins (after warmup drains),
        # so max_duration_ms applies only to the perf phase, not warmup.
        global_timeout_handle = None
        _timeout_done = False
        max_duration_ms = ctx.rt_settings.max_duration_ms

        def _on_global_timeout() -> None:
            if not _timeout_done:
                logger.warning(
                    "Performance phase duration limit reached (%d ms); stopping performance phase.",
                    max_duration_ms,
                )
                session.cancel_current_strategy()

        completed_accuracy_phases: list[PhaseResult] = []

        def _on_phase_start(phase: PhaseConfig) -> None:
            nonlocal global_timeout_handle
            if (
                phase.phase_type == PhaseType.PERFORMANCE
                and max_duration_ms is not None
            ):
                global_timeout_handle = loop.call_later(
                    max_duration_ms / 1000.0, _on_global_timeout
                )

        def _on_phase_complete(
            phase: PhaseConfig, phase_result: PhaseResult | None
        ) -> None:
            if phase.phase_type != PhaseType.ACCURACY or phase_result is None:
                return
            completed_accuracy_phases.append(phase_result)
            eval_cfg = next(
                (e for e in ctx.eval_configs if e.dataset_name == phase.name),
                None,
            )
            if eval_cfg is None:
                return
            _score_accuracy_phase_incremental(
                ctx, eval_cfg, phase_result, completed_accuracy_phases, tmpfs_dir
            )

        loop.add_signal_handler(signal.SIGINT, session.stop)
        try:
            result = await session.run(
                phases,
                on_phase_start=_on_phase_start,
                on_phase_complete=_on_phase_complete,
            )
        except Exception as e:
            raise ExecutionError(f"Benchmark execution failed: {e}") from e
        finally:
            _timeout_done = True
            if global_timeout_handle is not None:
                global_timeout_handle.cancel()
            loop.remove_signal_handler(signal.SIGINT)
            logger.info("Cleaning up...")
            try:
                if http_client:
                    await http_client.shutdown_async()
            except Exception as e:
                logger.warning(f"Client cleanup error: {e}")
            logger.info(
                "Closing publisher (buffer=%d, pending=%d)...",
                publisher.buffered_count,
                publisher.pending_count,
            )
            publisher.close()
            logger.info("Waiting for services to finish processing...")
            await asyncio.to_thread(launcher.wait_for_exit, None)

            # Source the snapshot dict for Report:
            # 1. Preferred: the JSON file the aggregator atomically wrote
            #    in publish_final (ENDED-driven or signal-handler-driven).
            # 2. Fallback: convert the last live snapshot from pub/sub to
            #    its dict form. Only reached when the aggregator was killed
            #    by an uncatchable signal (SIGKILL / OOM) before its
            #    handler could write. Report will be marked incomplete
            #    because state will be LIVE / DRAINING, not "complete".
            snap_dict: dict[str, Any] | None = _load_final_snapshot_from_disk(
                metrics_output_dir / "final_snapshot.json"
            )
            if snap_dict is not None:
                logger.info("Built report from final_snapshot.json")
            elif metrics_subscriber.latest is not None:
                snap_dict = snapshot_to_dict(metrics_subscriber.latest)
                logger.warning(
                    "No final_snapshot.json on disk; falling back to last "
                    "pub/sub snapshot (state may or may not be terminal)"
                )
            else:
                logger.error("No metrics snapshot available; cannot build report")

            if snap_dict is not None:
                try:
                    report = Report.from_snapshot(snap_dict)
                    if not report.complete:
                        logger.warning(
                            "Report is incomplete (state=%s, n_pending_tasks=%d)",
                            report.state,
                            snap_dict.get("n_pending_tasks", 0),
                        )
                except Exception as e:  # noqa: BLE001 — best-effort report build.
                    logger.warning(f"Failed to build report from snapshot: {e}")

            metrics_subscriber.close()
            pbar.close()

    return BenchmarkResult(
        session=result,
        collector=collector,
        report=report,
        tmpfs_dir=tmpfs_dir,
    )


def run_benchmark_async(ctx: BenchmarkContext) -> BenchmarkResult:
    """Run async benchmark. Sync entry point — drives the event loop."""
    loop = LoopManager().default_loop
    return loop.run_until_complete(_run_benchmark_async(ctx, loop))


def _score_eval_cfg(eval_cfg: AccuracyConfiguration) -> dict[str, Any]:
    """Score one accuracy dataset from events.jsonl + sample_idx_map.json."""
    scorer_instance = eval_cfg.scorer(
        eval_cfg.dataset_name,
        eval_cfg.dataset,
        eval_cfg.report_dir,
        extractor=eval_cfg.extractor,
        ground_truth_column=eval_cfg.ground_truth_column,
        **eval_cfg.extras,
    )
    score, n_repeats = scorer_instance.score()
    assert eval_cfg.dataset.data is not None
    return {
        "dataset_name": eval_cfg.dataset_name,
        "num_samples": len(eval_cfg.dataset.data),
        "extractor": (
            eval_cfg.extractor.__name__ if eval_cfg.extractor is not None else None
        ),
        "ground_truth_column": eval_cfg.ground_truth_column,
        "score": score,
        "n_repeats": n_repeats,
    }


def _write_accuracy_sample_idx_map(
    report_dir: Path, accuracy_phase_results: list[PhaseResult]
) -> None:
    sample_idx_map = {pr.name: pr.uuid_to_index for pr in accuracy_phase_results}
    map_path = report_dir / "sample_idx_map.json"
    with map_path.open("wb") as f:
        f.write(msgspec.json.format(msgspec.json.encode(sample_idx_map), indent=2))


def _complete_uuids_in_event_log(events_path: Path) -> set[str]:
    """Return sample UUIDs with a COMPLETE record in the event log."""
    if not events_path.exists():
        return set()
    decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
    complete: set[str] = set()
    with events_path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = decoder.decode(stripped)
            if (
                record.event_type == SampleEventType.COMPLETE
                and record.sample_uuid
            ):
                complete.add(record.sample_uuid)
    return complete


def _wait_for_phase_event_log(
    events_path: Path,
    phase_result: PhaseResult,
    timeout_s: float = 60.0,
) -> bool:
    """Wait until the event logger has flushed COMPLETE records for a phase."""
    expected_uuids = set(phase_result.uuid_to_index.keys())
    if not expected_uuids:
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready = _complete_uuids_in_event_log(events_path) & expected_uuids
        if len(ready) >= phase_result.issued_count:
            return True
        time.sleep(0.25)
    ready = _complete_uuids_in_event_log(events_path) & expected_uuids
    logger.warning(
        "Timed out waiting for phase %s events (%d/%d COMPLETE in log)",
        phase_result.name,
        len(ready),
        phase_result.issued_count,
    )
    return False


def _score_accuracy_phase_incremental(
    ctx: BenchmarkContext,
    eval_cfg: AccuracyConfiguration,
    phase_result: PhaseResult,
    accuracy_phase_results: list[PhaseResult],
    tmpfs_dir: Path,
) -> None:
    """Persist events and score one accuracy phase before later phases run."""
    _write_accuracy_sample_idx_map(ctx.report_dir, accuracy_phase_results)
    events_path = tmpfs_dir / "events" / "events.jsonl"
    expected_repeats = eval_cfg.num_repeats

    partial_path = ctx.report_dir / "accuracy_scores_partial.json"
    partial_scores: dict[str, Any] = {}
    if partial_path.exists():
        partial_scores = json.loads(partial_path.read_text())

    deadline = time.monotonic() + 120.0
    last_error: Exception | None = None
    entry: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if not _wait_for_phase_event_log(
            events_path, phase_result, timeout_s=30.0
        ):
            last_error = TimeoutError(
                f"Timed out waiting for {phase_result.issued_count} COMPLETE events"
            )
            time.sleep(0.5)
            continue
        _salvage_tmpfs(ctx.report_dir, tmpfs_dir)
        try:
            candidate = _score_eval_cfg(eval_cfg)
        except (FileNotFoundError, KeyError) as e:
            last_error = e
            time.sleep(0.5)
            continue
        except Exception as e:
            last_error = e
            if "events.jsonl" in str(e).lower() or "sample_uuid" in str(e):
                time.sleep(0.5)
                continue
            logger.error(
                "Incremental accuracy scoring failed for %s: %s",
                eval_cfg.dataset_name,
                e,
            )
            return
        if candidate["n_repeats"] != expected_repeats:
            last_error = RuntimeError(
                f"Expected {expected_repeats} repeats, got {candidate['n_repeats']}"
            )
            time.sleep(0.5)
            continue
        entry = candidate
        break

    if entry is None:
        logger.error(
            "Incremental accuracy scoring failed for %s: %s",
            eval_cfg.dataset_name,
            last_error,
        )
        return

    partial_scores[eval_cfg.dataset_name] = entry
    partial_path.write_text(json.dumps(partial_scores, indent=2))
    logger.info(
        "Incremental score for %s: %s (%s repeats) -> %s",
        eval_cfg.dataset_name,
        entry["score"],
        entry["n_repeats"],
        partial_path,
    )


def _write_scoring_artifacts(
    ctx: BenchmarkContext,
    result: SessionResult,
    tmpfs_dir: Path,
) -> None:
    """Write sample_idx_map.json and copy events.jsonl for Scorer consumption.

    events.jsonl is written by EventLoggerService to tmpfs during the benchmark.
    We copy it to report_dir (typically on disk) during finalization.
    """

    accuracy_phase_results = [
        pr for pr in result.phase_results if pr.phase_type == PhaseType.ACCURACY
    ]
    _write_accuracy_sample_idx_map(ctx.report_dir, accuracy_phase_results)
    logger.debug("Wrote %s", ctx.report_dir / "sample_idx_map.json")

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


def finalize_benchmark(ctx: BenchmarkContext, bench: BenchmarkResult) -> None:
    """Score accuracy, aggregate results, write JSON."""
    config = ctx.config
    result = bench.session
    collector = bench.collector
    report = bench.report

    # Display report if available (from MetricsAggregator pub/sub snapshot)
    if report is not None:
        report.display(fn=lambda s: logger.info(s), summary_only=True)
        report.to_json(save_to=ctx.report_dir / "result_summary.json")

        # Write human-readable report.txt
        report_txt = ctx.report_dir / "report.txt"
        with report_txt.open("w") as f:
            report.display(fn=lambda s: print(s, file=f))
        logger.info(f"Report written to {report_txt}")

    # Write scoring artifacts + copy event log from tmpfs to disk
    _write_scoring_artifacts(ctx, result, bench.tmpfs_dir)

    # Accuracy scoring
    accuracy_scores: dict[str, Any] = {}
    for eval_cfg in ctx.eval_configs:
        try:
            accuracy_scores[eval_cfg.dataset_name] = _score_eval_cfg(eval_cfg)
        except Exception as e:
            logger.error("Accuracy scoring failed for %s: %s", eval_cfg.dataset_name, e)
            continue
        entry = accuracy_scores[eval_cfg.dataset_name]
        logger.info(
            "Score for %s: %s (%s repeats)",
            eval_cfg.dataset_name,
            entry["score"],
            entry["n_repeats"],
        )

    # Report metrics: prefer Report from MetricsSnapshot, fall back to SessionResult
    if report is not None and report.duration_ns is not None:
        perf_elapsed = report.duration_ns / 1e9
        total_issued = report.n_samples_issued
        n_errors = report.n_samples_failed
        qps = report.qps() or 0.0
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
    logger.info(f"Results: {max(0, total_issued - n_errors)}/{total_issued} successful")
    if qps > 0:
        logger.info(f"Estimated QPS: {qps:.1f}")

    if collector.errors:
        logger.warning(f"Errors: {len(collector.errors)}")
        for err in collector.errors[:3]:
            logger.debug(f"  {err}")
        if len(collector.errors) > 3:
            logger.debug(f"  ... +{len(collector.errors) - 3} more")

    # Write results JSON
    try:
        results: dict[str, Any] = {
            "config": {
                "endpoint": config.endpoint_config.endpoints,
                "mode": ctx.test_mode,
                "target_qps": config.settings.load_pattern.target_qps,
            },
            "results": {
                "total": total_issued,
                "successful": max(0, total_issued - n_errors),
                "failed": n_errors,
                "elapsed_time": perf_elapsed,
                "qps": qps,
            },
        }
        if accuracy_scores:
            results["accuracy_scores"] = accuracy_scores
        if ctx.collect_responses:
            results["responses"] = collector.responses
        if collector.errors:
            results["errors"] = collector.errors

        results_path = ctx.report_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {results_path}")
    except Exception as e:
        logger.error(f"Save failed: {e}")


def run_benchmark(config: BenchmarkConfig, test_mode: TestMode) -> None:
    """Orchestrate setup → execute → finalize."""
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
        logger.warning("Benchmark interrupted by user")
    finally:
        if bench:
            if bench.tmpfs_dir.exists():
                _salvage_tmpfs(ctx.report_dir, bench.tmpfs_dir)
                shutil.rmtree(bench.tmpfs_dir, ignore_errors=True)
            logger.info(f"Partial results saved to {ctx.report_dir}")
