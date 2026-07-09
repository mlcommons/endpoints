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
from typing import Any, TextIO
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urljoin

import msgspec
import msgspec.json
from huggingface_hub import model_info
from tqdm import tqdm
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
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.compliance import AuditRunSpec
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    BenchmarkConfig,
    DatasetType,
    LoadPattern,
    LoadPatternType,
    ProfilerEngine,
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
        if eval_cfg.dataset_name == "performance":
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


# (start_path, stop_path) for each supported inference engine's profiling
# protocol. Add a row when introducing a new ProfilerEngine variant.
_PROFILE_PATHS: dict[ProfilerEngine, tuple[str, str]] = {
    ProfilerEngine.VLLM: ("/start_profile", "/stop_profile"),
}


def _derive_profile_urls(
    endpoints: list[str], engine: ProfilerEngine, action: str
) -> list[str]:
    """One profile URL per endpoint, derived from the engine's HTTP protocol.

    For vLLM: strip a trailing ``/v1`` from each endpoint and append
    ``/{start,stop}_profile``. ``action`` is ``"start"`` or ``"stop"``.
    """
    if not endpoints:
        raise ValueError(
            f"profiling.engine={engine.value} but endpoint_config.endpoints "
            f"is empty; cannot derive {action} URLs"
        )
    start_path, stop_path = _PROFILE_PATHS[engine]
    path = start_path if action == "start" else stop_path
    urls: list[str] = []
    for ep in endpoints:
        base = ep.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        urls.append(f"{base.rstrip('/')}{path}")
    return urls


def _post_profile(url: str) -> dict[str, Any]:
    """POST {url} with empty body; never raises. Returns a record dict suitable
    for report.txt rendering and profiling.json serialization."""
    record: dict[str, Any] = {
        "url": url,
        "sent_at_ns": time.monotonic_ns(),
        "sent_at_iso": datetime.now().isoformat(timespec="milliseconds"),
        "status": None,
        "error": None,
    }
    req = urllib_request.Request(url, method="POST", data=b"")
    try:
        with urllib_request.urlopen(req, timeout=2) as resp:
            record["status"] = resp.status
    except urllib_error.HTTPError as e:
        record["status"] = e.code
        record["error"] = f"{e.code} {e.reason}"
    except Exception as e:  # noqa: BLE001 — profile failures must never abort a run
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def _render_profile_status(rec: dict[str, Any]) -> str:
    status = rec.get("status")
    error = rec.get("error")
    if status == 200:
        return "200 OK"
    if status == 404:
        return (
            "404 (profiling not enabled on server — pass "
            "--profiler-config.profiler=... to server)"
        )
    if error:
        return error
    if status is not None:
        return str(status)
    return "ERROR"


def _write_profiling_section(f: TextIO, profiling: dict[str, Any]) -> None:
    """Append the Profiling section to report.txt (called after report.display)."""
    starts = profiling.get("starts", [])
    stops = profiling.get("stops", [])
    f.write("\n------------------- Profiling -------------------\n")
    f.write(f"Engine: {profiling.get('engine', 'unknown')}\n")
    f.write("Start:\n")
    for rec in starts:
        f.write(
            f"  POST {rec['url']} @ {rec['sent_at_iso']} → "
            f"{_render_profile_status(rec)}\n"
        )
    if stops:
        f.write("Stop:\n")
        for rec in stops:
            suffix = (
                " (from abort handler)" if rec.get("stop_reason") == "abort" else ""
            )
            f.write(
                f"  POST {rec['url']} @ {rec['sent_at_iso']} → "
                f"{_render_profile_status(rec)}{suffix}\n"
            )
    if starts and stops:
        first_start = min(r["sent_at_ns"] for r in starts)
        last_stop = max(r["sent_at_ns"] for r in stops)
        f.write(f"Trigger span: {(last_stop - first_start) / 1e9:.2f} s\n")
    f.write(
        "\nNote: actual trace window is bounded by server-side "
        "--profiler-config.delay_iterations and "
        "--profiler-config.max_iterations.\n"
        "Trace artifact path is in server stdout.\n"
    )


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
    tmpfs_dir: Path | None = None
    try:
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
            aggregator_args.extend(
                ["--drain-timeout", str(config.settings.drain.metrics_drain_timeout_s)]
            )
            aggregator_args.extend(
                [
                    "--tokenizer-workers",
                    str(config.settings.drain.metrics_tokenizer_workers),
                ]
            )

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
                timeout=config.settings.service_ready_timeout_s,
            )

            # Create endpoint client on the shared loop
            endpoints = config.endpoint_config.endpoints
            logger.info(f"Connecting: {endpoints}")
            http_client: HTTPEndpointClient | None = None
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
            except Exception as e:
                pbar.close()
                publisher.close()
                launcher.kill_all()
                raise SetupError(f"Failed to connect to endpoint: {e}") from e

            # Build agentic inference strategy if the performance dataset uses it.
            agentic_inference_strategy: AgenticInferenceStrategy | None = None
            if isinstance(ctx.dataloader, AgenticInferenceDataset):
                agentic_cfg = None
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
                        agentic_cfg = perf_ds_cfg.agentic_inference
                assert ctx.dataloader.conversation_metadata is not None
                agentic_inference_strategy = AgenticInferenceStrategy(
                    conversation_manager=ConversationManager(),
                    dataset_metadata=ctx.dataloader.conversation_metadata,
                    agentic_inference_config=agentic_cfg,
                    target_concurrency=ctx.config.settings.load_pattern.target_concurrency,
                )

            _on_sample_complete: Callable[[QueryResult], None]
            if agentic_inference_strategy is not None:

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
                        logger.exception(
                            "collector.on_complete_hook failed (result=%s)", result.id
                        )

                agentic_inference_strategy._session_on_sample_complete = (
                    _on_sample_complete
                )
                agentic_inference_strategy._session_publisher = publisher

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

            phases = _build_phases(ctx, perf_strategy=agentic_inference_strategy)
            report: Report | None = None

            _timeout_done = False
            max_duration_ms = (
                ctx.rt_settings.max_duration_ms if ctx.rt_settings is not None else None
            )

            # Profile trigger state. Pre-derive URLs once so a bad config
            # (engine set but no endpoints) fails before the run.
            profiling_cfg = config.settings.profiling
            profile_start_urls: list[str] = []
            profile_stop_urls: list[str] = []
            profile_starts: list[dict[str, Any]] = []
            profile_stops: list[dict[str, Any]] = []
            if profiling_cfg.engine is not None:
                profile_endpoints = (
                    profiling_cfg.urls or config.endpoint_config.endpoints
                )
                profile_start_urls = _derive_profile_urls(
                    profile_endpoints, profiling_cfg.engine, "start"
                )
                profile_stop_urls = _derive_profile_urls(
                    profile_endpoints, profiling_cfg.engine, "stop"
                )
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
                # issued, so the server is armed when traffic begins. Blocks
                # the loop briefly (sub-100ms per URL); strategy task hasn't
                # been created yet so nothing is starved.
                for url in profile_start_urls:
                    rec = _post_profile(url)
                    if rec["status"] == 200:
                        logger.info("Profile start: %s -> 200 OK", url)
                    else:
                        logger.warning(
                            "Profile start: %s -> %s",
                            url,
                            rec["error"] or rec["status"],
                        )
                    profile_starts.append(rec)

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
                # Unifies the clean phase-end path and the abort path —
                # both reach this block, both fire stops.
                if profile_starts:
                    stop_reason = "phase_end" if session_completed_normally else "abort"
                    for i, start_rec in enumerate(profile_starts):
                        if start_rec["status"] != 200 or i >= len(profile_stop_urls):
                            continue
                        rec = _post_profile(profile_stop_urls[i])
                        rec["stop_reason"] = stop_reason
                        if rec["status"] == 200:
                            logger.info(
                                "Profile stop: %s -> 200 OK", profile_stop_urls[i]
                            )
                        else:
                            logger.warning(
                                "Profile stop: %s -> %s",
                                profile_stop_urls[i],
                                rec["error"] or rec["status"],
                            )
                        profile_stops.append(rec)
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
                        runtime = ctx.config.settings.runtime
                        warmup = ctx.config.settings.warmup
                        load_pattern = ctx.config.settings.load_pattern
                        report = Report.from_snapshot(
                            snap_dict,
                            seeds={
                                "scheduler_random_seed": runtime.scheduler_random_seed,
                                "dataloader_random_seed": runtime.dataloader_random_seed,
                                "warmup_random_seed": warmup.warmup_random_seed,
                            },
                            use_legacy_loadgen_qps_metrics=(
                                load_pattern.type == LoadPatternType.POISSON
                                and load_pattern.use_legacy_loadgen_qps_metrics
                            ),
                        )
                        if not report.complete:
                            logger.warning(
                                "Report is incomplete (state=%s, n_pending_tasks=%d)",
                                report.state,
                                snap_dict.get("n_pending_tasks", 0),
                            )
                        if report.legacy_loadgen_window_duration_ns is not None:
                            logger.warning(
                                "Reporting QPS/TPS with the legacy MLPerf LoadGen Server "
                                "'completed' definition (deprecated; to be removed once a "
                                "formal tail-cutting mechanism lands). Pass "
                                "--no-use-legacy-loadgen-qps-metrics for endpoints-native "
                                "metrics."
                            )
                    except Exception as e:  # noqa: BLE001 — best-effort report build.
                        logger.warning(f"Failed to build report from snapshot: {e}")

                metrics_subscriber.close()
                pbar.close()
    except BaseException:
        # tmpfs_dir may still be None if the exception hit before it was
        # created (e.g. ZMQ context setup), in which case there is nothing
        # to clean up.
        if tmpfs_dir is not None and tmpfs_dir.exists():
            _salvage_tmpfs(ctx.report_dir, tmpfs_dir)
            shutil.rmtree(tmpfs_dir, ignore_errors=True)
        raise

    profiling_payload: dict[str, Any] | None = None
    if profiling_cfg.engine is not None:
        profiling_payload = {
            "engine": profiling_cfg.engine.value,
            "starts": profile_starts,
            "stops": profile_stops,
        }

    return BenchmarkResult(
        session=result,
        collector=collector,
        report=report,
        tmpfs_dir=tmpfs_dir,
        profiling=profiling_payload,
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


def finalize_benchmark(ctx: BenchmarkContext, bench: BenchmarkResult) -> None:
    """Score accuracy, aggregate results, write JSON."""
    config = ctx.config
    result = bench.session
    collector = bench.collector
    report = bench.report

    # Display report if available (from MetricsAggregator pub/sub snapshot).
    # result_summary.json is the self-complete machine-readable report (carries
    # qps/tps/seeds via Report.to_json); report.txt is the full human-readable
    # dump (histograms + percentiles); the console log shows just the summary.
    if report is not None:
        report.display(fn=lambda s: logger.info(s), summary_only=True)
        report.to_json(save_to=ctx.report_dir / "result_summary.json")

        report_txt = ctx.report_dir / "report.txt"
        with report_txt.open("w") as f:
            report.display(fn=lambda s: print(s, file=f))
            if bench.profiling is not None:
                _write_profiling_section(f, bench.profiling)
        logger.info("Report written to %s", report_txt)

    # Sibling profiling.json — kept separate so Report stays a pure
    # snapshot-derived struct.
    if bench.profiling is not None:
        (ctx.report_dir / "profiling.json").write_text(
            json.dumps(bench.profiling, indent=2)
        )

    # Write scoring artifacts + copy event log from tmpfs to disk
    _write_scoring_artifacts(ctx, result, bench.tmpfs_dir)

    # Accuracy scoring
    accuracy_scores: dict[str, Any] = {}
    for eval_cfg in ctx.eval_configs:
        try:
            scorer_instance = eval_cfg.scorer(
                eval_cfg.dataset_name,
                eval_cfg.dataset,
                eval_cfg.report_dir,
                extractor=eval_cfg.extractor,
                ground_truth_column=eval_cfg.ground_truth_column,
                **eval_cfg.extras,
            )
        except TypeError as e:
            raise InputValidationError(
                f"Dataset '{eval_cfg.dataset_name}': invalid accuracy_config.extras "
                f"for scorer '{eval_cfg.scorer.__name__}': {e}"
            ) from e
        score, n_repeats = scorer_instance.score()
        assert eval_cfg.dataset.data is not None
        num_samples = len(eval_cfg.dataset.data)
        if eval_cfg.dataset_name == "performance":
            num_samples = sum(phase.issued_count for phase in result.perf_results)
        entry: dict[str, Any] = {
            "dataset_name": eval_cfg.dataset_name,
            "num_samples": num_samples,
            "extractor": (
                eval_cfg.extractor.__name__ if eval_cfg.extractor is not None else None
            ),
            "ground_truth_column": eval_cfg.ground_truth_column,
            "score": score,
            # False when the scorer produced only a partial headline (e.g.
            # LegacyMLPerfDeepSeekR1Scorer when the lcb-service container was unreachable),
            # so a partial number is never mistaken for a complete one.
            "complete": scorer_instance.complete,
        }
        breakdown = scorer_instance.score_breakdown()
        if breakdown is not None:
            entry["breakdown"] = breakdown
        accuracy_scores[eval_cfg.dataset_name] = entry
        logger.info(
            f"Score for {eval_cfg.dataset_name}: {score} "
            f"({n_repeats} repeats, complete={scorer_instance.complete})"
        )

    # Report metrics: prefer Report from MetricsSnapshot, fall back to SessionResult
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

    # Write results JSON
    try:
        results: dict[str, Any] = {
            "config": {
                "endpoint": config.endpoint_config.endpoints,
                "mode": ctx.test_mode,
                "accuracy_only": ctx.accuracy_only,
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
