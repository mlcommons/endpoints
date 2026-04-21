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
import platform
import shutil
import signal
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

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
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricCounterKey,
)
from inference_endpoint.async_utils.services.metrics_aggregator.kv_store import (
    BasicKVStoreReader,
)
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricSeriesKey,
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
from inference_endpoint.core.types import QueryResult
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
    metrics_dir: Path | None = None


@dataclass
class AccuracyConfiguration:
    scorer: type[Scorer]
    extractor: type[Extractor]
    dataset_name: str
    dataset: Dataset
    report_dir: Path
    ground_truth_column: str | None
    num_repeats: int


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
    """Check if a HuggingFace tokenizer exists for the model (API only, no download).

    Returns True if the model repo exists and has tokenizer files, False otherwise.
    This function is a probe — it never loads or downloads the tokenizer itself.
    Downstream consumers that need tokenization (e.g. the MetricsAggregator
    subprocess for ISL/OSL/TPOT, Harmony transforms for prompt preprocessing,
    and any future plugin with its own tokenization need) each load their own
    instance as required.
    """
    try:
        info = model_info(model_name)
        # Check for tokenizer files in the repo
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
            or acc_cfg.accuracy_config.extractor is None
        ):
            raise InputValidationError(
                f"Dataset '{acc_cfg.name}' requires accuracy_config with eval_method and extractor"
            )

        ds = DataLoaderFactory.create_loader(
            acc_cfg, num_repeats=acc_cfg.accuracy_config.num_repeats
        )
        accuracy_datasets.append(ds)
        # TODO add tests and defaults
        eval_configs.append(
            AccuracyConfiguration(
                Scorer.get(acc_cfg.accuracy_config.eval_method),
                Extractor.get(acc_cfg.accuracy_config.extractor),
                acc_cfg.name,
                ds,
                report_dir,
                acc_cfg.accuracy_config.ground_truth,
                acc_cfg.accuracy_config.num_repeats,
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

    # Tokenizer check (light API call, no download)
    model_name = config.model_params.name
    tokenizer_name = model_name if _check_tokenizer_exists(model_name) else None

    # Streaming
    logger.info(
        f"Streaming: {'enabled' if config.model_params.streaming == StreamingMode.ON else 'disabled'}"
        f" ({config.model_params.streaming.value})"
    )

    # Datasets
    dataloader, accuracy_datasets, eval_configs = _load_datasets(config, report_dir)

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


def _build_phases(ctx: BenchmarkContext) -> list[PhaseConfig]:
    """Build the phase list from BenchmarkContext."""
    phases: list[PhaseConfig] = []

    # Performance phase
    phases.append(
        PhaseConfig(
            "performance", ctx.rt_settings, ctx.dataloader, PhaseType.PERFORMANCE
        )
    )

    # Accuracy phases — use eval_cfg.dataset_name as phase name so it matches
    # what Scorer._load_sample_index_map() looks up in sample_idx_map.json
    for eval_cfg in ctx.eval_configs:
        acc_ds = eval_cfg.dataset
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
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        phases.append(
            PhaseConfig(eval_cfg.dataset_name, acc_settings, acc_ds, PhaseType.ACCURACY)
        )

    return phases


def _setup_kv_reader(
    metrics_dir: Path,
    streaming: bool,
) -> BasicKVStoreReader:
    """Create a KVStoreReader pre-registered with all metric keys."""
    reader = BasicKVStoreReader(metrics_dir)
    for counter_key in MetricCounterKey:
        reader.register_key(counter_key.value, "counter")
    _STREAMING_ONLY = {
        MetricSeriesKey.TTFT_NS,
        MetricSeriesKey.CHUNK_DELTA_NS,
        MetricSeriesKey.TPOT_NS,
    }
    _FLOAT_SERIES = {MetricSeriesKey.TPOT_NS}
    for series_key in MetricSeriesKey:
        if series_key in _STREAMING_ONLY and not streaming:
            continue
        dtype = float if series_key in _FLOAT_SERIES else int
        reader.register_key(series_key.value, "series", dtype=dtype)
    return reader


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

        # Tmpfs for high-frequency writes (metrics mmap + event log).
        # On ARM, metrics need an on-disk directory so msync provides
        # write ordering for cross-process mmap reads. Event logs are
        # append-only and don't have ordering requirements, so they
        # can stay on tmpfs.
        shm = Path("/dev/shm")
        use_shm = shm.exists()
        tmpfs_base = shm if use_shm else Path(tempfile.gettempdir())
        tmpfs_dir = tmpfs_base / f"benchmark_{session_id}"
        tmpfs_dir.mkdir(parents=True, exist_ok=True)

        # On ARM, mmap write ordering requires msync on a real filesystem.
        # msync is a no-op on tmpfs, so metrics must use an on-disk directory.
        if use_shm and platform.machine() != "x86_64":
            logger.info(
                "ARM platform: using on-disk metrics directory for mmap ordering"
            )
            metrics_dir = Path(
                tempfile.mkdtemp(prefix=f"metrics_{session_id}_", dir=".")
            )
        else:
            metrics_dir = tmpfs_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

        event_log_dir = tmpfs_dir / "events"
        event_log_dir.mkdir(parents=True, exist_ok=True)

        # Launch service subprocesses
        launcher = ServiceLauncher(zmq_ctx)
        if zmq_ctx.socket_dir is None:
            raise RuntimeError("ZMQ socket_dir must be set after publisher bind")
        aggregator_args: list[str] = [
            "--socket-dir",
            zmq_ctx.socket_dir,
            "--socket-name",
            pub_socket_name,
            "--metrics-dir",
            str(metrics_dir),
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
            http_config = config.settings.client.with_updates(
                endpoint_urls=[urljoin(e, api_type.default_route()) for e in endpoints],
                api_type=api_type,
                adapter=None,  # reset so validator re-resolves adapter from api_type
                accumulator=None,  # reset so validator re-resolves accumulator from api_type
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

        # Create session
        session = BenchmarkSession(
            issuer=issuer,
            event_publisher=publisher,
            loop=loop,
            on_sample_complete=collector.on_complete_hook,
            session_id=session_id,
        )

        phases = _build_phases(ctx)
        report: Report | None = None

        loop.add_signal_handler(signal.SIGINT, session.stop)
        try:
            result = await session.run(phases)
        except Exception as e:
            raise ExecutionError(f"Benchmark execution failed: {e}") from e
        finally:
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

            # Build report AFTER aggregator has exited — ensures all metrics
            # (TTFT, TPOT, OSL, latency) are fully written to KVStore.
            try:
                kv_reader = _setup_kv_reader(metrics_dir, ctx.enable_streaming)
                report = Report.from_kv_reader(kv_reader)
                kv_reader.close()
            except Exception as e:
                logger.warning(f"Failed to build report from metrics: {e}")

            pbar.close()

    # Track metrics_dir separately if it's not under tmpfs_dir (ARM on-disk case)
    separate_metrics = metrics_dir if metrics_dir.parent != tmpfs_dir else None
    return BenchmarkResult(
        session=result,
        collector=collector,
        report=report,
        tmpfs_dir=tmpfs_dir,
        metrics_dir=separate_metrics,
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

    # metrics mmap files (from MetricsAggregator KVStore)
    src_metrics = tmpfs_dir / "metrics"
    if src_metrics.exists():
        dst_metrics = report_dir / "metrics"
        dst_metrics.mkdir(parents=True, exist_ok=True)
        for f in src_metrics.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_metrics / f.name)
        logger.debug(f"Copied metrics from {src_metrics} -> {dst_metrics}")


def finalize_benchmark(ctx: BenchmarkContext, bench: BenchmarkResult) -> None:
    """Score accuracy, aggregate results, write JSON."""
    config = ctx.config
    result = bench.session
    collector = bench.collector
    report = bench.report

    # Display report if available (from MetricsAggregator KVStore)
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
        scorer_instance = eval_cfg.scorer(
            eval_cfg.dataset_name,
            eval_cfg.dataset,
            eval_cfg.report_dir,
            extractor=eval_cfg.extractor,
            ground_truth_column=eval_cfg.ground_truth_column,
        )
        score, n_repeats = scorer_instance.score()
        assert eval_cfg.dataset.data is not None
        accuracy_scores[eval_cfg.dataset_name] = {
            "dataset_name": eval_cfg.dataset_name,
            "num_samples": len(eval_cfg.dataset.data),
            "extractor": eval_cfg.extractor.__name__,
            "ground_truth_column": eval_cfg.ground_truth_column,
            "score": score,
            "n_repeats": n_repeats,
        }
        logger.info(f"Score for {eval_cfg.dataset_name}: {score} ({n_repeats} repeats)")

    # Report metrics: prefer Report from KVStore, fall back to SessionResult
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
            if bench.metrics_dir and bench.metrics_dir.exists():
                shutil.rmtree(bench.metrics_dir, ignore_errors=True)
            logger.info(f"Partial results saved to {ctx.report_dir}")
