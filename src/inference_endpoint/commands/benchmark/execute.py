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

"""Benchmark execution — phased architecture for threaded and future async runners.

Phases:
    1. setup_benchmark()        — load tokenizer, dataset, scheduler (no IO)
    2. run_benchmark_threaded() — HTTP client + BenchmarkSession (threaded IO)
    3. finalize_benchmark()     — accuracy scoring, results JSON
"""

from __future__ import annotations

import json
import logging
import signal
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    BenchmarkConfig,
    DatasetType,
    StreamingMode,
    SystemDefaults,
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
from inference_endpoint.load_generator import (
    BenchmarkSession,
    SampleEvent,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.scheduler import Scheduler

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
    tokenizer: AutoTokenizer | None
    dataloader: Dataset
    rt_settings: RuntimeSettings
    scheduler: Scheduler
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


def _load_tokenizer(model_name: str) -> AutoTokenizer | None:
    """Load HuggingFace tokenizer, warn on failure."""
    try:
        logger.info(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
        logger.warning("Continuing without tokenizer (report metrics may be limited)")
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


def _create_scheduler(
    config: BenchmarkConfig, rt_settings: RuntimeSettings
) -> Scheduler:
    """Create scheduler using __init_subclass__ registry."""
    load_pattern_type = config.settings.load_pattern.type
    try:
        scheduler_class = Scheduler.get_implementation(load_pattern_type)
        scheduler = scheduler_class(rt_settings, WithoutReplacementSampleOrder)
        logger.info(
            f"Scheduler: {scheduler_class.__name__} (pattern: {load_pattern_type.value})"
        )
        return scheduler
    except KeyError as e:
        raise SetupError(str(e)) from e


def setup_benchmark(config: BenchmarkConfig, test_mode: TestMode) -> BenchmarkContext:
    """Load tokenizer, dataset, create scheduler, setup report dir."""
    # CPU affinity
    affinity_plan = (
        pin_loadgen(config.settings.client.workers)
        if config.enable_cpu_affinity
        else None
    )

    # Report directory
    report_dir = (
        Path(config.report_dir) if config.report_dir else _default_report_path()
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(report_dir / "config.yaml")

    # Tokenizer (model name validated by BenchmarkConfig._resolve_and_validate)
    tokenizer = _load_tokenizer(config.model_params.name)

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

    scheduler = _create_scheduler(config, rt_settings)

    return BenchmarkContext(
        config=config,
        test_mode=test_mode,
        report_dir=report_dir,
        tokenizer=tokenizer,
        dataloader=dataloader,
        rt_settings=rt_settings,
        scheduler=scheduler,
        total_samples=total_samples,
        accuracy_datasets=accuracy_datasets,
        eval_configs=eval_configs,
        affinity_plan=affinity_plan,
    )


def run_benchmark_threaded(ctx: BenchmarkContext) -> tuple[Any, ResponseCollector]:
    """Run benchmark session with threaded HTTP client. Returns (report, collector)."""
    config = ctx.config

    # Setup response collector
    pbar = tqdm(
        desc=f"{config.model_params.name} (Streaming: {ctx.enable_streaming})",
        total=ctx.total_samples,
        smoothing=0,  # smoothing=0 shows average instead of EMA
    )
    collector = ResponseCollector(collect_responses=ctx.collect_responses, pbar=pbar)
    SampleEventHandler.register_hook(SampleEvent.COMPLETE, collector.on_complete_hook)

    # Create endpoint client
    endpoints = config.endpoint_config.endpoints
    logger.info(f"Connecting: {endpoints}")
    # Transport context is managed by WorkerManager (created from transport config).
    with tempfile.TemporaryDirectory(prefix="inference_endpoint_") as _tmp_dir:
        try:
            api_type: APIType = config.endpoint_config.api_type
            http_config = config.settings.client.with_updates(
                endpoint_urls=[urljoin(e, api_type.default_route()) for e in endpoints],
                api_type=api_type,
                api_key=config.endpoint_config.api_key,
                event_logs_dir=ctx.report_dir,
                cpu_affinity=ctx.affinity_plan,
            )
            http_client = HTTPEndpointClient(http_config)
            sample_issuer = HttpClientSampleIssuer(http_client)
        except Exception as e:
            raise SetupError(f"Failed to connect to endpoint: {e}") from e

        # Run benchmark
        logger.info("Running...")
        sess = None
        try:
            sess = BenchmarkSession.start(
                ctx.rt_settings,
                ctx.dataloader,
                sample_issuer,
                ctx.scheduler,
                name=f"cli_benchmark_{uuid.uuid4().hex[0:8]}",
                report_dir=ctx.report_dir,
                tokenizer_override=ctx.tokenizer,
                accuracy_datasets=ctx.accuracy_datasets,
                max_shutdown_timeout_s=config.timeout or SystemDefaults.DEFAULT_TIMEOUT,
                dump_events_log=True,
            )

            # Wait for test end with ability to interrupt
            def _raise_keyboard_interrupt(*_: object) -> None:
                raise KeyboardInterrupt

            old_handler = signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
            try:
                sess.wait_for_test_end()
            finally:
                # Always restore original handler
                signal.signal(signal.SIGINT, old_handler)

            # Prefer authoritative metrics from the session report
            report = getattr(sess, "report", None)
            if report is None:
                raise ExecutionError("Session report missing — cannot produce results")
            return report, collector

        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted by user")
            raise
        except ExecutionError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            raise ExecutionError(f"Benchmark execution failed: {e}") from e
        finally:
            # Cleanup - always execute
            logger.info("Cleaning up...")
            try:
                if sess is not None:
                    sess.stop()
                pbar.close()
                sample_issuer.shutdown()
                http_client.shutdown()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")


def finalize_benchmark(
    ctx: BenchmarkContext,
    report: Any,
    collector: ResponseCollector,
) -> None:
    """Score accuracy, aggregate results, write JSON."""
    config = ctx.config

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

    # Report metrics
    elapsed = report.duration_ns / 1e9 if report.duration_ns is not None else 0.0
    total_issued = report.n_samples_issued
    success = total_issued - report.n_samples_failed
    qps = report.qps or 0.0

    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"Results: {success}/{total_issued} successful")
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
                "successful": success,
                "failed": report.n_samples_failed,
                "elapsed_time": elapsed,
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
    report, collector = run_benchmark_threaded(ctx)
    finalize_benchmark(ctx, report, collector)
