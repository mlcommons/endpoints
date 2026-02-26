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

"""
TODO: PoC only, subject to change!

Benchmark command implementation."""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from inference_endpoint.commands.utils import get_default_report_path
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    BenchmarkConfig,
    ClientSettings,
    DatasetType,
    EndpointConfig,
    LoadPattern,
    LoadPatternType,
    Metrics,
    ModelParams,
    OSLDistribution,
    RuntimeConfig,
    Settings,
    StreamingMode,
    TestMode,
    TestType,
)
from inference_endpoint.config.schema import (
    Dataset as DatasetConfig,
)
from inference_endpoint.config.yaml_loader import ConfigError, ConfigLoader
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.cpu_affinity import pin_loadgen
from inference_endpoint.evaluation import Extractor
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.exceptions import (
    ExecutionError,
    InputValidationError,
    SetupError,
)
from inference_endpoint.load_generator import (
    SessionConfig,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.scheduler import Scheduler

# Suppress HuggingFace warnings about missing PyTorch/TensorFlow
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class ResponseCollector:
    """Collects query responses and errors for accuracy evaluation and reporting.

    TODO (to be deprecated): This is a temporary testing/validation feature. Once the full
    reporter functionality is implemented, this class will be deprecated in
    favor of the comprehensive reporting system.

    This collector acts as a callback handler for completed queries, tracking:
    - Total count of completed queries
    - Response outputs (when collect_responses is True)
    - Error messages from failed queries

    Used primarily in accuracy evaluation mode (TestMode.ACC/BOTH) where
    responses need to be stored for later comparison against ground truth.

    Attributes:
        collect_responses: Whether to store full response text (memory intensive).
        responses: Map of query_id -> response_output for successful queries.
        errors: List of error messages from failed queries.
        count: Total number of completed queries (success + failure).
    """

    def __init__(self, collect_responses: bool = False, pbar: tqdm | None = None):
        """Initialize response collector.

        Args:
            collect_responses: If True, stores full response text for each query.
                              If False, only tracks counts and errors (saves memory).
        """
        self.collect_responses = collect_responses
        self.responses: dict[str, str] = {}
        self.errors: list[str] = []
        self.count = 0

        self.pbar = pbar

    def on_complete_hook(self, result: QueryResult):
        """Callback invoked when a query completes (success or failure).

        This method is registered with SampleEventHandler and called automatically
        when a COMPLETE event fires. It updates internal counters and optionally
        stores the response text.

        Args:
            result: QueryResult containing response data or error information.
        """
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
    report_dir: os.PathLike
    ground_truth_column: str | None
    num_repeats: int


def setup_benchmark(
    config: BenchmarkConfig,
    test_mode: TestMode,
    benchmark_mode: TestType | None,
) -> SessionConfig:
    """Common setup for both sync and async benchmark runners.

    Handles: CPU affinity, tokenizer, report dir, streaming, datasets,
    scheduler, HTTP client config. Returns a SessionConfig with all
    prepared state.
    """
    collect_responses = test_mode in [TestMode.ACC, TestMode.BOTH]

    # CPU affinity
    affinity_plan = None
    if config.enable_cpu_affinity:
        affinity_plan = pin_loadgen(config.settings.client.workers)

    # Model name
    model_name = config.model_params.name
    if not model_name and config.submission_ref:
        model_name = config.submission_ref.model
        config.model_params.name = model_name
    if not model_name:
        raise InputValidationError("No model name provided")

    # Report directory
    report_dir = (
        Path(config.report_dir) if config.report_dir else get_default_report_path()
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(report_dir / "config.yaml")

    # Load tokenizer if model name is provided
    # Priority: CLI args (offline/online modes) > config submission_ref (from-config mode)
    tokenizer = None
    try:
        logger.info(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
        logger.warning("Continuing without tokenizer (report metrics may be limited)")

    # Determine if streaming should be enabled based on config
    streaming_mode = config.model_params.streaming
    if streaming_mode == StreamingMode.ON:
        enable_streaming = True
    elif streaming_mode == StreamingMode.OFF:
        enable_streaming = False
    else:
        enable_streaming = benchmark_mode == TestType.ONLINE
        config.model_params.streaming = (
            StreamingMode.ON if enable_streaming else StreamingMode.OFF
        )

    # Datasets
    accuracy_configs = [d for d in config.datasets if d.type == DatasetType.ACCURACY]
    performance_configs = [
        d for d in config.datasets if d.type == DatasetType.PERFORMANCE
    ]
    if not performance_configs and not accuracy_configs:
        raise InputValidationError("No performance or accuracy datasets provided")

    # Accuracy datasets
    # Pack the evaluation parameters for each accuracy dataset
    accuracy_datasets: list[Dataset] = []
    eval_configs: list[AccuracyConfiguration] = []
    for acc_config in accuracy_configs:
        # Type narrowing: ensure accuracy_config is not None
        assert (
            acc_config.accuracy_config is not None
        ), f"accuracy_config must be set for dataset {acc_config.name}"
        # Type narrowing: ensure required fields are not None
        assert (
            acc_config.accuracy_config.eval_method is not None
        ), f"eval_method must be set for dataset {acc_config.name}"
        assert (
            acc_config.accuracy_config.extractor is not None
        ), f"extractor must be set for dataset {acc_config.name}"

        dataset = DataLoaderFactory.create_loader(
            acc_config, num_repeats=acc_config.accuracy_config.num_repeats
        )
        accuracy_datasets.append(dataset)
        # TODO add tests and defaults
        eval_configs.append(
            AccuracyConfiguration(
                Scorer.get(acc_config.accuracy_config.eval_method),
                Extractor.get(acc_config.accuracy_config.extractor),
                acc_config.name,
                dataset,
                report_dir,
                acc_config.accuracy_config.ground_truth,
                acc_config.accuracy_config.num_repeats,
            )
        )
        dataset.load(
            api_type=config.endpoint_config.api_type,
            model_params=config.model_params,
        )
        logger.info(f"Loaded {dataset} - {dataset.num_samples()} samples")

    if not accuracy_configs:
        logger.info("No accuracy datasets provided")

    if len(performance_configs) > 1:
        logger.warning(
            "Multiple performance datasets provided, only the first one will be used"
        )

    # Performance dataset
    try:
        dataloader = DataLoaderFactory.create_loader(performance_configs[0])
        dataloader.load(
            api_type=config.endpoint_config.api_type, model_params=config.model_params
        )
        logger.info(f"Loaded {dataloader.num_samples()} samples")
    except FileNotFoundError as e:
        raise InputValidationError(
            f"Dataset file not found: {performance_configs[0].path}"
        ) from e
    except Exception as e:
        raise SetupError(f"Failed to load dataset: {e}") from e

    # Setup runtime settings using factory method
    rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
    load_pattern_type = config.settings.load_pattern.type

    # Calculate and display expected sample count
    total_samples = rt_settings.total_samples_to_issue()
    if accuracy_datasets:
        total_samples += sum(
            dataset.num_samples() * dataset.repeats for dataset in accuracy_datasets
        )

    # Create scheduler using __init_subclass__ registry
    try:
        scheduler_class = Scheduler.get_implementation(load_pattern_type)
        scheduler = scheduler_class(rt_settings, WithoutReplacementSampleOrder)
        logger.info(
            f"Scheduler: {scheduler_class.__name__} (pattern: {load_pattern_type.value})"
        )
    except KeyError as e:
        raise SetupError(str(e)) from e

    logger.info(
        f"Mode: {test_mode}, Target QPS: {config.settings.load_pattern.target_qps}, Responses: {collect_responses}"
    )
    logger.info(
        f"Min Duration: {rt_settings.min_duration_ms / 1000:.1f}s, Expected samples: {total_samples}"
    )

    # HTTP client config
    endpoints = config.endpoint_config.endpoints
    assert endpoints is not None
    api_type: APIType = config.endpoint_config.api_type

    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(e, api_type.default_route()) for e in endpoints],
        api_type=api_type,
        num_workers=config.settings.client.workers,
        record_worker_events=config.settings.client.record_worker_events,
        event_logs_dir=report_dir,
        log_level=config.settings.client.log_level,
        cpu_affinity=affinity_plan,
        warmup_connections=config.settings.client.warmup_connections,
        max_connections=config.settings.client.max_connections,
        api_key=config.endpoint_config.api_key,
    )

    return SessionConfig(
        config=config,
        tokenizer=tokenizer,
        report_dir=report_dir,
        dataloader=dataloader,
        scheduler=scheduler,
        rt_settings=rt_settings,
        http_config=http_config,
        total_samples=total_samples,
        collect_responses=collect_responses,
        enable_streaming=enable_streaming,
        affinity_plan=affinity_plan,
        accuracy_datasets=accuracy_datasets,
        eval_configs=eval_configs,
        model_name=model_name,
        load_pattern_type=load_pattern_type,
        endpoints=endpoints,
        test_mode=test_mode,
        benchmark_mode=benchmark_mode,
    )


def post_benchmark(
    setup: SessionConfig,
    report: Any,
    response_collector: ResponseCollector,
) -> None:
    """Shared post-benchmark processing: accuracy scoring, results JSON, error summary."""
    if report is None:
        logger.error(
            "Session report missing — benchmark reporter failed to produce results"
        )
        raise ExecutionError(
            "Session report missing — cannot produce benchmark results"
        )

    elapsed_time = report.duration_ns / 1e9
    total = report.n_samples_issued
    success_count = report.n_samples_completed
    # qps will be None if duration was 0, so fall back to 0.0
    estimated_qps = report.qps or 0.0

    # Report results
    logger.info(f"Completed in {elapsed_time:.1f}s")
    logger.info(f"Results: {success_count}/{total} successful")
    logger.info(f"Estimated QPS: {estimated_qps:.1f}")

    # Accuracy scoring
    accuracy_scores: dict[str, Any] = {}
    for eval_config in setup.eval_configs:
        scorer_instance = eval_config.scorer(
            eval_config.dataset_name,
            eval_config.dataset,
            eval_config.report_dir,
            extractor=eval_config.extractor,
            ground_truth_column=eval_config.ground_truth_column,
        )
        score, n_repeats = scorer_instance.score()
        assert eval_config.dataset.data is not None
        accuracy_scores[eval_config.dataset_name] = {
            "dataset_name": eval_config.dataset_name,
            "num_samples": len(eval_config.dataset.data),
            "extractor": eval_config.extractor.__name__,
            "ground_truth_column": eval_config.ground_truth_column,
            "score": score,
            "n_repeats": n_repeats,
        }
        logger.info(
            f"Score for {eval_config.dataset_name}: {score} ({n_repeats} repeats)"
        )

    # Error summary
    if response_collector.errors:
        logger.warning(f"Errors: {len(response_collector.errors)}")
        if setup.config.verbose:
            for error in response_collector.errors[:3]:
                logger.warning(f"  {error}")
            if len(response_collector.errors) > 3:
                logger.warning(f"  ... +{len(response_collector.errors) - 3} more")

    # Results JSON
    try:
        results: dict[str, Any] = {
            "config": {
                "endpoint": setup.endpoints,
                "mode": setup.test_mode.value
                if hasattr(setup.test_mode, "value")
                else str(setup.test_mode),
                "target_qps": setup.config.settings.load_pattern.target_qps,
            },
            "results": {
                "total": total,
                "successful": success_count,
                "failed": total - success_count,
                "elapsed_time": elapsed_time,
                "qps": estimated_qps,
            },
        }
        if accuracy_scores:
            results["accuracy_scores"] = accuracy_scores
        if setup.collect_responses:
            results["responses"] = response_collector.responses
        # Always save all errors (useful for debugging)
        if response_collector.errors:
            results["errors"] = response_collector.errors
        # Save results to JSON file
        results_path = setup.report_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {results_path}")
    except Exception as e:
        logger.error(f"Save failed: {e}")


async def run_benchmark_command(args: argparse.Namespace) -> None:
    """Run performance benchmark in offline, online, or YAML-configured mode.

    This is the main entry point for the benchmark command. It:
    1. Determines benchmark mode (offline/online/from-config)
    2. Builds or loads configuration (CLI args vs YAML)
    3. Validates configuration comprehensively
    4. Determines response collection strategy based on test mode
    5. Delegates to _run_benchmark() for execution

    Benchmark modes:
    - offline: Max throughput test (all queries at t=0)
    - online: Sustained QPS test (Poisson/concurrency-based scheduling)
    - from-config: Load all settings from YAML file

    Test modes (--mode):
    - perf: Performance metrics only (no response storage)
    - acc: Accuracy evaluation (collect responses)
    - both: Both performance and accuracy

    Args:
        args: Parsed command line arguments from argparse.
              Expected attributes vary by benchmark_mode:
              - offline/online: endpoint, model, dataset, qps, workers, etc.
              - from-config: config (YAML path)

    Raises:
        InputValidationError: If configuration is invalid or args are missing.
        SetupError: If initialization fails (dataset loading, connection, etc.).
        ExecutionError: If benchmark execution fails after successful setup.
    """

    # Determine benchmark mode
    benchmark_mode_str = getattr(
        args, "benchmark_mode", None
    )  # "offline", "online", "from-config", or None

    # Three subcommands:
    # - benchmark offline: CLI mode
    # - benchmark online: CLI mode
    # - benchmark from-config: YAML mode
    # Argparse enforces all arg validity per mode

    if benchmark_mode_str == "from-config":
        # ===== YAML MODE - Load from config file =====
        config_path = args.config  # Required by argparse
        try:
            effective_config: BenchmarkConfig = ConfigLoader.load_yaml(
                Path(config_path)
            )

            # Only auxiliary params allowed (output)
            mode_str = getattr(args, "mode", None)
            test_mode = (
                TestMode(mode_str)
                if mode_str
                else (
                    TestMode.BOTH
                    if effective_config.type == TestType.SUBMISSION
                    else TestMode.PERF
                )
            )

            # Get benchmark mode from config
            benchmark_mode = effective_config.get_benchmark_mode()
            if not benchmark_mode:
                raise InputValidationError(
                    "SUBMISSION configs must specify 'benchmark_mode' (offline or online)"
                )
        except ConfigError as e:
            logger.error(f"Config error: {e}")
            raise InputValidationError(f"Config error: {e}") from e

    elif benchmark_mode_str in ("offline", "online"):
        # ===== CLI MODE - Build config from CLI params =====
        benchmark_mode = TestType(benchmark_mode_str)  # TestType values are lowercase
        effective_config = _build_config_from_cli(args, benchmark_mode_str)
        test_mode = (
            TestMode(args.mode) if getattr(args, "mode", None) else TestMode.PERF
        )

    else:
        # Shouldn't happen with current argparse structure
        raise InputValidationError(
            "Unknown benchmark mode. Use: offline, online, or from-config"
        )

    # Validate configuration
    try:
        ConfigLoader.validate_config(effective_config, benchmark_mode)
    except ConfigError as e:
        logger.exception("Config validation error")
        raise InputValidationError(str(e)) from e

    # Common setup
    setup = setup_benchmark(effective_config, test_mode, benchmark_mode)

    # Select execution engine and run
    use_async = os.environ.get("INFERENCE_ENDPOINT_ASYNC", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_async:
        logger.info("Using single-loop async benchmark engine (uvloop)")
        from inference_endpoint.commands.benchmark_async import (
            run_benchmark as run_benchmark_async,
        )

        report, response_collector = await run_benchmark_async(setup)
    else:
        from inference_endpoint.commands.benchmark_sync import (
            run_benchmark as run_benchmark_sync,
        )

        report, response_collector = run_benchmark_sync(setup)

    # Shared post-processing: accuracy scoring, results.json, error summary
    post_benchmark(setup, report, response_collector)


def _build_config_from_cli(
    args: argparse.Namespace, benchmark_mode: str
) -> BenchmarkConfig:
    """Build BenchmarkConfig from CLI arguments (CLI mode only).

    Args:
        args: Parsed CLI arguments
        benchmark_mode: "online" or "offline"

    Returns:
        BenchmarkConfig built from CLI params

    Raises:
        InputValidationError: If required params missing
    """
    # Determine load pattern (CLI override or mode default)
    if load_pattern_arg := getattr(args, "load_pattern", None):
        load_pattern_type = LoadPatternType(load_pattern_arg)
    else:
        match benchmark_mode:
            case "offline":
                load_pattern_type = LoadPatternType.MAX_THROUGHPUT
            case "online" if getattr(args, "concurrency", None):
                load_pattern_type = LoadPatternType.CONCURRENCY
            case "online":
                load_pattern_type = LoadPatternType.POISSON
    report_dir = getattr(
        args,
        "report_dir",
        get_default_report_path(),
    )
    timeout = getattr(args, "timeout", None)
    verbose_level = getattr(args, "verbose", 0)
    api_type = APIType(getattr(args, "api_type", "openai"))
    # Build BenchmarkConfig from CLI params
    return BenchmarkConfig(
        name=f"cli_{benchmark_mode}",
        version="1.0",
        type=TestType.OFFLINE if benchmark_mode == "offline" else TestType.ONLINE,
        datasets=[
            DatasetConfig(
                name=args.dataset.stem,
                type=DatasetType.PERFORMANCE,
                path=str(args.dataset),
                format=None,  # Will be inferred by DataLoaderFactory
            )
        ],
        settings=Settings(
            load_pattern=LoadPattern(
                type=load_pattern_type,
                target_qps=getattr(args, "target_qps", None),
                target_concurrency=getattr(args, "concurrency", None),
            ),
            runtime=RuntimeConfig(
                min_duration_ms=args.duration * 1000
                if args.duration
                else 0,  # Default: 0 (sample count determined by n_samples_to_issue or dataset size)
                max_duration_ms=1800000,
                n_samples_to_issue=getattr(
                    args, "num_samples", None
                ),  # Map --num-samples to config
                scheduler_random_seed=42,
                dataloader_random_seed=42,
            ),
            client=ClientSettings(
                workers=args.workers if args.workers else -1,
                log_level="DEBUG" if verbose_level >= 2 else "INFO",
                warmup_connections=getattr(args, "warmup_connections", -1),
                max_connections=getattr(args, "max_connections", None) or -1,
            ),
        ),
        model_params=ModelParams(
            name=args.model,
            temperature=0.7,
            max_new_tokens=args.max_output_tokens if args.max_output_tokens else 1024,
            osl_distribution=OSLDistribution(
                min=args.min_output_tokens if args.min_output_tokens else 1,
                max=args.max_output_tokens if args.max_output_tokens else 2048,
            )
            if (args.min_output_tokens or args.max_output_tokens)
            else None,
            streaming=StreamingMode(getattr(args, "streaming", "auto")),
        ),
        endpoint_config=EndpointConfig(
            endpoints=[e.strip() for e in args.endpoints.split(",") if e.strip()],
            api_key=args.api_key,
            api_type=api_type,
        ),
        metrics=Metrics(),
        report_dir=report_dir,
        timeout=timeout,
        verbose=verbose_level > 0,
    )
