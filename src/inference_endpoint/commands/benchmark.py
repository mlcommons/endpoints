# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import signal
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    ClientSettings,
    Dataset,
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
from inference_endpoint.config.yaml_loader import ConfigError, ConfigLoader
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
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
        elif self.collect_responses:
            self.responses[result.id] = result.response_output

        if self.pbar:
            self.pbar.update(1)


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
        effective_config: BenchmarkConfig = _build_config_from_cli(
            args, benchmark_mode_str
        )
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

    # Determine if we should collect responses
    collect_responses = test_mode in [TestMode.ACC, TestMode.BOTH]

    # Run benchmark
    _run_benchmark(effective_config, collect_responses, test_mode, benchmark_mode)


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
    report_dir = getattr(args, "report_dir", None)
    timeout = getattr(args, "timeout", None)
    verbose = getattr(args, "verbose", False)
    output = getattr(args, "output", None)
    # Build BenchmarkConfig from CLI params
    return BenchmarkConfig(
        name=f"cli_{benchmark_mode}",
        version="1.0",
        type=TestType.OFFLINE if benchmark_mode == "offline" else TestType.ONLINE,
        datasets=[
            Dataset(
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
                workers=args.workers if args.workers else 4,
                max_concurrency=-1,  # client uses unlimited concurrency by default
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
        endpoint_config=EndpointConfig(endpoint=args.endpoint, api_key=args.api_key),
        metrics=Metrics(),
        baseline=None,  # CLI mode doesn't use baseline
        report_dir=report_dir,
        output=output,
        timeout=timeout,
        verbose=verbose,
    )


def _get_dataset_path(args: argparse.Namespace, config: BenchmarkConfig) -> Path:
    """Get dataset path from CLI args or config.

    CURRENT LIMITATION: Only supports single dataset execution.
    Priority: CLI args > config datasets[0]

    Args:
        args: Command arguments
        config: BenchmarkConfig

    Returns:
        Path to dataset file

    Raises:
        InputValidationError: If no dataset specified or file doesn't exist

    TODO: Multi-dataset support
    When implemented, this should:
    1. Return list[Path] for multiple datasets
    2. Validate all dataset paths exist
    3. Support dataset interleaving strategies
    """
    if hasattr(args, "dataset") and args.dataset:
        dataset_path = Path(args.dataset)
    else:
        # TODO: Multi-dataset - currently just picks single dataset
        single_dataset = config.get_single_dataset()
        if single_dataset:
            dataset_path = Path(single_dataset.path)
        else:
            logger.error("Dataset required: --dataset PATH or specify in config")
            raise InputValidationError(
                "Dataset required: --dataset PATH or specify in config"
            )

    # Validate file exists
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        raise InputValidationError(f"Dataset not found: {dataset_path}")

    return dataset_path


def _get_dataset_format(config: BenchmarkConfig, dataset_path: Path) -> str:
    """Get or infer dataset format.

    CURRENT LIMITATION: Only supports single dataset.

    Args:
        config: BenchmarkConfig
        dataset_path: Path to dataset file

    Returns:
        Dataset format string (e.g., "pkl", "hf")

    TODO: Multi-dataset support
    When implemented, this should:
    1. Return dict[Path, str] mapping dataset paths to formats
    2. Validate format compatibility across datasets
    """
    # Try to get format from config
    # TODO: Multi-dataset - currently just uses single dataset format
    single_dataset = config.get_single_dataset()
    if single_dataset and single_dataset.format:
        return single_dataset.format

    # Infer from file extension
    format_str = DataLoaderFactory.infer_format(dataset_path)
    logger.info(f"Inferred dataset format: {format_str}")
    return format_str


def _run_benchmark(
    config: BenchmarkConfig,
    collect_responses: bool,
    test_mode: TestMode,
    benchmark_mode: TestType | None,
) -> None:
    """Execute the actual benchmark with full lifecycle management.

    This function orchestrates the complete benchmark execution:
    1. Load tokenizer for the target model
    2. Load and validate dataset using DataLoaderFactory
    3. Setup runtime settings and scheduler
    4. Create HTTP endpoint client with multiprocessing workers
    5. Run benchmark session with signal handling
    6. Collect and report results
    7. Clean up resources (always, even on error)

    Architecture notes:
    - This is a SYNCHRONOUS function (not async) because HTTPEndpointClient
      manages its own event loop in a separate thread
    - Uses blocking operations: http_client.start(), sess.wait_for_test_end()
    - Signal handling: SIGINT (Ctrl+C) gracefully stops benchmark
    - Cleanup: Always executes via finally block

    Streaming behavior:
    - Enabled automatically for online mode (for TTFT metrics)
    - Disabled for offline mode (max throughput focus)

    Args:
        args: Command arguments containing output paths, verbosity, etc.
        config: Validated BenchmarkConfig (immutable Pydantic model).
               Contains all benchmark parameters from CLI or YAML.
        collect_responses: Whether to store full response text.
                          True for accuracy modes (TestMode.ACC/BOTH).
        test_mode: What to collect - PERF (metrics only), ACC (responses),
                  or BOTH (metrics + responses).
        benchmark_mode: Execution mode - OFFLINE (max throughput) or
                       ONLINE (sustained QPS). Affects streaming and scheduling.

    Raises:
        InputValidationError: If model/dataset cannot be loaded or validated.
        SetupError: If connection to endpoint fails or resources unavailable.
        ExecutionError: If benchmark execution fails after successful setup.
        KeyboardInterrupt: If user interrupts with Ctrl+C (re-raised for CLI handler).
    """

    # Load tokenizer if model name is provided
    # Priority: CLI args (offline/online modes) > config submission_ref (from-config mode)
    tokenizer = None
    model_name = config.model_params.name
    if not model_name and config.submission_ref:
        model_name = config.submission_ref.model
    if not model_name and config.model_params.name:
        model_name = config.model_params.name

    if config.report_dir:
        report_dir = Path(config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        config.to_yaml_file(report_dir / "config.yaml")

    max_tokens = config.model_params.max_new_tokens

    if model_name:
        try:
            logger.info(f"Loading tokenizer for model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            logger.warning(
                "Continuing without tokenizer (report metrics may be limited)"
            )
    else:
        # Throw exception if no model name is provided
        raise InputValidationError("No model name provided")

    # Get dataset - from CLI or from config
    # TODO: Dataset Logic is not yet fully implemented
    # dataset_path = _get_dataset_path(args, config)
    dataset_path = config.datasets[0].path

    # Load dataset using factory
    dataset_format = _get_dataset_format(config, dataset_path)
    logger.info(f"Loading: {dataset_path} (format: {dataset_format})")

    # Determine if streaming should be enabled based on config
    streaming_mode = config.model_params.streaming

    if streaming_mode == StreamingMode.ON:
        enable_streaming = True
        logger.info("Streaming: enabled (forced via --streaming=on)")
    elif streaming_mode == StreamingMode.OFF:
        enable_streaming = False
        logger.info("Streaming: disabled (forced via --streaming=off)")
    else:  # StreamingMode.AUTO
        enable_streaming = benchmark_mode == TestType.ONLINE
        if enable_streaming:
            logger.info("Streaming: enabled (auto, online mode)")
        else:
            logger.info("Streaming: disabled (auto, offline mode)")

    try:
        if any(d.parser for d in config.datasets):
            key_maps = [d.parser for d in config.datasets]
        else:
            key_maps = None
        logger.info(f"Parser key maps: {key_maps}")

        dataloader = DataLoaderFactory.create_loader(
            dataset_path,
            format=dataset_format,
            key_maps=key_maps,
            metadata={
                "model": model_name,
                "stream": enable_streaming,
                "max_completion_tokens": max_tokens,
                "temperature": config.model_params.temperature,
                "top_p": config.model_params.top_p,
                "top_k": config.model_params.top_k,
                "repetition_penalty": config.model_params.repetition_penalty,
            },
        )
        dataloader.load()
        logger.info(f"Loaded {dataloader.num_samples()} samples")
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {dataset_path}")
        raise InputValidationError(f"Dataset file not found: {dataset_path}") from e
    except NotImplementedError as e:
        logger.error(f"Dataset format not supported: {dataset_format}")
        raise SetupError(str(e)) from e
    except Exception as e:
        logger.error("Dataset load failed")
        raise SetupError(f"Failed to load dataset: {e}") from e

    # Setup runtime settings using factory method
    rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
    target_qps = config.settings.load_pattern.target_qps
    load_pattern_type = config.settings.load_pattern.type

    # Calculate and display expected sample count
    total_samples = rt_settings.total_samples_to_issue()
    duration_s = rt_settings.min_duration_ms / 1000

    logger.info(
        f"Mode: {test_mode}, Target QPS: {target_qps}, Responses: {collect_responses}"
    )
    logger.info(f"Min Duration: {duration_s:.1f}s, Expected samples: {total_samples}")

    # Create scheduler using __init_subclass__ registry
    try:
        scheduler_class = Scheduler.get_implementation(load_pattern_type)
        scheduler = scheduler_class(rt_settings, WithoutReplacementSampleOrder)
        logger.info(
            f"Scheduler: {scheduler_class.__name__} (pattern: {load_pattern_type.value})"
        )
    except KeyError as e:
        logger.exception("Scheduler not available")
        raise SetupError(str(e)) from e

    # Setup response collector
    pbar = tqdm(
        desc=f"{model_name} (Streaming: {enable_streaming})", total=total_samples
    )
    response_collector = ResponseCollector(
        collect_responses=collect_responses, pbar=pbar
    )
    SampleEventHandler.register_hook(
        SampleEvent.COMPLETE, response_collector.on_complete_hook
    )

    # Create endpoint client
    endpoint = config.endpoint_config.endpoint
    num_workers = config.settings.client.workers

    logger.info(f"Connecting: {endpoint}")
    logger.info(f"Client config: workers={num_workers}")

    tmp_dir = tempfile.mkdtemp(prefix="inference_endpoint_")

    try:
        http_config = HTTPClientConfig(
            endpoint_url=urljoin(endpoint, "/v1/chat/completions"),
            num_workers=num_workers,
            max_concurrency=-1,  # unlimited
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_dir}/req",
            zmq_response_queue_addr=f"ipc://{tmp_dir}/resp",
            zmq_readiness_queue_addr=f"ipc://{tmp_dir}/ready",
        )

        http_client = HTTPEndpointClient(http_config, aiohttp_config, zmq_config)
        sample_issuer = HttpClientSampleIssuer(http_client)

        http_client.start()
        sample_issuer.start()

    except Exception as e:
        logger.error("Connection failed")
        raise SetupError(f"Failed to connect to endpoint: {e}") from e

    # Run benchmark
    logger.info("Running...")
    start_time = time.time()

    sess = None
    try:
        sess = BenchmarkSession.start(
            rt_settings,
            dataloader,
            sample_issuer,
            scheduler,
            name=f"cli_benchmark_{uuid.uuid4().hex[0:8]}",
            stop_sample_issuer_on_test_end=False,
            report_dir=config.report_dir,
            tokenizer_override=tokenizer,
            max_shutdown_timeout_s=config.timeout if config.timeout else None,
        )

        # Wait for test end with ability to interrupt
        def signal_handler(signum, frame):
            logger.warning("Interrupt signal received, stopping benchmark...")
            # Raise KeyboardInterrupt to break out of wait_for_test_end()
            raise KeyboardInterrupt()

        # Install our handler, save old one
        old_handler = signal.signal(signal.SIGINT, signal_handler)
        try:
            sess.wait_for_test_end()
        finally:
            # Always restore original handler
            signal.signal(signal.SIGINT, old_handler)

        elapsed_time = time.time() - start_time
        success_count = response_collector.count - len(response_collector.errors)
        estimated_qps = (
            response_collector.count / elapsed_time if elapsed_time > 0 else 0
        )

        # Report results
        logger.info(f"Completed in {elapsed_time:.1f}s")
        logger.info(
            f"Results: {success_count}/{scheduler.total_samples_to_issue} successful"
        )
        logger.info(f"Estimated QPS: {estimated_qps:.1f}")

        if response_collector.errors:
            logger.warning(f"Errors: {len(response_collector.errors)}")
            if config.verbose:
                for error in response_collector.errors[:3]:
                    logger.warning(f"  {error}")
                if len(response_collector.errors) > 3:
                    logger.warning(f"  ... +{len(response_collector.errors) - 3} more")

        # Save results if requested
        if config.output:
            try:
                results = {
                    "config": {
                        "endpoint": endpoint,
                        "mode": test_mode,
                        "target_qps": target_qps,
                    },
                    "results": {
                        "total": scheduler.total_samples_to_issue,
                        "successful": success_count,
                        "failed": len(response_collector.errors),
                        "elapsed_time": elapsed_time,
                        "qps": estimated_qps,
                    },
                }

                if collect_responses:
                    results["responses"] = response_collector.responses

                # Always save all errors (useful for debugging)
                if response_collector.errors:
                    results["errors"] = response_collector.errors

                with open(config.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved: {config.output}")
            except Exception as e:
                logger.error(f"Save failed: {e}")

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        # Will be re-raised by CLI main() for proper exit
        raise
    except ExecutionError:
        # Re-raise our own exceptions
        raise
    except Exception as e:
        logger.error("Benchmark failed")
        raise ExecutionError(f"Benchmark execution failed: {e}") from e
    finally:
        # Cleanup - always execute
        logger.info("Cleaning up...")
        try:
            pbar.close()
            sample_issuer.shutdown()
            http_client.shutdown()
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            if config.verbose:
                logger.warning(f"Cleanup error: {e}")
