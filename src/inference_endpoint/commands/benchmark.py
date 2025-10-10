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
from pathlib import Path

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
    """Collects responses for accuracy evaluation."""

    def __init__(self, collect_responses: bool = False):
        self.collect_responses = collect_responses
        self.responses: dict[str, str] = {}
        self.errors: list[str] = []
        self.count = 0

    def on_complete_hook(self, result: QueryResult):
        """Hook called when a sample completes."""
        self.count += 1
        if result.error:
            self.errors.append(f"Sample {result.id}: {result.error}")
        elif self.collect_responses:
            self.responses[result.id] = result.response_output


async def run_benchmark_command(args: argparse.Namespace) -> None:
    """Run benchmark (offline, online, or YAML-configured)."""

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
            effective_config = ConfigLoader.load_yaml(Path(config_path))

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

    # Validate configuration (comprehensive validation with warnings)
    try:
        ConfigLoader.validate_config(effective_config, benchmark_mode)
    except ConfigError as e:
        logger.exception("Config validation error")
        raise InputValidationError(str(e)) from e

    # Determine if we should collect responses
    collect_responses = test_mode in [TestMode.ACC, TestMode.BOTH]

    # Run benchmark
    _run_benchmark(args, effective_config, collect_responses, test_mode, benchmark_mode)


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
    load_pattern_arg = getattr(args, "load_pattern", None)
    if load_pattern_arg:
        load_pattern_type = LoadPatternType(load_pattern_arg)
    else:
        load_pattern_type = (
            LoadPatternType.MAX_THROUGHPUT
            if benchmark_mode == "offline"
            else LoadPatternType.POISSON
        )

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
                format="pkl",  # Will be inferred by DataLoaderFactory
            )
        ],
        settings=Settings(
            load_pattern=LoadPattern(
                type=load_pattern_type, qps=args.qps if args.qps else 10.0
            ),
            runtime=RuntimeConfig(
                min_duration_ms=args.duration * 1000
                if args.duration
                else 10000,  # Default: 10s for quick testing (TODO: Make configurable)
                max_duration_ms=1800000,
                scheduler_random_seed=42,
                dataloader_random_seed=42,
            ),
            client=ClientSettings(
                workers=args.workers if args.workers else 4,
                max_concurrency=args.concurrency if args.concurrency else -1,
            ),
        ),
        model_params=ModelParams(
            temperature=0.7,
            max_new_tokens=args.max_tokens if args.max_tokens else 1024,
            osl_distribution=OSLDistribution(
                min=args.min_tokens if args.min_tokens else 1,
                max=args.max_tokens if args.max_tokens else 2048,
            )
            if (args.min_tokens or args.max_tokens)
            else None,
        ),
        endpoint_config=EndpointConfig(endpoint=args.endpoint, api_key=args.api_key),
        metrics=Metrics(),
        baseline=None,  # CLI mode doesn't use baseline
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
    # Priority: CLI args > config
    if args.dataset:
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
    args: argparse.Namespace,
    config: BenchmarkConfig,
    collect_responses: bool,
    test_mode: TestMode,
    benchmark_mode: TestType | None,
) -> None:
    """Execute the actual benchmark.

    Note: This function uses sync methods (http_client.start(), sess.wait_for_test_end())
    because HTTPEndpointClient manages its own event loop in a separate thread.
    It should NOT be async since it performs blocking operations.

    Args:
        args: Command arguments
        config: Validated BenchmarkConfig (immutable Pydantic model)
        collect_responses: Whether to collect responses (for accuracy mode)
        test_mode: TestMode enum (PERF, ACC, or BOTH)
        benchmark_mode: TestType enum (OFFLINE or ONLINE)
    """

    # Load tokenizer if model name is provided
    # Priority: CLI args (offline/online modes) > config submission_ref (from-config mode)
    tokenizer = None
    model_name = getattr(args, "model", None)
    if not model_name and config.submission_ref:
        model_name = config.submission_ref.model

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

    # Get report path if specified
    report_path = getattr(args, "report_path", None)
    if report_path:
        logger.info(f"Report will be saved to: {report_path}")

    # Get dataset - from CLI or from config
    # TODO: Dataset Logic is not yet fully implemented
    dataset_path = _get_dataset_path(args, config)

    # Load dataset using factory
    dataset_format = _get_dataset_format(config, dataset_path)
    logger.info(f"Loading: {dataset_path.name} (format: {dataset_format})")

    # Determine if streaming should be enabled (only for online mode)
    enable_streaming = benchmark_mode == TestType.ONLINE
    if enable_streaming:
        logger.info("Streaming enabled for TTFT metrics (online mode)")

    try:
        # Create loader using factory
        def parser(x):
            return {
                "prompt": x.text_input,
                "output": x.ref_output,
                "model": model_name,
                "stream": enable_streaming,  # Enable streaming only for online mode
            }

        dataloader = DataLoaderFactory.create_loader(
            dataset_path, format=dataset_format, parser=parser
        )
        dataloader.load()
        logger.info(f"Loaded {dataloader.num_samples()} samples")
    except NotImplementedError as e:
        logger.error(f"Dataset format not supported: {dataset_format}")
        raise SetupError(str(e)) from e
    except Exception as e:
        logger.error("Dataset load failed")
        raise SetupError(f"Failed to load dataset: {e}") from e

    # Setup runtime settings using factory method
    rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
    qps = config.settings.load_pattern.qps
    load_pattern_type = config.settings.load_pattern.type

    # Calculate and display expected sample count
    total_samples = rt_settings.total_samples_to_issue()
    duration_s = rt_settings.min_duration_ms / 1000

    logger.info(f"Mode: {test_mode}, QPS: {qps}, Responses: {collect_responses}")
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
    response_collector = ResponseCollector(collect_responses=collect_responses)
    SampleEventHandler.register_hook(
        SampleEvent.COMPLETE, response_collector.on_complete_hook
    )

    # Create endpoint client
    endpoint = config.endpoint_config.endpoint
    num_workers = config.settings.client.workers
    max_concurrency = config.settings.client.max_concurrency

    logger.info(f"Connecting: {endpoint}")

    tmp_dir = tempfile.mkdtemp(prefix="inference_endpoint_")

    try:
        http_config = HTTPClientConfig(
            endpoint_url=f"{endpoint}/v1/chat/completions",
            num_workers=num_workers,
            max_concurrency=max_concurrency,
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
            name="cli_benchmark",
            stop_sample_issuer_on_test_end=False,
            report_path=report_path,
            tokenizer_override=tokenizer,
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
            if args.verbose:
                for error in response_collector.errors[:3]:
                    logger.warning(f"  {error}")
                if len(response_collector.errors) > 3:
                    logger.warning(f"  ... +{len(response_collector.errors) - 3} more")

        # Save results if requested
        if hasattr(args, "output") and args.output:
            try:
                results = {
                    "config": {
                        "endpoint": endpoint,
                        "mode": test_mode,
                        "qps": qps,
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

                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved: {args.output}")
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
            sample_issuer.shutdown()
            http_client.shutdown()
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            if args.verbose:
                logger.warning(f"Cleanup error: {e}")
