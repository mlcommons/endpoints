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

"""Benchmark command implementation."""

import argparse
import json
import logging
import random
import shutil
import signal
import tempfile
import time
from pathlib import Path

from inference_endpoint import metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.config.schema import TestMode, TestType
from inference_endpoint.config.yaml_config import ConfigError, ConfigLoader
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.loadgen import HttpClientSampleIssuer
from inference_endpoint.exceptions import (
    ExecutionError,
    InputValidationError,
    SetupError,
)
from inference_endpoint.load_generator import (
    BenchmarkSession,
    MaxThroughputScheduler,
    PoissonDistributionScheduler,
    SampleEvent,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)

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
    )  # "offline", "online", or None (YAML)
    benchmark_mode = TestType(benchmark_mode_str) if benchmark_mode_str else None
    config_path = getattr(args, "config", None)

    # Extract CLI overrides once
    cli_overrides = _extract_cli_overrides(args)

    # Load or create config
    if config_path:
        # YAML-based benchmark
        try:
            yaml_config = ConfigLoader.load_yaml(Path(config_path))
            effective_config = ConfigLoader.merge_with_cli_args(
                yaml_config, cli_overrides
            )
            # Determine test mode
            mode_str = getattr(args, "mode", None)
            if mode_str:
                test_mode = TestMode(mode_str)
            else:
                # Default: BOTH for submission, PERF otherwise
                test_mode = (
                    TestMode.BOTH
                    if yaml_config.type == TestType.SUBMISSION
                    else TestMode.PERF
                )
            # Use yaml type for validation if no explicit benchmark_mode
            if not benchmark_mode:
                # Get benchmark mode from config
                benchmark_mode = yaml_config.get_benchmark_mode()

                # For SUBMISSION without benchmark_mode set, error
                if yaml_config.type == TestType.SUBMISSION and not benchmark_mode:
                    raise InputValidationError(
                        "SUBMISSION configs must specify 'benchmark_mode' (offline or online) "
                        "to indicate whether to run offline or online performance benchmarks"
                    )
        except ConfigError as e:
            logger.error(f"Config error: {e}")
            raise InputValidationError(f"Config error: {e}") from e
    else:
        # Quick benchmark (offline/online) - no YAML, no locking concerns
        if not benchmark_mode:
            logger.error("Specify benchmark mode: offline, online, or use --config")
            raise InputValidationError(
                "Benchmark mode required: offline, online, or --config PATH"
            )

        # Create default config and apply CLI overrides using helper
        effective_config = ConfigLoader.create_default_config(benchmark_mode)
        ConfigLoader.apply_cli_overrides_to_dict(effective_config, cli_overrides)

        # Determine test mode
        mode_str = getattr(args, "mode", None)
        test_mode = TestMode(mode_str) if mode_str else TestMode.PERF

    # Validate required fields FIRST (fail fast)
    if not effective_config.get("endpoint"):
        logger.error("Endpoint required: --endpoint URL or specify in YAML config")
        raise InputValidationError(
            "Endpoint required: --endpoint URL or specify in YAML config"
        )

    # Model is required for actual endpoints (not strictly required for echo server testing)
    model = effective_config.get("model") or (
        effective_config.get("baseline", {}).get("model")
        if effective_config.get("baseline")
        else None
    )
    if not model:
        logger.warning(
            "No model specified. Using default 'gpt-3.5-turbo'. "
            "Specify with --model or in YAML config for production use."
        )
        effective_config["model"] = "gpt-3.5-turbo"

    # Validate configuration consistency
    try:
        ConfigLoader.validate_config(effective_config, benchmark_mode)
    except ConfigError as e:
        logger.error(f"Config validation error: {e}")
        raise InputValidationError(f"Config validation failed: {e}") from e

    # Determine if we should collect responses
    collect_responses = test_mode in [TestMode.ACC, TestMode.BOTH]

    # Run benchmark
    _run_benchmark(args, effective_config, collect_responses, test_mode, benchmark_mode)


def _extract_cli_overrides(args: argparse.Namespace) -> dict:
    """Extract CLI arguments that can override YAML config.

    Returns only non-None values from CLI arguments.
    """
    # Define all possible override fields
    override_fields = [
        "endpoint",
        "api_key",
        "model",
        "qps",
        "concurrency",
        "workers",
        "duration",
        "min_tokens",
        "max_tokens",
    ]

    # Extract non-None values - no mapping needed, use field names directly
    return {
        field: value
        for field in override_fields
        if (value := getattr(args, field, None)) is not None
    }


def _run_benchmark(
    args: argparse.Namespace,
    config: dict,
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
        config: Merged configuration
        collect_responses: Whether to collect responses (for accuracy mode)
        test_mode: TestMode enum (PERF, ACC, or BOTH)
        benchmark_mode: TestType enum (OFFLINE or ONLINE)
    """

    # Get dataset - from CLI or from config
    if hasattr(args, "dataset") and args.dataset:
        dataset_path = args.dataset
    elif "datasets" in config and config["datasets"]:
        # Use first performance dataset from config
        perf_datasets = [
            d for d in config["datasets"] if d.get("type") == "performance"
        ]
        if perf_datasets:
            dataset_path = Path(perf_datasets[0]["path"])
        else:
            dataset_path = Path(config["datasets"][0]["path"])
    else:
        logger.error("Dataset required: --dataset PATH or specify in config")
        raise InputValidationError(
            "Dataset required: --dataset PATH or specify in config"
        )

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        raise InputValidationError(f"Dataset not found: {dataset_path}")

    # Load dataset using factory
    logger.info(f"Loading: {dataset_path.name}")
    try:
        # Get dataset format from config or infer from file extension
        if "datasets" in config and config["datasets"]:
            # Get format from first dataset config
            dataset_format = config["datasets"][0].get("format", None)
        else:
            dataset_format = None

        # Infer format if not specified
        if not dataset_format:
            dataset_format = DataLoaderFactory.infer_format(dataset_path)
            logger.info(f"Inferred dataset format: {dataset_format}")

        # Create loader using factory
        def parser(x):
            return {
                "prompt": x.text_input,
                "output": x.ref_output,
                "model": config.get("model", "gpt-3.5-turbo"),
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

    # Setup runtime settings
    runtime_config = config.get("runtime", {})
    load_pattern_config = config.get("load_pattern", {})
    qps = load_pattern_config["qps"]  # No fallback - schema ensures this exists

    logger.info(f"Mode: {test_mode}, QPS: {qps}, Responses: {collect_responses}")

    rt_settings = RuntimeSettings(
        metrics.Throughput(qps),
        [metrics.Throughput(qps)],
        min_duration_ms=runtime_config.get("min_duration_ms", 600000),
        max_duration_ms=runtime_config.get("max_duration_ms", 1800000),
        n_samples_from_dataset=dataloader.num_samples(),
        n_samples_to_issue=dataloader.num_samples(),
        min_sample_count=1,
        rng_sched=random.Random(runtime_config.get("random_seed", 42)),
        rng_sample_index=random.Random(runtime_config.get("random_seed", 42)),
    )

    # Create scheduler based on load pattern type
    # Validate that pattern matches benchmark mode
    load_pattern_type = load_pattern_config.get("type", "max_throughput")

    if load_pattern_type == "max_throughput":
        # Offline mode: max throughput (all queries at t=0)
        # Validate: should only be used with offline mode
        if benchmark_mode == TestType.ONLINE:
            logger.warning(
                "Using max_throughput pattern in online mode - this is offline behavior"
            )

        scheduler = MaxThroughputScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
        )
        logger.info(
            "Scheduler: MaxThroughputScheduler (offline burst mode, all queries at t=0)"
        )

    elif load_pattern_type == "poisson":
        # Online mode: Poisson distribution - fixed QPS
        # Validate: should only be used with online mode
        if benchmark_mode == TestType.OFFLINE:
            raise InputValidationError(
                "Cannot use 'poisson' pattern with offline mode. "
                "Offline must use 'max_throughput' pattern."
            )

        scheduler = PoissonDistributionScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
        )
        logger.info(
            f"Scheduler: PoissonDistributionScheduler (online mode, {qps} QPS target)"
        )

    elif load_pattern_type == "concurrency":
        # Online mode: Fixed concurrency - not yet implemented
        # Validate: should only be used with online mode
        if benchmark_mode == TestType.OFFLINE:
            raise InputValidationError(
                "Cannot use 'concurrency' pattern with offline mode. "
                "Offline must use 'max_throughput' pattern."
            )

        # TODO: Implement ConcurrencyScheduler
        # In this mode:
        # - Maintain exactly N concurrent requests
        # - Issue new request when one completes
        # - QPS is not directly controlled (dominated by concurrency and latency)
        # - Useful for measuring latency at specific concurrency levels
        logger.error("Load pattern 'concurrency' not yet implemented")
        logger.error(
            "This mode would maintain fixed concurrent requests (N in-flight at all times)"
        )
        logger.error("TODO: Implement ConcurrencyScheduler")
        raise NotImplementedError(
            "Concurrency-based scheduler not yet implemented. "
            "Use 'poisson' for fixed-QPS or 'max_throughput' for offline."
        )

    else:
        # Unknown or unimplemented pattern
        logger.error(f"Unknown/unimplemented load pattern: '{load_pattern_type}'")
        logger.error("Available: max_throughput (offline), poisson (online fixed-QPS)")
        logger.error("Not yet implemented: concurrency (online fixed-concurrency)")
        raise InputValidationError(f"Load pattern '{load_pattern_type}' not supported")

    # Setup response collector
    response_collector = ResponseCollector(collect_responses=collect_responses)
    SampleEventHandler.register_hook(
        SampleEvent.COMPLETE, response_collector.on_complete_hook
    )

    # Create endpoint client
    endpoint = config["endpoint"]
    logger.info(f"Connecting: {endpoint}")

    client_config = config.get("client", {})
    tmp_dir = tempfile.mkdtemp(prefix="inference_endpoint_")

    try:
        http_config = HTTPClientConfig(
            endpoint_url=f"{endpoint}/v1/chat/completions",
            num_workers=client_config.get("workers", 4),
            max_concurrency=client_config.get("max_concurrency", 50),
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
        actual_qps = response_collector.count / elapsed_time if elapsed_time > 0 else 0

        # Report results
        logger.info(f"Completed in {elapsed_time:.1f}s")
        logger.info(
            f"Results: {success_count}/{scheduler.total_samples_to_issue} successful"
        )
        logger.info(f"QPS: {actual_qps:.1f}")

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
                        "qps": actual_qps,
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
