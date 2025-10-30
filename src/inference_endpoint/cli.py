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

Command Line Interface for the MLPerf Inference Endpoint Benchmarking System.

This module provides CLI for performance benchmarking and accuracy evaluation.
"""

import argparse
import asyncio
import logging
import sys
import traceback
from pathlib import Path

from inference_endpoint import __version__
from inference_endpoint.commands import (
    run_benchmark_command,
    run_eval_command,
    run_info_command,
    run_init_command,
    run_probe_command,
    run_validate_command,
)
from inference_endpoint.exceptions import (
    CLIError,
    ExecutionError,
    InputValidationError,
    SetupError,
)
from inference_endpoint.load_generator.scheduler import Scheduler
from inference_endpoint.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Inference Endpoint Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    # ===== Benchmark command =====
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run performance benchmark"
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_mode")

    # benchmark offline
    offline_parser = benchmark_subparsers.add_parser(
        "offline",
        help="Offline benchmark (max throughput)",
        description="Offline mode: Issues all queries at t=0 for max throughput. "
        "QPS is used to calculate total queries (QPS × duration).",
    )
    _add_shared_benchmark_args(offline_parser)
    _add_auxiliary_args(offline_parser)

    # benchmark online
    online_parser = benchmark_subparsers.add_parser(
        "online",
        help="Online benchmark (sustained QPS)",
        description="Online mode: Issues queries at target QPS using Poisson distribution.",
    )
    _add_shared_benchmark_args(online_parser)
    _add_online_specific_args(online_parser)
    _add_auxiliary_args(online_parser)

    # benchmark from-config (YAML mode)
    # Clean solution: Third subcommand for YAML mode
    # Now offline/online don't see --config at all (can't be accidentally set)
    from_config_parser = benchmark_subparsers.add_parser(
        "from-config",
        help="Run benchmark from YAML config",
        description="YAML mode: Load all configuration from YAML file.",
    )
    from_config_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="YAML config file"
    )
    _add_auxiliary_args(from_config_parser)

    # ===== Eval command =====
    eval_parser = subparsers.add_parser(
        "eval", help="Run accuracy evaluation (CLI-only)"
    )
    eval_parser.add_argument(
        "--dataset", type=str, help="Dataset name(s) or path (comma-separated)"
    )
    eval_parser.add_argument(
        "--endpoint", "-e", type=str, required=True, help="Endpoint URL"
    )
    eval_parser.add_argument("--api-key", type=str, help="API key")
    # Note: --config removed - eval is CLI-only for simplicity
    eval_parser.add_argument("--output", "-o", type=Path, help="Output file")
    eval_parser.add_argument("--judge", type=str, help="Judge model (future)")

    # ===== Probe command =====
    probe_parser = subparsers.add_parser("probe", help="Test endpoint connectivity")
    probe_parser.add_argument(
        "--endpoint", "-e", type=str, required=True, help="Endpoint URL"
    )
    probe_parser.add_argument("--api-key", type=str, help="API key")
    probe_parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., llama-2-70b)"
    )
    probe_parser.add_argument(
        "--requests", type=int, default=10, help="Number of test requests"
    )
    probe_parser.add_argument(
        "--prompt",
        type=str,
        default="Please write me a joke in 30 words.",
        help="Test prompt text",
    )

    # ===== Utility commands =====
    subparsers.add_parser("info", help="Show system info")

    validate_parser = subparsers.add_parser("validate", help="Validate YAML config")
    validate_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="Config file"
    )

    init_parser = subparsers.add_parser("init", help="Generate config template")
    init_parser.add_argument(
        "--template",
        choices=["offline", "online", "eval", "submission"],
        required=True,
        help="Template type",
    )
    init_parser.add_argument("--output", "-o", type=str, help="Output filename")

    return parser


def _add_shared_benchmark_args(parser):
    """Add shared benchmark arguments (used by both offline and online)."""
    parser.add_argument(
        "--endpoint", "-e", type=str, required=True, help="Endpoint URL"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., llama-2-70b)"
    )
    parser.add_argument(
        "--dataset", "-d", type=Path, required=True, help="Dataset file"
    )
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--qps", type=float, help="Target QPS (default: 10.0)")
    parser.add_argument("--workers", type=int, help="HTTP workers (default: 4)")
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Max concurrent requests (default: -1 unlimited)",
    )
    parser.add_argument(
        "--duration", type=int, help="Duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--mode", choices=["perf", "acc", "both"], help="Test mode (default: perf)"
    )
    parser.add_argument("--min-tokens", type=int, help="Min output tokens")
    parser.add_argument("--max-tokens", type=int, help="Max output tokens")


def _add_online_specific_args(parser):
    """Add online-specific arguments."""
    # Derive choices from Scheduler._IMPL_MAP (single source of truth via __init_subclass__)
    available_patterns = [p.value for p in Scheduler._IMPL_MAP.keys()]
    parser.add_argument(
        "--load-pattern",
        choices=available_patterns,
        help=f"Load pattern (default: poisson, available: {', '.join(available_patterns)})",
    )


def _add_auxiliary_args(parser):
    """Add auxiliary arguments (output-related, no benchmark impact)."""
    parser.add_argument("--output", "-o", type=Path, help="Results output file")
    parser.add_argument(
        "--report-path", type=Path, help="Path to save detailed benchmark report"
    )


# Argparse structure enforces arg validity - no manual validation needed


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging based on verbosity
    # Default to INFO so users see important execution info (duration, samples, results)
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.INFO  # Default: INFO (not WARNING) for user-facing info

    setup_logging()
    logging.getLogger().setLevel(log_level)

    # Dispatch commands
    try:
        if args.command == "benchmark":
            await run_benchmark_command(args)
        elif args.command == "eval":
            await run_eval_command(args)
        elif args.command == "probe":
            await run_probe_command(args)
        elif args.command == "info":
            await run_info_command(args)
        elif args.command == "validate":
            await run_validate_command(args)
        elif args.command == "init":
            await run_init_command(args)
        elif not args.command:
            parser.print_help()
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        sys.exit(0)
    except InputValidationError as e:
        # User input errors - log without stack trace
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except SetupError as e:
        # Setup/initialization errors - log with optional stack trace
        logger.error(f"Setup failed: {e}")
        if args.verbose >= 2:
            traceback.print_exc()
        sys.exit(1)
    except ExecutionError as e:
        # Execution errors - log with optional stack trace
        logger.error(f"Execution failed: {e}")
        if args.verbose >= 2:
            traceback.print_exc()
        sys.exit(1)
    except NotImplementedError as e:
        # Feature not implemented
        logger.error(f"Not implemented: {e}")
        sys.exit(1)
    except CLIError as e:
        # Generic CLI error
        logger.error(f"Error: {e}")
        if args.verbose >= 2:
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        # Unexpected error - always show details
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
