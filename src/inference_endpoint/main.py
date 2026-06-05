#!/usr/bin/env python3
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

"""Main entry point — app definition, error formatter, command registration, and dispatch.

Benchmark commands are in commands/benchmark/cli.py (lazy-loaded).
Simple commands (probe, info, validate-yaml, init, eval) are defined here.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Annotated

import cyclopts

from inference_endpoint import __version__
from inference_endpoint.commands.info import execute_info
from inference_endpoint.commands.init import execute_init
from inference_endpoint.commands.probe import ProbeConfig, execute_probe
from inference_endpoint.commands.validate import execute_validate
from inference_endpoint.config.utils import cli_error_formatter
from inference_endpoint.exceptions import (
    CLIError,
    ExecutionError,
    InputValidationError,
    SetupError,
)
from inference_endpoint.utils import trace
from inference_endpoint.utils.logging import setup_logging
from inference_endpoint.utils.trace import bootstrap as _bootstrap_trace

logger = logging.getLogger(__name__)


app = cyclopts.App(
    name="inference-endpoint",
    help="MLPerf Inference Endpoint Benchmarking System.",
    version=__version__,
    error_formatter=cli_error_formatter,
)


@app.meta.default
def launcher(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[
        int,
        cyclopts.Parameter(
            name="--verbose",
            alias="-v",
            count=True,
            help="Verbosity level (-v info, -vv debug, -vvv trace)",
        ),
    ] = 0,
):
    """Global options applied before any command."""
    # -vvv only spawns the trace pipeline (FIFO, dashboard subprocess,
    # stdout/stderr redirect) for the `benchmark` subcommand. Other
    # commands (info, init, probe, validate-yaml) cap the verbosity at
    # DEBUG so the FIFO/dashboard/log-redirect path is skipped —
    # running e.g. `inference-endpoint -vvv info` should print info,
    # not hijack the terminal.
    is_benchmark = bool(tokens) and tokens[0] == "benchmark"
    setup_logging(level=_bootstrap_trace(verbose if is_benchmark else min(verbose, 2)))
    try:
        app(tokens)
    finally:
        # Guarantees FIFO/dir cleanup when the benchmark fails before
        # reaching run_benchmark_async's own teardown (e.g. config /
        # dataset / endpoint setup raises). Idempotent off-trace.
        trace.cleanup()


# Benchmark subcommands — lazy-loaded from commands/benchmark/cli.py
app.command("inference_endpoint.commands.benchmark.cli:benchmark_app", name="benchmark")


# --- Misc commands ---


@app.command
def probe(*, config: ProbeConfig):
    """Test endpoint connectivity."""
    execute_probe(config)


@app.command
def info():
    """Show system information."""
    execute_info()


@app.command(name="validate-yaml")
def validate_yaml(
    *, config: Annotated[Path, cyclopts.Parameter(name=["--config", "-c"])]
):
    """Validate YAML configuration file."""
    execute_validate(config)


@app.command(name="init")
def init_cmd(template: str):
    """Generate config template.

    Args:
        template: Template type (offline, online, concurrency, eval, submission).
    """
    execute_init(template)


@app.command
def eval(
    endpoints: str,
    *,
    dataset: str | None = None,
    api_key: str | None = None,
    output: Path | None = None,
    judge: str | None = None,
):
    """Run accuracy evaluation."""
    raise CLIError(
        "Accuracy evaluation is not yet implemented. "
        "Track progress at: https://github.com/mlcommons/endpoints/issues/4"
    )


def run() -> None:
    """Entry point."""
    try:
        app.meta()
    except SystemExit as e:
        sys.exit(e.code or 0)
    except KeyboardInterrupt:
        sys.exit(130)
    except NotImplementedError as e:
        logger.error(str(e))
        sys.exit(1)
    except InputValidationError as e:
        logger.error(str(e))
        sys.exit(2)
    except SetupError as e:
        logger.error(str(e))
        sys.exit(3)
    except ExecutionError as e:
        logger.error(str(e))
        sys.exit(4)
    except CLIError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()
