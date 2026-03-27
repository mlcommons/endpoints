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

"""Benchmark CLI subcommands — offline, online, from-config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cyclopts
import yaml
from pydantic import ValidationError  # noqa: F401 (used in from_config)

from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    OfflineBenchmarkConfig,
    OnlineBenchmarkConfig,
    TestMode,
    TestType,
)
from inference_endpoint.exceptions import DatasetValidationError, InputValidationError

benchmark_app = cyclopts.App(name="benchmark", help="Run benchmarks.")


def _run(config: BenchmarkConfig, dataset: list[str], mode: TestMode) -> None:
    """Unified entry point: inject CLI datasets if needed, then run."""
    if not config.datasets and dataset:
        try:
            # Raw strings are parsed by BenchmarkConfig._coerce_dataset_strings validator
            config = config.with_updates(datasets=dataset)
        except ValidationError as e:
            msgs = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            raise DatasetValidationError(f"Invalid --dataset: {msgs}") from e
        except ValueError as e:
            raise DatasetValidationError(f"Invalid --dataset: {e}") from e
    run_benchmark(config, mode)


@benchmark_app.command
def offline(
    *,
    config: OfflineBenchmarkConfig,
    dataset: Annotated[
        list[str],
        cyclopts.Parameter(
            help="Dataset(s) as [perf|acc:]<path>[,key=val...]", negative=""
        ),
    ],
    mode: Annotated[
        TestMode,
        cyclopts.Parameter(help="Test mode: perf, acc, or both"),
    ] = TestMode.PERF,
):
    """Offline benchmark — all queries at t=0 for max throughput."""
    _run(config, dataset, mode)


@benchmark_app.command(name="online")
def online(
    *,
    config: OnlineBenchmarkConfig,
    dataset: Annotated[
        list[str],
        cyclopts.Parameter(
            help="Dataset(s) as [perf|acc:]<path>[,key=val...]", negative=""
        ),
    ],
    mode: Annotated[
        TestMode,
        cyclopts.Parameter(help="Test mode: perf, acc, or both"),
    ] = TestMode.PERF,
):
    """Online benchmark — sustained QPS with load pattern."""
    _run(config, dataset, mode)


@benchmark_app.command(name="from-config")
def from_config(
    *,
    config: Annotated[Path, cyclopts.Parameter(name=["--config", "-c"])],
    timeout: float | None = None,
    mode: TestMode | None = None,
):
    """Run benchmark from YAML config file."""
    try:
        resolved = BenchmarkConfig.from_yaml_file(config)
    except (yaml.YAMLError, ValidationError, ValueError, FileNotFoundError) as e:
        raise InputValidationError(f"Config error: {e}") from e
    if timeout is not None:
        resolved = resolved.with_updates(timeout=timeout)
    test_mode = mode or (
        TestMode.BOTH if resolved.type == TestType.SUBMISSION else TestMode.PERF
    )
    _run(resolved, [], test_mode)
