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

"""Generic compliance audit orchestrator.

run_audit(config) drives all phases of a compliance audit test back-to-back
against the same endpoint, then verifies the results and writes the result.

run_audit returns an AuditResult; cli.py maps PASS/FAIL and main.run() maps
exceptions to process exit codes:
  0  PASS           — result.passed is True
  1  FAIL           — result.passed is False (cli.py raises CLIError)
  3  SetupError     — config invalid for the audit (bad sample count/index)
  4  ExecutionError — a phase run failed or produced partial data
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..compliance import AuditRunArtifacts, get_audit_test
from ..compliance.result import AuditResult, write_result
from ..config.schema import BenchmarkConfig, DatasetType
from ..exceptions import ExecutionError, SetupError
from .benchmark.execute import (
    BenchmarkResult,
    TestMode,
    _salvage_tmpfs,
    finalize_benchmark,
    run_benchmark_async,
    setup_benchmark,
)

logger = logging.getLogger(__name__)


def run_audit(config: BenchmarkConfig, base_report_dir: Path) -> AuditResult:
    """Orchestrate the planned audit phases and return the result.

    All phases run back-to-back against the same endpoint, each under its
    own subdirectory of ``base_report_dir``. If any phase raises, the error
    is re-raised without verifying (a crashed phase must not produce a result).

    Args:
        config: Main benchmark config (must have config.audit set).
        base_report_dir: Audit output directory (e.g. ``<report_dir>/audit``);
            the per-phase subdirs and verify_<TEST>.txt + audit_result.json all
            live here.

    Returns:
        AuditResult — always returned; caller maps passed/failed to exit code.

    Raises:
        SetupError: Config invalid for audit (missing audit block, bad sample
            count/index for the dataset).
        ExecutionError: A phase benchmark run failed.
    """
    assert config.audit is not None, "run_audit called with config.audit=None"
    base_report_dir.mkdir(parents=True, exist_ok=True)
    audit_cfg = config.audit
    test = get_audit_test(audit_cfg.test)

    specs = test.plan_runs(audit_cfg)

    perf_datasets = [d for d in config.datasets if d.type == DatasetType.PERFORMANCE]
    if not perf_datasets:
        raise SetupError("Audit requires at least one performance dataset")
    accuracy_datasets = [d for d in config.datasets if d.type == DatasetType.ACCURACY]

    # Execute each phase back-to-back. The first phase's setup_benchmark loads
    # the dataset; reuse that size so the test can bounds-check all of its
    # phases before any of them actually runs. setup_benchmark only loads data
    # (it spawns no workers), so a failed validation here costs one load and
    # nothing more.
    artifacts: list[AuditRunArtifacts] = []
    dataset_size: int | None = None
    for spec in specs:
        phase_dir = base_report_dir / spec.label
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Per-phase config; datasets depend on the spec's own test_mode.
        phase_datasets = (
            perf_datasets
            if spec.test_mode == TestMode.PERF
            else perf_datasets + accuracy_datasets
        )
        phase_config = config.with_updates(
            report_dir=phase_dir, audit=None, datasets=phase_datasets
        )

        bench: BenchmarkResult | None = None
        try:
            ctx = setup_benchmark(phase_config, spec.test_mode, audit_run_spec=spec)
            if ctx.dataloader is None:
                raise SetupError(
                    f"Audit phase '{spec.label}' loaded no performance dataset"
                )
            if dataset_size is None:
                dataset_size = ctx.dataloader.num_samples()
                test.validate(
                    audit_cfg, dataset_size, config.settings.load_pattern.type
                )
            bench = run_benchmark_async(ctx)
            finalize_benchmark(ctx, bench)
        except (SetupError, ExecutionError):
            raise
        except Exception as exc:
            raise ExecutionError(f"Audit phase '{spec.label}' failed: {exc}") from exc
        finally:
            # Bypasses run_benchmark()'s cleanup, so each phase does its own.
            if bench is not None and bench.tmpfs_dir.exists():
                _salvage_tmpfs(ctx.report_dir, bench.tmpfs_dir)
                shutil.rmtree(bench.tmpfs_dir, ignore_errors=True)

        report = bench.report
        if report is None:
            raise ExecutionError(f"Audit phase '{spec.label}' produced no report")
        # A SIGINT/SIGTERM during a (long) audit phase is turned into a graceful
        # stop, so the phase returns with an "interrupted" report. Propagate it
        # as KeyboardInterrupt so the CLI exits 130 (interrupted), not as a
        # generic ExecutionError (exit 4) indistinguishable from a phase crash.
        if report.state == "interrupted":
            raise KeyboardInterrupt(f"Audit phase '{spec.label}' interrupted")
        # A drain-timeout (state complete but async tasks still pending) yields
        # partial stats; certifying a result from it would let an incomplete
        # run pass compliance.
        if not report.complete:
            raise ExecutionError(
                f"Audit phase '{spec.label}' did not complete cleanly "
                "(metrics drain timed out); "
                "refusing to certify a result from partial data"
            )
        # When the spec didn't fix a count (None = full dataset), the requested
        # count is the number actually issued this phase.
        n_requested = (
            spec.n_samples if spec.n_samples is not None else report.n_samples_issued
        )
        artifacts.append(
            AuditRunArtifacts(
                label=spec.label,
                report_dir=phase_dir,
                report=report,
                n_requested=n_requested,
            )
        )

    # Normalizes verify()'s zero-QPS ValueError to exit 4, not a traceback.
    try:
        result = test.verify(artifacts, audit_cfg)
    except (SetupError, ExecutionError):
        raise
    except Exception as exc:
        raise ExecutionError(f"Audit verification failed: {exc}") from exc
    write_result(result, base_report_dir)

    status = "PASS" if result.passed else "FAIL"
    logger.info(
        "Audit %s %s — %s",
        audit_cfg.test,
        status,
        result.details.get("reason", ""),
    )
    return result
