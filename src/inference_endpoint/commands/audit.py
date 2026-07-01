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
  3  SetupError     — config invalid for audit (bad load pattern / sample_index)
  4  ExecutionError — a phase run failed or produced partial data
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..compliance import AuditRunArtifacts, get_audit_test
from ..compliance.result import AuditResult, write_result
from ..config.schema import BenchmarkConfig, LoadPatternType
from ..exceptions import ExecutionError, SetupError
from .benchmark.execute import (
    TestMode,
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
        SetupError: Config invalid for audit (missing audit block, paced load, bad index).
        ExecutionError: A phase benchmark run failed.
    """
    assert config.audit is not None, "run_audit called with config.audit=None"
    base_report_dir.mkdir(parents=True, exist_ok=True)
    audit_cfg = config.audit
    test = get_audit_test(audit_cfg.test)

    # Validate load pattern. The output-caching audit (MLPerf TEST04) compares
    # the audit phase's achieved QPS against the reference phase's, and report.qps
    # is completed-samples / duration. That maps onto the MLPerf scenarios whose
    # score is a throughput rate:
    #   - max_throughput → Offline ("Samples per second").
    #   - concurrency / poisson → Server ("Completed samples per second"); both
    #     phases reuse the same settings, so the arrival rate matches by
    #     construction (upstream's TEST04 pins target_latency for the same reason).
    # agentic_inference (conversation-turn semantics) makes a fixed-index phase
    # meaningless, and burst/step are not implemented. Allow-list the valid
    # patterns rather than enumerate the rejects.
    load_type = config.settings.load_pattern.type
    if load_type not in (
        LoadPatternType.MAX_THROUGHPUT,
        LoadPatternType.CONCURRENCY,
        LoadPatternType.POISSON,
    ):
        raise SetupError(
            "Compliance audit requires a max_throughput, concurrency, or poisson "
            f"load pattern. Got: {load_type.value}"
        )

    specs = test.plan_runs(audit_cfg)

    perf_datasets = [d for d in config.datasets if d.type.value == "performance"]
    if not perf_datasets:
        raise SetupError("Audit requires at least one performance dataset")

    # Three distinct sample counts are in play here; keep them straight:
    #   - dataset_size: rows in the loaded dataset (the bound a fixed
    #     sample_index must fall within).
    #   - spec.n_samples: the count a phase requests (None = full/default).
    #   - report.n_samples_issued: the count a phase actually issued at runtime.
    #
    # The first phase's setup_benchmark loads the dataset; reuse that size to
    # bounds-check the fixed indices before any phase actually runs.
    # setup_benchmark only loads data (it spawns no workers), so a failed bounds
    # check here costs one load and nothing more.
    artifacts: list[AuditRunArtifacts] = []
    dataset_size: int | None = None
    for spec in specs:
        phase_dir = base_report_dir / spec.label
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Build a per-phase config: phase subdirectory, no nested audit, and
        # performance datasets only. An audit phase is performance-only; leaving
        # accuracy datasets attached would make setup_benchmark append ACCURACY
        # phases, re-issuing and re-scoring the full accuracy set on every phase.
        phase_config = config.with_updates(
            report_dir=phase_dir, audit=None, datasets=perf_datasets
        )

        try:
            ctx = setup_benchmark(phase_config, TestMode.PERF, run_spec=spec)
            if dataset_size is None:
                dataset_size = ctx.dataloader.num_samples()
                # Validate each distinct fixed index once (deduped across specs).
                fixed_indices = {
                    s.sample_order.fixed_index
                    for s in specs
                    if s.sample_order.fixed_index is not None
                }
                for idx in sorted(fixed_indices):
                    if not 0 <= idx < dataset_size:
                        raise SetupError(
                            f"Audit sample_index={idx} is out of range "
                            f"[0, {dataset_size}) for dataset with "
                            f"{dataset_size} samples"
                        )
                # A without-replacement (reference) phase must not request more
                # rows than the dataset holds: WithoutReplacementSampleOrder
                # reshuffles and repeats once exhausted, so an oversized count
                # re-issues earlier rows and the "distinct samples" baseline
                # becomes cacheable too — masking caching or flipping the verdict.
                for s in specs:
                    if (
                        s.sample_order.fixed_index is None
                        and s.n_samples is not None
                        and s.n_samples > dataset_size
                    ):
                        raise SetupError(
                            f"Audit phase '{s.label}': requested {s.n_samples} "
                            f"distinct samples exceeds dataset size {dataset_size}"
                        )
            bench = run_benchmark_async(ctx)
            finalize_benchmark(ctx, bench)
        except (SetupError, ExecutionError):
            raise
        except Exception as exc:
            raise ExecutionError(f"Audit phase '{spec.label}' failed: {exc}") from exc

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

    result = test.verify(artifacts, audit_cfg)
    write_result(result, base_report_dir)

    status = "PASS" if result.passed else "FAIL"
    logger.info(
        "Audit %s %s — %s",
        audit_cfg.test,
        status,
        result.details.get("reason", ""),
    )
    return result
