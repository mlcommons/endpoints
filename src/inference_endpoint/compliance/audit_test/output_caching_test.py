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

"""Output-caching audit (MLPerf TEST04).

Detects throughput inflation from duplicate-query caching by issuing the
same sample repeatedly (the audit phase) and comparing QPS against a
reference run of distinct samples. This re-implements the intent of the
MLPerf Inference TEST04 compliance test.

Pass criterion (MLCommons-faithful):
  Each phase completed ≥ requested * (1 - threshold)
  AND audit_qps < ref_qps * (1 + threshold)  [caching inflates audit QPS → FAIL]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...config.runtime_settings import SampleOrderSpec
from ...config.schema import AuditTestId, LoadPatternType
from ...exceptions import SetupError
from .. import AuditRunArtifacts, AuditRunSpec
from ..result import AuditResult

if TYPE_CHECKING:
    from ...config.schema import AuditConfig, OutputCachingTestConfig
    from ...metrics.report import Report

# Only load patterns where achieved QPS is bounded by the SUT's own serving
# capacity, not by the load generator's arrival schedule. max_throughput
# (Offline) issues everything at once; concurrency keeps a fixed number of
# requests in flight, so QPS = concurrency / latency directly reflects SUT
# speed. A rate-paced pattern (poisson) pins the arrival rate independently
# of the SUT: if that rate is at or below what the SUT can already sustain,
# a caching-induced speedup never surfaces as higher completed QPS — both
# phases just complete at the pinned rate, masking the exact signal this
# audit exists to detect.
_VALID_LOAD_PATTERNS = (LoadPatternType.MAX_THROUGHPUT, LoadPatternType.CONCURRENCY)


@dataclass(frozen=True, slots=True)
class AuditRunStats:
    """Per-phase throughput stats for the output-caching QPS comparison."""

    qps: float
    n_completed: int
    n_requested: int

    @classmethod
    def from_report(cls, report: Report, n_requested: int) -> AuditRunStats:
        qps = report.qps
        if qps is None:
            raise ValueError("Report has no duration — cannot compute QPS")
        if qps <= 0:
            raise ValueError(
                f"Report has non-positive throughput (qps={qps}); the run "
                "completed no samples, so an output-caching comparison is impossible"
            )
        return cls(
            qps=qps, n_completed=report.n_samples_completed, n_requested=n_requested
        )


class OutputCachingAudit:
    """Output-caching audit (MLPerf TEST04)."""

    test_id: ClassVar[AuditTestId] = AuditTestId.OUTPUT_CACHING_TEST

    def plan_runs(self, cfg: AuditConfig) -> list[AuditRunSpec]:
        c: OutputCachingTestConfig = cfg  # type: ignore[assignment]
        # samples is required, so the reference phase always has an explicit
        # count; the audit phase defaults to the same count when omitted.
        ref_n = c.samples
        audit_n = c.audit_samples if c.audit_samples is not None else ref_n
        return [
            AuditRunSpec(
                label="reference",
                n_samples=ref_n,
                sample_order=SampleOrderSpec.without_replacement(),
            ),
            AuditRunSpec(
                label="output_caching",
                n_samples=audit_n,
                sample_order=SampleOrderSpec.single(c.sample_index),
            ),
        ]

    def validate(
        self, cfg: AuditConfig, dataset_size: int, load_pattern: LoadPatternType
    ) -> None:
        """Bounds-check the planned phases and load pattern.

        The reference phase draws distinct samples without replacement; if its
        count exceeds the dataset, the order wraps and re-issues samples, making
        the baseline partially cacheable and able to mask (or invert) a caching
        speedup. The audit phase pins one fixed index, which must exist — but its
        count may exceed the dataset, since it repeats that single sample.

        ``dataset_size`` is the performance dataset **as loaded** (i.e. after its
        own ``samples:`` truncation), not the underlying file's row count — a
        ``sample_index`` bound-checked here may still be out of range against
        the raw file if the dataset's ``samples:`` is smaller than the file.
        """
        if load_pattern not in _VALID_LOAD_PATTERNS:
            valid = ", ".join(p.value for p in _VALID_LOAD_PATTERNS)
            raise SetupError(
                f"Output-caching audit requires a load pattern where achieved "
                f"QPS reflects SUT capacity ({valid}); got {load_pattern.value!r}, "
                "which would mask a caching-induced speedup"
            )
        for spec in self.plan_runs(cfg):
            idx = spec.sample_order.fixed_index
            if idx is None:
                if spec.n_samples is not None and spec.n_samples > dataset_size:
                    raise SetupError(
                        f"Audit phase '{spec.label}': n_samples={spec.n_samples} "
                        f"exceeds dataset size {dataset_size}; a distinct-sample "
                        "phase would wrap and re-issue samples"
                    )
            elif not (0 <= idx < dataset_size):
                raise SetupError(
                    f"Audit phase '{spec.label}': sample_index={idx} is out of "
                    f"range [0, {dataset_size}) for dataset with {dataset_size} "
                    "samples"
                )

    def verify(self, runs: list[AuditRunArtifacts], cfg: AuditConfig) -> AuditResult:
        if len(runs) != 2:
            raise ValueError(
                "Output-caching verify expects exactly 2 phases (reference, "
                f"output_caching); got {len(runs)}"
            )
        c: OutputCachingTestConfig = cfg  # type: ignore[assignment]
        ref_arts, audit_arts = runs[0], runs[1]
        return verify_output_caching(
            AuditRunStats.from_report(ref_arts.report, ref_arts.n_requested),
            AuditRunStats.from_report(audit_arts.report, audit_arts.n_requested),
            threshold=c.threshold,
        )


def verify_output_caching(
    ref: AuditRunStats,
    audit: AuditRunStats,
    threshold: float = 0.10,
) -> AuditResult:
    """Core output-caching (MLPerf TEST04) result logic — pure function, no I/O.

    Pass iff:
      1. Each phase completed ≥ (1 - threshold) of its requested queries.
      2. audit_qps < ref_qps * (1 + threshold)
    """
    min_completion = 1.0 - threshold
    ref_ok = ref.n_completed >= ref.n_requested * min_completion
    audit_ok = audit.n_completed >= audit.n_requested * min_completion

    if not ref_ok or not audit_ok:
        passed = False
        reason = (
            f"Phase incomplete: reference {ref.n_completed}/{ref.n_requested}, "
            f"audit {audit.n_completed}/{audit.n_requested} "
            f"(threshold {threshold:.0%})"
        )
    else:
        # Matches upstream compliance/TEST04/verify_performance.py's strict `<`:
        # a run exactly on the boundary is not "faster" enough to pass.
        limit = ref.qps * (1.0 + threshold)
        passed = audit.qps < limit
        reason = (
            f"audit_qps={audit.qps:.4f} {'<' if passed else '>='} "
            f"ref_qps * (1 + {threshold:.0%}) = {limit:.4f}"
        )

    return AuditResult(
        test_id=AuditTestId.OUTPUT_CACHING_TEST.value,
        passed=passed,
        details={
            "ref_qps": ref.qps,
            "audit_qps": audit.qps,
            "threshold": threshold,
            "ref_completed": ref.n_completed,
            "ref_requested": ref.n_requested,
            "audit_completed": audit.n_completed,
            "audit_requested": audit.n_requested,
            "reason": reason,
        },
    )
