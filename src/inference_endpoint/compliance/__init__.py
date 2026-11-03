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

"""Compliance for benchmark runs: audit tests + submission checker.

Audit framework: AuditTest protocol + AuditRunSpec/AuditRunArtifacts types +
test registry, run via ``commands/audit.py:run_audit``.

Submission checker (``checker.py``): validates a completed run's report
directory against a registered ruleset — config-lock (deterministic/
single-stream settings), the accuracy gate, and run-validity rules (e.g. 0
dropped turns for the agentic performance run).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from ..config.schema import TestMode
from ..exceptions import SetupError
from .result import AuditResult

if TYPE_CHECKING:
    from ..config.runtime_settings import SampleOrderSpec
    from ..config.schema import AuditConfig, AuditTestId, LoadPatternType
    from ..metrics.report import Report


@dataclass(frozen=True, slots=True)
class AuditRunSpec:
    """Declarative description of one audit phase.

    ``n_samples = None`` means "issue the benchmark's default count" (full
    dataset / duration-driven) — it flows through to
    ``RuntimeSettings.n_samples_to_issue`` unchanged.

    ``test_mode`` lets a phase opt into accuracy scoring (``ACC``/``BOTH``)
    instead of the ``PERF``-only default — the orchestrator reads this
    per-phase rather than hardcoding ``TestMode.PERF`` for every audit test.
    """

    label: str
    n_samples: int | None
    sample_order: SampleOrderSpec
    test_mode: TestMode = TestMode.PERF


@dataclass(frozen=True, slots=True)
class AuditRunArtifacts:
    """Collected output of one audit phase — passed to AuditTest.verify()."""

    label: str
    report_dir: Path
    report: Report
    n_requested: int


class AuditTest(Protocol):
    test_id: ClassVar[AuditTestId]

    def plan_runs(self, cfg: AuditConfig) -> list[AuditRunSpec]: ...

    def verify(
        self, runs: list[AuditRunArtifacts], cfg: AuditConfig
    ) -> AuditResult: ...

    def validate(
        self, cfg: AuditConfig, dataset_size: int, load_pattern: LoadPatternType
    ) -> None:
        """Raise SetupError if cfg is invalid for a dataset_size / load_pattern.

        Owns every precondition that is specific to this audit (e.g. which
        sample counts/indices its phases can use, which load patterns produce
        a meaningful comparison) so the generic orchestrator stays test-agnostic.
        """
        ...


def get_audit_test(test_id: AuditTestId) -> AuditTest:
    try:
        return AUDIT_TESTS[test_id]
    except KeyError:
        raise SetupError(f"No audit test registered for '{test_id.value}'") from None


# Implementations are imported at module end — after the protocol and dataclasses
# above are defined — because each audit module imports those names from this
# package; importing at the top would be circular. This is module-scope wiring,
# not a lazy in-function import.
from .audit_test.output_caching_test import OutputCachingAudit  # noqa: E402
from .checker import (  # noqa: E402
    Check,
    ComplianceReport,
    check_accuracy,
    check_config_lock,
    check_perf_validity,
    check_submission,
)

AUDIT_TESTS: dict[AuditTestId, AuditTest] = {
    OutputCachingAudit.test_id: OutputCachingAudit(),
}


__all__ = [
    "AuditTest",
    "AuditResult",
    "AuditRunArtifacts",
    "AuditRunSpec",
    "AUDIT_TESTS",
    "get_audit_test",
    "Check",
    "ComplianceReport",
    "check_accuracy",
    "check_config_lock",
    "check_perf_validity",
    "check_submission",
]
