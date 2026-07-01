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

"""Compliance audit framework.

AuditTest protocol + AuditRunSpec/AuditRunStats/AuditRunArtifacts types + test registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from ..metrics.report import Report
from .result import AuditResult

if TYPE_CHECKING:
    from ..config.runtime_settings import SampleOrderSpec
    from ..config.schema import AuditConfig, AuditTestId


@dataclass(frozen=True, slots=True)
class AuditRunSpec:
    """Declarative description of one audit phase.

    ``n_samples = None`` means "issue the benchmark's default count" (full
    dataset / duration-driven) — it flows through to
    ``RuntimeSettings.n_samples_to_issue`` unchanged.
    """

    label: str
    n_samples: int | None
    sample_order: SampleOrderSpec


@dataclass(frozen=True, slots=True)
class AuditRunStats:
    """Per-phase throughput stats consumed by AuditTest.verify()."""

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


@dataclass(frozen=True, slots=True)
class AuditRunArtifacts:
    """Collected output of one audit phase — passed to AuditTest.verify()."""

    label: str
    report_dir: Path
    report: Report
    n_requested: int

    def stats(self) -> AuditRunStats:
        return AuditRunStats.from_report(self.report, self.n_requested)


class AuditTest(Protocol):
    test_id: ClassVar[AuditTestId]

    def plan_runs(self, cfg: AuditConfig) -> list[AuditRunSpec]: ...

    def verify(
        self, runs: list[AuditRunArtifacts], cfg: AuditConfig
    ) -> AuditResult: ...


_REGISTRY: dict[str, AuditTest] = {}


def register(test: AuditTest) -> None:
    # Key on the enum .value (e.g. "output_caching_test"); str() on a (str, Enum)
    # member yields "AuditTestId.OUTPUT_CACHING_TEST", not the value.
    _REGISTRY[test.test_id.value] = test


def get_audit_test(test_id: AuditTestId) -> AuditTest:
    key = test_id.value
    if key not in _REGISTRY:
        raise KeyError(f"No audit test registered for '{key}'")
    return _REGISTRY[key]


__all__ = [
    "AuditTest",
    "AuditResult",
    "AuditRunArtifacts",
    "AuditRunSpec",
    "AuditRunStats",
    "get_audit_test",
    "register",
]

# Import audit-test implementations so their register(...) calls populate
# _REGISTRY on package import. Kept at the bottom — after the names above are
# defined — because each test module imports AuditRunSpec/register from here;
# a top-of-file import would be circular.
from . import audit_test  # noqa: E402, F401
