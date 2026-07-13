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

"""Compliance-audit configuration (the YAML-only ``audit:`` block)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import AuditTestId


class OutputCachingTestConfig(BaseModel):
    """Configuration for the output-caching audit (MLPerf TEST04).

    The output-caching test runs two back-to-back phases — a reference run of
    distinct samples and an audit run that repeats one fixed sample — then
    checks that the audit QPS does not exceed the reference QPS by more than
    ``threshold``. A large speedup indicates the SUT is caching responses.

    samples: reference-phase query count (required — an explicit count keeps
        the per-phase completion check meaningful; a duration-driven phase has
        no independent target to validate completion against)
    audit_samples: audit-phase query count (None → equals samples)
    sample_index: which dataset row is repeated (MLCommons performance_issue_same_index)
    threshold: tolerance shared by both pass checks — each phase must complete
        ≥ requested * (1 - threshold), and audit_qps must stay < ref_qps * (1 + threshold)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    test: Literal[AuditTestId.OUTPUT_CACHING_TEST]
    only: bool = Field(
        False,
        description="Run only the audit — skip the main benchmark (upstream-style standalone TEST04)",
    )
    samples: int = Field(..., ge=1, description="Reference phase query count")
    audit_samples: int | None = Field(
        None, ge=1, description="Audit phase query count (default: equals samples)"
    )
    sample_index: int = Field(
        0, ge=0, description="Dataset row index repeated in the audit phase"
    )
    threshold: float = Field(
        0.10,
        gt=0,
        lt=1,
        description=(
            "Tolerance for both checks: each phase must complete "
            "≥ requested * (1 - threshold), and audit_qps must stay "
            "< ref_qps * (1 + threshold)"
        ),
    )


# Single member today; becomes
# Annotated[OutputCachingTestConfig | ..., Field(discriminator="test")]
# when additional audit tests are added.
AuditConfig = OutputCachingTestConfig
