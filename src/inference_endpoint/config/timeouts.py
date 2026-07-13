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

"""All global time knobs in one frozen model — durations and deadlines.

Two disjoint categories live here, and they must not be conflated:

- **Workload durations** (``min_duration_ms``/``max_duration_ms``): part of
  the benchmark definition — they shape sample-count math and bound the
  performance phase only. Reaching them is a *normal* end of the run.
- **Give-up deadlines** (everything else): failure handling. Reaching one
  means something is stuck. ``run_timeout_s`` is the whole-run watchdog:
  when it fires the run is aborted and the report is marked INTERRUPTED —
  it never derives or caps the per-stage deadlines below it.

``None`` means "wait indefinitely" for every optional deadline. Deadlines
are resolved to plain floats before the run starts; nothing in the hot
path reads this model.

Dataset-scoped time knobs (e.g. agentic ``turn_timeout_s``) stay in their
dataset config blocks — they are per-workload behavior, not global.
"""

from __future__ import annotations

from typing import Annotated, Self

import cyclopts
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils import WithUpdatesMixin


@cyclopts.Parameter(name="*")
class Timeouts(WithUpdatesMixin, BaseModel):
    """Global durations and deadlines (see module docstring for semantics)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Workload durations (benchmark definition, not failure handling) ---
    min_duration_ms: Annotated[
        int,
        cyclopts.Parameter(
            alias="--duration", help="Min duration (ms, or with suffix: 600s, 10m)"
        ),
    ] = Field(600000, ge=0)
    max_duration_ms: int = Field(
        0,
        ge=0,
        description="Maximum performance-phase duration in ms (0 for no limit)",
    )

    # --- Whole-run watchdog ---
    run_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--timeout",
            help=(
                "Whole-run watchdog in seconds (None = off). Firing aborts the "
                "run and marks the report INTERRUPTED."
            ),
        ),
    ] = Field(
        None,
        gt=0,
        description=(
            "Whole-run watchdog in seconds (None = off). Covers every phase "
            "including drains; firing aborts the run, marks the report "
            "INTERRUPTED, and exits non-zero. Never derives per-stage deadlines."
        ),
    )

    # --- Startup deadlines ---
    service_ready_timeout_s: Annotated[
        float,
        cyclopts.Parameter(
            alias="--service-ready-timeout",
            help="Seconds to wait for metrics/event-logger services to start",
        ),
    ] = Field(
        30.0,
        ge=0,
        description="Seconds to wait for metrics-aggregator/event-logger services to become ready.",
    )
    # --- Drain deadlines (None = wait indefinitely) ---
    warmup_drain_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--warmup-drain-timeout",
            help="Warmup drain timeout in seconds (None = wait indefinitely)",
        ),
    ] = Field(
        240.0,
        gt=0,
        description="Warmup drain timeout in seconds (None = wait indefinitely)",
    )
    performance_drain_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--performance-drain-timeout",
            help="Performance drain timeout in seconds (None = wait indefinitely)",
        ),
    ] = Field(
        240.0,
        gt=0,
        description="Performance drain timeout in seconds (None = wait indefinitely)",
    )
    accuracy_drain_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--accuracy-drain-timeout",
            help="Accuracy drain timeout in seconds (None = wait indefinitely)",
        ),
    ] = Field(
        None,
        gt=0,
        description=(
            "Accuracy drain timeout in seconds (None = wait indefinitely; "
            "accuracy is unbounded by default because every sample must complete)"
        ),
    )
    metrics_drain_timeout_s: Annotated[
        float | None,
        cyclopts.Parameter(
            alias="--metrics-drain-timeout",
            help=(
                "Wall-clock budget (seconds) for the metrics aggregator to finish "
                "tokenizing buffered samples after the run ends "
                "(None = wait indefinitely)"
            ),
        ),
    ] = Field(
        None,
        gt=0,
        description=(
            "Wall-clock budget (seconds) to finish tokenizing buffered samples "
            "after ENDED (None = wait indefinitely). An incomplete drain is "
            "surfaced via n_pending_tasks > 0, never silently dropped."
        ),
    )

    @field_validator("min_duration_ms", "max_duration_ms", mode="before")
    @classmethod
    def _parse_duration_suffix(cls, v: object) -> object:
        """Accept duration with unit suffix: 600s, 10m, 600000ms, or plain int (ms)."""
        if isinstance(v, str):
            v = v.strip()
            if v.endswith("ms"):
                return int(v[:-2])
            if v.endswith("m"):
                return int(float(v[:-1]) * 60_000)
            if v.endswith("s"):
                return int(float(v[:-1]) * 1000)
        return v

    @model_validator(mode="after")
    def _validate_durations(self) -> Self:
        if self.max_duration_ms != 0 and self.max_duration_ms < self.min_duration_ms:
            raise ValueError(
                f"max_duration_ms ({self.max_duration_ms}) must be >= "
                f"min_duration_ms ({self.min_duration_ms})"
            )
        return self
