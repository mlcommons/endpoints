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

"""Global waits and deadlines (the ``settings.timeouts`` block).

Split criterion: one module per config domain; every global time knob that
bounds how long the harness waits — startup readiness, per-phase drains, the
worker lifecycle, and the whole-run watchdog — lives here. Workload durations
(``runtime.max_duration_ms``) are part of the benchmark definition, not waits,
and stay in ``runtime``. Dataset-scoped time knobs (e.g. agentic
``turn_timeout_s``) stay in their dataset config blocks.
"""

from __future__ import annotations

from typing import Annotated

import cyclopts
from pydantic import BaseModel, ConfigDict, Field

from ..utils import WithUpdatesMixin


@cyclopts.Parameter(name="*")
class Timeouts(WithUpdatesMixin, BaseModel):
    """All global waits and deadlines. ``None`` = wait indefinitely / off.

    Reaching an optional deadline means something is stuck; ``run_timeout_s``
    is the whole-run watchdog — when it fires the run is aborted and the
    report is marked INTERRUPTED. It never derives or caps the other
    deadlines. Workload durations (``runtime.max_duration_ms``) are NOT
    timeouts and do not live here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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
        None,
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
            "after ENDED (None = wait indefinitely). An incomplete drain fails "
            "the run: artifacts are written with complete: false, then "
            "run_benchmark exits non-zero."
        ),
    )
    worker_initialization_timeout_s: float = Field(
        60.0, ge=0, description="Endpoint-client worker init timeout (seconds)"
    )
    worker_graceful_shutdown_wait_s: float = Field(
        0.5,
        ge=0,
        description="Endpoint-client post-run graceful shutdown wait (seconds)",
    )
    worker_force_kill_timeout_s: float = Field(
        0.5,
        ge=0,
        description="Endpoint-client force kill timeout after graceful wait (seconds)",
    )
