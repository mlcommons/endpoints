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

"""Async load generator for the MLPerf Inference Endpoint Benchmarking System.

See docs/load_generator/design.md for the full design.
"""

from .delay import make_delay_fn, poisson_delay_fn
from .sample_order import (
    SampleOrder,
    WithoutReplacementSampleOrder,
    WithReplacementSampleOrder,
    create_sample_order,
)
from .session import (
    BenchmarkSession,
    PhaseConfig,
    PhaseIssuer,
    PhaseResult,
    PhaseType,
    SessionResult,
)
from .strategy import (
    BurstStrategy,
    ConcurrencyStrategy,
    LoadStrategy,
    TimedIssueStrategy,
    create_load_strategy,
)

__all__ = [
    # New async API
    "BenchmarkSession",
    "PhaseConfig",
    "PhaseType",
    "PhaseResult",
    "SessionResult",
    "PhaseIssuer",
    "LoadStrategy",
    "TimedIssueStrategy",
    "BurstStrategy",
    "ConcurrencyStrategy",
    "create_load_strategy",
    "SampleOrder",
    "WithoutReplacementSampleOrder",
    "WithReplacementSampleOrder",
    "create_sample_order",
    "make_delay_fn",
    "poisson_delay_fn",
]
