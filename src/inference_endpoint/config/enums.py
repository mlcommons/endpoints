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

"""Shared enums for the benchmark configuration schema."""

from __future__ import annotations

from enum import Enum


class LoadPatternType(str, Enum):
    """Load pattern types."""

    MAX_THROUGHPUT = "max_throughput"  # Offline: all queries at t=0
    POISSON = "poisson"  # Online: fixed QPS with Poisson distribution
    CONCURRENCY = "concurrency"  # Online: fixed concurrent requests
    AGENTIC_INFERENCE = (
        "agentic_inference"  # Agentic inference conversations with turn sequencing
    )
    BURST = "burst"  # Burst pattern (TODO)
    STEP = "step"  # Step pattern (TODO)


class OSLDistributionType(str, Enum):
    """Output Sequence Length distribution types."""

    ORIGINAL = "original"  # Use original distribution from dataset (default)
    FIXED = "fixed"  # Fixed length for all outputs
    UNIFORM = "uniform"  # Uniform distribution between min and max
    NORMAL = "normal"  # Normal/Gaussian distribution


class DatasetType(str, Enum):
    """Dataset purpose type."""

    PERFORMANCE = "performance"
    ACCURACY = "accuracy"


class EvalMethod(str, Enum):
    """Evaluation methods for accuracy testing."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    JUDGE = "judge"


class ScorerMethod(str, Enum):
    """Registered scorer methods for accuracy evaluation."""

    PASS_AT_1 = "pass_at_1"
    STRING_MATCH = "string_match"
    ROUGE = "rouge"
    CODE_BENCH = "code_bench_scorer"
    SHOPIFY_CATEGORY_F1 = "shopify_category_f1"
    AGENTIC_INFERENCE_INLINE = "agentic_inference_inline"
    VBENCH = "vbench"
    BFCL_V4 = "bfcl_v4"
    LEGACY_MLPERF_DEEPSEEK_R1 = "legacy_mlperf_deepseek_r1"


class AuditTestId(str, Enum):
    """Registered compliance audit test identifiers."""

    # Output-caching audit — MLPerf TEST04 (duplicate-query caching detection).
    OUTPUT_CACHING_TEST = "output_caching_test"


class TestMode(str, Enum):
    """Test mode determining what to collect.

    - PERF: Performance metrics only (no response storage)
    - ACC: Accuracy metrics (collect and evaluate responses)
    - BOTH: Both performance and accuracy (selective collection by dataset type)
    """

    PERF = "perf"
    ACC = "acc"
    BOTH = "both"


class StreamingMode(str, Enum):
    """Streaming mode for response handling.

    - AUTO: Automatically enable for online mode, disable for offline mode
    - ON: Force streaming enabled (for TTFT metrics)
    - OFF: Force streaming disabled
    """

    AUTO = "auto"
    ON = "on"
    OFF = "off"


class TestType(str, Enum):
    """Test type for both config classification and execution mode.

    - OFFLINE: Max throughput benchmark (all queries at t=0)
    - ONLINE: Sustained QPS benchmark (Poisson or concurrency-based)
    - EVAL: Accuracy evaluation
    - SUBMISSION: Official submission (may include both perf and accuracy)
    """

    OFFLINE = "offline"
    ONLINE = "online"
    EVAL = "eval"
    SUBMISSION = "submission"


class ProfilerEngine(str, Enum):
    """Inference engine whose profiling protocol the client should drive.

    Selects the HTTP path layout used to derive start/stop URLs from
    ``endpoint_config.endpoints``. Each value corresponds to one server-side
    profiling protocol; add a new variant + ``_PROFILE_PATHS`` row to support
    another engine.
    """

    VLLM = "vllm"
