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

"""HTTP endpoint client configuration.

Single Pydantic model for both CLI/YAML (via cyclopts) and runtime.
Internal fields use ``cyclopts.Parameter(parse=False)`` so they are
invisible to the parser but can be set programmatically.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Literal

import cyclopts
from pydantic import BaseModel, Field, field_validator, model_validator

from inference_endpoint.async_utils.transport.zmq import ZMQTransportConfig
from inference_endpoint.core.types import APIType

from .cpu_affinity import AffinityPlan, get_cpus_in_numa_node, get_current_numa_node
from .utils import get_ephemeral_port_limit, get_ephemeral_port_range

ADAPTER_MAP = {
    APIType.OPENAI: "inference_endpoint.openai.openai_msgspec_adapter.OpenAIMsgspecAdapter",
    APIType.SGLANG: "inference_endpoint.sglang.adapter.SGLangGenerateAdapter",
}

ACCUMULATOR_MAP = {
    APIType.OPENAI: "inference_endpoint.openai.accumulator.OpenAISSEAccumulator",
    APIType.SGLANG: "inference_endpoint.sglang.accumulator.SGLangSSEAccumulator",
}


class HTTPClientConfig(BaseModel):
    """HTTP endpoint client configuration.

    User-facing fields are exposed to CLI/YAML via cyclopts.
    Internal fields use ``parse=False`` — set programmatically only.
    """

    model_config = {"extra": "forbid"}

    # =========================================================================
    # User-facing fields (exposed to CLI/YAML)
    # =========================================================================

    workers: Annotated[
        int, cyclopts.Parameter(alias="--workers", help="Worker processes (-1=auto)")
    ] = Field(-1, ge=-1)

    record_worker_events: bool = Field(False, description="Record per-worker events")
    log_level: str = Field("INFO", description="Worker log level")

    # Pre-establish TCP connections during init for reuse at runtime.
    # Reduces p99/max latency from cold-start connections.
    #
    # Values:
    #   -1 = auto (50% of pool, safe default - 100% can overwhelm some servers)
    #    0 = disabled
    #   >0 = explicit total connection count to warmup (split across workers)
    warmup_connections: int = Field(
        -1, description="Pre-establish TCP connections (-1=auto, 0=disabled)"
    )

    # Maximum concurrent TCP connections.
    # Performance sweetspot is often a low number compared to port limit ~1024.
    #
    # Values:
    #   - >0 = explicit max size of TCP connection pool
    #   - -1: unlimited (bound by system ephemeral_port_limit)
    max_connections: Annotated[
        int,
        cyclopts.Parameter(
            alias="--max-connections", help="Max TCP connections (-1=unlimited)"
        ),
    ] = -1

    # Transport-specific configuration.
    # When adding new transports, convert to discriminated union on ``type``.
    transport: ZMQTransportConfig | None = None

    # WARNING: Use with caution
    # Can cause large performance overhead on main-thread (user / Loadgen)
    #
    # When enabled, all chunks will be made available via get_ready_responses() ASAP
    # When disabled, only first chunk of every response will arrive via get_ready_responses()
    #
    # NOTE:
    #   - StreamChunk.metadata['first_chunk'] is set for first chunk of every response
    #   - At end of stream, QueryResult is returned with the entire response content
    stream_all_chunks: bool = False

    # Worker lifecycle timeouts
    worker_initialization_timeout: float = 60.0  # init
    worker_graceful_shutdown_wait: float = 0.5  # post-run
    worker_force_kill_timeout: float = 0.5  # post-run

    # Connection idle timeout - discard connections idle longer than this.
    # Two fold benefits:
    # 1. Prevents keep-alive race condition where server closes idle connection
    #    at the exact moment client sends a new request (half-closed TCP).
    # 2. Early discard connections which are likely disconnected by the server already
    max_idle_time: float = 4.0  # seconds

    # Minimum required connections for http-client to initialize.
    # Will log warning if not enough ephemeral ports are available during warmup.
    #
    # Values:
    #   - >0 = explicit minimum required connections
    #   - 0 = disable check (no warning if ports unavailable)
    #   - -1 = auto (defaults to 12.5% of system ephemeral port range)
    min_required_connections: int = -1

    # GC strategy for worker processes to reduce latency spikes from collection pauses
    #
    # Values:
    #   - "disabled": GC completely disabled (risky for long-running benchmarks)
    #   - "relaxed": GC enabled with 50x higher threshold (less aggressive)
    #   - "system": Standard Python GC with default thresholds
    worker_gc_mode: Literal["disabled", "relaxed", "system"] = "relaxed"

    # =========================================================================
    # Internal fields (parse=False — set programmatically, not via CLI/YAML)
    # =========================================================================

    endpoint_urls: Annotated[list[str], cyclopts.Parameter(parse=False)] = Field(
        default_factory=list
    )
    api_type: Annotated[APIType, cyclopts.Parameter(parse=False)] = APIType.OPENAI
    api_key: Annotated[str | None, cyclopts.Parameter(parse=False)] = None

    event_logs_dir: Annotated[Path | None, cyclopts.Parameter(parse=False)] = None

    # CPU affinity plan for worker processes (computed by caller, e.g. benchmark command).
    # None = disabled (no worker pinning)
    cpu_affinity: Annotated[AffinityPlan | None, cyclopts.Parameter(parse=False)] = (
        Field(default=None, exclude=True)
    )

    # Request adapter for Query/Response <-> Payload/Response bytes
    # Resolved from api_type in _resolve_defaults validator
    adapter: Annotated[Any, cyclopts.Parameter(parse=False)] = Field(
        default=None, exclude=True
    )

    # SSE accumulator for streaming responses
    # Resolved from api_type in _resolve_defaults validator
    accumulator: Annotated[Any, cyclopts.Parameter(parse=False)] = Field(
        default=None, exclude=True
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("workers")
    @classmethod
    def _workers_not_zero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("workers must be -1 (auto) or >= 1, got 0")
        return v

    @model_validator(mode="after")
    def _resolve_defaults(self) -> HTTPClientConfig:
        """Resolve auto-detect values and lazy defaults."""
        if isinstance(self.api_type, str):
            self.api_type = APIType(self.api_type)

        if self.workers == -1:
            self.workers = _get_auto_num_workers()

        if self.adapter is None:
            adapter_path = ADAPTER_MAP.get(self.api_type)
            if not adapter_path:
                raise ValueError(f"Invalid or unsupported API type: {self.api_type}")
            module_path, class_name = adapter_path.rsplit(".", 1)
            module = import_module(module_path)
            self.adapter = getattr(module, class_name)

        if self.accumulator is None:
            accumulator_path = ACCUMULATOR_MAP.get(
                self.api_type, ACCUMULATOR_MAP[APIType.OPENAI]
            )
            module_path, class_name = accumulator_path.rsplit(".", 1)
            module = import_module(module_path)
            self.accumulator = getattr(module, class_name)

        if self.transport is None:
            from inference_endpoint.async_utils.transport.zmq import (
                ZMQTransportConfig,
            )

            self.transport = ZMQTransportConfig()

        # Only resolve ports when endpoint_urls are set (runtime config, not settings default)
        if self.endpoint_urls:
            low, high = get_ephemeral_port_range()
            system_maximum_ports = high - low + 1
            available_ports = get_ephemeral_port_limit()

            if self.max_connections == -1:
                self.max_connections = available_ports
            elif self.max_connections > 0:
                if self.max_connections > available_ports:
                    raise RuntimeError(
                        f"--max-connections ({self.max_connections}) exceeds ephemeral port limit ({available_ports}). "
                        f"Either reduce --max-connections or increase system port limit."
                    )

            if self.min_required_connections == -1:
                self.min_required_connections = int(system_maximum_ports * 0.125)

        return self


def _get_auto_num_workers() -> int:
    """
    Compute optimal number of workers based on NUMA topology.

    Defaults to NUMA domain size (min 10, max 24) for optimal memory locality.
    Users can override with explicit num_workers to use more cores (workers
    will be pinned to additional cores outside NUMA domain if needed).

    Returns:
        Number of workers to use when num_workers is -1 (auto).
    """
    min_workers = 10
    max_workers = 24

    numa_node = get_current_numa_node()
    if numa_node is None:
        return min_workers

    numa_cpus = get_cpus_in_numa_node(numa_node)
    if not numa_cpus:
        return min_workers

    return min(max(min_workers, len(numa_cpus)), max_workers)


__all__ = ["HTTPClientConfig"]
