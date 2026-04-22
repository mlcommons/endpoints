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

import functools
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Literal

import cyclopts
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inference_endpoint.async_utils.transport import AnyTransportConfig
from inference_endpoint.async_utils.transport.protocol import TransportConfig
from inference_endpoint.core.types import APIType
from inference_endpoint.utils import WithUpdatesMixin

from .accumulator_protocol import SSEAccumulatorProtocol
from .adapter_protocol import HttpRequestAdapter
from .cpu_affinity import (
    AffinityPlan,
    UnsupportedPlatformError,
    get_cpus_in_numa_node,
    get_current_numa_node,
)
from .utils import get_ephemeral_port_limit, get_ephemeral_port_range

ADAPTER_MAP = {
    APIType.OPENAI: "inference_endpoint.openai.openai_msgspec_adapter.OpenAIMsgspecAdapter",
    APIType.SGLANG: "inference_endpoint.sglang.adapter.SGLangGenerateAdapter",
}

ACCUMULATOR_MAP = {
    APIType.OPENAI: "inference_endpoint.openai.accumulator.OpenAISSEAccumulator",
    APIType.SGLANG: "inference_endpoint.sglang.accumulator.SGLangSSEAccumulator",
}


class HTTPClientConfig(WithUpdatesMixin, BaseModel):
    """HTTP endpoint client configuration.

    User-facing fields are exposed to CLI/YAML via cyclopts.
    Internal fields use ``parse=False`` — set programmatically only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    # =========================================================================
    # User-facing fields (exposed to CLI/YAML)
    # =========================================================================

    num_workers: Annotated[
        int,
        cyclopts.Parameter(
            alias=["--workers", "--num-workers"],
            help="Worker processes (-1=auto)",
        ),
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
        -1, ge=-1, description="Pre-establish TCP connections (-1=auto, 0=disabled)"
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
    ] = Field(-1, ge=-1)

    # Transport configuration
    transport: AnyTransportConfig = Field(
        default_factory=TransportConfig.create_default
    )

    # WARNING: Use with caution
    # Can cause large performance overhead on main-thread (user / Loadgen)
    #
    # When enabled, all chunks will be made available via get_ready_responses() ASAP
    # When disabled, only first chunk of every response will arrive via get_ready_responses()
    #
    # NOTE:
    #   - StreamChunk.metadata['first_chunk'] is set for first chunk of every response
    #   - At end of stream, QueryResult is returned with the entire response content
    stream_all_chunks: bool = Field(
        False, description="Stream all chunks to main thread (caution: perf overhead)"
    )

    # Worker lifecycle timeouts
    worker_initialization_timeout: float = Field(
        60.0, description="Worker init timeout (seconds)"
    )
    worker_graceful_shutdown_wait: float = Field(
        0.5, description="Post-run graceful shutdown wait (seconds)"
    )
    worker_force_kill_timeout: float = Field(
        0.5, description="Force kill timeout after graceful wait (seconds)"
    )

    # Connection idle timeout - discard connections idle longer than this.
    # Two fold benefits:
    # 1. Prevents keep-alive race condition where server closes idle connection
    #    at the exact moment client sends a new request (half-closed TCP).
    # 2. Early discard connections which are likely disconnected by the server already
    max_idle_time: float = Field(
        4.0, description="Discard connections idle longer than this (seconds)"
    )

    # Minimum required connections for http-client to initialize.
    # Will log warning if not enough ephemeral ports are available during warmup.
    #
    # Values:
    #   - >0 = explicit minimum required connections
    #   - 0 = disable check (no warning if ports unavailable)
    #   - -1 = auto (defaults to 12.5% of system ephemeral port range)
    min_required_connections: int = Field(
        -1, description="Min connections to initialize (-1=auto, 0=disabled)"
    )

    # GC strategy for worker processes to reduce latency spikes from collection pauses
    #
    # Values:
    #   - "disabled": GC completely disabled (risky for long-running benchmarks)
    #   - "relaxed": GC enabled with 50x higher threshold (less aggressive)
    #   - "system": Standard Python GC with default thresholds
    worker_gc_mode: Literal["disabled", "relaxed", "system"] = Field(
        "relaxed", description="Worker GC strategy"
    )

    # =========================================================================
    # Internal fields (parse=False — set programmatically, not via CLI/YAML)
    # =========================================================================

    endpoint_urls: Annotated[list[str], cyclopts.Parameter(parse=False)] = Field(
        default_factory=list, exclude=True
    )
    api_type: Annotated[APIType, cyclopts.Parameter(parse=False)] = Field(
        default=APIType.OPENAI, exclude=True
    )
    api_key: Annotated[str | None, cyclopts.Parameter(parse=False)] = Field(
        default=None, exclude=True
    )

    event_logs_dir: Annotated[Path | None, cyclopts.Parameter(parse=False)] = Field(
        default=None, exclude=True
    )

    # CPU affinity plan for worker processes (computed by caller, e.g. benchmark command).
    # None = disabled (no worker pinning)
    cpu_affinity: Annotated[AffinityPlan | None, cyclopts.Parameter(parse=False)] = (
        Field(default=None, exclude=True)
    )

    # Request adapter for Query/Response <-> Payload/Response bytes
    # Resolved from api_type in _resolve_defaults validator
    adapter: Annotated[
        type[HttpRequestAdapter] | None, cyclopts.Parameter(parse=False)
    ] = Field(default=None, exclude=True)

    # SSE accumulator for streaming responses
    # Resolved from api_type in _resolve_defaults validator
    accumulator: Annotated[
        type[SSEAccumulatorProtocol] | None, cyclopts.Parameter(parse=False)
    ] = Field(default=None, exclude=True)

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("num_workers", "max_connections")
    @classmethod
    def _resolve_zeros(cls, v: int, info: Any) -> int:
        if v == 0:
            raise ValueError(f"{info.field_name} must be -1 (auto) or >= 1, got 0")
        return v

    @model_validator(mode="after")
    def _resolve_defaults(self) -> HTTPClientConfig:
        """Resolve auto-detect values and lazy defaults."""
        if isinstance(self.api_type, str):
            object.__setattr__(self, "api_type", APIType(self.api_type))

        if self.num_workers == -1:
            object.__setattr__(self, "num_workers", _get_auto_num_workers())

        if self.adapter is None:
            adapter_path = ADAPTER_MAP.get(self.api_type)
            if not adapter_path:
                raise ValueError(f"Invalid or unsupported API type: {self.api_type}")
            module_path, class_name = adapter_path.rsplit(".", 1)
            module = import_module(module_path)
            object.__setattr__(self, "adapter", getattr(module, class_name))

        if self.accumulator is None:
            accumulator_path = ACCUMULATOR_MAP.get(
                self.api_type, ACCUMULATOR_MAP[APIType.OPENAI]
            )
            module_path, class_name = accumulator_path.rsplit(".", 1)
            module = import_module(module_path)
            object.__setattr__(self, "accumulator", getattr(module, class_name))

        # Only resolve ports when endpoint_urls are set (runtime config, not settings default)
        if self.endpoint_urls:
            low, high = get_ephemeral_port_range()
            system_maximum_ports = high - low + 1
            available_ports = get_ephemeral_port_limit()

            if self.max_connections == -1:
                object.__setattr__(self, "max_connections", available_ports)
            elif self.max_connections > 0:
                if self.max_connections > available_ports:
                    raise RuntimeError(
                        f"--max-connections ({self.max_connections}) exceeds ephemeral port limit ({available_ports}). "
                        f"Either reduce --max-connections or increase system port limit."
                    )

            if self.min_required_connections == -1:
                object.__setattr__(
                    self, "min_required_connections", int(system_maximum_ports * 0.125)
                )

        return self


@functools.lru_cache(maxsize=1)
def _get_auto_num_workers() -> int:
    """
    Compute optimal number of workers based on NUMA topology.

    Defaults to NUMA domain size (min 10, max 24) for optimal memory locality.
    Users can override with explicit num_workers to use more cores (workers
    will be pinned to additional cores outside NUMA domain if needed).

    On non-Linux platforms (NUMA probing is Linux-only) falls back to
    ``min_workers`` so the config can still be constructed for local
    development, template regeneration, and tests.

    Returns:
        Number of workers to use when num_workers is -1 (auto).
    """
    min_workers = 10
    max_workers = 24

    try:
        numa_node = get_current_numa_node()
        if numa_node is None:
            return min_workers
        numa_cpus = get_cpus_in_numa_node(numa_node)
    except UnsupportedPlatformError:
        return min_workers

    if not numa_cpus:
        return min_workers

    return min(max(min_workers, len(numa_cpus)), max_workers)


__all__ = ["HTTPClientConfig"]
