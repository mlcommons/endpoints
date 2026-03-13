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

"""Configuration classes for HTTP endpoint client."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal

from inference_endpoint.async_utils.transport.protocol import WorkerPoolTransport
from inference_endpoint.config.schema import APIType

from .accumulator_protocol import SSEAccumulatorProtocol
from .adapter_protocol import HttpRequestAdapter
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


@dataclass
class HTTPClientConfig:
    """Configuration for the HTTP endpoint client."""

    endpoint_urls: list[str]
    api_type: APIType = APIType.OPENAI
    api_key: str | None = None

    # Number of worker processes (-1 for automatic detection)
    #   - -1 for "auto": min(max(8, loadgen_numa_domain_size - 1), 24)
    num_workers: int = -1

    record_worker_events: bool = False
    event_logs_dir: Path | None = None
    log_level: str = "INFO"

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

    # CPU affinity plan for worker processes (computed by caller, e.g. benchmark command).
    # None = disabled (no worker pinning)
    cpu_affinity: AffinityPlan | None = None

    # Worker lifecycle timeouts
    worker_initialization_timeout: float = 40.0  # init
    worker_graceful_shutdown_wait: float = 0.5  # post-run
    worker_force_kill_timeout: float = 0.5  # post-run

    # Connection idle timeout - discard connections idle longer than this.
    # Two fold benefits:
    # 1. Prevents keep-alive race condition where server closes idle connection
    #    at the exact moment client sends a new request (half-closed TCP).
    # 2. Early discard connections which are likely disconnected by the server already
    max_idle_time: float = 4.0  # seconds

    # Pre-establish TCP connections during init for reuse at runtime.
    # Reduces p99/max latency from cold-start connections.
    #
    # Values:
    #   -1 = auto (50% of pool, safe default - 100% can overwhelm some servers)
    #    0 = disabled
    #   >0 = explicit total connection count to warmup (split across workers)
    warmup_connections: int = -1

    # Maximum concurrent TCP connections.
    # Performance sweetspot is often a low number compared to port limit ~1024.
    #
    # Values:
    #   - >0 = explicit max size of TCP connection pool
    #   - -1: unlimited (bound by system ephemeral_port_limit)
    max_connections: int = -1

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

    # Request adapter for Query/Response <-> Payload/Response bytes
    # Default in __post_init__ if None
    adapter: type[HttpRequestAdapter] = None  # type: ignore[assignment]

    # SSE accumulator for streaming responses
    # Default in __post_init__ if None
    accumulator: type[SSEAccumulatorProtocol] = None  # type: ignore[assignment]

    # Worker pool transport class for worker IPC
    # Default in __post_init__ if None
    worker_pool_transport: type[WorkerPoolTransport] = None  # type: ignore[assignment]

    def __post_init__(self):
        # set default adapter in __post_init__ to avoid circular dependency
        if isinstance(self.api_type, str):
            self.api_type = APIType(self.api_type)

        if self.num_workers == -1:
            self.num_workers = _get_auto_num_workers()

        if self.adapter is None:
            adapter_path = ADAPTER_MAP.get(self.api_type)
            if not adapter_path:
                raise ValueError(f"Invalid or unsupported API type: {self.api_type}")

            module_path, class_name = adapter_path.rsplit(".", 1)
            module = import_module(module_path)
            self.adapter = getattr(module, class_name)

        if self.accumulator is None:
            # Default to OpenAI accumulator for unrecognized API types
            accumulator_path = ACCUMULATOR_MAP.get(
                self.api_type, ACCUMULATOR_MAP[APIType.OPENAI]
            )
            module_path, class_name = accumulator_path.rsplit(".", 1)
            module = import_module(module_path)
            self.accumulator = getattr(module, class_name)

        if self.worker_pool_transport is None:
            # Default to ZMQ worker pool transport
            from inference_endpoint.async_utils.transport import ZmqWorkerPoolTransport

            self.worker_pool_transport = ZmqWorkerPoolTransport

        low, high = get_ephemeral_port_range()
        system_maximum_ports = high - low + 1
        available_ports = get_ephemeral_port_limit()

        if self.max_connections == -1:
            # Auto: use available ephemeral ports
            self.max_connections = available_ports

            # Resolve min_required_connections: -1 means auto (12.5% of system max)
            if self.min_required_connections == -1:
                self.min_required_connections = int(system_maximum_ports * 0.125)
        else:
            # User specified explicit max_connections - validate against port limit
            if self.max_connections > available_ports:
                raise RuntimeError(
                    f"--max-connections ({self.max_connections}) exceeds ephemeral port limit ({available_ports}). "
                    f"Either reduce --max-connections or increase system port limit."
                )


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
