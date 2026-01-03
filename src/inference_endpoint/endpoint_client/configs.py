# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import socket
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import aiohttp
import zmq

from ..config.schema import APIType
from .adapter_protocol import HttpRequestAdapter

ADAPTER_MAP = {
    APIType.OPENAI: "inference_endpoint.openai.openai_msgspec_adapter.OpenAIMsgspecAdapter",
    APIType.SGLANG: "inference_endpoint.sglang.adapter.SGLangGenerateAdapter",
}


@dataclass
class HTTPClientConfig:
    """Configuration for the HTTP endpoint client."""

    endpoint_url: str
    api_type: APIType = APIType.OPENAI
    num_workers: int = 4

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

    # Worker lifecycle timeouts
    worker_initialization_timeout: float = (
        120.0  # initialize and warmup TCP connections
    )
    worker_graceful_shutdown_wait: float = 0.5  # post-run
    worker_force_kill_timeout: float = 0.5  # post-run

    # Pre-Open TCP connection sockets during init for resuse at runtime.
    # Can reduce p99,max-latency hit from cold-start connections or bursty traffic.
    #
    # Values:
    #   - "auto" (default) = ephemeral_port_limit // num_workers
    #   - "auto-min" = 0.10 * (ephemeral_port_limit // num_workers))
    #   - None or 0 = disable warmup
    #   - >0 = explicit number of connections to warm up
    warmup_connections: str | int | None = "auto"

    # GC strategy for worker processes to reduce latency spikes from collection pauses
    #
    # Values:
    #   - "disabled": GC completely disabled (risky for long-running benchmarks)
    #   - "relaxed": GC enabled with 50x higher thresholds (less aggressive)
    #   - "system": Standard Python GC with default thresholds
    worker_gc_mode: str = "relaxed"

    # TODO(vir):
    #   -  move streaming to HttpClient config (not per-query)
    #   -  add max-sequence-length to HttpClient config (not per-query), base streaming_buffer_size on it
    streaming_buffer_size: int = 128 * 1024  # 128KB buffer for streaming tokens

    # Request adapter for Query/Response <-> Payload/Response bytes
    adapter: type[HttpRequestAdapter] | None = field(default=None, init=False)

    def __post_init__(self):
        # set default adapter in __post_init__ to avoid circular dependency
        if isinstance(self.api_type, str):
            self.api_type = APIType(self.api_type)

        if self.adapter is None:
            adapter_path = ADAPTER_MAP.get(self.api_type)
            if not adapter_path:
                raise ValueError(f"Invalid or unsupported API type: {self.api_type}")

            module_path, class_name = adapter_path.rsplit(".", 1)
            module = import_module(module_path)
            self.adapter = getattr(module, class_name)


@dataclass
class SocketConfig:
    """Default values for socket options."""

    # Nagle's algorithm batches small packets to improve network efficiency
    # TCP_NODELAY disables Nagle's algorithm lower latency in both directions
    # Causes increased CPU usage due to more packets being sent
    tcp_nodelay: int = 1

    # Quick ACK mode (Linux-specific)
    # Forces immediate acknowledgment of received packets
    # instead of the default delayed ACK behavior.
    tcp_quickack: int = 1

    # Connection keepalive settings for long-lived connections
    so_keepalive: int = 1  # Enable keepalive at socket level
    tcp_keepidle: int = 30  # Start keepalive probes after 30 seconds idle
    tcp_keepcnt: int = 1  # 1 failed keepalive probes = dead
    tcp_keepintvl: int = 30  # Send probes every 30 seconds

    # Make sure socket buffers are never the bottle neck
    # With HTTP/1.1, a TCP socket will only be used for a single request
    # Largest message size would be server response in Offline Mode
    # 4MB /4 bytes per token = 1M tokens in any given packet
    so_rcvbuf: int = 1024 * 1024 * 10  # 4MB receive buffer
    so_sndbuf: int = 1024 * 1024 * 10  # 4MB send buffer

    # Linux-specific
    # Causes socket to be closed if no data is received for the specified time
    #
    # WARNING:
    # offline-mode might suffer dropped conections due to this timeout
    tcp_user_timeout: int = 30000  # 30 seconds

    def apply_to_socket(self, sock: socket.socket) -> None:
        """
        Apply the default socket options to the given socket.
        Socket is used by aiohttp.ClientSession to create TCP connections for HTTP requests.

        With socket-reuse enabled, TCP connection can be maintained across requests.
        """

        # Low-latency optimizations for streaming
        sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, self.tcp_nodelay)

        # Connection keepalive settings for long-lived SSE connections
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, self.so_keepalive)

        # Fine-tune keepalive timing
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, self.tcp_keepidle)
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, self.tcp_keepintvl)
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, self.tcp_keepcnt)

        # Buffer size optimizations for streaming
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.so_rcvbuf)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.so_sndbuf)

        # Enable Quick ACK mode
        if hasattr(socket, "TCP_QUICKACK"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, self.tcp_quickack)

        # Set idle connection timeout
        if hasattr(socket, "TCP_USER_TIMEOUT"):
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_USER_TIMEOUT, self.tcp_user_timeout
            )


@dataclass
class AioHttpConfig:
    """Configuration for aiohttp client session and connectors."""

    # lifetime timeouts
    client_timeout_total: float | None = None  # None means no timeout
    client_timeout_connect: float | None = None
    client_timeout_sock_read: float | None = None

    # skip these headers in requests
    skip_auto_headers: list[str] = field(
        default_factory=lambda: ["User-Agent", "Accept-Encoding"]
    )

    # TCP Connection Pooling
    # NOTE: limit=0 means unlimited; None defaults to aiohttp's 100 which causes pool starvation
    tcp_connector_limit: int = 0  # 0 = unlimited (respects system fd limit)
    tcp_connector_limit_per_host: int = 0  # 0 = unlimited per host
    tcp_connector_force_close: bool = False  # Enable connection pooling
    tcp_connector_keepalive_timeout: int = 86400  # 24 hours - keep connections alive
    tcp_connector_enable_cleanup_closed: bool = (
        False  # Skip SSL transport cleanup overhead
    )

    # DNS caching
    tcp_connector_use_dns_cache: bool = True
    tcp_connector_ttl_dns_cache: int = 300  # 5 min

    # Happy Eyeballs effects the staggering of connection attempts (RFC 8305)
    # When disabled, we issue connection in traditional sequential manner (IPv4 first, then IPv6)
    # This offers better connection latency when host is known reliable
    tcp_connector_happy_eyeballs_delay: float | None = None  # None = disabled

    # AF_UNSPEC = IPv4 and IPv6, AF_INET = IPv4 only, AF_INET6 = IPv6 only
    tcp_connector_family: int = socket.AF_UNSPEC

    # Socket defaults
    socket_defaults: SocketConfig = field(default_factory=SocketConfig)

    def create_tcp_connector(self, **kwargs) -> aiohttp.TCPConnector:
        """Create a TCP connector with this configuration."""

        def socket_factory(addr_info):
            """TCP socket factory optimized for HTTP performance."""
            family, sock_type, proto, _, _ = addr_info
            sock = socket.socket(family=family, type=sock_type, proto=proto)
            self.socket_defaults.apply_to_socket(sock)
            return sock

        connector_kwargs: dict[str, Any] = {
            "limit": self.tcp_connector_limit,
            "limit_per_host": self.tcp_connector_limit_per_host,
            "force_close": self.tcp_connector_force_close,
            "keepalive_timeout": self.tcp_connector_keepalive_timeout,
            "enable_cleanup_closed": self.tcp_connector_enable_cleanup_closed,
            "ttl_dns_cache": self.tcp_connector_ttl_dns_cache,
            "use_dns_cache": self.tcp_connector_use_dns_cache,
            "happy_eyeballs_delay": self.tcp_connector_happy_eyeballs_delay,
            "family": self.tcp_connector_family,
            "socket_factory": socket_factory,
        }

        # Allow overrides from kwargs
        connector_kwargs.update(kwargs)
        return aiohttp.TCPConnector(**connector_kwargs)


@dataclass
class ZMQConfig:
    """Configuration for ZMQ sockets and communication."""

    # Main ZMQ settings
    zmq_io_threads: int = 4  # Number of ZMQ IO threads ; TODO(vir): needs to scale?
    zmq_high_water_mark: int = 0  # Max queue size per socket (0=unlimited)

    # ZMQ addresses (use None for auto-generated prefixes using PID)
    zmq_request_queue_prefix: str | None = None
    zmq_response_queue_addr: str | None = None
    zmq_readiness_queue_addr: str | None = None

    # ZMQ socket options
    zmq_linger: int = 0  # 0 = Don't block on close
    zmq_immediate: int = 1  # ensure messages only enqueued on READY connections
    zmq_send_timeout: int = -1  # -1 = Non-blocking send
    zmq_recv_timeout: int = 1  # Timeout on receive() in ms

    zmq_recv_buffer_size: int = 10 * 1024 * 1024  # 10MB receive buffer (OS level)
    zmq_send_buffer_size: int = 10 * 1024 * 1024  # 10MB send buffer (OS level)

    def __post_init__(self):
        """Generate portable ZMQ socket paths if not provided."""
        if self.zmq_request_queue_prefix is None:
            self.zmq_request_queue_prefix = ZMQConfig._get_ipc_path(
                "http_worker_requests"
            )
        assert (
            len(self.zmq_request_queue_prefix) <= zmq.IPC_PATH_MAX_LEN
        ), "ZMQ request queue prefix is too long"

        if self.zmq_response_queue_addr is None:
            self.zmq_response_queue_addr = ZMQConfig._get_ipc_path(
                "http_worker_responses"
            )
        assert (
            len(self.zmq_response_queue_addr) <= zmq.IPC_PATH_MAX_LEN
        ), "ZMQ response queue address is too long"

        if self.zmq_readiness_queue_addr is None:
            self.zmq_readiness_queue_addr = ZMQConfig._get_ipc_path(
                "http_worker_readiness"
            )
        assert (
            len(self.zmq_readiness_queue_addr) <= zmq.IPC_PATH_MAX_LEN
        ), "ZMQ readiness queue address is too long"

    @staticmethod
    def _get_ipc_path(name: str) -> str:
        """Generate an IPC socket path.

        Args:
            name: Base name for the socket

        Returns:
            IPC socket path with PID to avoid conflicts
        """
        assert os.name != "nt", "Windows not yet supported"

        # Include PID to avoid conflicts between processes
        pid = os.getpid()
        ipc_path = f"ipc:///tmp/mlperf_endpoint_{name}_{pid}"
        assert len(ipc_path) <= zmq.IPC_PATH_MAX_LEN, "ZMQ socket path is too long"
        return ipc_path


__all__ = ["HTTPClientConfig", "AioHttpConfig", "ZMQConfig", "SocketConfig"]
