"""Configuration classes for HTTP endpoint client."""

import os
import socket
from dataclasses import dataclass, field
from typing import Any

import aiohttp


@dataclass
class HTTPClientConfig:
    """Configuration for the HTTP endpoint client."""

    endpoint_url: str
    num_workers: int = 4
    max_concurrency: int = (
        -1
    )  # -1: unlimited, else: limit concurrent requests via semaphore

    # Worker lifecycle timeouts
    worker_initialization_timeout: float = 10.0  # init
    worker_health_check_interval: float = 2.0  # runtime
    worker_graceful_shutdown_wait: float = 0.5  # post-run
    worker_force_kill_timeout: float = 1.0  # post-run

    # Response handling timeouts, for signal handling
    response_handler_timeout: float = 1.0
    worker_request_timeout: float = 1.0


@dataclass
class SocketConfig:
    """Linux socket optimizations for HTTP performance."""

    # Essential socket options
    tcp_nodelay: bool = True  # Disable Nagle's algorithm for lower latency
    so_keepalive: bool = True  # Enable TCP keepalive
    so_reuseaddr: bool = True  # Allow socket reuse

    # Socket buffer sizes (0 means use system default)
    so_rcvbuf: int = 0  # Receive buffer size
    so_sndbuf: int = 0  # Send buffer size

    # TCP keepalive parameters (Linux-specific)
    tcp_keepidle: int = 60  # Seconds before sending keepalive probes
    tcp_keepintvl: int = 10  # Interval between keepalive probes
    tcp_keepcnt: int = 6  # Number of keepalive probes before closing

    def apply_to_socket(self, sock: socket.socket) -> None:
        """Apply Linux socket optimizations."""
        # TCP_NODELAY - disable Nagle's algorithm
        if self.tcp_nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # SO_KEEPALIVE and related options
        if self.so_keepalive:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.tcp_keepidle)
            sock.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.tcp_keepintvl
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.tcp_keepcnt)

        # SO_REUSEADDR - allow socket reuse
        if self.so_reuseaddr:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Socket buffer sizes
        if self.so_rcvbuf > 0:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.so_rcvbuf)
        if self.so_sndbuf > 0:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.so_sndbuf)


@dataclass
class AioHttpConfig:
    """Configuration for aiohttp client session and connectors."""

    # TCP Connector settings

    # aiohttp.ClientSession configs
    client_timeout_total: float | None = None  # None means no timeout
    client_timeout_connect: float | None = None
    client_timeout_sock_read: float | None = None
    client_session_connector_owner: bool = (
        False  # TCP connector is owned by USER directly
    )
    skip_auto_headers: list[str] = field(
        default_factory=lambda: ["User-Agent", "Accept-Encoding"]
    )

    # aiohttp.TCPConnector configs
    tcp_connector_use_dns_cache: bool = True
    tcp_connector_ttl_dns_cache: int = 300
    tcp_connector_enable_cleanup_closed: bool = True
    tcp_connector_limit: int = 0  # 0 means unlimited
    tcp_connector_limit_per_host: int = 0  # 0 means unlimited per host
    tcp_connector_keepalive_timeout: int = (
        30  # Keep TCP connections alive for 30 seconds
    )
    tcp_connector_force_close: bool = (
        False  # Keep TCP connections alive, reuse across requests
    )
    tcp_connector_enable_tcp_nodelay: bool = True  # Disable Nagle's algorithm
    tcp_connector_happy_eyeballs_delay: float = (
        0.25  # Delay for IPv4/IPv6 connection race
    )
    tcp_connector_family: int = 0  # 0 = AF_UNSPEC (both IPv4 and IPv6)

    # Streaming configs
    streaming_buffer_size: int = 64 * 1024  # 64KB buffer for streaming

    # Socket-level defaults
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
            "ttl_dns_cache": self.tcp_connector_ttl_dns_cache,
            "use_dns_cache": self.tcp_connector_use_dns_cache,
            "force_close": self.tcp_connector_force_close,
            "keepalive_timeout": self.tcp_connector_keepalive_timeout,
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
    zmq_io_threads: int = 4  # Number of ZMQ IO threads
    zmq_high_water_mark: int = 10_000  # max msg queue size

    # ZMQ addresses (use None for auto-generation)
    zmq_request_queue_prefix: str | None = None
    zmq_response_queue_addr: str | None = None
    zmq_readiness_queue_addr: str | None = None

    # ZMQ socket options
    zmq_linger: int = 0  # Don't block on close
    zmq_send_timeout: int = -1  # Non-blocking send
    zmq_recv_timeout: int = -1  # Blocking receive
    zmq_recv_buffer_size: int = 10 * 1024 * 1024  # 10MB receive buffer
    zmq_send_buffer_size: int = 10 * 1024 * 1024  # 10MB send buffer

    def __post_init__(self):
        """Generate portable ZMQ socket paths if not provided."""
        if self.zmq_request_queue_prefix is None:
            self.zmq_request_queue_prefix = ZMQConfig._get_ipc_path(
                "http_worker_requests"
            )

        if self.zmq_response_queue_addr is None:
            self.zmq_response_queue_addr = ZMQConfig._get_ipc_path(
                "http_worker_responses"
            )

        if self.zmq_readiness_queue_addr is None:
            self.zmq_readiness_queue_addr = ZMQConfig._get_ipc_path(
                "http_worker_readiness"
            )

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
        return f"ipc:///tmp/mlperf_endpoint_{name}_{pid}"


__all__ = ["HTTPClientConfig", "AioHttpConfig", "ZMQConfig", "SocketConfig"]
