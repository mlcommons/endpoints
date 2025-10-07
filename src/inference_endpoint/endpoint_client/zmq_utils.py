"""ZMQ utilities for endpoint client communication."""

import pickle
from typing import Any

import zmq
import zmq.asyncio

from inference_endpoint.endpoint_client.configs import ZMQConfig
from inference_endpoint.profiling import profile


class ZMQSocket:
    """Base class for async ZMQ sockets."""

    def __init__(
        self,
        context: zmq.asyncio.Context,
        socket_type: int,
        address: str,
        config: ZMQConfig,
        bind: bool = False,
    ):
        """Initialize ZMQ socket with common configuration."""
        self.socket = context.socket(socket_type)
        self.address = address

        try:
            if bind:
                self.socket.bind(address)
            else:
                self.socket.connect(address)
        except zmq.ZMQError as e:
            if bind:
                action = "bind to"
                sol = "Ensure the address is not already in use."
            else:
                action = "connect to"
                sol = "Ensure the target socket exists"
            raise zmq.ZMQError(
                f"Failed to {action} {address}: {e.strerror}. \n{sol}"
            ) from e

        # Common socket options
        self.socket.setsockopt(zmq.LINGER, config.zmq_linger)

        # Set type-specific options
        self._set_socket_options(config)

    def _set_socket_options(self, config: ZMQConfig) -> None:
        """Override in subclasses to set specific socket options."""
        pass

    def close(self, linger_ms: int | None = None) -> None:
        """Close socket cleanly."""
        if linger_ms is not None:
            self.socket.setsockopt(zmq.LINGER, linger_ms)
        self.socket.close()


class ZMQPushSocket(ZMQSocket):
    """Async wrapper for ZMQ PUSH socket."""

    def __init__(self, context: zmq.asyncio.Context, address: str, config: ZMQConfig):
        """Initialize ZMQ push socket."""
        super().__init__(context, zmq.PUSH, address, config, bind=False)

    def _set_socket_options(self, config: ZMQConfig) -> None:
        """Set PUSH socket specific options."""
        self.socket.setsockopt(zmq.SNDHWM, config.zmq_high_water_mark)
        self.socket.setsockopt(zmq.SNDBUF, config.zmq_send_buffer_size)
        self.socket.setsockopt(zmq.SNDTIMEO, config.zmq_send_timeout)

    @profile
    async def send(self, data: Any) -> None:
        """Serialize and send data through push socket (non-blocking)."""
        serialized = pickle.dumps(data)
        await self.socket.send(serialized, zmq.NOBLOCK)


class ZMQPullSocket(ZMQSocket):
    """Async wrapper for ZMQ PULL socket."""

    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        config: ZMQConfig,
        bind: bool = False,
    ):
        """Initialize ZMQ pull socket."""
        super().__init__(context, zmq.PULL, address, config, bind=bind)

    def _set_socket_options(self, config: ZMQConfig) -> None:
        """Set PULL socket specific options."""
        self.socket.setsockopt(zmq.RCVHWM, config.zmq_high_water_mark)
        self.socket.setsockopt(zmq.RCVBUF, config.zmq_recv_buffer_size)
        self.socket.setsockopt(zmq.RCVTIMEO, config.zmq_recv_timeout)

    @profile
    async def receive(self) -> Any | None:
        """
        Receive and deserialize data from pull socket.
        Returns None on timeout.

        Returns:
            Deserialized data or None on timeout.
        """
        try:
            serialized = await self.socket.recv()
            return pickle.loads(serialized)
        except zmq.Again:
            # Timeout occurred
            return None
