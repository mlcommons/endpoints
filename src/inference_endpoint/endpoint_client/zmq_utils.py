"""ZMQ utilities for endpoint client communication using aiozmq."""

import asyncio
from typing import Any

import aiozmq
import msgspec
import zmq

from inference_endpoint.endpoint_client.configs import ZMQConfig
from inference_endpoint.profiling import profile


class _ZMQSocket:
    """Base class for ZMQ sockets using aiozmq."""

    __slots__ = ("address", "config", "_stream")

    def __init__(self, address: str, config: ZMQConfig):
        self.address = address
        self.config = config
        self._stream: aiozmq.ZmqStream | None = None

    def close(self, linger_ms: int | None = None) -> None:
        if self._stream:
            if linger_ms is not None:
                self._stream.transport.setsockopt(zmq.LINGER, linger_ms)
            self._stream.close()

    async def __aenter__(self):
        if self._stream is None:
            await self.initialize(loop=asyncio.get_event_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ZMQPushSocket(_ZMQSocket):
    """Non-blocking PUSH socket using aiozmq."""

    __slots__ = ("_encoder",)

    def __init__(self, address: str, config: ZMQConfig):
        super().__init__(address, config)
        self._encoder = msgspec.msgpack.Encoder()

    async def initialize(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialize stream. Must be called before send()."""
        self._stream = await aiozmq.create_zmq_stream(
            zmq.PUSH, connect=self.address, loop=loop
        )
        self._stream.transport.setsockopt(zmq.LINGER, self.config.zmq_linger)
        self._stream.transport.setsockopt(zmq.SNDHWM, self.config.zmq_high_water_mark)
        self._stream.transport.setsockopt(zmq.SNDBUF, self.config.zmq_send_buffer_size)
        self._stream.transport.setsockopt(zmq.SNDTIMEO, self.config.zmq_send_timeout)

    @classmethod
    async def create(
        cls,
        address: str,
        config: ZMQConfig,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> "ZMQPushSocket":
        """Create and initialize in one step."""
        instance = cls(address, config)
        await instance.initialize(loop=loop)
        return instance

    @profile
    async def send(self, data: Any) -> None:
        """Non-blocking zero-copy send."""
        assert self._stream is not None
        serialized = self._encoder.encode(data)

        # NOTE(vir): aiozmq does not support zero copy like pyzmq
        self._stream.write((memoryview(serialized),))


class ZMQPullSocket(_ZMQSocket):
    """Zero-copy PULL socket using aiozmq."""

    __slots__ = ("bind", "_decoder")

    def __init__(
        self,
        address: str,
        config: ZMQConfig,
        bind: bool = False,
        decoder_type: type | None = None,
    ):
        super().__init__(address, config)
        self.bind = bind
        self._decoder = (
            msgspec.msgpack.Decoder(type=decoder_type)
            if decoder_type
            else msgspec.msgpack.Decoder()
        )

    async def initialize(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialize stream. Must be called before receive()."""
        if self.bind:
            self._stream = await aiozmq.create_zmq_stream(
                zmq.PULL, bind=self.address, loop=loop
            )
        else:
            self._stream = await aiozmq.create_zmq_stream(
                zmq.PULL, connect=self.address, loop=loop
            )

        self._stream.transport.setsockopt(zmq.LINGER, self.config.zmq_linger)
        self._stream.transport.setsockopt(zmq.RCVHWM, self.config.zmq_high_water_mark)
        self._stream.transport.setsockopt(zmq.RCVBUF, self.config.zmq_recv_buffer_size)
        self._stream.transport.setsockopt(zmq.RCVTIMEO, self.config.zmq_recv_timeout)

    @classmethod
    async def create(
        cls,
        address: str,
        config: ZMQConfig,
        bind: bool = False,
        decoder_type: type | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> "ZMQPullSocket":
        """Create and initialize in one step."""
        instance = cls(address, config, bind=bind, decoder_type=decoder_type)
        await instance.initialize(loop=loop)
        return instance

    @profile
    async def receive(self) -> Any | None:
        """Receive and return decoded data. Returns None on timeout."""
        assert self._stream is not None

        try:
            msg = await self._stream.read()
            return self._decoder.decode(msg[0])
        except zmq.Again:
            return None
