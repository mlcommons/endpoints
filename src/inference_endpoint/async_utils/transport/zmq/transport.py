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

"""
ZMQ transport implementation for worker pool IPC.

    Main Process                      Worker Processes
    ┌─────────────────┐              ┌─────────────────┐
    │ ZmqWorkerPool   │              │ Worker 0        │
    │ Transport       │              │                 │
    │                 │   PUSH/PULL  │  ┌───────────┐  │
    │  request_sender ├──────────────┼──► receiver  │  │
    │       [0]       │   (per-wkr)  │  └───────────┘  │
    │                 │              │                 │
    │  response_recv  ◄──────────────┼── sender        │
    │   (fan-in)      │   PULL/PUSH  │                 │
    └─────────────────┘              └─────────────────┘

Notes:
    - Edge-Triggered FDs: ZMQ sockets use edge-triggered notifications.
      We drain all messages on each callback and reschedule via ``call_soon``
      to catch messages arriving during processing.
    - Direct Event Loop Integration: Uses ``add_reader``/``add_writer`` instead
      of asyncio ZMQ. This gives us control over the hot path and avoids extra layers.
    - Uses msgspec.msgpack serialization.
    - All transports are single-threaded and must be used from the event loop thread.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq

from inference_endpoint.core.types import Query, QueryResult, StreamChunk

from ..protocol import (
    ReceiverTransport,
    SenderTransport,
    WorkerConnector,
    WorkerPoolTransport,
)
from .context import ManagedZMQContext
from .ready_check import ReadyCheckReceiver, send_ready_signal

logger = logging.getLogger(__name__)


__all__ = [
    "ZmqWorkerPoolTransport",
]


@dataclass
class _ZMQSocketConfig:
    """Internal: ZMQ socket configuration with tuned defaults."""

    # NOTE(vir): ZMQ Background I/O Threads
    # Controls the size of the ZMQ background I/O thread pool (in C++).
    # These threads handle non-blocking I/O, message queuing, and transport
    # operations off the main Python thread.
    #
    # Performance characteristics:
    #   - only created in Main Process (owning WorkerPoolTransport)
    #   - each requires separate physical core for peak-performance, not hyperthreads
    #   - possibly effected by msg-size, messaging rate, NOT worker count (since 4-threads works fine with 100 workers)
    #   - tested io_threads=4 for upto 100 worker processes (on 224xCore x86 System)
    io_threads: int = 4

    high_water_mark: int = 0  # 0 = unlimited
    linger: int = -1  # Block indefinitely on close to send pending messages
    immediate: int = 1  # Only enqueue on ready connections
    # Default 4MB; increase for multimodal (VL) payloads via HTTPClientConfig / YAML / CLI.
    recv_buffer_size: int = 4 * 1024 * 1024  # 4MB
    send_buffer_size: int = 4 * 1024 * 1024  # 4MB

    def apply_recv(self, sock: zmq.Socket) -> None:
        """Apply receiver socket options."""
        sock.setsockopt(zmq.LINGER, self.linger)
        sock.setsockopt(zmq.RCVHWM, self.high_water_mark)
        sock.setsockopt(zmq.RCVBUF, self.recv_buffer_size)

    def apply_send(self, sock: zmq.Socket) -> None:
        """Apply sender socket options."""
        sock.setsockopt(zmq.LINGER, self.linger)
        sock.setsockopt(zmq.SNDHWM, self.high_water_mark)
        sock.setsockopt(zmq.SNDBUF, self.send_buffer_size)
        sock.setsockopt(zmq.IMMEDIATE, self.immediate)


class _ZmqReceiverTransport(ReceiverTransport):
    """
    ZMQ PULL socket receiver with event-driven receive.

    ZMQ's FD is edge-triggered (signals on state change, not data presence).
    This requires special handling:

    1. On FD readable, we drain ALL available messages (not just one)
    2. After draining, we reschedule via `call_soon` to catch messages
       that arrived during processing (race condition mitigation)
    3. The `_soon_call` handle prevents duplicate schedules and enables
       clean cancellation on close

    This pattern is adapted from aiozmq's `_ZmqLooplessTransportImpl`.
    (see: https://github.com/aio-libs/aiozmq/blob/bc471896c3b200fd052beccfac32bd4cd79e24d5/aiozmq/core.py#L678)
    """

    __slots__ = (
        "_loop",
        "_sock",
        "_fd",
        "_decoder",
        "_deque",
        "_waiter",
        "_closing",
        "_soon_call",
        "_recv_buf",
        "_recv_view",
    )

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        sock: zmq.Socket,
        decoder: msgspec.msgpack.Decoder,
    ):
        self._loop = loop
        self._sock = sock
        self._fd = sock.getsockopt(zmq.FD)
        self._decoder = decoder
        # Fast queue: deque + Future (no lock overhead vs asyncio.Queue)
        self._deque: deque[Any] = deque()
        self._waiter: asyncio.Future[None] | None = None
        self._closing = False
        self._soon_call: asyncio.Handle | None = None

        # NOTE(vir):
        # zmq recv_into with Pre-allocated buffer.
        # msgspec can decode in-place, avoiding per-message bytes allocation.
        recv_buf_size = sock.getsockopt(zmq.RCVBUF)
        self._recv_buf = bytearray(recv_buf_size)
        self._recv_view = memoryview(self._recv_buf)

        self._loop.add_reader(self._fd, self._on_readable)

    def _on_readable(self) -> None:
        """
        Handle FD readable event - drain all messages.

        Called by event loop when ZMQ's internal state changes. Because ZMQ
        uses edge-triggering, we must:

        1. Clear `_soon_call` (we're now executing)
        2. Drain ALL available messages in a tight loop
        3. Wake waiter ONCE after draining (batched notification)
        4. Reschedule via `call_soon` if we did work (catch racing messages)

        The reschedule step handles this race:
            t0: FD becomes readable, event loop queues our callback
            t1: We start draining messages
            t2: New message arrives (no new edge notification)
            t3: We finish draining, think we're done
            t4: call_soon fires, we drain the t2 message

        Without step 4, the t2 message would sit unprocessed until the next
        unrelated edge trigger.
        """
        self._soon_call = None

        if self._closing:
            return

        count = 0
        recv_buf = self._recv_buf
        recv_view = self._recv_view
        buf_len = len(recv_buf)
        try:
            while True:
                nbytes = self._sock.recv_into(recv_buf, flags=zmq.NOBLOCK)
                if nbytes > buf_len:
                    raise RuntimeError(
                        f"ZMQ message truncated ({nbytes} > {buf_len} bytes). "
                        f"Increase recv_buffer_size in _ZMQSocketConfig."
                    )
                self._deque.append(self._decoder.decode(recv_view[:nbytes]))
                count += 1
        except zmq.Again:
            # Normal: no more messages
            pass
        except zmq.ZMQError as e:
            if e.errno not in (errno.EAGAIN, errno.EINTR, errno.ENOTSOCK):
                logger.error(f"ZMQ recv error: {e}")
        except msgspec.DecodeError as e:
            logger.error(f"Decode error: {e}")

        # Wake waiter once after draining (not per message)
        if count > 0:
            if self._waiter is not None and not self._waiter.done():
                self._waiter.set_result(None)

            # Reschedule to catch messages that arrived during processing
            if self._soon_call is None:
                try:
                    if self._sock.getsockopt(zmq.EVENTS) & zmq.POLLIN:
                        self._soon_call = self._loop.call_soon(self._on_readable)
                except zmq.ZMQError:
                    # Socket may be closed; ignore.
                    pass

    def poll(self) -> Any | None:
        """Non-blocking poll. Returns item if available, None otherwise."""
        if self._deque:
            return self._deque.popleft()
        return None

    async def recv(self) -> Any | None:
        """Receive a message. Returns None when closed."""
        # Fast path: items already in deque
        if self._deque:
            return self._deque.popleft()

        # Check if closed with empty queue
        if self._closing:
            return None

        # Wait for items
        while not self._deque:
            if self._closing:
                return None
            self._waiter = self._loop.create_future()
            try:
                await self._waiter
            finally:
                self._waiter = None

        return self._deque.popleft()

    def close(self) -> None:
        """Close the transport. Idempotent."""
        if self._closing:
            return
        self._closing = True

        # Cancel pending callback
        if self._soon_call is not None:
            self._soon_call.cancel()
            self._soon_call = None

        # Remove from event loop
        try:
            self._loop.remove_reader(self._fd)
        except (ValueError, OSError):
            # Already removed or invalid fd
            pass

        # Socket is closed by ManagedZMQContext.cleanup() when the context scope exits.

        # Wake waiter so receive() can return None
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)


class _ZmqSenderTransport(SenderTransport):
    """
    ZMQ PUSH socket sender with buffered non-blocking writes.

    ZMQ's FD is edge-triggered (signals on state change, not data presence).
    This requires special handling:
    1. Fast path: Direct send when buffer empty and socket ready
    2. Slow path: Buffer message, register writer callback
    3. Writer callback drains buffer, reschedules if more work
    """

    __slots__ = (
        "_loop",
        "_sock",
        "_fd",
        "_encoder",
        "_buffer",
        "_closing",
        "_writing",
        "_soon_call",
    )

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        sock: zmq.Socket,
        encoder: msgspec.msgpack.Encoder,
    ):
        self._loop = loop
        self._sock = sock
        self._fd = sock.getsockopt(zmq.FD)
        self._encoder = encoder
        self._buffer: deque[bytes] = deque()
        self._closing = False
        self._writing = False
        self._soon_call: asyncio.Handle | None = None

    def send(self, data: Any) -> None:
        """Send a message. Non-blocking, buffers if socket would block."""
        if self._closing:
            return

        serialized = self._encoder.encode(data)

        # Fast path: direct send when buffer is empty
        if not self._buffer:
            try:
                self._sock.send(serialized, zmq.NOBLOCK, copy=False, track=False)
                return
            except zmq.Again:
                # Socket would block; fall through to buffer and use writer.
                pass
            except zmq.ZMQError as e:
                if e.errno not in (errno.EAGAIN, errno.EINTR):
                    logger.error(f"ZMQ send error: {e}")
                return

        # Slow path: buffer and register writer
        self._buffer.append(serialized)
        if not self._writing:
            self._writing = True
            self._loop.add_writer(self._fd, self._on_writable)

    def _on_writable(self) -> None:
        """Called by event loop when socket is writable.

        ZMQ's FD is edge-triggered, so we must:
        1. Drain the buffer as much as possible
        2. Reschedule via call_soon if buffer still has items
        """
        self._soon_call = None

        if self._closing:
            return

        try:
            while self._buffer:
                self._sock.send(self._buffer[0], zmq.NOBLOCK, copy=False, track=False)
                self._buffer.popleft()
        except zmq.Again:
            # Normal: socket would block
            pass
        except zmq.ZMQError as e:
            if e.errno not in (errno.EAGAIN, errno.EINTR, errno.ENOTSOCK):
                logger.error(f"ZMQ send error in buffer drain: {e}")
            self._buffer.clear()
            self._stop_writing()
            return

        if not self._buffer:
            self._stop_writing()
            return

        # Buffer has remaining items - always reschedule to drain
        if self._soon_call is None:
            self._soon_call = self._loop.call_soon(self._on_writable)

    def _stop_writing(self) -> None:
        """Stop watching for writability."""
        if self._writing:
            try:
                self._loop.remove_writer(self._fd)
            except (ValueError, OSError):
                # Already removed or invalid fd
                pass
            self._writing = False

    def close(self) -> None:
        """Close the transport. Idempotent."""
        if self._closing:
            return
        self._closing = True

        # Cancel pending callback
        if self._soon_call is not None:
            self._soon_call.cancel()
            self._soon_call = None

        self._stop_writing()
        self._buffer.clear()

        # Socket is closed by ManagedZMQContext.cleanup() when the context scope exits.


# =============================================================================
# Factory Functions (internal)
# =============================================================================


def _create_receiver(
    loop: asyncio.AbstractEventLoop,
    path: str,
    zmq_context: ManagedZMQContext,
    config: _ZMQSocketConfig,
    message_type: type | None = None,
    bind: bool = False,
) -> _ZmqReceiverTransport:
    """Create a ZMQ receiver transport.

    Args:
        loop: Event loop for transport registration.
        path: Socket path for IPC address construction (via zmq_context).
        zmq_context: Managed ZMQ context (use .socket() for tracked cleanup).
        config: Socket configuration.
        message_type: Type hint for msgspec decoder. Can be a single type, Union type, or None.
        bind: Whether to bind (True) or connect (False).

    Returns:
        Configured receiver transport.
    """
    sock = zmq_context.socket(zmq.PULL)
    config.apply_recv(sock)

    if bind:
        zmq_context.bind(sock, path)
    else:
        zmq_context.connect(sock, path)

    decoder = (
        msgspec.msgpack.Decoder(type=message_type)  # type: ignore[arg-type]
        if message_type
        else msgspec.msgpack.Decoder()
    )

    return _ZmqReceiverTransport(loop, sock, decoder)


def _create_sender(
    loop: asyncio.AbstractEventLoop,
    path: str,
    zmq_context: ManagedZMQContext,
    config: _ZMQSocketConfig,
    bind: bool = False,
) -> _ZmqSenderTransport:
    """Create a ZMQ sender transport."""
    sock = zmq_context.socket(zmq.PUSH)
    config.apply_send(sock)

    if bind:
        zmq_context.bind(sock, path)
    else:
        zmq_context.connect(sock, path)

    encoder = msgspec.msgpack.Encoder()
    return _ZmqSenderTransport(loop, sock, encoder)


# =============================================================================
# Worker Connector (passed to worker processes)
# =============================================================================


@dataclass(slots=True)
class _ZmqWorkerConnector(WorkerConnector):
    """Internal: Picklable connector for worker processes.

    Contains socket_dir and path components for IPC address construction.
    Passed to workers via multiprocessing.
    """

    config: _ZMQSocketConfig
    socket_dir: str
    request_paths: list[str]
    response_path: str
    readiness_path: str

    @asynccontextmanager
    async def connect(
        self, worker_id: int, zmq_context: ManagedZMQContext
    ) -> AsyncIterator[tuple[_ZmqReceiverTransport, _ZmqSenderTransport]]:
        """Connect worker transports and signal readiness.

        Startup Sequence
        -------------------
        Main Process              Worker Process
            │                         │
            │  spawn(connector)       │
            ├────────────────────────►│
            │                         │
            │                    connect to request addr
            │                    connect to response addr
            │                    connect to readiness addr
            │                         │
            │◄────── READY signal ────┤
            │                         │
        wait_for_workers_ready()      │
        returns                   (start processing)

        Args:
            worker_id: Unique identifier for this worker.
            zmq_context: Managed ZMQ context (e.g. from ManagedZMQContext.scoped() in this process).
                Must have socket_dir set to the same directory as the main process.

        Yields:
            Tuple of (request_receiver, response_sender) transports.
        """
        loop = asyncio.get_running_loop()
        request_path = self.request_paths[worker_id]

        logger.debug("Worker %d request path: %s", worker_id, request_path)
        logger.debug("Worker %d response path: %s", worker_id, self.response_path)

        # Worker CONNECTS (main process BINDS)
        requests = _create_receiver(
            loop, request_path, zmq_context, self.config, Query, bind=False
        )
        responses = _create_sender(
            loop, self.response_path, zmq_context, self.config, bind=False
        )

        try:
            await send_ready_signal(zmq_context, self.readiness_path, worker_id)

            yield requests, responses
        finally:
            requests.close()
            responses.close()


# =============================================================================
# Worker Pool Transport (main process)
# =============================================================================


class ZmqWorkerPoolTransport(WorkerPoolTransport):
    """ZMQ implementation of WorkerPoolTransport.

    Main process transport for worker pool communication.
    Provides fan-out (send to workers) and fan-in (receive from workers).

    The caller must pass a ManagedZMQContext (e.g. from ManagedZMQContext.scoped())
    and scope the transport lifetime within that context for proper cleanup.

    Usage:
        with ManagedZMQContext.scoped(io_threads=4) as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 4, zmq_ctx)
            for i in range(4):
                spawn_worker(i, pool.worker_connector, ...)
            await pool.wait_for_workers_ready(timeout=30)
            pool.send(worker_id, query)
            result = pool.poll()        # Non-blocking
            result = await pool.recv()  # Blocking
            pool.cleanup()
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        zmq_context: ManagedZMQContext,
        config: _ZMQSocketConfig,
        num_workers: int,
    ) -> None:
        self._loop = loop
        self._config = config
        self._num_workers = num_workers
        self._closed = False
        suffix = uuid.uuid4().hex[:8]

        # Generate path components (address construction is handled by zmq_context)
        request_paths = [f"req_{suffix}_{i}" for i in range(num_workers)]
        response_path = f"resp_{suffix}"
        readiness_path = f"ready_{suffix}"

        # Create transports (main process BINDS — this sets socket_dir if needed)
        self._request_senders = [
            _create_sender(loop, path, zmq_context, config, bind=True)
            for path in request_paths
        ]
        self._response_receiver = _create_receiver(
            loop,
            response_path,
            zmq_context,
            config,
            QueryResult | StreamChunk,  # type: ignore[arg-type]
            bind=True,
        )
        self._ready_check = ReadyCheckReceiver(readiness_path, zmq_context, num_workers)

        # socket_dir is now guaranteed set (bind() created it if needed).
        # Store resolved addresses for debugging and tests.
        assert zmq_context.socket_dir is not None
        self._request_addrs = [zmq_context._make_address(p) for p in request_paths]
        self._response_addr = zmq_context._make_address(response_path)
        self._worker_connector = _ZmqWorkerConnector(
            config=config,
            socket_dir=zmq_context.socket_dir,
            request_paths=request_paths,
            response_path=response_path,
            readiness_path=readiness_path,
        )

    @classmethod
    def create(
        cls,
        loop: asyncio.AbstractEventLoop,
        num_workers: int,
        zmq_context: ManagedZMQContext,
        *args: Any,
        **kwargs: Any,
    ) -> ZmqWorkerPoolTransport:
        """Factory to create ZmqWorkerPoolTransport.

        Signature matches WorkerPoolTransport.create(loop, num_workers, *args, **kwargs).
        Expects zmq_context as first extra positional arg.

        Args:
            loop: Event loop for transport registration.
            num_workers: Number of workers (required).
            zmq_context: Managed ZMQ context (e.g. from ManagedZMQContext.scoped()).
            *args: Ignored - prevents any errors with extraneous args and adheres with WorkerPoolTransport.create().
            **kwargs: Optional _ZMQSocketConfig overrides (e.g. ``recv_buffer_size``, ``send_buffer_size``).

        Returns:
            Configured ZmqWorkerPoolTransport instance.
        """
        if os.name == "nt":
            raise RuntimeError("Windows not yet supported for ZMQ transport")

        config = _ZMQSocketConfig(**kwargs)
        return cls(loop, zmq_context, config, num_workers)

    @property
    def worker_connector(self) -> WorkerConnector:
        """Connector to pass to worker processes."""
        return self._worker_connector

    def send(self, worker_id: int, query: Query) -> None:
        """Send request to specific worker."""
        self._request_senders[worker_id].send(query)

    def poll(self) -> QueryResult | StreamChunk | None:
        """Non-blocking poll. Returns response if available, None otherwise."""
        return self._response_receiver.poll()

    async def recv(self) -> QueryResult | StreamChunk | None:
        """Blocking receive. Waits for next response."""
        return await self._response_receiver.recv()

    async def wait_for_workers_ready(self, timeout: float | None = None) -> None:
        """Block until all workers signal readiness.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.

        Raises:
            TimeoutError: If workers don't signal in time (only if timeout is set).
        """
        await self._ready_check.wait(timeout=timeout)

    def cleanup(self) -> None:
        """Close all transports and release resources. Idempotent."""
        if self._closed:
            return
        self._closed = True

        # Close all transports (each is idempotent).
        for sender in self._request_senders:
            sender.close()
        self._response_receiver.close()
        self._ready_check.close()

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            # Never raise from __del__
            pass
