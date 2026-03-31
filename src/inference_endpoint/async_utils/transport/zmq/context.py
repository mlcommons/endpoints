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

"""Process-wide ZMQ context and socket directory for the zmq transport submodule.

This module provides ManagedZMQContext, a per-process singleton that holds
a ZMQ context and an optional socket directory. Publishers, the main process
ZmqWorkerPoolTransport, and (in their own process) workers and subscribers
each receive a ManagedZMQContext and use it to create sockets.

All socket binding and connecting should go through ctx.bind() and
ctx.connect() so that address construction is centralized and IPC socket
directories are managed automatically.

Scope the lifetime of ZMQ objects with ManagedZMQContext.scoped() so that
cleanup (context termination and socket directory removal) runs when the
context manager exits.

This module does not import from the rest of the zmq package to avoid
cyclic imports.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlunparse

import zmq
import zmq.asyncio

from inference_endpoint.utils import SingletonMixin


class ManagedZMQContext(SingletonMixin):
    """A managed ZMQ context. If the context is not created, it will be created on first use.

    This class is a per-process singleton, and is compatible with both 'spawn' and 'fork'
    multiprocessing start methods.
    """

    def __init__(self, io_threads: int = 1, socket_dir: str | None = None) -> None:
        """Creates a new ManagedZMQContext.

        Args:
            io_threads: The number of IO threads to use for the ZMQ context.
            socket_dir: Directory for IPC socket files. If None, a temporary
                directory is created on first bind(). For connect(), socket_dir
                must be set (either here or by a prior bind()).
        """
        if getattr(self, "_initialized", False):
            # If mp.start_method is 'spawn', the child process will always create its own singleton
            # since it will not share the parent's memory.
            # However, if mp.start_method is 'fork', we need to check if the PID matches and create
            # a new singleton if we are in a child process.
            if os.getpid() == self.pid:
                if socket_dir is not None and socket_dir != self.socket_dir:
                    raise ValueError(
                        f"ManagedZMQContext singleton already initialized with "
                        f"socket_dir={self.socket_dir!r}, cannot reinitialize "
                        f"with socket_dir={socket_dir!r}"
                    )
                return
        self.pid: int = os.getpid()

        self.ctx: zmq.Context | None = zmq.Context(io_threads=io_threads)
        self.socket_dir: str | None = (
            socket_dir.rstrip("/") if socket_dir else socket_dir
        )
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._sockets: list[zmq.Socket] = []

        self._initialized = True

    def socket(self, socket_type: int) -> zmq.Socket:
        """Create a ZMQ socket and register it for cleanup.

        Use this instead of self.ctx.socket() so that cleanup() can close all
        sockets before terminating the context, avoiding hangs on context.term().
        """
        if not self._initialized:
            raise RuntimeError("ManagedZMQContext is not initialized")

        if self.ctx is None:
            raise RuntimeError(
                "ZMQ context is not initialized, but initialized is True"
            )

        sock = self.ctx.socket(socket_type)
        self._sockets.append(sock)
        return sock

    def async_socket(self, socket_type: int) -> zmq.asyncio.Socket:
        """Create an async ZMQ socket (supports ``await sock.send()``/``recv()``).

        Uses a shadow ``zmq.asyncio.Context`` over the existing context so
        the underlying C context is shared. The socket is registered for
        cleanup like regular sockets.
        """
        if not self._initialized:
            raise RuntimeError("ManagedZMQContext is not initialized")

        if self.ctx is None:
            raise RuntimeError(
                "ZMQ context is not initialized, but initialized is True"
            )

        async_ctx = zmq.asyncio.Context(shadow=self.ctx)
        sock = async_ctx.socket(socket_type)
        self._sockets.append(sock)
        return sock

    def _ipc_socket_path(self, path: str) -> str:
        """Return the filesystem path for an IPC socket: ``<socket_dir>/<path>``."""
        return f"{self.socket_dir}/{path}"

    def _make_address(self, path: str, scheme: str = "ipc") -> str:
        """Construct a full ZMQ address from path and scheme.

        For IPC, ``urlunparse`` is called with ``socket_dir`` as the netloc
        and ``path`` as the path component. Because ``ipc`` is not a
        registered URI scheme, ``urlparse`` does *not* recover the netloc on
        round-trip — the full filesystem path ends up in ``parsed.path`` with
        ``parsed.netloc == ""``. This is expected for non-registered URI
        schemes and is fine because the produced string
        ``ipc://<socket_dir>/<path>`` is the correct ZMQ address regardless.

        For TCP, ``path`` is the netloc (host:port).

        Args:
            path: For IPC, the socket filename (e.g. "ev_pub_abc123").
                  For TCP, the netloc including port (e.g. "127.0.0.1:5555").
            scheme: Transport protocol (default "ipc"). Also supports "tcp".

        Returns:
            Full ZMQ address string.
        """
        if scheme == "ipc":
            if self.socket_dir is None:
                raise ValueError("socket_dir is required for IPC addresses")
            socket_path = self._ipc_socket_path(path)
            if len(socket_path) > zmq.IPC_PATH_MAX_LEN:
                raise ValueError(
                    f"IPC socket path too long "
                    f"({len(socket_path)} > {zmq.IPC_PATH_MAX_LEN}): {socket_path}"
                )
            return urlunparse((scheme, self.socket_dir, path, "", "", ""))
        elif scheme == "tcp":
            return urlunparse((scheme, path, "", "", "", ""))
        else:
            raise ValueError(
                f"Unsupported scheme: {scheme!r}. Expected 'ipc' or 'tcp'."
            )

    def bind(self, sock: zmq.Socket, path: str, scheme: str = "ipc") -> str:
        """Construct address and bind socket.

        For IPC: if socket_dir is None, creates a temporary directory and sets
        socket_dir. If socket_dir is a string, ensures the directory exists
        via ``mkdir(parents=True, exist_ok=True)``.

        Args:
            sock: ZMQ socket to bind.
            path: Socket filename (IPC) or host:port (TCP).
            scheme: Transport protocol (default "ipc"). Also supports "tcp".

        Returns:
            The full address string that was bound.
        """
        if scheme == "ipc":
            if self.socket_dir is None:
                self._tmp_dir = tempfile.TemporaryDirectory(prefix="zmq_")
                self.socket_dir = self._tmp_dir.name
            else:
                Path(self.socket_dir).mkdir(parents=True, exist_ok=True)
        addr = self._make_address(path, scheme)
        sock.bind(addr)
        return addr

    def connect(self, sock: zmq.Socket, path: str, scheme: str = "ipc") -> str:
        """Construct address and connect socket.

        For IPC: socket_dir must already be set (either via ``__init__`` or a
        prior ``bind()`` call), and the socket file must exist and be readable.

        Args:
            sock: ZMQ socket to connect.
            path: Socket filename (IPC) or host:port (TCP).
            scheme: Transport protocol (default "ipc"). Also supports "tcp".

        Returns:
            The full address string that was connected.

        Raises:
            ValueError: If scheme is "ipc" and socket_dir is None.
            FileNotFoundError: If the IPC socket file does not exist.
        """
        if scheme == "ipc" and self.socket_dir is None:
            raise ValueError(
                "socket_dir is required for IPC connect. "
                "Pass socket_dir to ManagedZMQContext() or call bind() first."
            )
        addr = self._make_address(path, scheme)
        sock.connect(addr)
        return addr

    def cleanup(self) -> None:
        # Close all tracked sockets before terminating the context, so
        # ctx.term() does not hang when Python objects still hold socket refs.
        if self._sockets:
            for sock in self._sockets:
                try:
                    sock.close()
                except (zmq.ZMQError, OSError):
                    # Socket already closed or invalid. Ignore.
                    pass
            self._sockets.clear()

        # Destroy the context and temp directory
        if self.ctx is not None:
            try:
                self.ctx.term()
            except (zmq.ZMQError, OSError):
                # Context already closed or process tearing down. Ignore.
                pass
            finally:
                self.ctx = None

        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except (OSError, FileNotFoundError):
                # Temp directory already cleaned up or inaccessible at process exit.
                pass
            finally:
                self._tmp_dir = None
        self.socket_dir = None

        self._initialized = False

    @classmethod
    @contextmanager
    def scoped(
        cls, *args: Any, **kwargs: Any
    ) -> Generator[ManagedZMQContext, None, None]:
        """
        Context manager for a scoped ZMQ context.

        This context manager will create a new ZMQ context and socket directory for the
        duration of the context manager.

        If this context created the singleton instance, it will be cleaned up when the context manager is exited.
        Otherwise, the context will be left to be cleaned up by the first creator of the singleton instance.
        """
        # Properly handle ownership check for the mp.start_method == "fork" case
        context = cls._instance
        own = True
        if context is not None and getattr(context, "_initialized", False):
            own = False

        if own:
            # Instantiate the concrete ManagedZMQContext directly to avoid
            # relying on potential non-callable behavior from SingletonMixin.
            context = ManagedZMQContext(*args, **kwargs)

        try:
            yield context
        finally:
            if own:
                context.cleanup()
