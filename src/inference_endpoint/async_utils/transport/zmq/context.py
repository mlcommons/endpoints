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
a ZMQ context and a temporary socket directory. Publishers, the main process
ZmqWorkerPoolTransport, and (in their own process) workers and subscribers
each receive a ManagedZMQContext and use it to create sockets.

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

import zmq

from inference_endpoint.utils import SingletonMixin


class ManagedZMQContext(SingletonMixin):
    """A managed ZMQ context. If the context is not created, it will be created on first use.

    This class is a per-process singleton, and is compatible with both 'spawn' and 'fork'
    multiprocessing start methods.
    """

    def __init__(self, io_threads: int = 1, socket_dir: str | None = None) -> None:
        """Creates a new ManagedZMQContext.

        Args:
            io_threads (int): The number of IO threads to use for the ZMQ context.
            socket_dir (str | None): The directory to use for the ZMQ sockets. If None, a temporary directory will be created.
                If set, the directory must be writable and must already exist.
                If not set, a temporary directory will be created and used.
                This directory will be cleaned up when the context is cleaned up.
        """
        if getattr(self, "_initialized", False):
            # If mp.start_method is 'spawn', the child process will always create its own singleton
            # since it will not share the parent's memory.
            # However, if mp.start_method is 'fork', we need to check if the PID matches and create
            # a new singleton if we are in a child process.
            if os.getpid() == self.pid:
                return
        self.pid: int = os.getpid()

        self.ctx: zmq.Context | None = zmq.Context(io_threads=io_threads)

        if socket_dir is None:
            self._tmp_dir: tempfile.TemporaryDirectory | None = (
                tempfile.TemporaryDirectory(prefix="zmq_")
            )
            self.socket_dir: str | None = self._tmp_dir.name
        else:
            path = Path(socket_dir)
            if not path.exists():
                raise FileNotFoundError(f"Socket directory {path} does not exist")
            if not path.is_dir():
                raise NotADirectoryError(f"Socket directory {path} is not a directory")
            if not os.access(path, os.W_OK):
                raise PermissionError(f"Socket directory {path} is not writable")
            self._tmp_dir = None
            self.socket_dir: str | None = path.as_posix()  # type: ignore[no-redef]
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
