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

"""Unit tests for ManagedZMQContext."""

import os
import tempfile
import time

import pytest
import zmq
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext

# =============================================================================
# Creation and attributes
# =============================================================================


class TestManagedZMQContextCreation:
    """Tests for ManagedZMQContext creation and attributes."""

    def test_scoped_yields_context_with_ctx(self):
        """scoped() yields a context with valid .ctx. socket_dir is None until bind()."""
        with ManagedZMQContext.scoped() as ctx:
            assert ctx.ctx is not None
            assert ctx.socket_dir is None  # not set until bind()

    def test_bind_creates_temp_socket_dir(self):
        """bind() creates a temp socket_dir when socket_dir is None."""
        with ManagedZMQContext.scoped() as ctx:
            sock = ctx.socket(zmq.PUB)
            ctx.bind(sock, "test_socket")
            assert ctx.socket_dir is not None
            assert isinstance(ctx.socket_dir, str)
            assert os.path.isdir(ctx.socket_dir)
            assert "zmq_" in ctx.socket_dir
            assert ctx.socket_dir.startswith(tempfile.gettempdir())

    def test_scoped_accepts_io_threads(self):
        """scoped(io_threads=N) creates context without error."""
        with ManagedZMQContext.scoped(io_threads=2) as ctx:
            assert ctx.ctx is not None
            sock = ctx.socket(zmq.PUB)
            ctx.bind(sock, "test")
            # Socket closed by context cleanup on scope exit

    def test_socket_dir_is_writable(self):
        """socket_dir is a writable directory (for IPC sockets)."""
        with ManagedZMQContext.scoped() as ctx:
            # Trigger socket_dir creation via bind
            sock = ctx.socket(zmq.PUB)
            ctx.bind(sock, "test_writable")
            path = os.path.join(ctx.socket_dir, "test_socket")
            with open(path, "w") as f:
                f.write("test")
            assert os.path.isfile(path)
            os.unlink(path)


# =============================================================================
# socket() and tracking
# =============================================================================


class TestManagedZMQContextSocket:
    """Tests for ManagedZMQContext.socket() and socket tracking."""

    def test_socket_returns_zmq_socket(self):
        """socket(type) returns a ZMQ socket of the given type."""
        with ManagedZMQContext.scoped() as ctx:
            pub = ctx.socket(zmq.PUB)
            sub = ctx.socket(zmq.SUB)
            assert pub is not None
            assert sub is not None
            assert pub.getsockopt(zmq.TYPE) == zmq.PUB
            assert sub.getsockopt(zmq.TYPE) == zmq.SUB

    def test_socket_registers_for_cleanup(self):
        """Sockets created via .socket() are tracked in _sockets."""
        with ManagedZMQContext.scoped() as ctx:
            sock = ctx.socket(zmq.PUB)
            assert len(ctx._sockets) == 1
            assert sock in ctx._sockets
            _ = ctx.socket(zmq.SUB)
            assert len(ctx._sockets) == 2

    def test_socket_can_bind_and_connect(self):
        """Socket from .socket() can bind/connect via ctx.bind()/ctx.connect()."""
        with ManagedZMQContext.scoped() as ctx:
            pub = ctx.socket(zmq.PUB)
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.RCVTIMEO, 1000)
            ctx.bind(pub, "test_bind_connect")
            ctx.connect(sub, "test_bind_connect")
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            time.sleep(0.05)  # Allow slow-joiner to establish
            pub.send_string("hello")
            msg = sub.recv_string()
            assert msg == "hello"


# =============================================================================
# cleanup()
# =============================================================================


class TestManagedZMQContextCleanup:
    """Tests for ManagedZMQContext.cleanup()."""

    def test_cleanup_clears_ctx_and_socket_dir(self):
        """cleanup() sets ctx to None and socket_dir to None."""
        with ManagedZMQContext.scoped() as ctx:
            _ = ctx.socket(zmq.PUB)
        # After scoped exit, cleanup() was called (own=True). Singleton may be
        # the same object but re-initialized on next use. Create a fresh scoped
        # and inspect after explicit cleanup.
        with ManagedZMQContext.scoped() as ctx:
            ctx.cleanup()
            assert ctx.ctx is None
            assert ctx.socket_dir is None
            assert ctx._tmp_dir is None
            assert ctx._sockets == []
            assert ctx._initialized is False

    def test_cleanup_idempotent(self):
        """Calling cleanup() twice does not raise."""
        with ManagedZMQContext.scoped() as ctx:
            ctx.socket(zmq.PUB)
            ctx.cleanup()
            ctx.cleanup()
            ctx.cleanup()
        # Second scoped to leave singleton in a valid state for other tests
        with ManagedZMQContext.scoped():
            pass

    def test_scoped_exit_closes_sockets_no_hang(self):
        """Exiting scoped() with open sockets does not hang (sockets closed before term)."""
        with ManagedZMQContext.scoped() as ctx:
            pub = ctx.socket(zmq.PUB)
            sub = ctx.socket(zmq.SUB)
            ctx.bind(pub, "hang_test")
            ctx.connect(sub, "hang_test")
            # Do not close sockets manually; context cleanup should close them
        # If we get here without hanging, the test passed


# =============================================================================
# scoped() ownership
# =============================================================================


class TestManagedZMQContextScoped:
    """Tests for ManagedZMQContext.scoped() ownership and reuse."""

    def test_scoped_exit_with_own_cleans_up(self):
        """When scoped() owns the context, exit runs cleanup."""
        with ManagedZMQContext.scoped() as ctx:
            ctx.socket(zmq.PUB)
            assert ctx.ctx is not None
        # After exit, next scoped() creates a fresh context (previous was cleaned)
        with ManagedZMQContext.scoped() as ctx2:
            assert ctx2.ctx is not None
            ctx2.socket(zmq.PULL)

    def test_two_scoped_share_singleton_second_does_not_cleanup(self):
        """Nested or sequential scoped() with existing singleton: second does not cleanup."""
        # First scoped creates and owns
        with ManagedZMQContext.scoped() as ctx1:
            # Trigger socket_dir creation
            sock = ctx1.socket(zmq.PUB)
            ctx1.bind(sock, "test_nested")
            dir1 = ctx1.socket_dir
            # Second scoped: singleton already exists and initialized, so own=False
            with ManagedZMQContext.scoped() as ctx2:
                assert ctx1 is ctx2
                assert ctx2.socket_dir == dir1
            # After inner scoped exit, we did NOT call cleanup() (own=False)
            assert ctx1.ctx is not None
            assert ctx1.socket_dir == dir1
        # After outer exit, cleanup() ran (own=True)


# =============================================================================
# Mismatched socket_dir detection
# =============================================================================


class TestManagedZMQContextMismatchedSocketDir:
    """Tests for ManagedZMQContext raising on mismatched socket_dir."""

    def test_different_socket_dir_raises(self, tmp_path):
        """Re-init with a different explicit socket_dir raises ValueError."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        with ManagedZMQContext.scoped(socket_dir=str(dir_a)) as ctx:
            assert ctx.socket_dir == str(dir_a)
            with pytest.raises(ValueError, match="cannot reinitialize"):
                ManagedZMQContext(socket_dir=str(dir_b))

    def test_none_socket_dir_does_not_raise(self):
        """Re-init with socket_dir=None does not raise (caller doesn't care)."""
        with ManagedZMQContext.scoped() as ctx:
            # Trigger socket_dir creation
            sock = ctx.socket(zmq.PUB)
            ctx.bind(sock, "test_none")
            original_dir = ctx.socket_dir
            ManagedZMQContext(socket_dir=None)
            assert ctx.socket_dir == original_dir

    def test_same_socket_dir_does_not_raise(self, tmp_path):
        """Re-init with the same explicit socket_dir does not raise."""
        sock_dir = tmp_path / "same_dir"
        sock_dir.mkdir()

        with ManagedZMQContext.scoped(socket_dir=str(sock_dir)) as ctx:
            assert ctx.socket_dir == str(sock_dir)
            ManagedZMQContext(socket_dir=str(sock_dir))
