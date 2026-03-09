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

"""Unit tests for ZMQ transport layer.

Tests the ZmqWorkerPoolTransport and related components in isolation,
without requiring external HTTP servers or real child processes.
"""

import asyncio
import errno
from unittest.mock import patch

import pytest
import pytest_asyncio
import zmq
from inference_endpoint.async_utils.transport import ZmqWorkerPoolTransport
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.types import Query, QueryResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_loop():
    """Provide a sync event loop for non-async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def zmq_pool():
    """Provide a ZmqWorkerPoolTransport with auto-cleanup, scoped to ManagedZMQContext."""
    loop = asyncio.get_running_loop()
    with ManagedZMQContext.scoped() as zmq_ctx:
        zmq_pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
        yield zmq_pool
        zmq_pool.cleanup()


# =============================================================================
# Creation Tests
# =============================================================================


class TestZmqPoolCreation:
    """Tests for ZmqWorkerPoolTransport.create() factory."""

    def test_create_with_defaults(self, event_loop):
        """Basic creation with defaults."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            zmq_pool = ZmqWorkerPoolTransport.create(event_loop, 4, zmq_ctx)
            try:
                assert zmq_pool is not None
                assert zmq_pool.worker_connector is not None
            finally:
                zmq_pool.cleanup()

    def test_create_with_overrides(self, event_loop):
        """Config overrides are applied."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            zmq_pool = ZmqWorkerPoolTransport.create(
                event_loop, 2, zmq_ctx, io_threads=8
            )
            try:
                assert zmq_pool._config.io_threads == 8
            finally:
                zmq_pool.cleanup()


# =============================================================================
# Communication Tests
# =============================================================================


class TestZmqCommunication:
    """Tests for send/receive functionality."""

    @pytest.mark.asyncio
    async def test_send_recv_roundtrip(self, zmq_pool):
        """Basic send→recv roundtrip."""
        connector = zmq_pool.worker_connector
        zmq_ctx = ManagedZMQContext()

        async with connector.connect(0, zmq_ctx) as (worker_recv, worker_send):
            # Main sends request
            query = Query(id="test-1", data={"prompt": "hello"})
            zmq_pool.send(0, query)

            # Worker receives
            received = await worker_recv.recv()
            assert received.id == "test-1"
            assert received.data["prompt"] == "hello"

            # Worker sends response
            result = QueryResult(id="test-1", response_output="world")
            worker_send.send(result)

            # Main receives via recv()
            response = await zmq_pool.recv()
            assert response.id == "test-1"
            assert response.response_output == "world"

    @pytest.mark.asyncio
    async def test_poll_nonblocking(self, zmq_pool):
        """poll() returns None when empty, item when available."""
        connector = zmq_pool.worker_connector
        zmq_ctx = ManagedZMQContext()

        async with connector.connect(0, zmq_ctx) as (_, worker_send):
            # Empty - poll returns None immediately
            assert zmq_pool.poll() is None

            # Worker sends response
            worker_send.send(QueryResult(id="test", response_output="hi"))
            await asyncio.sleep(0.01)  # Let event loop process

            # Available - poll returns item
            result = zmq_pool.poll()
            assert result is not None
            assert result.id == "test"

            # Empty again
            assert zmq_pool.poll() is None

    @pytest.mark.asyncio
    async def test_multiple_workers_readiness(self):
        """Multiple workers can signal readiness."""
        loop = asyncio.get_running_loop()
        num_workers = 4
        with ManagedZMQContext.scoped() as zmq_ctx:
            zmq_pool = ZmqWorkerPoolTransport.create(loop, num_workers, zmq_ctx)
            connector = zmq_pool.worker_connector

            async def simulate_worker(worker_id: int):
                async with connector.connect(worker_id, zmq_ctx):
                    await asyncio.sleep(1.0)

            worker_tasks = [
                asyncio.create_task(simulate_worker(i)) for i in range(num_workers)
            ]

            try:
                await zmq_pool.wait_for_workers_ready(timeout=0.5)
            finally:
                for task in worker_tasks:
                    task.cancel()
                await asyncio.gather(*worker_tasks, return_exceptions=True)
                zmq_pool.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {"id": "simple", "prompt": "Hello"},
            {"id": "large", "prompt": "x" * 10000},
            {"id": "unicode", "prompt": "你好 🚀"},
        ],
    )
    async def test_payload_variants(self, payload):
        """Various payload types serialize correctly."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            zmq_pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            try:
                async with zmq_pool.worker_connector.connect(0, zmq_ctx) as (
                    worker_recv,
                    _,
                ):
                    query = Query(id=payload["id"], data={"prompt": payload["prompt"]})
                    zmq_pool.send(0, query)

                    received = await worker_recv.recv()
                    assert received.id == payload["id"]
                    assert received.data["prompt"] == payload["prompt"]
            finally:
                zmq_pool.cleanup()

    @pytest.mark.asyncio
    async def test_messages_preserve_order(self, zmq_pool):
        """Multiple queued messages are received in order."""
        zmq_ctx = ManagedZMQContext()
        async with zmq_pool.worker_connector.connect(0, zmq_ctx) as (
            worker_recv,
            _,
        ):
            num_messages = 10
            for i in range(num_messages):
                zmq_pool.send(0, Query(id=f"msg-{i}", data={"seq": i}))

            for i in range(num_messages):
                received = await asyncio.wait_for(worker_recv.recv(), timeout=0.5)
                assert received.id == f"msg-{i}"
                assert received.data["seq"] == i


# =============================================================================
# Robustness Tests
# =============================================================================


class TestZmqRobustness:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers_started,expected_ready",
        [(0, "0/2"), (1, "1/3")],
        ids=["none_ready", "partial_ready"],
    )
    async def test_wait_for_workers_timeout(self, workers_started, expected_ready):
        """Timeout when not all workers connect."""
        loop = asyncio.get_running_loop()
        total_workers = 2 if workers_started == 0 else 3
        with ManagedZMQContext.scoped() as zmq_ctx:
            zmq_pool = ZmqWorkerPoolTransport.create(loop, total_workers, zmq_ctx)
            connector = zmq_pool.worker_connector

            async def simulate_worker(wid):
                async with connector.connect(wid, zmq_ctx):
                    await asyncio.sleep(1.0)

            tasks = [
                asyncio.create_task(simulate_worker(i)) for i in range(workers_started)
            ]

            try:
                with pytest.raises(TimeoutError) as exc_info:
                    await zmq_pool.wait_for_workers_ready(timeout=0.05)
                assert expected_ready in str(exc_info.value)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                zmq_pool.cleanup()

    @pytest.mark.asyncio
    async def test_close_wakes_pending_recv(self, zmq_pool):
        """Closing receiver wakes pending recv() with None."""
        zmq_ctx = ManagedZMQContext()
        async with zmq_pool.worker_connector.connect(0, zmq_ctx) as (
            worker_recv,
            _,
        ):
            worker_recv.close()
            result = await asyncio.wait_for(worker_recv.recv(), timeout=0.5)
            assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self):
        """cleanup() and close() are idempotent."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            zmq_pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)

            async with zmq_pool.worker_connector.connect(0, zmq_ctx) as (
                recv,
                send,
            ):
                # Transport close is idempotent
                recv.close()
                recv.close()
                send.close()
                send.close()

            # Pool cleanup is idempotent
            zmq_pool.cleanup()
            zmq_pool.cleanup()
            zmq_pool.cleanup()

    @pytest.mark.asyncio
    async def test_operations_after_cleanup(self):
        """Operations after cleanup are safe."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            zmq_pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            zmq_pool.cleanup()

            # send is silent
            zmq_pool.send(0, Query(id="x", data={}))

            # recv returns None
            result = await asyncio.wait_for(zmq_pool.recv(), timeout=0.1)
            assert result is None

            # poll returns None
            assert zmq_pool.poll() is None


class TestReceiverCloseBehavior:
    """Tests for _ZmqReceiverTransport close and shutdown paths."""

    @pytest.mark.asyncio
    async def test_receiver_recv_returns_none_when_closed(self):
        """recv() returns None immediately after close() with empty deque."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            # Close the receiver
            receiver.close()

            # recv should return None immediately
            result = await receiver.recv()
            assert result is None

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_receiver_on_readable_returns_early_when_closing(self):
        """_on_readable returns early without draining when _closing is True."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            # Set closing flag
            receiver._closing = True

            # Call _on_readable directly -- it should return early
            # without touching the socket or deque
            deque_len_before = len(receiver._deque)
            receiver._on_readable()
            assert len(receiver._deque) == deque_len_before

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_receiver_close_cancels_pending_soon_call(self):
        """close() cancels a pending _soon_call handle."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            # Manually set a pending callback to simulate reschedule
            receiver._soon_call = loop.call_soon(lambda: None)
            assert receiver._soon_call is not None

            # Close should cancel it
            receiver.close()
            assert receiver._soon_call is None
            assert receiver._closing

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_receiver_close_handles_already_removed_reader(self):
        """close() handles ValueError/OSError from remove_reader gracefully."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            # Remove the reader before close, so close() will get an error
            try:
                loop.remove_reader(receiver._fd)
            except (ValueError, OSError):
                pass

            # close() should not raise even though reader is already removed
            receiver.close()
            assert receiver._closing

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_receiver_recv_returns_none_when_closed_during_wait(self):
        """recv() returns None when close() is called while waiting."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            # Schedule close after a short delay
            async def close_after_delay():
                await asyncio.sleep(0.01)
                receiver.close()

            close_task = asyncio.create_task(close_after_delay())

            # recv() should block until close wakes the waiter
            result = await asyncio.wait_for(receiver.recv(), timeout=1.0)
            assert result is None

            await close_task
            pool.cleanup()


class TestReceiverErrorHandling:
    """Tests for ZMQ error handling paths in _ZmqReceiverTransport."""

    @pytest.mark.asyncio
    async def test_receiver_on_readable_zmq_error_logged(self):
        """_on_readable logs ZMQError with unexpected errno."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            # Mock the socket to raise ZMQError with an unexpected errno
            original_recv = receiver._sock.recv
            call_count = 0

            def mock_recv(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    err = zmq.ZMQError(errno.ECONNREFUSED, "Connection refused")
                    raise err
                raise zmq.Again()

            receiver._sock.recv = mock_recv

            # Should not raise -- error is logged
            receiver._on_readable()

            # Restore
            receiver._sock.recv = original_recv
            pool.cleanup()

    @pytest.mark.asyncio
    async def test_receiver_on_readable_zmq_error_ignored_errno(self):
        """_on_readable silently handles ZMQError with EINTR/ENOTSOCK errno."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            for err_no in (errno.EINTR, errno.ENOTSOCK):
                original_recv = receiver._sock.recv

                def mock_recv(*args, _err_no=err_no, **kwargs):
                    raise zmq.ZMQError(_err_no)

                receiver._sock.recv = mock_recv

                # Should not raise -- these errnos are silently ignored
                receiver._on_readable()

                receiver._sock.recv = original_recv

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_receiver_on_readable_reschedule_zmq_error(self):
        """_on_readable handles ZMQError from getsockopt during reschedule."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            receiver = pool._response_receiver

            import msgspec

            # Create a valid msgpack payload that the decoder can handle
            encoder = msgspec.msgpack.Encoder()
            fake_result = QueryResult(id="fake", response_output="x")
            fake_payload = encoder.encode(fake_result)

            # Mock recv: succeed once with fake data, then raise Again
            original_recv = receiver._sock.recv
            call_count = 0

            def mock_recv(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return fake_payload
                raise zmq.Again()

            # Mock getsockopt to raise ZMQError for EVENTS
            original_getsockopt = receiver._sock.getsockopt

            def mock_getsockopt(opt):
                if opt == zmq.EVENTS:
                    raise zmq.ZMQError(errno.ENOTSOCK, "Socket closed")
                return original_getsockopt(opt)

            receiver._sock.recv = mock_recv
            receiver._sock.getsockopt = mock_getsockopt

            # Call _on_readable -- should succeed on recv, then catch
            # ZMQError from getsockopt without raising
            receiver._on_readable()

            # Verify that a message was drained (count > 0 path was taken)
            assert len(receiver._deque) == 1

            # Restore
            receiver._sock.recv = original_recv
            receiver._sock.getsockopt = original_getsockopt

            pool.cleanup()


class TestSenderBehavior:
    """Tests for _ZmqSenderTransport internal methods."""

    @pytest.mark.asyncio
    async def test_sender_on_writable_returns_when_closing(self):
        """_on_writable returns immediately when _closing is True."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Buffer a message and set closing
            sender._buffer.append(b"test-data")
            sender._closing = True

            sender._on_writable()

            # Buffer should NOT be drained (returned early)
            assert len(sender._buffer) == 1

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_sender_on_writable_drains_buffer(self):
        """_on_writable drains buffered messages and stops writing."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Connect a worker so the PUSH socket has a peer
            connector = pool.worker_connector
            async with connector.connect(0, zmq_ctx) as (worker_recv, _):
                await asyncio.sleep(0.01)  # Let connection establish

                # Manually buffer a message (simulating slow path)
                encoded = sender._encoder.encode({"test": "data"})
                sender._buffer.append(encoded)
                sender._writing = True
                loop.add_writer(sender._fd, sender._on_writable)

                # Call _on_writable directly
                sender._on_writable()

                # Buffer should be drained and writing stopped
                assert len(sender._buffer) == 0
                assert not sender._writing

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_sender_on_writable_zmq_again_keeps_buffer(self):
        """_on_writable handles zmq.Again by keeping remaining buffer items."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Mock socket to always raise Again
            original_send = sender._sock.send

            def mock_send(*args, **kwargs):
                raise zmq.Again()

            sender._sock.send = mock_send
            sender._buffer.append(b"data1")
            sender._buffer.append(b"data2")
            sender._writing = True

            sender._on_writable()

            # Buffer should still have items (Again means socket would block)
            assert len(sender._buffer) == 2
            # _soon_call should be set for reschedule
            assert sender._soon_call is not None

            # Cancel the scheduled callback to avoid it firing
            sender._soon_call.cancel()
            sender._soon_call = None

            sender._sock.send = original_send
            pool.cleanup()

    @pytest.mark.asyncio
    async def test_sender_on_writable_zmq_error_clears_buffer(self):
        """_on_writable clears buffer and stops writing on unexpected ZMQError."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Mock socket to raise ZMQError with unexpected errno
            original_send = sender._sock.send

            def mock_send(*args, **kwargs):
                raise zmq.ZMQError(errno.ECONNREFUSED, "Connection refused")

            sender._sock.send = mock_send
            sender._buffer.append(b"data1")
            sender._buffer.append(b"data2")
            sender._writing = True
            loop.add_writer(sender._fd, sender._on_writable)

            sender._on_writable()

            # Buffer should be cleared and writing stopped
            assert len(sender._buffer) == 0
            assert not sender._writing

            sender._sock.send = original_send
            pool.cleanup()

    @pytest.mark.asyncio
    async def test_sender_on_writable_reschedules_with_remaining_buffer(self):
        """_on_writable reschedules via call_soon when buffer has remaining items."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Mock socket: succeed on first, raise Again on second
            original_send = sender._sock.send
            call_count = 0

            def mock_send(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return original_send(*args, **kwargs)
                raise zmq.Again()

            # Connect a worker so send can succeed
            connector = pool.worker_connector
            async with connector.connect(0, zmq_ctx) as (worker_recv, _):
                await asyncio.sleep(0.01)

                sender._sock.send = mock_send
                sender._buffer.append(b"data1")
                sender._buffer.append(b"data2")
                sender._writing = True

                sender._on_writable()

                # First message sent, second blocked by Again
                # Buffer should have 1 remaining
                assert len(sender._buffer) == 1
                # _soon_call should be set for reschedule
                assert sender._soon_call is not None

                # Cancel the scheduled callback
                sender._soon_call.cancel()
                sender._soon_call = None

                sender._sock.send = original_send

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_sender_stop_writing_handles_already_removed_writer(self):
        """_stop_writing handles ValueError/OSError from remove_writer."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Set writing flag but don't actually add a writer
            sender._writing = True

            # _stop_writing should handle the error from remove_writer gracefully
            sender._stop_writing()
            assert not sender._writing

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_sender_close_cancels_pending_callback(self):
        """close() cancels pending _soon_call handle on sender."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Manually set a pending callback
            sender._soon_call = loop.call_soon(lambda: None)
            assert sender._soon_call is not None

            # Close should cancel it
            sender.close()
            assert sender._soon_call is None
            assert sender._closing

            pool.cleanup()

    @pytest.mark.asyncio
    async def test_sender_send_fast_path_zmq_error(self):
        """send() handles ZMQError on fast-path direct send."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            sender = pool._request_senders[0]

            # Mock socket to raise ZMQError with unexpected errno on send
            original_send = sender._sock.send

            def mock_send(*args, **kwargs):
                raise zmq.ZMQError(errno.ECONNREFUSED, "Connection refused")

            sender._sock.send = mock_send

            # send should not raise -- error path returns silently
            query = Query(id="test-err", data={"prompt": "hello"})
            sender.send(query)

            # Buffer should be empty (error on fast path returns, does not buffer)
            assert len(sender._buffer) == 0

            sender._sock.send = original_send
            pool.cleanup()


class TestZmqWorkerPoolTransportCreation:
    """Tests for ZmqWorkerPoolTransport factory and lifecycle methods."""

    def test_create_rejects_windows(self, event_loop):
        """create() raises RuntimeError on Windows."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            with patch(
                "inference_endpoint.async_utils.transport.zmq.transport.os"
            ) as mock_os:
                mock_os.name = "nt"
                with pytest.raises(RuntimeError, match="Windows not yet supported"):
                    ZmqWorkerPoolTransport.create(event_loop, 1, zmq_ctx)

    def test_create_rejects_long_ipc_paths(self, event_loop):
        """create() raises ValueError when IPC path exceeds max length."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            # Override socket_dir with a very long path to trigger the check
            original_socket_dir = zmq_ctx.socket_dir
            zmq_ctx.socket_dir = "/tmp/" + "a" * 200

            with pytest.raises(ValueError, match="IPC path too long"):
                ZmqWorkerPoolTransport.create(event_loop, 1, zmq_ctx)

            zmq_ctx.socket_dir = original_socket_dir

    @pytest.mark.asyncio
    async def test_wait_for_workers_ready_no_timeout(self):
        """wait_for_workers_ready with timeout=None uses plain recv()."""
        loop = asyncio.get_running_loop()
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(loop, 1, zmq_ctx)
            connector = pool.worker_connector

            async def simulate_worker():
                async with connector.connect(0, zmq_ctx):
                    await asyncio.sleep(0.5)

            worker_task = asyncio.create_task(simulate_worker())

            try:
                # Use asyncio.wait_for to prevent hanging if something goes wrong
                await asyncio.wait_for(
                    pool.wait_for_workers_ready(timeout=None),
                    timeout=2.0,
                )
            finally:
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
                pool.cleanup()

    def test_del_calls_cleanup(self, event_loop):
        """__del__ calls cleanup without raising."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(event_loop, 1, zmq_ctx)

            # Call __del__ directly -- should not raise
            pool.__del__()
            assert pool._closed

    def test_del_suppresses_exceptions(self, event_loop):
        """__del__ suppresses exceptions from cleanup()."""
        with ManagedZMQContext.scoped() as zmq_ctx:
            pool = ZmqWorkerPoolTransport.create(event_loop, 1, zmq_ctx)

            # Mock cleanup to raise an exception
            original_cleanup = pool.cleanup

            def raising_cleanup():
                raise RuntimeError("Simulated cleanup failure")

            pool.cleanup = raising_cleanup

            # __del__ should not raise
            pool.__del__()

            # Restore and do real cleanup
            pool.cleanup = original_cleanup
            pool.cleanup()
