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

import pytest
import pytest_asyncio
from inference_endpoint.async_utils.transport import ZmqWorkerPoolTransport
from inference_endpoint.async_utils.transport.zmq.transport import ZMQTransportConfig
from inference_endpoint.core.types import Query, QueryResult, TextModelOutput

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
    """Provide a ZmqWorkerPoolTransport with auto-cleanup."""
    loop = asyncio.get_running_loop()
    zmq_pool = ZmqWorkerPoolTransport.create(loop, 1)
    yield zmq_pool
    zmq_pool.cleanup()


# =============================================================================
# Creation Tests
# =============================================================================


class TestZmqPoolCreation:
    """Tests for ZmqWorkerPoolTransport.create() factory."""

    def test_create_with_defaults(self, event_loop):
        """Basic creation with defaults."""
        zmq_pool = ZmqWorkerPoolTransport.create(event_loop, 4)
        try:
            assert zmq_pool is not None
            assert zmq_pool.worker_connector is not None
        finally:
            zmq_pool.cleanup()

    def test_create_with_overrides(self, event_loop):
        """Config overrides are applied."""
        config = ZMQTransportConfig(io_threads=8)
        zmq_pool = ZmqWorkerPoolTransport.create(event_loop, 2, config=config)
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

        async with connector.connect(0) as (worker_recv, worker_send):
            # Main sends request
            query = Query(id="test-1", data={"prompt": "hello"})
            zmq_pool.send(0, query)

            # Worker receives
            received = await worker_recv.recv()
            assert received.id == "test-1"
            assert received.data["prompt"] == "hello"

            # Worker sends response
            result = QueryResult(
                id="test-1", response_output=TextModelOutput(output="world")
            )
            worker_send.send(result)

            # Main receives via recv()
            response = await zmq_pool.recv()
            assert response.id == "test-1"
            assert response.response_output == TextModelOutput(output="world")

    @pytest.mark.asyncio
    async def test_poll_nonblocking(self, zmq_pool):
        """poll() returns None when empty, item when available."""
        connector = zmq_pool.worker_connector

        async with connector.connect(0) as (_, worker_send):
            # Empty - poll returns None immediately
            assert zmq_pool.poll() is None

            # Worker sends response
            worker_send.send(
                QueryResult(id="test", response_output=TextModelOutput(output="hi"))
            )
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
        zmq_pool = ZmqWorkerPoolTransport.create(loop, num_workers)
        connector = zmq_pool.worker_connector

        async def simulate_worker(worker_id: int):
            async with connector.connect(worker_id):
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
        zmq_pool = ZmqWorkerPoolTransport.create(loop, 1)

        try:
            async with zmq_pool.worker_connector.connect(0) as (
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
        async with zmq_pool.worker_connector.connect(0) as (
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
        [(0, "/2"), (1, "/3")],
        ids=["none_ready", "partial_ready"],
    )
    async def test_wait_for_workers_timeout(self, workers_started, expected_ready):
        """Timeout when not all workers connect."""
        loop = asyncio.get_running_loop()
        total_workers = 2 if workers_started == 0 else 3
        zmq_pool = ZmqWorkerPoolTransport.create(loop, total_workers)
        connector = zmq_pool.worker_connector

        async def simulate_worker(wid):
            async with connector.connect(wid):
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
        async with zmq_pool.worker_connector.connect(0) as (
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
        zmq_pool = ZmqWorkerPoolTransport.create(loop, 1)
        connector = zmq_pool.worker_connector

        async def simulate_worker():
            async with connector.connect(0) as (recv, send):
                # Transport close is idempotent
                recv.close()
                recv.close()
                send.close()
                send.close()

        worker_task = asyncio.create_task(simulate_worker())
        await zmq_pool.wait_for_workers_ready(timeout=5.0)
        await worker_task

        # Pool cleanup is idempotent
        zmq_pool.cleanup()
        zmq_pool.cleanup()
        zmq_pool.cleanup()

    @pytest.mark.asyncio
    async def test_operations_after_cleanup(self):
        """Operations after cleanup are safe."""
        loop = asyncio.get_running_loop()
        zmq_pool = ZmqWorkerPoolTransport.create(loop, 1)
        zmq_pool.cleanup()

        # send is silent
        zmq_pool.send(0, Query(id="x", data={}))

        # recv returns None
        result = await asyncio.wait_for(zmq_pool.recv(), timeout=0.1)
        assert result is None

        # poll returns None
        assert zmq_pool.poll() is None
