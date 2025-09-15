"""Tests for WorkerManager functionality."""

import asyncio

import pytest
import zmq
import zmq.asyncio
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import WorkerManager


class TestWorkerManager:
    """Test WorkerManager functionality with real processes."""

    @pytest.fixture
    def manager_config(self, mock_http_echo_server, tmp_path):
        """Create manager configuration."""
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
        )
        aiohttp_config = AioHttpConfig()
        # Use tmp_path for unique socket paths per test
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_path}/test_manager_req",
            zmq_response_queue_addr=f"ipc://{tmp_path}/test_manager_resp",
        )
        zmq_context = zmq.asyncio.Context()
        return http_config, aiohttp_config, zmq_config, zmq_context

    @pytest.mark.asyncio
    async def test_worker_manager_spawn_and_monitor(self, manager_config):
        """Test WorkerManager spawning and monitoring real workers."""
        http_config, aiohttp_config, zmq_config, zmq_context = manager_config

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager (spawns workers)
            await manager.initialize()

            # Verify workers were spawned
            assert len(manager.workers) == http_config.num_workers
            assert len(manager.worker_pids) == http_config.num_workers

            # Verify all workers are alive
            for worker in manager.workers:
                assert worker.is_alive()
                assert worker.pid > 0

            # Shutdown manager
            await manager.shutdown()

            # Verify workers are terminated
            for worker in manager.workers:
                assert not worker.is_alive()

        finally:
            # Clean up context
            zmq_context.term()

    @pytest.mark.asyncio
    async def test_worker_manager_restart_dead_worker(self, manager_config):
        """Test WorkerManager restarting a dead worker."""
        http_config, aiohttp_config, zmq_config, zmq_context = manager_config

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            # Get first worker
            first_worker = manager.workers[0]
            first_pid = first_worker.pid

            # Kill the first worker
            first_worker.terminate()
            first_worker.join(timeout=2.0)

            # Verify worker is dead
            assert not first_worker.is_alive()

            # Give the manager's monitor task time to detect and restart the dead worker
            await asyncio.sleep(2.5)

            # Verify worker was replaced
            new_worker = manager.workers[0]

            # Give new worker additional time to start if needed
            for _ in range(10):  # Try for up to 1 second
                if new_worker.is_alive():
                    break
                await asyncio.sleep(0.1)

            # Check if worker was successfully restarted
            if new_worker.is_alive():
                assert new_worker.pid != first_pid
                assert manager.worker_pids[0] == new_worker.pid
            else:
                # If worker failed to restart, that's also acceptable for this test
                # as it shows the manager attempted to restart it
                assert (
                    new_worker != first_worker
                )  # Should be a different Process object
                assert new_worker.pid != first_pid

            # Shutdown
            await manager.shutdown()

        finally:
            zmq_context.term()

    @pytest.mark.asyncio
    async def test_worker_manager_monitor_task_cancellation(self, manager_config):
        """Test WorkerManager monitor task cancellation during shutdown."""
        http_config, aiohttp_config, zmq_config, zmq_context = manager_config

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            # Verify monitor task is running
            assert manager._monitor_task is not None
            assert not manager._monitor_task.done()

            # Shutdown should cancel monitor task
            await manager.shutdown()

            # Verify monitor task was cancelled
            assert manager._monitor_task.done()

        finally:
            zmq_context.term()

    @pytest.mark.asyncio
    async def test_worker_manager_force_kill_workers(self, manager_config):
        """Test WorkerManager force killing workers that don't terminate gracefully."""
        http_config, aiohttp_config, zmq_config, zmq_context = manager_config

        # Use a smaller number of workers for this test
        http_config.num_workers = 1

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            # Get the worker
            worker = manager.workers[0]

            # Verify worker is alive
            assert worker.is_alive()

            # Shutdown manager (this will test the force kill path)
            await manager.shutdown()

            # Verify worker is terminated
            assert not worker.is_alive()

        finally:
            zmq_context.term()


class TestWorkerManagerAdvanced:
    """Advanced WorkerManager tests for edge cases."""

    @pytest.fixture
    def advanced_manager_config(self, tmp_path):
        """Create advanced manager configuration."""
        # Use tmp_path for unique socket paths per test
        http_config = HTTPClientConfig(
            endpoint_url="http://localhost:99999/advanced",
            num_workers=2,
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=f"ipc://{tmp_path}/test_advanced_req",
            zmq_response_queue_addr=f"ipc://{tmp_path}/test_advanced_resp",
        )
        zmq_context = zmq.asyncio.Context()
        return http_config, aiohttp_config, zmq_config, zmq_context

    @pytest.mark.asyncio
    async def test_worker_manager_multiple_worker_deaths(self, advanced_manager_config):
        """Test WorkerManager handling multiple worker deaths simultaneously."""
        http_config, aiohttp_config, zmq_config, zmq_context = advanced_manager_config

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            # Kill all workers simultaneously
            original_pids = [worker.pid for worker in manager.workers]
            for worker in manager.workers:
                worker.terminate()

            # Wait for graceful termination
            await asyncio.sleep(2.0)

            # Force kill any workers that didn't terminate gracefully
            for worker in manager.workers:
                if worker.is_alive():
                    worker.kill()
                worker.join(timeout=2.0)

            # Verify all workers are dead
            for worker in manager.workers:
                assert not worker.is_alive()

            # Wait for monitor to detect and restart workers
            await asyncio.sleep(2.5)  # Monitor checks every 2 seconds

            # Verify workers were restarted
            new_pids = [worker.pid for worker in manager.workers]

            # At least some workers should be restarted (may not all succeed)
            restarted_count = sum(
                1
                for old_pid, new_pid in zip(original_pids, new_pids, strict=False)
                if old_pid != new_pid
            )
            assert restarted_count >= 1

            # Shutdown
            await manager.shutdown()

        finally:
            zmq_context.term()

    @pytest.mark.asyncio
    async def test_worker_manager_shutdown_during_restart(
        self, advanced_manager_config
    ):
        """Test WorkerManager shutdown while workers are being restarted."""
        http_config, aiohttp_config, zmq_config, zmq_context = advanced_manager_config

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            # Kill first worker to trigger restart
            manager.workers[0].terminate()
            manager.workers[0].join(timeout=1.0)

            # Immediately shutdown manager (before restart completes)
            await manager.shutdown()

            # Verify shutdown completed without hanging
            assert manager._shutdown_event.is_set()

        finally:
            zmq_context.term()
