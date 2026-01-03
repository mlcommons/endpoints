# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Integration tests for HttpClient worker manager functionality."""

import asyncio
import os
import signal
import subprocess

import pytest
import zmq
import zmq.asyncio
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker import WorkerManager

from ...test_helpers import get_test_socket_path

# timeout for OS to handle process signals
TEST_WORKER_POST_KILL_DELAY_S = 0.5


def check_for_zombies(pids: list[int], timeout: float = 1.0) -> list[int]:
    """
    Check which PIDs are zombie processes (non-asserting query).

    Uses ps command to check process state. Process states:
      - R: Running
      - S: Sleeping
      - Z: Zombie (defunct, waiting for parent to call join() to reap)
      - T: Stopped
      - +: Foreground process group

    Args:
        pids: List of process IDs to check
        timeout: Timeout for subprocess calls

    Returns:
        List of PIDs that are zombies (empty list if none found)

    Raises:
        RuntimeError: If ps command fails with unexpected error
    """
    zombie_pids = []

    for pid in pids:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        match result.returncode:
            case 0:
                # Process exists - check if it's a zombie
                stat = result.stdout.strip()
                if "Z" in stat:
                    zombie_pids.append(pid)
            case 1:
                # Process doesn't exist (already reaped) - this is fine
                pass
            case _:
                # Unexpected error
                raise RuntimeError(
                    f"ps command failed for PID {pid} with code {result.returncode}: "
                    f"{result.stderr}"
                )

    return zombie_pids


def assert_no_zombies(pids: list[int], timeout: float = 1.0) -> None:
    """
    Assert that none of the given PIDs are zombie processes.

    Convenience wrapper around check_for_zombies() that fails the test
    if any zombies are detected.

    Args:
        pids: List of process IDs to check
        timeout: Timeout for subprocess calls

    Raises:
        AssertionError: If any process is a zombie
        RuntimeError: If ps command fails with unexpected error
    """
    zombie_pids = check_for_zombies(pids, timeout)
    assert (
        len(zombie_pids) == 0
    ), f"Found {len(zombie_pids)} zombie process(es): {zombie_pids}"


class TestWorkerLifecycle:
    """Test basic worker spawning, lifecycle, and shutdown."""

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
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_manager", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_manager", "_resp"
            ),
        )
        zmq_context = zmq.asyncio.Context()
        return http_config, aiohttp_config, zmq_config, zmq_context

    @pytest.mark.asyncio
    async def test_spawn_workers_and_graceful_shutdown(self, manager_config):
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
            zmq_context.destroy(linger=0)

    @pytest.mark.parametrize(
        "signal_type,signal_method",
        [
            ("SIGTERM", "terminate"),
            ("SIGINT", signal.SIGINT),
            ("SIGKILL", "kill"),
        ],
        ids=["sigterm", "sigint", "sigkill"],
    )
    @pytest.mark.asyncio
    async def test_signal_handling_and_zombie_reaping(
        self, manager_config, signal_type, signal_method
    ):
        """Test workers handle signals correctly and are reaped without leaving zombies."""
        http_config, aiohttp_config, zmq_config, zmq_context = manager_config
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

            worker = manager.workers[0]
            worker_pid = worker.pid
            assert worker.is_alive()

            # Send signal based on type
            if signal_method == "terminate":
                worker.terminate()  # SIGTERM
            elif signal_method == "kill":
                worker.kill()  # SIGKILL
            elif isinstance(signal_method, int):
                os.kill(worker_pid, signal_method)  # SIGINT or other

            # Give time for signal to be processed
            await asyncio.sleep(TEST_WORKER_POST_KILL_DELAY_S)

            # Check zombie state before shutdown
            zombies_before = check_for_zombies([worker_pid])
            if zombies_before:
                print(f"✓ Worker {worker_pid} became zombie after {signal_type}")
            elif not worker.is_alive():
                print(f"✓ Worker {worker_pid} exited after {signal_type}")
            else:
                print(f"⚠ Worker {worker_pid} still running after {signal_type}")

            # Shutdown should handle the worker properly
            await manager.shutdown()

            # Verify worker is dead and reaped
            assert not worker.is_alive(), f"Worker should be dead after {signal_type}"
            assert (
                worker.exitcode is not None
            ), f"Worker should be reaped after {signal_type}"
            assert_no_zombies([worker_pid])

            print(f"✓ Worker properly handled {signal_type} and was reaped")

        finally:
            zmq_context.destroy(linger=0)

    @pytest.mark.asyncio
    async def test_multiple_workers_with_mixed_signals(self, manager_config):
        """Test shutdown handles multiple workers killed with different signals simultaneously."""
        http_config, aiohttp_config, zmq_config, zmq_context = manager_config
        http_config.num_workers = 3

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            assert len(manager.workers) == 3
            worker_pids = [w.pid for w in manager.workers]

            # Kill workers with different signals (realistic mixed scenario)
            manager.workers[0].kill()  # SIGKILL - immediate death
            manager.workers[1].terminate()  # SIGTERM - graceful
            os.kill(manager.workers[2].pid, signal.SIGINT)  # SIGINT - interrupt

            # Give time for signal to be processed
            await asyncio.sleep(TEST_WORKER_POST_KILL_DELAY_S)

            # Check which became zombies
            zombies_before = check_for_zombies(worker_pids)
            print(
                f"Mixed signals: {len(zombies_before)}/{len(worker_pids)} zombies before shutdown"
            )

            # Shutdown should handle all workers regardless of how they died
            await manager.shutdown()

            # Verify all workers are dead and reaped
            for i, worker in enumerate(manager.workers):
                assert not worker.is_alive(), f"Worker {i} should be dead"
                assert worker.exitcode is not None, f"Worker {i} should be reaped"

            # No zombies should remain
            assert_no_zombies(worker_pids)
            print(
                f"✓ All {len(worker_pids)} workers properly reaped after mixed signals"
            )

        finally:
            zmq_context.destroy(linger=0)


class TestWorkerDeathScenarios:
    """Test edge cases: multiple worker deaths, concurrent failures, and cleanup."""

    @pytest.fixture
    def worker_death_config(self, tmp_path, mock_http_echo_server):
        """Create configuration for worker death scenario tests."""
        # Use tmp_path for unique socket paths per test
        http_config = HTTPClientConfig(
            endpoint_url=f"{mock_http_echo_server.url}/v1/chat/completions",
            num_workers=2,
        )
        aiohttp_config = AioHttpConfig()
        zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_advanced", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_advanced", "_resp"
            ),
        )
        zmq_context = zmq.asyncio.Context()
        return http_config, aiohttp_config, zmq_config, zmq_context

    @pytest.mark.asyncio
    async def test_all_workers_killed_simultaneously(self, worker_death_config):
        """Test shutdown reaps all zombies when all workers are killed at once."""
        http_config, aiohttp_config, zmq_config, zmq_context = worker_death_config

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            # Kill all workers forcefully to create zombies
            original_pids = [worker.pid for worker in manager.workers]
            for worker in manager.workers:
                worker.kill()

            # Give time for signal to be processed
            await asyncio.sleep(TEST_WORKER_POST_KILL_DELAY_S)

            # Verify all workers are dead
            for worker in manager.workers:
                assert not worker.is_alive()

            # Verify workers are same objects (not replaced)
            for i, worker in enumerate(manager.workers):
                assert worker.pid == original_pids[i]
                assert not worker.is_alive()

            # CRITICAL: Check for zombies BEFORE shutdown
            zombies_before = check_for_zombies(original_pids)
            print(
                f"Workers before shutdown: {len(original_pids)} total, {len(zombies_before)} zombies"
            )
            if zombies_before:
                print(
                    f"✓ Verified: {len(zombies_before)} zombie(s) exist before shutdown: {zombies_before}"
                )
            else:
                print("⚠ All workers auto-reaped by OS before shutdown check")

            # Shutdown should reap all zombies
            await manager.shutdown()

            # CRITICAL: Verify NO zombies after shutdown
            zombies_after = check_for_zombies(original_pids)
            assert (
                len(zombies_after) == 0
            ), f"Shutdown failed to reap {len(zombies_after)} zombie(s): {zombies_after}"

            # Verify all zombies were reaped using assert_no_zombies
            assert_no_zombies(original_pids)

            if zombies_before:
                print(
                    f"✓ Verified: All {len(zombies_before)} zombie(s) were reaped by shutdown"
                )

        finally:
            zmq_context.destroy(linger=0)

    @pytest.mark.asyncio
    async def test_shutdown_with_preexisting_dead_worker(self, worker_death_config):
        """Test shutdown gracefully handles workers that died before shutdown was called."""
        http_config, aiohttp_config, zmq_config, zmq_context = worker_death_config

        manager = WorkerManager(
            http_config=http_config,
            aiohttp_config=aiohttp_config,
            zmq_config=zmq_config,
            zmq_context=zmq_context,
        )

        try:
            # Initialize manager
            await manager.initialize()

            # Track all PIDs for zombie verification
            all_pids = [worker.pid for worker in manager.workers]
            dead_pid = manager.workers[0].pid

            # Kill first worker
            manager.workers[0].terminate()

            # Give time for signal to be processed
            await asyncio.sleep(TEST_WORKER_POST_KILL_DELAY_S)

            # Check for zombies before shutdown
            zombies_before = check_for_zombies([dead_pid])
            if zombies_before:
                print(f"✓ Verified: Worker {dead_pid} is zombie before shutdown")

            # Immediately shutdown manager (should handle dead worker gracefully)
            await manager.shutdown()

            # Verify shutdown completed without hanging
            assert manager._shutdown_event.is_set()

            # Verify all workers are dead and reaped (no zombies)
            for worker in manager.workers:
                assert not worker.is_alive()
                assert (
                    worker.exitcode is not None
                ), "Worker should have exit code (been reaped)"

            # Verify no zombies in process table after shutdown
            assert_no_zombies(all_pids)

            if zombies_before:
                zombies_after = check_for_zombies([dead_pid])
                assert (
                    len(zombies_after) == 0
                ), f"Shutdown failed to reap zombie {dead_pid}"
                print(f"✓ Verified: Zombie {dead_pid} was reaped by shutdown")

        finally:
            zmq_context.destroy(linger=0)
