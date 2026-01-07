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

"""
Worker process isolated throughput benchmarks.

Tests:
  - Issue Rate: Max ZMQ send rate to worker (time-based)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvloop
import zmq.asyncio
from inference_endpoint.core.types import Query, QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.worker_manager import WorkerManager
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket
from inference_endpoint.testing.bare_response_server import BareResponseServerProcess

from tests.test_helpers import get_test_socket_path

# =============================================================================
# Configuration
# =============================================================================

# Issue rate: measure for this duration
ISSUE_RATE_DURATION_S = 5.0

# Prompt sizes in characters
PROMPT_SIZES = [128, 1024 * 4, 1024 * 16, 1024 * 32]

# =============================================================================
# Helpers
# =============================================================================


@dataclass
class Result:
    """Benchmark result with count and elapsed time."""

    count: int
    elapsed: float

    @property
    def qps(self) -> float:
        return self.count / self.elapsed if self.elapsed > 0 else 0


async def _setup_worker(server_url: str, tmp_path: Path):
    """Initialize worker and return (push, pull, manager, context)."""
    zmq_cfg = ZMQConfig(
        zmq_request_queue_prefix=get_test_socket_path(tmp_path, "worker", "_req"),
        zmq_response_queue_addr=get_test_socket_path(tmp_path, "worker", "_resp"),
        zmq_readiness_queue_addr=get_test_socket_path(tmp_path, "worker", "_ready"),
    )
    http_cfg = HTTPClientConfig(
        endpoint_url=f"{server_url}/v1/chat/completions",
        num_workers=1,
        log_level="ERROR",
        warmup_connections="auto-min",
    )

    ctx = zmq.asyncio.Context()
    push = ZMQPushSocket(ctx, f"{zmq_cfg.zmq_request_queue_prefix}_0_requests", zmq_cfg)
    pull = ZMQPullSocket(
        ctx,
        zmq_cfg.zmq_response_queue_addr,
        zmq_cfg,
        bind=True,
        decoder_type=QueryResult | StreamChunk,
    )

    mgr = WorkerManager(http_cfg, AioHttpConfig(), zmq_cfg, ctx)
    await mgr.initialize()

    return push, pull, mgr, ctx


async def _teardown(push, pull, mgr, ctx):
    """Shutdown worker and cleanup resources."""
    await mgr.shutdown()
    pull.close()
    push.close()
    ctx.destroy(linger=0)


async def _measure_issue_rate(
    streaming: bool,
    prompt_size: int,
    num_chunks: int,
    duration: float,
    server_url: str,
    tmp_path: Path,
) -> Result:
    """Measure max query issue rate (ZMQ send speed) for given duration."""
    push, pull, mgr, ctx = await _setup_worker(server_url, tmp_path)

    prompt = "x" * prompt_size
    count = 0
    t0 = time.perf_counter()
    t_end = t0 + duration

    while time.perf_counter() < t_end:
        await push.send(
            Query(id=f"q-{count}", data={"prompt": prompt, "stream": streaming})
        )
        count += 1

    elapsed = time.perf_counter() - t0
    await _teardown(push, pull, mgr, ctx)
    return Result(count, elapsed)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def event_loop():
    """Create uvloop event loop for each test."""
    loop = uvloop.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Tests - Issue Rate
# =============================================================================


@pytest.mark.timeout(0)
class TestWorkerIssueRate:
    """Test max ZMQ send rate to worker process."""

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.parametrize("prompt_size", PROMPT_SIZES)
    def test_issue_rate(self, prompt_size: int, event_loop, tmp_path):
        """Measure max issue rate for given prompt size."""

        async def run():
            async with BareResponseServerProcess(
                streaming=False, num_chunks=1, response_size=prompt_size
            ) as srv:
                return await _measure_issue_rate(
                    False, prompt_size, 1, ISSUE_RATE_DURATION_S, srv.url, tmp_path
                )

        result = event_loop.run_until_complete(run())
        print(f"\n  Issue rate prompt={prompt_size}: {result.qps:,.0f} QPS")
        assert result.qps > 0
