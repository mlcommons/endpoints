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
Response handler throughput benchmark.

Measures the roofline performance of HttpClientSampleIssuer._handle_responses
by pre-loading responses into ZMQ and benchmarking how fast they can be processed.

Uses the actual _handle_responses method from HttpClientSampleIssuer with proper
EventRecorder and SampleEventHandler setup matching production benchmark runs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvloop
import zmq.asyncio
from inference_endpoint.core.types import QueryResult, StreamChunk
from inference_endpoint.endpoint_client.configs import ZMQConfig
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.endpoint_client.zmq_utils import ZMQPullSocket, ZMQPushSocket
from inference_endpoint.load_generator.events import SampleEvent, SessionEvent
from inference_endpoint.load_generator.sample import SampleEventHandler
from inference_endpoint.metrics.recorder import EventRecorder

from tests.test_helpers import get_test_socket_path

# =============================================================================
# Configuration
# =============================================================================

NUM_RESPONSES = 25_000

# Response sizes to test (characters)
RESPONSE_SIZES = [128, 1024 * 4, 1024 * 16, 1024 * 32]

# Streaming rates: fraction of messages that are StreamChunks
# NOTE(vir):
# 0.99 excluded - at 25K requests with 99 chunks each, generates ~2.5M events
# EventRecorder writer thread commits 1000 events/batch and cannot drain the queue
# before the 10s close_timeout_s, causing RuntimeError on context exit.
STREAMING_RATES = [0.0, 0.50, 0.75, 0.90]


# =============================================================================
# Helpers
# =============================================================================


def streaming_rate_to_chunks(rate: float) -> int:
    """Convert streaming rate to chunks per request.

    streaming_rate = chunks / (chunks + 1)
    Solving for chunks: chunks = rate / (1 - rate)
    """
    if rate <= 0.0:
        return 0
    if rate >= 1.0:
        raise ValueError("streaming_rate must be < 1.0")
    return round(rate / (1.0 - rate))


@dataclass(slots=True)
class CompletionCounter:
    """Tracks completion count for verification."""

    count: int = 0
    target: int = 0
    done_event: threading.Event | None = None

    def increment(self, _: QueryResult) -> None:
        self.count += 1
        if self.done_event and self.count >= self.target:
            self.done_event.set()


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    count: int
    elapsed: float

    @property
    def rate(self) -> float:
        return self.count / self.elapsed if self.elapsed > 0 else 0


class MockHttpClient:
    """
    Mock HTTP client for testing HttpClientSampleIssuer._handle_responses.

    Provides the interface that HttpClientSampleIssuer expects:
    - loop: event loop for scheduling coroutines
    - try_receive(): async method to receive responses from ZMQ
    """

    def __init__(self, pull_socket: ZMQPullSocket, loop: asyncio.AbstractEventLoop):
        self._pull_socket = pull_socket
        self.loop = loop

    async def try_receive(self) -> QueryResult | StreamChunk | None:
        return await self._pull_socket.receive()


def _create_messages(
    num_requests: int,
    response_size: int,
    chunks_per_request: int,
) -> tuple[list[StreamChunk | QueryResult], int]:
    """
    Create test messages for benchmarking.

    Args:
        num_requests: Number of requests to simulate
        response_size: Size of response content in characters
        chunks_per_request: Number of chunks before final QueryResult (0 = non-streaming)

    Returns:
        Tuple of (messages list, total message count)
    """
    content = "x" * response_size
    messages: list[StreamChunk | QueryResult] = []

    for i in range(num_requests):
        # Add streaming chunks if configured
        for c in range(chunks_per_request):
            messages.append(
                StreamChunk(
                    id=f"q-{i}",
                    metadata={"first_chunk": c == 0},
                    response_chunk=content,
                )
            )

        # Always end with QueryResult
        messages.append(QueryResult(id=f"q-{i}", response_output=content))

    total_messages = num_requests * (chunks_per_request + 1)
    return messages, total_messages


def _run_handle_responses_benchmark(
    tmp_path: Path,
    test_name: str,
    num_requests: int,
    response_size: int,
    chunks_per_request: int,
) -> BenchmarkResult:
    """
    Run _handle_responses benchmark with given parameters.

    Sets up ZMQ, EventRecorder, and HttpClientSampleIssuer exactly as production.
    Sends messages while consumer drains, measures throughput.
    """
    zmq_cfg = ZMQConfig(
        zmq_request_queue_prefix=get_test_socket_path(tmp_path, test_name, "_req"),
        zmq_response_queue_addr=get_test_socket_path(tmp_path, test_name, "_resp"),
    )

    # Create event loop with uvloop (matching production)
    loop = uvloop.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    ctx = zmq.asyncio.Context()

    push_socket = ZMQPushSocket(
        ctx, zmq_cfg.zmq_response_queue_addr, zmq_cfg, bind=True
    )
    pull_socket = ZMQPullSocket(
        ctx,
        zmq_cfg.zmq_response_queue_addr,
        zmq_cfg,
        bind=False,
        decoder_type=QueryResult | StreamChunk,
    )

    mock_client = MockHttpClient(pull_socket, loop)

    # Create messages
    messages, total_messages = _create_messages(
        num_requests, response_size, chunks_per_request
    )

    # Track QueryResult completions (one per request)
    done_event = threading.Event()
    counter = CompletionCounter(target=num_requests, done_event=done_event)

    issuer = None
    try:
        with EventRecorder():
            SampleEventHandler.register_hook(SampleEvent.COMPLETE, counter.increment)

            # Record LOADGEN_ISSUE_CALLED for each request (to balance inflight counter)
            # This simulates what LoadGenerator does when issuing samples
            issue_timestamp = time.monotonic_ns()
            for i in range(num_requests):
                EventRecorder.record_event(
                    SessionEvent.LOADGEN_ISSUE_CALLED,
                    issue_timestamp,
                    sample_uuid=f"q-{i}",
                )

            # Create issuer FIRST - starts _handle_responses consumer on mock_client.loop
            # This prevents ZMQ buffer overflow when sending large messages
            issuer = HttpClientSampleIssuer(mock_client)

            # Send messages and measure time to process all
            # Retry on EAGAIN (buffer full) with small delay to let consumer drain
            async def send_messages():
                import zmq

                for msg in messages:
                    while True:
                        try:
                            await push_socket.send(msg)
                            break
                        except zmq.Again:
                            await asyncio.sleep(0.001)  # 1ms backoff

            t_start = time.perf_counter()
            asyncio.run_coroutine_threadsafe(send_messages(), loop).result()

            # Wait for all responses to be processed
            done_event.wait(timeout=120.0)
            elapsed = time.perf_counter() - t_start

            # Shutdown issuer inside EventRecorder context to avoid race
            issuer.shutdown()
            try:
                issuer._response_task.result(timeout=2.0)
            except Exception:
                pass  # Task was cancelled, which is expected

        return BenchmarkResult(total_messages, elapsed)

    finally:
        SampleEventHandler.clear_hooks(SampleEvent.COMPLETE)
        # Ensure issuer is shutdown even on exception
        if issuer is not None:
            try:
                issuer.shutdown()
            except Exception:
                pass
        push_socket.close()
        pull_socket.close()
        ctx.destroy(linger=0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.timeout(0)
class TestHandleResponsesRoofline:
    """
    Benchmark roofline performance of HttpClientSampleIssuer._handle_responses.

    Measures how fast _handle_responses can drain pre-loaded ZMQ messages
    and route them to SampleEventHandler.
    """

    @pytest.mark.performance
    @pytest.mark.xdist_group(name="serial_performance")
    @pytest.mark.parametrize("response_size", RESPONSE_SIZES)
    @pytest.mark.parametrize("streaming_rate", STREAMING_RATES)
    def test_throughput(
        self, response_size: int, streaming_rate: float, tmp_path: Path
    ):
        """Benchmark throughput across response sizes and streaming rates."""
        chunks_per_request = streaming_rate_to_chunks(streaming_rate)

        # Fixed number of requests, total messages varies with streaming rate
        messages_per_request = chunks_per_request + 1
        num_requests = NUM_RESPONSES
        expected_messages = num_requests * messages_per_request

        result = _run_handle_responses_benchmark(
            tmp_path,
            test_name=f"throughput_{response_size}_{int(streaming_rate * 100)}",
            num_requests=num_requests,
            response_size=response_size,
            chunks_per_request=chunks_per_request,
        )

        if streaming_rate == 0.0:
            label = "non-streaming"
        else:
            label = f"stream_rate={streaming_rate:.0%}"

        print(f"\n  size={response_size} {label}: " f"{result.rate:,.0f} msg/s")
        assert result.count == expected_messages
        assert result.rate > 0
