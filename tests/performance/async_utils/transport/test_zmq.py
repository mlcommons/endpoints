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
Performance tests for ZMQ transport layer.

Measures peak issue rate (main proc) and recv rate (worker proc) for each
message type (Query, QueryResult, StreamChunk) across payload sizes.
"""

import asyncio
import tempfile
import time
from dataclasses import dataclass

import msgspec
import pytest
import uvloop
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.transport import (
    _create_receiver,
    _create_sender,
    _ZMQSocketConfig,
)
from inference_endpoint.core.types import Query, QueryResult, StreamChunk

# =============================================================================
# Config
# =============================================================================

# Test duration in seconds
TEST_DURATION_SECONDS = 5.0

WARMUP_MESSAGES = 100

# Payload sizes in chars
PAYLOAD_SIZES_CHARS = [32, 128, 512, 1024, 4096, 16384, 32768]


# =============================================================================
# Message factories
# =============================================================================


def make_query(payload_chars: int, idx: int) -> Query:
    return Query(
        id=str(idx),
        data={"prompt": "x" * payload_chars, "model": "m", "stream": False},
        headers={"Content-Type": "application/json"},
    )


def make_query_result(payload_chars: int, idx: int) -> QueryResult:
    return QueryResult(
        id=str(idx),
        response_output="x" * payload_chars,
    )


def make_stream_chunk(payload_chars: int, idx: int) -> StreamChunk:
    return StreamChunk(
        id=str(idx),
        response_chunk="x" * payload_chars,
        is_complete=False,
    )


MESSAGE_FACTORIES = {
    "Query": (Query, make_query),
    "QueryResult": (QueryResult, make_query_result),
    "StreamChunk": (StreamChunk, make_stream_chunk),
}


# =============================================================================
# Result
# =============================================================================


@dataclass
class PerfResult:
    msg_type: str
    payload_chars: int
    msg_bytes: int  # serialized message size
    duration_sec: float
    issued: int
    received: int

    @property
    def issue_rate(self) -> float:
        return self.issued / self.duration_sec

    @property
    def recv_rate(self) -> float:
        return self.received / self.duration_sec

    @property
    def issue_us(self) -> float:
        return (self.duration_sec / self.issued) * 1e6 if self.issued else 0

    @property
    def recv_us(self) -> float:
        return (self.duration_sec / self.received) * 1e6 if self.received else 0

    @property
    def issue_mbs(self) -> float:
        return (self.issued * self.msg_bytes) / self.duration_sec / 1e6

    @property
    def recv_mbs(self) -> float:
        return (self.received * self.msg_bytes) / self.duration_sec / 1e6


# =============================================================================
# Benchmark
# =============================================================================


async def benchmark(
    msg_type_name: str,
    payload_chars: int,
    duration_sec: float,
    warmup: int,
) -> PerfResult:
    """Measure peak issue rate (sender) and recv rate (receiver) for duration."""
    msg_type, make_msg = MESSAGE_FACTORIES[msg_type_name]

    loop = asyncio.get_running_loop()

    config = _ZMQSocketConfig()

    with ManagedZMQContext.scoped(io_threads=config.io_threads) as zmq_ctx:
        with tempfile.TemporaryDirectory(prefix="zmq_") as tmp:
            addr = f"ipc://{tmp}/bench"

            sender = _create_sender(loop, addr, zmq_ctx, config, bind=True)
            receiver = _create_receiver(
                loop, addr, zmq_ctx, config, msg_type, bind=False
            )

            await asyncio.sleep(0.01)

            # Pre-create message pool
            pool_size = 10000
            msg_pool = [make_msg(payload_chars, i) for i in range(pool_size)]

            # Measure serialized message size
            encoder = msgspec.msgpack.Encoder()
            msg_bytes = len(encoder.encode(msg_pool[0]))

            issued = 0
            received = 0
            stop = False

            try:
                # Warmup
                for i in range(warmup):
                    sender.send(msg_pool[i % pool_size])
                for _ in range(warmup):
                    await receiver.recv()

                # Benchmark
                async def sender_loop():
                    nonlocal issued, stop
                    idx = 0
                    while not stop:
                        sender.send(msg_pool[idx % pool_size])
                        idx += 1
                        if idx % 1000 == 0:
                            await asyncio.sleep(0)
                    issued = idx

                async def receiver_loop():
                    nonlocal received, stop
                    start = time.perf_counter()
                    count = 0
                    while time.perf_counter() - start < duration_sec:
                        if await receiver.recv() is not None:
                            count += 1
                    stop = True
                    received = count

                await asyncio.gather(sender_loop(), receiver_loop())

            finally:
                sender.close()
                receiver.close()

    return PerfResult(
        msg_type=msg_type_name,
        payload_chars=payload_chars,
        msg_bytes=msg_bytes,
        duration_sec=duration_sec,
        issued=issued,
        received=received,
    )


# =============================================================================
# Parameterized test
# =============================================================================


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("msg_type", ["Query", "QueryResult", "StreamChunk"])
@pytest.mark.parametrize("payload_chars", PAYLOAD_SIZES_CHARS)
def test_zmq_transport_throughput(msg_type: str, payload_chars: int):
    """Measure peak issue/recv rates for TEST_DURATION_SECONDS."""

    async def run():
        return await benchmark(
            msg_type, payload_chars, TEST_DURATION_SECONDS, WARMUP_MESSAGES
        )

    result = uvloop.run(run())

    print(
        f"\n  {msg_type:<12} {payload_chars:>6} chars ({result.msg_bytes:>5} B): "
        f"issue={result.issue_rate:>9,.0f} msg/s {result.issue_mbs:>7.1f} MB/s, "
        f"recv={result.recv_rate:>9,.0f} msg/s {result.recv_mbs:>7.1f} MB/s"
    )
