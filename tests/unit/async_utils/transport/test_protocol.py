# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections import deque

import pytest
from inference_endpoint.async_utils.transport.protocol import MessageSubscriber


class _IntCodec:
    def encode(self, item: int) -> tuple[bytes, bytes]:
        return b"test____", str(item).encode()

    def decode(self, payload: bytes) -> int:
        return int(payload)

    def on_decode_error(self, payload: bytes, exc: Exception) -> int | None:
        return None


class _QueueSubscriber(MessageSubscriber[int]):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        payloads: list[bytes],
        *,
        max_read_batch_size: int,
    ) -> None:
        super().__init__(_IntCodec(), "test://subscriber", loop)
        self._payloads = deque(payloads)
        self._max_read_batch_size = max_read_batch_size
        self.batches: list[list[int]] = []
        self.received: list[int] = []
        self.done = asyncio.Event()
        self.expected = len(payloads)
        self.release = asyncio.Event()
        self.block_processing = False
        self.active = 0
        self.max_active = 0

    def receive(self) -> bytes | None:
        if not self._payloads:
            raise StopIteration
        return self._payloads.popleft()

    async def process(self, items: list[int]) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.block_processing:
            await self.release.wait()
        self.batches.append(items)
        self.received.extend(items)
        self.active -= 1
        if len(self.received) >= self.expected:
            self.done.set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subscriber_caps_each_read_and_reschedules_without_new_edge():
    subscriber = _QueueSubscriber(
        asyncio.get_running_loop(),
        [str(i).encode() for i in range(5)],
        max_read_batch_size=2,
    )

    subscriber._on_readable()
    await asyncio.wait_for(subscriber.done.wait(), timeout=1)

    assert subscriber.received == [0, 1, 2, 3, 4]
    assert subscriber.batches == [[0, 1], [2, 3], [4]]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subscriber_processes_batches_single_flight_in_fifo_order():
    subscriber = _QueueSubscriber(
        asyncio.get_running_loop(), [b"1"], max_read_batch_size=4
    )
    subscriber.block_processing = True
    subscriber.expected = 2

    subscriber._on_readable()
    subscriber._payloads.append(b"2")
    subscriber._on_readable()
    await asyncio.sleep(0)
    subscriber.release.set()
    await asyncio.wait_for(subscriber.done.wait(), timeout=1)

    assert subscriber.received == [1, 2]
    assert subscriber.max_active == 1


@pytest.mark.unit
def test_none_payloads_count_toward_read_budget_and_close_cancels_resume():
    subscriber = _QueueSubscriber(
        asyncio.new_event_loop(),
        [None, None, b"3"],  # type: ignore[list-item]
        max_read_batch_size=2,
    )
    try:
        subscriber._on_readable()

        assert list(subscriber._payloads) == [b"3"]
        assert subscriber._read_continuation is not None

        subscriber.close()
        assert subscriber._read_continuation is None
    finally:
        subscriber.loop.close()
