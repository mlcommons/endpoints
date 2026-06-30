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

"""Unit tests for the metrics-aggregator __main__ SIGTERM handler.

The SIGTERM path spawns an asyncio task that writes the INTERRUPTED
final snapshot and signals shutdown. asyncio tracks tasks only via
weakrefs, so user code must hold a strong reference to the spawned
task — otherwise GC can drop it mid-flight (Python asyncio docs),
losing the INTERRUPTED delivery the handler exists to provide.
"""

from __future__ import annotations

import asyncio
import gc
import weakref
from unittest.mock import AsyncMock, MagicMock

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator import (
    __main__ as agg_main,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sigterm_handler_holds_strong_reference_to_finalize_task():
    """SIGTERM-spawned _signal_finalize task must be held in a strong-ref set.

    Reproduces the discarded-create_task() bug: without a strong
    reference, Python's GC may drop the task mid-flight (loop tracks
    tasks via weakref only). This breaks the entire INTERRUPTED
    delivery contract the SIGTERM handler exists to provide.
    """
    loop = asyncio.get_event_loop()

    registry = MagicMock()
    table = MagicMock()
    table.total_tracked_duration_ns = 0
    table.in_flight_tasks_count = 0

    # publish_final blocks on an event so we can observe the task
    # mid-execution and exercise the strong-ref contract.
    publish_gate = asyncio.Event()

    async def _slow_publish(*args, **kwargs):
        await publish_gate.wait()

    publisher = MagicMock()
    publisher.publish_final = AsyncMock(side_effect=_slow_publish)

    shutdown_event = asyncio.Event()

    on_sigterm, pending = agg_main._make_sigterm_handler(
        loop=loop,
        registry=registry,
        publisher=publisher,
        table=table,
        shutdown_event=shutdown_event,
    )

    on_sigterm()

    # Right after the synchronous handler returns, the spawned task
    # MUST be in the strong-ref container — otherwise asyncio docs
    # say it is GC-vulnerable.
    assert len(pending) == 1, (
        "SIGTERM handler must hold a strong reference to the spawned task; "
        f"pending set has {len(pending)} entries"
    )

    task = next(iter(pending))
    weak = weakref.ref(task)
    del task

    # Force GC: the strong-ref set must keep the task alive.
    gc.collect()
    assert weak() is not None, (
        "task was garbage-collected despite the strong-ref set — "
        "the SIGTERM finalize would have been lost mid-flight"
    )
    assert len(pending) == 1

    # Allow publish_final to complete; done-callback must remove the
    # task from the set (otherwise the set grows unboundedly across
    # multiple SIGTERMs, which is itself a leak).
    publish_gate.set()
    await shutdown_event.wait()
    # Yield once so the done-callback (scheduled after the awaitable
    # resolves) gets a chance to run.
    await asyncio.sleep(0)

    assert len(pending) == 0, (
        "task must self-remove from the strong-ref set via done-callback "
        f"after completion; pending set has {len(pending)} entries"
    )
    publisher.publish_final.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sigterm_handler_refreshes_tracked_duration():
    """Handler must mirror the ENDED path: refresh tracked_duration_ns AND
    the legacy LoadGen window from the table BEFORE publish_final, so an
    interrupted run whose STOP_PERFORMANCE_TRACKING never fired still reports
    a sensible QPS and does not silently fall back from legacy to native.
    """
    loop = asyncio.get_event_loop()

    registry = MagicMock()
    table = MagicMock()
    table.total_tracked_duration_ns = 12345
    table.total_loadgen_window_ns = 67890
    table.in_flight_tasks_count = 3

    publisher = MagicMock()
    publisher.publish_final = AsyncMock()

    shutdown_event = asyncio.Event()

    on_sigterm, _ = agg_main._make_sigterm_handler(
        loop=loop,
        registry=registry,
        publisher=publisher,
        table=table,
        shutdown_event=shutdown_event,
    )
    on_sigterm()
    await shutdown_event.wait()
    await asyncio.sleep(0)

    refreshed = dict(c.args for c in registry.set_counter.call_args_list)
    tracked = next(v for n, v in refreshed.items() if "tracked_duration" in n)
    window = next(v for n, v in refreshed.items() if "loadgen_window" in n)
    assert tracked == 12345
    assert window == 67890
    publisher.publish_final.assert_awaited_once()
    assert publisher.publish_final.await_args.kwargs == {
        "n_pending_tasks": 3,
        "interrupted": True,
    }
