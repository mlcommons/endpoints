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

"""Per-sample metrics table, trigger system, and trigger implementations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
from inference_endpoint.core.record import SampleEventType, SessionEventType
from inference_endpoint.core.types import PromptData, TextModelOutput

if TYPE_CHECKING:
    from inference_endpoint.async_utils.services.metrics_aggregator.emitter import (
        MetricEmitter,
    )
    from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
        TokenizePool,
    )
    from inference_endpoint.core.record import EventRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SampleRow
# ---------------------------------------------------------------------------


class SampleRow(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    """Per-sample state for metric computation.

    Pure data container — no methods, no trigger awareness.
    Fields are set by MetricsTable.set_field() which dispatches triggers.

    gc=False is safe: no mutable container fields that could form reference cycles.
    """

    sample_uuid: str
    tracked_block_idx: int = -1
    issued_ns: int | None = None
    recv_first_ns: int | None = None
    last_recv_ns: int | None = None
    complete_ns: int | None = None


# ---------------------------------------------------------------------------
# TrackedBlock
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrackedBlock:
    """A single START_PERFORMANCE_TRACKING → (last sample completion) window.

    Duration extends to the last tracked sample completion, not to
    STOP_PERFORMANCE_TRACKING. Empty blocks have duration 0.
    """

    start_ns: int
    last_complete_ns: int
    completed_samples: int = 0

    @property
    def duration_ns(self) -> int:
        return self.last_complete_ns - self.start_ns


# ---------------------------------------------------------------------------
# EmitTrigger
# ---------------------------------------------------------------------------


class EmitTrigger(ABC):
    """A metric computation that fires when a SampleRow field is set.

    Runtime deps (emitter, pool, loop) are bound at construction.
    ``fire()`` receives only event-specific context.
    """

    def __init__(self, metric_name: str, requires: tuple[str, ...] = ()):
        self.metric_name = metric_name
        self.requires = requires

    @abstractmethod
    def fire(
        self,
        ev_rec: EventRecord,
        row: SampleRow,
        pre_change: dict[str, Any],
    ) -> asyncio.Task | None:
        """Must be non-blocking. Return a Task if async work was scheduled."""
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Timing triggers (sync)
# ---------------------------------------------------------------------------


class TtftTrigger(EmitTrigger):
    """TTFT = recv_first_ns (new, from ev_rec) - issued_ns."""

    def __init__(self, emitter: MetricEmitter):
        super().__init__("ttft_ns", requires=("issued_ns",))
        self._emitter = emitter

    def fire(self, ev_rec, row, pre_change):
        issued_ns = pre_change.get("issued_ns")
        if issued_ns is not None:
            self._emitter.emit(
                row.sample_uuid, "ttft_ns", ev_rec.timestamp_ns - issued_ns
            )
        return None


class ChunkDeltaTrigger(EmitTrigger):
    """chunk_delta_ns = new timestamp - previous last_recv_ns.

    Skips when pre-change last_recv_ns is None (first recv via RECV_FIRST).
    """

    def __init__(self, emitter: MetricEmitter):
        super().__init__("chunk_delta_ns", requires=("last_recv_ns",))
        self._emitter = emitter

    def fire(self, ev_rec, row, pre_change):
        prev = pre_change.get("last_recv_ns")
        if prev is None:
            return None
        self._emitter.emit(
            row.sample_uuid, "chunk_delta_ns", ev_rec.timestamp_ns - prev
        )
        return None


class SampleLatencyTrigger(EmitTrigger):
    """sample_latency_ns = complete_ns (new) - issued_ns."""

    def __init__(self, emitter: MetricEmitter):
        super().__init__("sample_latency_ns", requires=("issued_ns",))
        self._emitter = emitter

    def fire(self, ev_rec, row, pre_change):
        issued_ns = pre_change.get("issued_ns")
        if issued_ns is not None:
            self._emitter.emit(
                row.sample_uuid,
                "sample_latency_ns",
                ev_rec.timestamp_ns - issued_ns,
            )
        return None


# ---------------------------------------------------------------------------
# Token triggers (async)
# ---------------------------------------------------------------------------


class IslTrigger(EmitTrigger):
    """ISL from PromptData: len(token_ids) sync, or token_count(text) async."""

    def __init__(
        self,
        emitter: MetricEmitter,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
    ):
        super().__init__("isl", requires=())
        self._emitter = emitter
        self._pool = tokenize_pool
        self._loop = loop

    def fire(self, ev_rec, row, pre_change):
        if not isinstance(ev_rec.data, PromptData):
            return None
        if ev_rec.data.token_ids is not None:
            self._emitter.emit(row.sample_uuid, "isl", len(ev_rec.data.token_ids))
            return None
        if (
            ev_rec.data.text is not None
            and self._pool is not None
            and self._loop is not None
        ):
            text = ev_rec.data.text
            uuid = row.sample_uuid
            pool, loop, emitter = self._pool, self._loop, self._emitter

            async def _compute() -> None:
                try:
                    count = await pool.token_count_async(text, loop)
                    emitter.emit(uuid, "isl", count)
                except Exception:
                    logger.exception("ISL tokenization failed for %s", uuid)

            return loop.create_task(_compute())
        return None


class OslTrigger(EmitTrigger):
    """OSL = token_count(full output text) from COMPLETE event data."""

    def __init__(
        self,
        emitter: MetricEmitter,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
    ):
        super().__init__("osl", requires=())
        self._emitter = emitter
        self._pool = tokenize_pool
        self._loop = loop

    def fire(self, ev_rec, row, pre_change):
        if self._pool is None or self._loop is None:
            return None
        if not isinstance(ev_rec.data, TextModelOutput):
            return None
        output_text = str(ev_rec.data)
        if not output_text:
            return None

        uuid = row.sample_uuid
        pool, loop, emitter = self._pool, self._loop, self._emitter

        async def _compute() -> None:
            try:
                osl = await pool.token_count_async(output_text, loop)
                emitter.emit(uuid, "osl", osl)
            except Exception:
                logger.exception("OSL tokenization failed for %s", uuid)

        return loop.create_task(_compute())


class TpotTrigger(EmitTrigger):
    """TPOT = (complete_ns - recv_first_ns) / token_count(text_after_first_chunk).

    Only registered when streaming mode is enabled. Computes the TPOT denominator
    directly from TextModelOutput.text_after_first_chunk() at COMPLETE time,
    avoiding any dependency on RECV_FIRST tokenization state.

    # NOTE(agents): This trigger tokenizes text_after_first_chunk independently
    # from OslTrigger, which tokenizes the full output. This means the output is
    # tokenized twice at COMPLETE time for streaming samples. This is intentional:
    # OSL is always required (non-streaming and streaming), while TPOT is
    # streaming-only. Keeping them as separate triggers allows conditional
    # registration via the streaming flag. If tokenization throughput becomes a
    # bottleneck, consider merging OSL and TPOT into a single trigger that
    # tokenizes once and derives both metrics.
    """

    def __init__(
        self,
        emitter: MetricEmitter,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
    ):
        super().__init__("tpot_ns", requires=("recv_first_ns",))
        self._emitter = emitter
        self._pool = tokenize_pool
        self._loop = loop

    def fire(self, ev_rec, row, pre_change):
        if self._pool is None or self._loop is None:
            return None
        recv_first_ns = pre_change.get("recv_first_ns")
        if recv_first_ns is None:
            return None
        if not isinstance(ev_rec.data, TextModelOutput):
            return None
        after_first = ev_rec.data.text_after_first_chunk()
        if not after_first:
            return None

        uuid = row.sample_uuid
        complete_ns = ev_rec.timestamp_ns
        pool, loop, emitter = self._pool, self._loop, self._emitter

        async def _compute() -> None:
            try:
                tokens_after_first = await pool.token_count_async(after_first, loop)
                if tokens_after_first > 0:
                    tpot = (complete_ns - recv_first_ns) / tokens_after_first
                    emitter.emit(uuid, "tpot_ns", tpot)
            except Exception:
                logger.exception("TPOT tokenization failed for %s", uuid)

        return loop.create_task(_compute())


# ---------------------------------------------------------------------------
# MetricsTable
# ---------------------------------------------------------------------------


class MetricsTable:
    """Stores in-flight sample rows, session state, and dispatches triggers.

    Row lifecycle is managed internally via ``set_field``:
    - ISSUED: creates the row if tracking is on, assigns block index.
    - COMPLETE: fires triggers, sets field, updates tracked block, removes row.
    - Other events: fires triggers and sets field. No-op if row doesn't exist.

    Session state is updated via ``handle_session_event``.
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, SampleRow] = {}
        self._triggers: dict[str, list[EmitTrigger]] = {}
        self._in_flight_tasks: set[asyncio.Task] = set()

        # Session-level state
        self.is_tracking: bool = False
        self.session_started_ns: int | None = None
        self.tracked_blocks: list[TrackedBlock] = []

    # --- Trigger registration ---

    def add_trigger(self, field_name: str, trigger: EmitTrigger) -> None:
        """Register a trigger for a SampleRow field."""
        self._triggers.setdefault(field_name, []).append(trigger)

    # --- Session event handling ---

    def handle_session_event(self, ev_rec: EventRecord) -> None:
        """Update session-level state from a session event."""
        ev = ev_rec.event_type
        if ev == SessionEventType.STARTED:
            self.session_started_ns = ev_rec.timestamp_ns
        elif ev == SessionEventType.START_PERFORMANCE_TRACKING:
            if not self.is_tracking:
                self.is_tracking = True
                self.tracked_blocks.append(
                    TrackedBlock(
                        start_ns=ev_rec.timestamp_ns,
                        last_complete_ns=ev_rec.timestamp_ns,
                    )
                )
        elif ev == SessionEventType.STOP_PERFORMANCE_TRACKING:
            self.is_tracking = False

    # --- Row access ---

    def get_row(self, sample_uuid: str) -> SampleRow | None:
        return self._in_flight.get(sample_uuid)

    def __len__(self) -> int:
        return len(self._in_flight)

    # --- Tracked duration ---

    @property
    def total_tracked_duration_ns(self) -> int:
        """Sum of all tracking block durations."""
        return sum(b.duration_ns for b in self.tracked_blocks)

    @property
    def total_completed_tracked_samples(self) -> int:
        """Total samples completed across all tracking blocks."""
        return sum(b.completed_samples for b in self.tracked_blocks)

    # --- Field updates ---

    def set_field(
        self,
        sample_uuid: str,
        field_name: str,
        value: Any,
        ev_rec: EventRecord,
    ) -> None:
        """Update a sample field, handling row lifecycle and trigger dispatch.

        - ISSUED: creates the row if tracking is on, assigns current block index.
          No-op if tracking is off.
        - COMPLETE: fires triggers, sets field, updates tracked block, removes row.
        - Other events: fires triggers and sets field.
          No-op if the row doesn't exist (untracked sample).
        """
        row: SampleRow | None
        ev = ev_rec.event_type

        if ev == SampleEventType.ISSUED:
            if not self.is_tracking:
                return
            row = self._create_row(sample_uuid)
            row.tracked_block_idx = len(self.tracked_blocks) - 1
        else:
            row = self._in_flight.get(sample_uuid)
            if row is None:
                return

        self._fire_triggers(row, field_name, ev_rec)
        setattr(row, field_name, value)

        if ev == SampleEventType.COMPLETE:
            self._update_tracked_block(row, ev_rec.timestamp_ns)
            self._in_flight.pop(sample_uuid, None)

    # --- Task draining ---

    async def drain_tasks(self) -> None:
        """Await all in-flight async trigger tasks."""
        if self._in_flight_tasks:
            await asyncio.gather(*self._in_flight_tasks, return_exceptions=True)
            self._in_flight_tasks.clear()

    # --- Internal ---

    def _create_row(self, sample_uuid: str) -> SampleRow:
        if sample_uuid in self._in_flight:
            logger.warning(
                "Duplicate ISSUED for sample %s, possibly due to retry - skipping",
                sample_uuid,
            )
            return self._in_flight[sample_uuid]
        row = SampleRow(sample_uuid=sample_uuid)
        self._in_flight[sample_uuid] = row
        return row

    def _fire_triggers(
        self, row: SampleRow, field_name: str, ev_rec: EventRecord
    ) -> None:
        for trigger in self._triggers.get(field_name, ()):
            pre_change = {attr: getattr(row, attr) for attr in trigger.requires}
            task = trigger.fire(ev_rec, row, pre_change)
            if task is not None:
                self._in_flight_tasks.add(task)
                task.add_done_callback(self._in_flight_tasks.discard)

    def _update_tracked_block(self, row: SampleRow, complete_ns: int) -> None:
        """Extend the sample's tracked block duration and increment count."""
        idx = row.tracked_block_idx
        if 0 <= idx < len(self.tracked_blocks):
            block = self.tracked_blocks[idx]
            if complete_ns > block.last_complete_ns:
                block.last_complete_ns = complete_ns
            block.completed_samples += 1
