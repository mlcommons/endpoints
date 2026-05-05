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
from enum import Enum
from typing import TYPE_CHECKING, Any

import msgspec
from inference_endpoint.core.record import SampleEventType, SessionEventType
from inference_endpoint.core.types import PromptData, TextModelOutput

if TYPE_CHECKING:
    from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
        MetricsRegistry,
    )
    from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
        TokenizePool,
    )
    from inference_endpoint.core.record import EventRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SampleField enum
# ---------------------------------------------------------------------------


class SampleField(str, Enum):
    """SampleRow field names that triggers can be registered on."""

    ISSUED_NS = "issued_ns"
    RECV_FIRST_NS = "recv_first_ns"
    LAST_RECV_NS = "last_recv_ns"
    COMPLETE_NS = "complete_ns"


class MetricSeriesKey(str, Enum):
    """Series metric keys written by triggers to the registry."""

    ISL = "isl"
    OSL = "osl"
    SAMPLE_LATENCY_NS = "sample_latency_ns"
    TTFT_NS = "ttft_ns"
    CHUNK_DELTA_NS = "chunk_delta_ns"
    TPOT_NS = "tpot_ns"


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
# EmitTrigger base classes
# ---------------------------------------------------------------------------


class EmitTrigger(ABC):
    """A metric computation that fires when a SampleRow field is set.

    Each trigger has a ``metric_name`` and a ``registry`` reference. When
    ``fire()`` computes a value, it writes directly via
    ``self.registry.record(self.metric_name, value)``. Series registration
    (with HDR bounds, dtype, etc.) is the aggregator's responsibility —
    the trigger itself never registers metrics.
    """

    def __init__(
        self,
        metric_name: str,
        registry: MetricsRegistry,
        requires: tuple[str, ...] = (),
        dtype: type = int,
    ):
        # Resolve enum to its value string so registry names match the
        # registered series names (e.g. "ttft_ns" not "MetricSeriesKey.TTFT_NS").
        self.metric_name = (
            metric_name.value if isinstance(metric_name, Enum) else metric_name
        )
        self.registry = registry
        self.requires = requires
        self.dtype = dtype

    @abstractmethod
    def fire(
        self,
        ev_rec: EventRecord,
        row: SampleRow,
        pre_change: dict[str, Any],
    ) -> asyncio.Task | None:
        """Must be non-blocking. Return a Task if async work was scheduled."""
        raise NotImplementedError()


class TimeDeltaTrigger(EmitTrigger):
    """Sync trigger: emits ev_rec.timestamp_ns - pre_change[delta_start_fieldname].

    The emitted metric is a time delta: the firing event marks the end of the
    delta, and ``delta_start_fieldname`` names the SampleField whose timestamp
    marks the start. Skips silently if the start field is None (the delta has
    not yet opened for this sample).
    """

    def __init__(
        self,
        metric_name: str,
        registry: MetricsRegistry,
        delta_start_fieldname: str,
    ):
        super().__init__(metric_name, registry, requires=(delta_start_fieldname,))
        self._delta_start_fieldname = delta_start_fieldname

    def fire(self, ev_rec, row, pre_change):
        baseline = pre_change.get(self._delta_start_fieldname)
        if baseline is not None:
            self.registry.record(self.metric_name, ev_rec.timestamp_ns - baseline)
        return None


class AsyncTokenTrigger(EmitTrigger):
    """Base for triggers that need async tokenization.

    Subclasses implement ``_extract_text()`` to pull the text to tokenize
    from the event record. If text is returned, an async task is created
    to tokenize and emit. Subclasses can override ``_compute_value()`` to
    transform the token count before storing.
    """

    def __init__(
        self,
        metric_name: str,
        registry: MetricsRegistry,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
        requires: tuple[str, ...] = (),
        dtype: type = int,
    ):
        super().__init__(metric_name, registry, requires=requires, dtype=dtype)
        self._pool = tokenize_pool
        self._loop = loop

    @abstractmethod
    def _extract_text(
        self, ev_rec: EventRecord, row: SampleRow, pre_change: dict[str, Any]
    ) -> str | None:
        """Return the text to tokenize, or None to skip."""
        raise NotImplementedError()

    def _compute_value(
        self, token_count: int, ev_rec: EventRecord, pre_change: dict[str, Any]
    ) -> int | float | None:
        """Transform token count into the metric value. Default: count as-is."""
        return token_count

    def fire(self, ev_rec, row, pre_change):
        if self._pool is None or self._loop is None:
            return None
        text = self._extract_text(ev_rec, row, pre_change)
        if not text:
            return None

        pool, loop = self._pool, self._loop
        registry, name = self.registry, self.metric_name
        uuid = row.sample_uuid

        async def _tokenize_and_emit() -> None:
            try:
                count = await pool.token_count_async(text, loop)
                value = self._compute_value(count, ev_rec, pre_change)
                if value is not None:
                    registry.record(name, value)
            except Exception:
                logger.exception("%s tokenization failed for %s", name, uuid)

        return loop.create_task(_tokenize_and_emit())


# ---------------------------------------------------------------------------
# Timing triggers (sync)
# ---------------------------------------------------------------------------


class TtftTrigger(TimeDeltaTrigger):
    """TTFT = recv_first_ns (new) - issued_ns."""

    def __init__(self, registry: MetricsRegistry):
        super().__init__(
            MetricSeriesKey.TTFT_NS,
            registry,
            delta_start_fieldname=SampleField.ISSUED_NS,
        )


class ChunkDeltaTrigger(TimeDeltaTrigger):
    """chunk_delta_ns = new timestamp - previous last_recv_ns.

    Skips when pre-change last_recv_ns is None (first recv via RECV_FIRST).
    """

    def __init__(self, registry: MetricsRegistry):
        super().__init__(
            MetricSeriesKey.CHUNK_DELTA_NS,
            registry,
            delta_start_fieldname=SampleField.LAST_RECV_NS,
        )


class SampleLatencyTrigger(TimeDeltaTrigger):
    """sample_latency_ns = complete_ns (new) - issued_ns."""

    def __init__(self, registry: MetricsRegistry):
        super().__init__(
            MetricSeriesKey.SAMPLE_LATENCY_NS,
            registry,
            delta_start_fieldname=SampleField.ISSUED_NS,
        )


# ---------------------------------------------------------------------------
# Token triggers (async)
# ---------------------------------------------------------------------------


class IslTrigger(AsyncTokenTrigger):
    """ISL from PromptData: len(token_ids) sync, or token_count(text) async."""

    def __init__(
        self,
        registry: MetricsRegistry,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
    ):
        super().__init__(MetricSeriesKey.ISL, registry, tokenize_pool, loop)

    def fire(self, ev_rec, row, pre_change):
        # Sync fast path: any backend that pre-populates token_ids (e.g. SGLang).
        if isinstance(ev_rec.data, PromptData) and ev_rec.data.token_ids is not None:
            self.registry.record(self.metric_name, len(ev_rec.data.token_ids))
            return None
        # Async path: tokenize raw text — used when token_ids are unavailable
        # (e.g. OpenAI-compatible endpoints). Handled by the base class.
        return super().fire(ev_rec, row, pre_change)

    def _extract_text(self, ev_rec, row, pre_change):
        if isinstance(ev_rec.data, PromptData) and ev_rec.data.text is not None:
            return ev_rec.data.text
        return None


class OslTrigger(AsyncTokenTrigger):
    """OSL = token_count(full output text) from COMPLETE event data."""

    def __init__(
        self,
        registry: MetricsRegistry,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
    ):
        super().__init__(MetricSeriesKey.OSL, registry, tokenize_pool, loop)

    def _extract_text(self, ev_rec, row, pre_change):
        if isinstance(ev_rec.data, TextModelOutput):
            text = str(ev_rec.data)
            return text if text else None
        return None


class TpotTrigger(AsyncTokenTrigger):
    """TPOT = (complete_ns - recv_first_ns) / token_count(text_after_first_chunk).

    Only registered when streaming mode is enabled.

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
        registry: MetricsRegistry,
        tokenize_pool: TokenizePool | None,
        loop: asyncio.AbstractEventLoop | None,
    ):
        super().__init__(
            MetricSeriesKey.TPOT_NS,
            registry,
            tokenize_pool,
            loop,
            requires=(SampleField.RECV_FIRST_NS,),
            dtype=float,
        )

    def _extract_text(self, ev_rec, row, pre_change):
        if pre_change.get(SampleField.RECV_FIRST_NS) is None:
            return None
        if isinstance(ev_rec.data, TextModelOutput):
            return ev_rec.data.text_after_first_chunk() or None
        return None

    def _compute_value(self, token_count, ev_rec, pre_change):
        if token_count <= 0:
            return None
        recv_first_ns = pre_change[SampleField.RECV_FIRST_NS]
        return (ev_rec.timestamp_ns - recv_first_ns) / token_count


# ---------------------------------------------------------------------------
# MetricsTable
# ---------------------------------------------------------------------------


class MetricsTable:
    """Stores in-flight sample rows, session state, and dispatches triggers.

    Takes a ``MetricsRegistry`` for metric storage. Triggers are passed to
    ``add_trigger`` already wired against the registry. The table does NOT
    register the underlying series — the aggregator pre-registers all
    series with explicit HDR bounds before constructing triggers.

    Row lifecycle is managed internally via ``set_field``:
    - ISSUED: creates the row if tracking is on, assigns block index.
    - COMPLETE: fires triggers, sets field, updates tracked block, removes row.
    - Other events: fires triggers and sets field. No-op if row doesn't exist.

    Session state is updated via ``handle_session_event``.
    """

    def __init__(self, registry: MetricsRegistry) -> None:
        self._registry = registry
        self._in_flight: dict[str, SampleRow] = {}
        self._triggers: dict[str, list[EmitTrigger]] = {}
        self._in_flight_tasks: set[asyncio.Task] = set()

        # Session-level state
        self.is_tracking: bool = False
        self.session_started_ns: int | None = None
        self.tracked_blocks: list[TrackedBlock] = []

    # --- Trigger registration ---

    def add_trigger(self, field_name: str, trigger: EmitTrigger) -> None:
        """Register a trigger for a SampleRow field.

        The trigger's underlying series MUST already be registered on the
        registry by the aggregator (which knows the right HDR bounds and
        dtype). The table only stores the trigger reference.
        """
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
