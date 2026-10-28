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

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import msgspec
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    PromptInput,
    TextInput,
    TokenIdsInput,
    TokenizationInput,
    extract_tokenization_input,
    has_countable_output,
)
from inference_endpoint.core.record import SampleEventType, SessionEventType
from inference_endpoint.core.types import PromptData, TextModelOutput, TokenCountSource
from inference_endpoint.metrics.token_count_basis import (
    ServerUsage,
    classify_server_usage,
    missing_usage_message,
)

if TYPE_CHECKING:
    from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
        MetricsRegistry,
    )
    from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
        TokenBatchQueue,
    )
    from inference_endpoint.core.record import EventRecord
    from inference_endpoint.metrics.steady_state_diagnostics import SuperPassCollector

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
    """Series metric keys written by triggers to the registry.

    ``ISL``, ``OSL`` and ``TPOT_NS`` are base names: each token-count basis
    records them under ``basis_key`` (unsuffixed on the primary basis).
    """

    ISL = "isl"
    OSL = "osl"
    SAMPLE_LATENCY_NS = "sample_latency_ns"
    TTFT_NS = "ttft_ns"
    CHUNK_DELTA_NS = "chunk_delta_ns"
    TPOT_NS = "tpot_ns"


class MetricCounterKey(str, Enum):
    """Counter metric keys tracked by the aggregator.

    Total counters include all samples (warmup + tracked).
    Tracked counters only include samples within performance tracking windows.
    """

    TOTAL_SAMPLES_ISSUED = "total_samples_issued"
    TOTAL_SAMPLES_COMPLETED = "total_samples_completed"
    TOTAL_SAMPLES_FAILED = "total_samples_failed"
    TRACKED_SAMPLES_ISSUED = "tracked_samples_issued"
    TRACKED_SAMPLES_COMPLETED = "tracked_samples_completed"
    # Failed samples that were within a performance-tracking window.
    # Counted at ERROR-event time; correctness depends on
    # session.py:_handle_response emitting ERROR before COMPLETE so the
    # tracked row still exists when the aggregator sees the ERROR.
    TRACKED_SAMPLES_FAILED = "tracked_samples_failed"
    TRACKED_FINISH_REASON_STOP = "tracked_finish_reason_stop"
    TRACKED_FINISH_REASON_LENGTH = "tracked_finish_reason_length"
    TRACKED_FINISH_REASON_TOOL_CALLS = "tracked_finish_reason_tool_calls"
    TRACKED_FINISH_REASON_CONTENT_FILTER = "tracked_finish_reason_content_filter"
    TRACKED_FINISH_REASON_FUNCTION_CALL = "tracked_finish_reason_function_call"
    TRACKED_FINISH_REASON_OTHER = "tracked_finish_reason_other"
    TRACKED_DURATION_NS = "tracked_duration_ns"
    # Legacy MLPerf LoadGen Server "completed" window (final_query_all_samples_done_time).
    LEGACY_LOADGEN_WINDOW_DURATION_NS = "legacy_loadgen_window_duration_ns"
    # Total wall-clock duration since session start. Updated on every event as
    # max(current, event_timestamp - session_start). Stored as a counter
    # rather than computed from (now - start) at read time because
    # time.monotonic_ns() has a process-local epoch — a reader in another
    # process would get a meaningless value.
    TOTAL_DURATION_NS = "total_duration_ns"
    # Tracked successful samples whose COMPLETE carried no usable server usage
    # (``ServerUsage.MISSING``), whatever the primary basis, and the subset
    # whose output was empty (in-band error frames, cut streams).
    TRACKED_SAMPLES_MISSING_USAGE = "tracked_samples_missing_usage"
    TRACKED_SAMPLES_MISSING_USAGE_EMPTY_OUTPUT = (
        "tracked_samples_missing_usage_empty_output"
    )
    # Samples with both a tokenizer OSL and usable server usage, and the two
    # token totals over exactly those samples.
    OSL_PAIRED_SAMPLES = "osl_paired_samples"
    OSL_PAIRED_CLIENT_TOKENS = "osl_paired_client_tokens"
    OSL_PAIRED_SERVER_TOKENS = "osl_paired_server_tokens"


class BasisSeriesNames(msgspec.Struct, frozen=True, kw_only=True):  # type: ignore[call-arg]
    """The series one token-count basis records into, named at registration.

    ``None`` for a series the run does not record: ``isl`` without ISL
    collection, ``tpot_ns`` without streaming. Field names must equal the
    ``MetricSeriesKey`` values in ``TOKEN_SERIES``: the aggregator builds and
    reads this struct by ``spec.key.value``.
    """

    osl: str
    isl: str | None = None
    tpot_ns: str | None = None


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
    # Super-pass this sample was issued into; -1 means not collected.
    # Lives on SampleRow so the collector stays stateless per sample: a uuid map
    # in SuperPassCollector would duplicate this table for the length of the run.
    superpass_index: int = -1
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
    ) -> None:
        """Must be non-blocking.

        Sync triggers record into the registry directly. Token triggers
        enqueue onto the shared ``TokenBatchQueue`` for batched tokenization
        at the next flush; neither path schedules per-event tasks.
        """
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


class TokenTrigger(EmitTrigger):
    """Base for triggers whose metric needs tokenization.

    Subclasses return one explicit tokenization input. ``fire()`` dispatches
    already-tokenized IDs synchronously and sends every other variant through
    the shared ``TokenBatchQueue``. The queue is ``None`` when no tokenizer was
    configured or discovered for the run. In that state, pre-tokenized IDs are
    still counted synchronously, while text and structured token metrics are
    unavailable and therefore skipped.
    """

    def __init__(
        self,
        metric_name: str,
        registry: MetricsRegistry,
        queue: TokenBatchQueue | None,
        requires: tuple[str, ...] = (),
        dtype: type = int,
    ):
        super().__init__(metric_name, registry, requires=requires, dtype=dtype)
        self._queue = queue

    @abstractmethod
    def _extract_tokenization_input(
        self, ev_rec: EventRecord, row: SampleRow, pre_change: dict[str, Any]
    ) -> TokenizationInput | None:
        """Return the operation-specific tokenization input, or None to skip."""
        raise NotImplementedError()

    def _compute_value(
        self, token_count: int, ev_rec: EventRecord, pre_change: dict[str, Any]
    ) -> int | float | None:
        """Transform token count into the metric value. Default: count as-is."""
        return token_count

    def _make_recorder(
        self, ev_rec: EventRecord, row: SampleRow, pre_change: dict[str, Any]
    ) -> Callable[[int], None]:
        """Build the callback the queue runs once the token count is known.

        Subclasses may need in-flight state before the next flush. At flush time
        ``set_field`` may have dropped the row, so the caller passes it here.
        """
        registry, name = self.registry, self.metric_name

        def record(count: int) -> None:
            value = self._compute_value(count, ev_rec, pre_change)
            if value is not None:
                registry.record(name, value)

        return record

    def fire(self, ev_rec, row, pre_change):
        item = self._extract_tokenization_input(ev_rec, row, pre_change)
        if item is None:
            return
        if isinstance(item, TokenIdsInput):
            self.registry.record(self.metric_name, len(item.token_ids))
        elif isinstance(item, TextInput | MessageInput | PromptInput):
            if self._queue is not None:
                self._queue.enqueue(item, self._make_recorder(ev_rec, row, pre_change))


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
# Token triggers (batched)
# ---------------------------------------------------------------------------


class IslTrigger(TokenTrigger):
    """ISL from one prompt representation, in explicit priority order.

    ``PromptData`` produced by ``PhaseIssuer`` normally has exactly one populated
    representation. For defensive handling, token IDs take precedence over
    plain text, which takes precedence over structured messages. An explicitly
    empty token-ID tuple is therefore counted as zero rather than falling
    through to another representation.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        queue: TokenBatchQueue | None,
        *,
        metric_name: str,
    ):
        super().__init__(metric_name, registry, queue)

    def _extract_tokenization_input(self, ev_rec, row, pre_change):
        if not isinstance(ev_rec.data, PromptData):
            return None
        if ev_rec.data.token_ids is not None:
            return TokenIdsInput(ev_rec.data.token_ids)
        if ev_rec.data.text:
            return TextInput(ev_rec.data.text)
        if ev_rec.data.messages is not None:
            return PromptInput(
                ev_rec.data.messages,
                ev_rec.data.tools,
                ev_rec.data.chat_template_kwargs,
                ev_rec.data.chat_template,
                ev_rec.data.tool_choice,
            )
        return None


class OslTrigger(TokenTrigger):
    """OSL from one complete model output representation.

    An output containing reasoning or tool calls uses ``MessageInput`` so the
    chat template renders all structured fields, including ordinary content.
    Otherwise, non-empty output content uses ``TextInput``. ``TextModelOutput``
    does not carry server token IDs, so OSL has no token-ID/text precedence.

    A sample whose COMPLETE carries usable server usage
    (``ServerUsage.USABLE``) also adds its tokenizer count and the server
    ``completion_tokens`` to the ``osl_paired_*`` counters, so both totals
    cover exactly the same samples. Usage is classified here only for outputs
    sent to the tokenizer queue, so a run without a tokenizer never does it.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        queue: TokenBatchQueue | None,
        *,
        metric_name: str,
    ):
        super().__init__(metric_name, registry, queue)

    def _make_recorder(self, ev_rec, row, pre_change):
        server_count = ev_rec.completion_tokens
        usage = classify_server_usage(ev_rec.prompt_tokens, server_count, ev_rec.data)
        if usage is not ServerUsage.USABLE:
            return super()._make_recorder(ev_rec, row, pre_change)
        registry, name = self.registry, self.metric_name

        def record(count: int) -> None:
            registry.record(name, count)
            registry.increment(MetricCounterKey.OSL_PAIRED_SAMPLES.value)
            registry.increment(MetricCounterKey.OSL_PAIRED_CLIENT_TOKENS.value, count)
            registry.increment(
                MetricCounterKey.OSL_PAIRED_SERVER_TOKENS.value, server_count
            )

        return record

    def _extract_tokenization_input(self, ev_rec, row, pre_change):
        if not isinstance(ev_rec.data, TextModelOutput):
            return None
        return extract_tokenization_input(ev_rec.data)


class TpotTrigger(TokenTrigger):
    """TPOT = (complete_ns - recv_first_ns) / output token count.

    The window starts at ``recv_first_ns`` (first chunk), so TPOT **excludes
    TTFT** — it measures inter-token time over the post-first-chunk output,
    not time from issue. Streaming-only. Tokenizes that post-first-chunk
    output independently of ``OslTrigger`` (full output), so streaming samples
    are tokenized twice — intentional: OSL is always required, TPOT is
    conditional on the streaming flag.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        queue: TokenBatchQueue | None,
        *,
        metric_name: str,
        collector: SuperPassCollector | None,
    ):
        super().__init__(
            metric_name,
            registry,
            queue,
            requires=(SampleField.RECV_FIRST_NS,),
            dtype=float,
        )
        self._collector = collector

    def _make_recorder(self, ev_rec, row, pre_change):
        collector = self._collector
        # Read the super-pass while the sample is still in flight. The token
        # count arrives at the drain flush; by then set_field has dropped the
        # row. Carry only the int in the closure, so the row is not retained
        # until the flush.
        superpass_index = row.superpass_index
        if collector is None or superpass_index < 0:
            return super()._make_recorder(ev_rec, row, pre_change)
        registry, name = self.registry, self.metric_name

        def record(count: int) -> None:
            value = self._compute_value(count, ev_rec, pre_change)
            if value is not None:
                registry.record(name, value)
                collector.add_tpot(superpass_index, value, count)

        return record

    def _extract_tokenization_input(self, ev_rec, row, pre_change):
        if pre_change.get(SampleField.RECV_FIRST_NS) is None:
            return None
        if not isinstance(ev_rec.data, TextModelOutput):
            return None
        if ev_rec.data.reasoning or ev_rec.data.tool_calls:
            return MessageInput(*ev_rec.data.as_message_parts_after_first_chunk())
        text = ev_rec.data.text_after_first_chunk()
        if text:
            return TextInput(text)
        return None

    def _compute_value(self, token_count, ev_rec, pre_change):
        if token_count <= 0:
            return None
        recv_first_ns = pre_change[SampleField.RECV_FIRST_NS]
        return (ev_rec.timestamp_ns - recv_first_ns) / token_count


# ---------------------------------------------------------------------------
# Server-usage trigger (sync)
# ---------------------------------------------------------------------------


class ServerUsageTrigger(EmitTrigger):
    """Server-basis ISL, OSL and TPOT from the usage on the COMPLETE record.

    One ``classify_server_usage`` call per successful sample decides all
    three, so the server series always cover the same samples:

    - ``USABLE``: records ``prompt_tokens`` as ISL, ``completion_tokens`` as
      OSL and, when streaming, TPOT by LoadGen's rule
      ``(complete_ns - recv_first_ns) / (completion_tokens - 1)``: the first
      visible chunk is excluded from both the window and the count. TPOT is
      skipped without a first chunk, for ``completion_tokens <= 1`` and for a
      non-positive window.
    - ``TRUE_ZERO``: records nothing; no server-basis series records an empty
      completion, and client OSL and TPOT also skip empty output.
    - ``MISSING``: counts the sample in ``tracked_samples_missing_usage`` (and
      ``..._empty_output`` when its output is empty) and logs the first miss
      of the run: a WARNING when the server basis is primary, else INFO.

    Errors and agentic timeouts complete with ``data=None`` and are skipped.
    Usage exists only on COMPLETE, and no tokenizer is involved, so nothing
    here adds to ``n_pending_tasks``.

    The TPOT window can be non-positive because the two ends are stamped in
    different places: COMPLETE carries ``QueryResult.completed_at``, set when
    the transport decodes the response, while RECV_FIRST carries the time the
    session dequeues the chunk. A short stream decoded in one transport drain
    can therefore "complete" before its first chunk.

    It fires inside ``set_field(COMPLETE)`` while the row is still in the
    table, so it reads ``row.superpass_index`` directly and feeds ``collector``
    (given only when the server basis is primary) synchronously, with the
    post-first-chunk count ``completion_tokens - 1``.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        *,
        names: BasisSeriesNames,
        collector: SuperPassCollector | None,
        primary: TokenCountSource,
    ):
        super().__init__(names.osl, registry, requires=(SampleField.RECV_FIRST_NS,))
        self._isl = names.isl
        self._tpot = names.tpot_ns
        self._collector = collector
        self._primary = primary
        self._miss_logged = False

    def fire(self, ev_rec, row, pre_change):
        output = ev_rec.data
        if not isinstance(output, TextModelOutput):
            return
        prompt, completion = ev_rec.prompt_tokens, ev_rec.completion_tokens
        usage = classify_server_usage(prompt, completion, output)
        if usage is not ServerUsage.USABLE:
            if usage is ServerUsage.MISSING:
                self._count_missing(ev_rec, output)
            return
        registry = self.registry
        if self._isl is not None:
            registry.record(self._isl, prompt)
        registry.record(self.metric_name, completion)
        if self._tpot is None or completion <= 1:
            return
        recv_first_ns = pre_change[SampleField.RECV_FIRST_NS]
        if recv_first_ns is None:
            return
        delta = ev_rec.timestamp_ns - recv_first_ns
        if delta <= 0:
            return
        value = delta / (completion - 1)
        registry.record(self._tpot, value)
        if self._collector is not None and row.superpass_index >= 0:
            self._collector.add_tpot(row.superpass_index, value, completion - 1)

    def _count_missing(self, ev_rec: EventRecord, output: TextModelOutput) -> None:
        """Count a sample without usable usage; log the run's first one.

        An empty output is counted on its own: in-band error frames and cut
        streams look like that, and client counting would hide them rather
        than fix them.
        """
        registry = self.registry
        registry.increment(MetricCounterKey.TRACKED_SAMPLES_MISSING_USAGE.value)
        empty_output = not has_countable_output(output)
        if empty_output:
            registry.increment(
                MetricCounterKey.TRACKED_SAMPLES_MISSING_USAGE_EMPTY_OUTPUT.value
            )
        if self._miss_logged:
            return
        # Once per run, so a long run shows the problem early without logging
        # per sample. The aggregator counts this sample's completion after its
        # triggers fire, hence the + 1.
        self._miss_logged = True
        n_succeeded = (
            registry.counter_value(MetricCounterKey.TRACKED_SAMPLES_COMPLETED.value)
            - registry.counter_value(MetricCounterKey.TRACKED_SAMPLES_FAILED.value)
            + 1
        )
        logger.log(
            (
                logging.WARNING
                if self._primary is TokenCountSource.SERVER
                else logging.INFO
            ),
            "Tracked sample %s completed without usable server token usage: %s. "
            "Later misses are only counted (tracked_samples_missing_usage).",
            ev_rec.sample_uuid,
            missing_usage_message(
                1, int(empty_output), int(n_succeeded), primary=self._primary
            ),
        )


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

        # Session-level state
        self.is_tracking: bool = False
        self.session_started_ns: int | None = None
        self.tracked_blocks: list[TrackedBlock] = []

        # LoadGen window anchors: start at the FIRST issued tracked request
        # (LoadGen t=0), end at the completion of the last-issued request that
        # completed. See total_loadgen_window_ns.
        self._loadgen_max_issued_ns: int = -1
        self._loadgen_window_end_ns: int | None = None
        self._loadgen_window_start_ns: int | None = None

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

    @property
    def total_loadgen_window_ns(self) -> int:
        """Window from the first issued tracked request (LoadGen t=0) to the
        completion of the last-issued request that completed — the legacy MLPerf
        LoadGen ``final_query_all_samples_done_time`` analog. Returns 0 (=>
        the legacy metric falls back to native) when no tracked request completed."""
        if self._loadgen_window_end_ns is None or self._loadgen_window_start_ns is None:
            return 0
        return max(0, self._loadgen_window_end_ns - self._loadgen_window_start_ns)

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
            if self._loadgen_window_start_ns is None:
                self._loadgen_window_start_ns = value
        else:
            row = self._in_flight.get(sample_uuid)
            if row is None:
                return

        self._fire_triggers(row, field_name, ev_rec)
        setattr(row, field_name, value)

        if ev == SampleEventType.COMPLETE:
            self._update_tracked_block(row, ev_rec.timestamp_ns)
            self._in_flight.pop(sample_uuid, None)

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
            trigger.fire(ev_rec, row, pre_change)

    def _update_tracked_block(self, row: SampleRow, complete_ns: int) -> None:
        """Extend the sample's tracked block duration and increment count."""
        idx = row.tracked_block_idx
        if 0 <= idx < len(self.tracked_blocks):
            block = self.tracked_blocks[idx]
            if complete_ns > block.last_complete_ns:
                block.last_complete_ns = complete_ns
            block.completed_samples += 1
        # End the legacy LoadGen window at the completion of the last-issued
        # (largest issued_ns) request that completed.
        if row.issued_ns is not None and row.issued_ns > self._loadgen_max_issued_ns:
            self._loadgen_max_issued_ns = row.issued_ns
            self._loadgen_window_end_ns = complete_ns
