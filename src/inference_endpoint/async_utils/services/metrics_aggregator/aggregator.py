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

"""MetricsAggregatorService: thin event router for real-time metrics."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Final

from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqMessageSubscriber,
)
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    EventRecordCodec,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import PhaseData
from inference_endpoint.metrics.steady_state_diagnostics import (
    PERFORMANCE_PHASE,
    PROFILES,
    SuperPassCollector,
    compute_steady_state_metrics,
)

from .metrics_table import (
    ChunkDeltaTrigger,
    IslTrigger,
    MetricSeriesKey,
    MetricsTable,
    OslTrigger,
    SampleField,
    SampleLatencyTrigger,
    TpotTrigger,
    TtftTrigger,
)
from .publisher import MetricsPublisher
from .registry import TOKEN_HDR_HIGH, TOKEN_HDR_LOW, MetricsRegistry
from .snapshot import SessionState
from .token_metrics import BatchTokenizer, TokenBatchQueue

logger = logging.getLogger(__name__)

# Ceiling on the steady-state analysis, which runs in front of the final snapshot
# write. Must stay comfortably below the parent's post-drain grace
# (``Timeouts.service_exit_grace_s``, 60s), because once that expires the parent
# kills this process -- and a kill before ``publish_final`` costs the run the
# snapshot its Report is built from, not merely its verdict.
STEADY_STATE_ANALYSIS_TIMEOUT_S: Final[float] = 30.0


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


_FINISH_REASON_COUNTERS: Final[dict[str, MetricCounterKey]] = {
    "stop": MetricCounterKey.TRACKED_FINISH_REASON_STOP,
    "length": MetricCounterKey.TRACKED_FINISH_REASON_LENGTH,
    "tool_calls": MetricCounterKey.TRACKED_FINISH_REASON_TOOL_CALLS,
    "content_filter": MetricCounterKey.TRACKED_FINISH_REASON_CONTENT_FILTER,
    "function_call": MetricCounterKey.TRACKED_FINISH_REASON_FUNCTION_CALL,
}


_TRACKED_SAMPLE_EVENTS = frozenset(
    {
        SampleEventType.ISSUED,
        SampleEventType.COMPLETE,
        SampleEventType.RECV_FIRST,
        SampleEventType.RECV_NON_FIRST,
    }
)


# HDR bounds per series — chosen conservatively so realistic benchmark
# values cannot fall outside [low, high]. Values outside the range are
# clamped on insert and a warning is logged once per series.
_NS_HDR_LOW: Final[int] = 1
_NS_HDR_HIGH: Final[int] = 3_600_000_000_000  # 1 hour in ns


class MetricsAggregatorService(ZmqMessageSubscriber[EventRecord]):
    """Subscribes to EventRecords and computes per-sample metrics in real time.

    The aggregator is a thin event router. All state management, trigger
    dispatch, and row lifecycle are handled by ``MetricsTable``. The
    ``MetricsRegistry`` holds counters and series; the ``MetricsPublisher``
    publishes ``MetricsSnapshot`` over pub/sub at a fixed cadence and
    mirrors the final snapshot to disk.
    """

    def __init__(
        self,
        *args,
        registry: MetricsRegistry,
        publisher: MetricsPublisher,
        publish_interval_s: float,
        sig_figs: int,
        n_histogram_buckets: int,
        tokenizer: BatchTokenizer | None = None,
        live_flush_interval_s: float | None = None,
        streaming: bool = False,
        shutdown_event: asyncio.Event | None = None,
        drain_timeout_s: float | None = None,
        enable_isl: bool = True,
        steady_state_profile: str | None = None,
        **kwargs,
    ):
        # drain_timeout_s is injected (not derived) because the right
        # value is workload-dependent and the aggregator can't measure it
        # ahead of time. Keeping it as an arg lets the __main__ CLI flag
        # plumb the user's choice through without coupling this class to
        # argparse. None (the default) means unlimited — wait until the
        # drain finishes; an incomplete drain is flagged via
        # n_pending_tasks, never silently dropped.
        super().__init__(EventRecordCodec(), *args, **kwargs)
        self._registry = registry
        self._publisher = publisher
        self._publish_interval_s = publish_interval_s
        # Token triggers enqueue onto this queue; it is flushed by the
        # queue's own live loop (start_live) and by the end-of-run drain.
        # None when no tokenizer is set (token metrics disabled), in which
        # case those triggers are no-ops.
        self._token_queue: TokenBatchQueue | None = (
            TokenBatchQueue(tokenizer, self.loop) if tokenizer is not None else None
        )
        # Cadence of the queue's live flush loop (None = no mid-run
        # tokenization; everything defers to the end-of-run drain).
        self._live_flush_interval_s = live_flush_interval_s
        self._streaming = streaming
        self._shutdown_event = shutdown_event
        self._shutdown_received = False
        # Latched by SessionEventType.INTERRUPTED (published by the session
        # right before ENDED on an aborted run) so the ENDED-driven finalize
        # below tags the snapshot state=interrupted, not COMPLETE.
        self._interrupted = False
        self._drain_timeout_s = drain_timeout_s

        self._session_start_ns: int | None = None
        self._total_duration_ns: int = 0
        self._total_processed = 0
        self._last_log_count = 0
        # Tracks the run's lifecycle state, surfaced on the wire as
        # MetricsSnapshot.state. Transitions are forward-only:
        # INITIALIZE → LIVE (on first STARTED) → DRAINING (on ENDED) →
        # COMPLETE (set implicitly via publish_final).
        self._session_state: SessionState = SessionState.INITIALIZE

        # Pre-register all metrics on the registry. Tests can introspect via
        # registry.has_counter / has_series.
        self._register_metrics(streaming, enable_isl, sig_figs, n_histogram_buckets)

        # Steady-state collection is opt-in, and the profile name is the opt-in:
        # it carries the CoV bounds and warmup driver the verdict is judged on, and
        # only the parent knows the load pattern they follow from. None = off.
        # An unsupported profile -- agentic, whose steady-state metric is
        # per-trajectory NATL and not what this collector produces, or offline --
        # is refused here as well as by the parent's gate, so neither side alone
        # can turn collection on for a workload it does not describe.
        self._steady_state_profile = (
            PROFILES[steady_state_profile] if steady_state_profile else None
        )
        if self._steady_state_profile is not None and not (
            self._steady_state_profile.supported
        ):
            logger.warning(
                "Steady-state collection refused: profile %r is not supported",
                steady_state_profile,
            )
            self._steady_state_profile = None
        # The bucket size arrives later, on PHASE_START, but TpotTrigger needs the
        # collector at registration time -- before any event.
        self._collector = (
            SuperPassCollector() if self._steady_state_profile is not None else None
        )

        self._table = MetricsTable(self._registry)
        self._register_triggers(streaming, enable_isl)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register_metrics(
        self,
        streaming: bool,
        enable_isl: bool,
        sig_figs: int,
        n_histogram_buckets: int,
    ) -> None:
        """Register all counters and series on the registry."""
        for key in MetricCounterKey:
            self._registry.register_counter(key.value)

        # Always-present series
        self._registry.register_series(
            MetricSeriesKey.SAMPLE_LATENCY_NS.value,
            hdr_low=_NS_HDR_LOW,
            hdr_high=_NS_HDR_HIGH,
            tail_latency=True,
            sig_figs=sig_figs,
            n_histogram_buckets=n_histogram_buckets,
        )
        if enable_isl:
            self._registry.register_series(
                MetricSeriesKey.ISL.value,
                hdr_low=TOKEN_HDR_LOW,
                hdr_high=TOKEN_HDR_HIGH,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
            )
        self._registry.register_series(
            MetricSeriesKey.OSL.value,
            hdr_low=TOKEN_HDR_LOW,
            hdr_high=TOKEN_HDR_HIGH,
            sig_figs=sig_figs,
            n_histogram_buckets=n_histogram_buckets,
        )

        # Streaming-only series
        if streaming:
            self._registry.register_series(
                MetricSeriesKey.TTFT_NS.value,
                hdr_low=_NS_HDR_LOW,
                hdr_high=_NS_HDR_HIGH,
                tail_latency=True,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
            )
            self._registry.register_series(
                MetricSeriesKey.CHUNK_DELTA_NS.value,
                hdr_low=_NS_HDR_LOW,
                hdr_high=_NS_HDR_HIGH,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
            )
            self._registry.register_series(
                MetricSeriesKey.TPOT_NS.value,
                hdr_low=_NS_HDR_LOW,
                hdr_high=_NS_HDR_HIGH,
                tail_latency=True,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
                dtype=float,
            )

    def _register_triggers(self, streaming: bool, enable_isl: bool) -> None:
        """Register metric triggers on the table.

        Streaming-only triggers (TTFT, chunk_delta, TPOT) are only registered
        when ``streaming=True``; the ISL trigger only when ``enable_isl=True``.
        """
        table = self._table
        registry = self._registry
        queue = self._token_queue

        # Latency and OSL are always registered; ISL is opt-out.
        if enable_isl:
            table.add_trigger(SampleField.ISSUED_NS, IslTrigger(registry, queue))
        table.add_trigger(SampleField.COMPLETE_NS, SampleLatencyTrigger(registry))
        table.add_trigger(SampleField.COMPLETE_NS, OslTrigger(registry, queue))

        # Streaming-only
        if streaming:
            table.add_trigger(SampleField.RECV_FIRST_NS, TtftTrigger(registry))
            table.add_trigger(SampleField.LAST_RECV_NS, ChunkDeltaTrigger(registry))
            table.add_trigger(
                SampleField.COMPLETE_NS, TpotTrigger(registry, queue, self._collector)
            )

    @property
    def table(self) -> MetricsTable:
        """The per-sample metrics table (read-only; for service wiring)."""
        return self._table

    @property
    def token_queue(self) -> TokenBatchQueue | None:
        """The token batch queue, if token metrics are enabled."""
        return self._token_queue

    @property
    def pending_tokens(self) -> int:
        """Enqueued tokenizations not yet recorded (the snapshot n_pending_tasks)."""
        return self._token_queue.pending if self._token_queue is not None else 0

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    async def process(self, records: list[EventRecord]) -> None:
        saw_shutdown = False
        table = self._table
        registry = self._registry
        collector = self._collector

        self._total_processed += len(records)
        if self._total_processed - self._last_log_count >= 10000:
            logger.debug(
                "Aggregator processed %d records (%d in this batch)",
                self._total_processed,
                len(records),
            )
            self._last_log_count = self._total_processed

        for record in records:
            if self._shutdown_received:
                break

            ev = record.event_type

            # Update total_duration_ns on every event
            if self._session_start_ns is not None:
                elapsed = record.timestamp_ns - self._session_start_ns
                if elapsed > self._total_duration_ns:
                    self._total_duration_ns = elapsed
                    registry.set_counter(
                        MetricCounterKey.TOTAL_DURATION_NS.value,
                        self._total_duration_ns,
                    )

            # --- Session events ---
            if isinstance(ev, SessionEventType):
                if ev == SessionEventType.ENDED:
                    logger.info("ENDED event received, shutting down aggregator")
                    self._shutdown_received = True
                    saw_shutdown = True
                elif ev == SessionEventType.INTERRUPTED:
                    logger.info(
                        "INTERRUPTED marker received — final snapshot will be "
                        "tagged state=interrupted"
                    )
                    self._interrupted = True
                elif ev == SessionEventType.PHASE_START:
                    # Performance phases only: warmup announces first, with its
                    # own dataset size, and the standalone parse reads the same
                    # phase. announce_phase adds the second guard -- a series
                    # already being bucketed cannot be re-sized.
                    if (
                        collector is not None
                        and isinstance(record.data, PhaseData)
                        and record.data.phase_type == PERFORMANCE_PHASE
                    ):
                        collector.announce_phase(record.data.num_turns)
                else:
                    if ev == SessionEventType.STARTED:
                        if self._session_start_ns is not None:
                            # A duplicate STARTED is a producer bug:
                            # re-assigning _session_start_ns would freeze
                            # total_duration_ns (the max-of-elapsed guard
                            # never updates once the start moves forward)
                            # and corrupt every downstream rate calc for
                            # the rest of the run. Surface loudly and
                            # ignore — the publisher.start guard already
                            # rejects the second tick-task spawn, but
                            # session-state must also be defended here.
                            logger.error(
                                "Duplicate STARTED event received "
                                "(original at ts=%d, duplicate at ts=%d); "
                                "ignoring — producer must emit STARTED "
                                "exactly once per session.",
                                self._session_start_ns,
                                record.timestamp_ns,
                            )
                        else:
                            self._session_start_ns = record.timestamp_ns
                            self._session_state = SessionState.LIVE
                            # Now that we have an event loop running, start
                            # the publisher tick task. The callable is
                            # invoked once per tick to capture the live
                            # (state, n_pending_tasks) pair at each emit.
                            self._publisher.start(
                                registry,
                                self._publish_interval_s,
                                get_runtime_state=lambda: (
                                    self._session_state,
                                    self.pending_tokens,
                                ),
                            )
                            if (
                                self._token_queue is not None
                                and self._live_flush_interval_s is not None
                            ):
                                self._token_queue.start_live(
                                    self._live_flush_interval_s
                                )
                    table.handle_session_event(record)
                    if ev == SessionEventType.STOP_PERFORMANCE_TRACKING:
                        registry.set_counter(
                            MetricCounterKey.TRACKED_DURATION_NS.value,
                            table.total_tracked_duration_ns,
                        )
                        registry.set_counter(
                            MetricCounterKey.LEGACY_LOADGEN_WINDOW_DURATION_NS.value,
                            table.total_loadgen_window_ns,
                        )
                logger.debug("Session event: %s", ev)
                continue

            # --- Error events ---
            # Counted BEFORE the COMPLETE event (session.py emits ERROR
            # first), so the tracked row still exists for tracked-failed
            # detection.
            if isinstance(ev, ErrorEventType):
                registry.increment(MetricCounterKey.TOTAL_SAMPLES_FAILED.value)
                if record.sample_uuid and table.get_row(record.sample_uuid) is not None:
                    registry.increment(MetricCounterKey.TRACKED_SAMPLES_FAILED.value)
                logger.debug("Error event: %s", record)
                continue

            # --- Sample events ---
            if (
                not isinstance(ev, SampleEventType)
                or ev not in _TRACKED_SAMPLE_EVENTS
                or not record.sample_uuid
            ):
                continue

            uuid = record.sample_uuid
            ts = record.timestamp_ns

            if ev == SampleEventType.ISSUED:
                table.set_field(uuid, SampleField.ISSUED_NS, ts, record)
                registry.increment(MetricCounterKey.TOTAL_SAMPLES_ISSUED.value)
                row = table.get_row(uuid)
                if row is not None:
                    registry.increment(MetricCounterKey.TRACKED_SAMPLES_ISSUED.value)
                    # is_tracking as well as the row: set_field returns early for an
                    # ISSUED outside the tracking window WITHOUT dropping the
                    # in-flight row, so a sample re-issued after
                    # STOP_PERFORMANCE_TRACKING still has one. The event-log parse
                    # skips it, and a bucket whose last_issue_ns came from outside
                    # the window reports a different system TPS.
                    if collector is not None and table.is_tracking:
                        # The row carries the super-pass, so a second issue of the
                        # same sample is a retry, not a new bucket slot.
                        if row.sp_index < 0:
                            row.sp_index = collector.assign(ts)
                        else:
                            collector.reissue(row.sp_index, ts)
            elif ev == SampleEventType.RECV_FIRST:
                # Read before set_field overwrites recv_first_ns: a retried sample
                # re-emits RECV_FIRST and must not contribute a second TTFT.
                row = table.get_row(uuid) if collector is not None else None
                first_chunk = row is not None and row.recv_first_ns is None
                table.set_field(uuid, SampleField.RECV_FIRST_NS, ts, record)
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
                if (
                    collector is not None
                    and row is not None
                    and row.sp_index >= 0
                    and row.issued_ns is not None
                ):
                    collector.on_recv_first(
                        row.sp_index, row.issued_ns, ts, record.turn, first=first_chunk
                    )
            elif ev == SampleEventType.RECV_NON_FIRST:
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
            elif ev == SampleEventType.COMPLETE:
                # Captured before set_field, which drops the row from the table.
                row = table.get_row(uuid)
                is_tracked = row is not None
                table.set_field(uuid, SampleField.COMPLETE_NS, ts, record)
                if (
                    collector is not None
                    and row is not None
                    and row.sp_index >= 0
                    and row.issued_ns is not None
                ):
                    collector.on_complete(
                        row.sp_index, row.issued_ns, row.recv_first_ns, ts
                    )
                registry.increment(MetricCounterKey.TOTAL_SAMPLES_COMPLETED.value)
                if is_tracked:
                    registry.increment(MetricCounterKey.TRACKED_SAMPLES_COMPLETED.value)
                    if isinstance(record.finish_reason, str):
                        counter = _FINISH_REASON_COUNTERS.get(
                            record.finish_reason,
                            MetricCounterKey.TRACKED_FINISH_REASON_OTHER,
                        )
                        registry.increment(counter.value)

        if saw_shutdown:
            # ENDED has been observed; transition to DRAINING so any tick
            # that fires before publish_final reflects the new state.
            self._session_state = SessionState.DRAINING
            logger.info("Draining %d pending tokenizations...", self.pending_tokens)
            # The drain and final publish are wrapped together so the aggregator
            # ALWAYS reaches _finalize (which sets the shutdown event); a
            # tokenizer failure during the drain must not skip publish_final and
            # leave main()'s `await shutdown_event.wait()` hanging.
            n_pending = self.pending_tokens
            try:
                # flush_remaining tokenizes the whole buffer in one batched pass,
                # bounded by the drain budget, and never raises: it returns the
                # count it could not finish (timeout or failure), which becomes
                # the snapshot's n_pending_tasks so Report flags an incomplete drain.
                if self._token_queue is not None:
                    n_pending = await self._token_queue.flush_remaining(
                        self._drain_timeout_s
                    )
                if n_pending > 0:
                    budget = (
                        f"{self._drain_timeout_s:.1f}s"
                        if self._drain_timeout_s is not None
                        else "unlimited"
                    )
                    logger.warning(
                        "tokenizer drain incomplete (budget %s); %d tokenizations "
                        "did not complete",
                        budget,
                        n_pending,
                    )
                else:
                    logger.info("Tokenizations fully drained (n_pending_tasks=0)")
                registry.set_counter(
                    MetricCounterKey.TRACKED_DURATION_NS.value,
                    table.total_tracked_duration_ns,
                )
                registry.set_counter(
                    MetricCounterKey.LEGACY_LOADGEN_WINDOW_DURATION_NS.value,
                    table.total_loadgen_window_ns,
                )
                await self._publisher.publish_final(
                    registry,
                    n_pending_tasks=n_pending,
                    interrupted=self._interrupted,
                    steady_state=await self._steady_state_verdict(n_pending),
                )
            finally:
                # The aggregator MUST close the publisher and signal shutdown even
                # if the drain/publish above failed — otherwise main()'s
                # `await shutdown_event.wait()` hangs forever. aclose is
                # independently wrapped: its failure must not prevent _finalize,
                # which is what sets the shutdown event.
                try:
                    await self._publisher.aclose()
                except Exception:  # noqa: BLE001 — best-effort cleanup.
                    logger.exception(
                        "metrics: publisher.aclose failed during ENDED finalize"
                    )
                self._finalize()

    async def _steady_state_verdict(self, n_pending: int) -> dict | None:
        """The verdict for the terminal snapshot, or None if it is not deserved.

        Only computed for a run the collected series actually describes: an
        interrupted run is truncated, and an incomplete drain means token counts
        are missing, so TPOT and OSL would be wrong.

        Runs before the snapshot is built so one atomic write carries it, which
        puts the analysis in front of the file the Report is built from. Nothing
        else bounds that: the drain budget covers ``flush_remaining`` and has
        already returned by now, and the parent kills this process once its own
        grace expires. So the analysis carries its own deadline, and a run that
        blows it publishes without a verdict rather than without a snapshot --
        no failure here may cost the run its snapshot, wall-clock included.

        Computed off the event loop so the deadline can be enforced at all: a
        synchronous call cannot be interrupted, and abandoning the thread is
        fine because the process exits moments later.
        """
        collector, profile = self._collector, self._steady_state_profile
        if collector is None or profile is None:
            return None
        if self._interrupted or n_pending > 0 or not collector.superpass_size:
            logger.info("Steady state not computed: the run is not described by it")
            return None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: dict(
                        compute_steady_state_metrics(
                            collector.series(),
                            superpass_size=collector.superpass_size,
                            cov_bounds=profile.cov_bounds,
                            warmup_driver=profile.warmup_driver,
                        )
                    )
                ),
                timeout=STEADY_STATE_ANALYSIS_TIMEOUT_S,
            )
        except TimeoutError:
            logger.warning(
                "Steady state not computed: analysis exceeded %.0fs; the snapshot "
                "is published without a verdict",
                STEADY_STATE_ANALYSIS_TIMEOUT_S,
            )
            return None
        except Exception:  # noqa: BLE001 — best-effort; never fail the run.
            logger.exception("metrics: steady-state detection failed")
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        logger.info(
            "Aggregator finalized: %d total records processed", self._total_processed
        )
        self.close()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        elif self.loop is not None and self.loop.is_running():
            self.loop.stop()

    def close(self) -> None:
        try:
            self._publisher.close()
        except Exception:  # noqa: BLE001 — close is best-effort during shutdown.
            logger.exception("metrics: publisher close failed")
        super().close()
