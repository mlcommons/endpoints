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
from dataclasses import dataclass
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
from inference_endpoint.core.types import (
    TOKEN_HDR_HIGH,
    TOKEN_HDR_LOW,
    PhaseData,
    PhaseType,
    TokenCountSource,
    basis_key,
)
from inference_endpoint.metrics.steady_state_diagnostics import (
    PROFILES,
    SteadyState,
    SuperPassCollector,
    compute_steady_state_metrics,
)

from .metrics_table import (
    BasisSeriesNames,
    ChunkDeltaTrigger,
    IslTrigger,
    MetricCounterKey,
    MetricSeriesKey,
    MetricsTable,
    OslTrigger,
    SampleField,
    SampleLatencyTrigger,
    ServerUsageTrigger,
    TpotTrigger,
    TtftTrigger,
)
from .publisher import MetricsPublisher
from .registry import MetricsRegistry
from .snapshot import SessionState
from .token_metrics import BatchTokenizer, TokenBatchQueue

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class TokenSeriesSpec:
    """One token metric, registered once per token-count basis."""

    key: MetricSeriesKey
    dtype: type
    hdr_low: int
    hdr_high: int
    # Recorded only for streaming runs.
    streaming_only: bool
    # Dropped on both bases when ISL collection is off.
    is_isl: bool
    # Gets early-stopping estimates, on the primary basis only.
    tail_latency: bool


TOKEN_SERIES: Final[tuple[TokenSeriesSpec, ...]] = (
    TokenSeriesSpec(
        key=MetricSeriesKey.ISL,
        dtype=int,
        hdr_low=TOKEN_HDR_LOW,
        hdr_high=TOKEN_HDR_HIGH,
        streaming_only=False,
        is_isl=True,
        tail_latency=False,
    ),
    TokenSeriesSpec(
        key=MetricSeriesKey.OSL,
        dtype=int,
        hdr_low=TOKEN_HDR_LOW,
        hdr_high=TOKEN_HDR_HIGH,
        streaming_only=False,
        is_isl=False,
        tail_latency=False,
    ),
    TokenSeriesSpec(
        key=MetricSeriesKey.TPOT_NS,
        dtype=float,
        hdr_low=_NS_HDR_LOW,
        hdr_high=_NS_HDR_HIGH,
        streaming_only=True,
        is_isl=False,
        tail_latency=True,
    ),
)


def series_names(
    basis: TokenCountSource,
    *,
    primary: TokenCountSource,
    enable_isl: bool,
    streaming: bool,
) -> BasisSeriesNames:
    """The ``TOKEN_SERIES`` names ``basis`` records into in this run.

    Names follow ``basis_key``: unsuffixed on the primary basis. A series the
    run does not record is left out (``None``).
    """
    return BasisSeriesNames(
        **{
            spec.key.value: basis_key(spec.key.value, basis, primary=primary)
            for spec in TOKEN_SERIES
            if (enable_isl or not spec.is_isl)
            and (streaming or not spec.streaming_only)
        }
    )


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
        token_count_source: TokenCountSource,
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
        # Tokenizer triggers enqueue onto this queue; it is flushed by the
        # queue's own live loop (start_live) and by the end-of-run drain.
        # None when no tokenizer is set, in which case the client-basis
        # triggers count only pre-tokenized prompts; the server-basis trigger
        # needs no tokenizer.
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
        # Both bases record every token series; the primary keeps the base
        # names and feeds early stopping and the steady-state collector.
        self._token_count_source = token_count_source
        self._series_names = {
            basis: series_names(
                basis,
                primary=token_count_source,
                enable_isl=enable_isl,
                streaming=streaming,
            )
            for basis in TokenCountSource
        }

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
        self._register_metrics(streaming, sig_figs, n_histogram_buckets)

        # A profile name enables steady-state collection. The profile carries
        # the CoV bounds and warmup driver used for the verdict. None leaves
        # collection off.
        #
        # Only the parent knows the load pattern, so it chooses the profile.
        # This subprocess also refuses unsupported profiles, so neither side
        # can enable collection for a run the verdict does not describe.
        #
        # Use .get so an unknown name degrades to collection off. Raising
        # KeyError here would kill the subprocess during startup and surface
        # as a launch timeout in the parent.
        self._steady_state_profile = (
            PROFILES.get(steady_state_profile) if steady_state_profile else None
        )
        if steady_state_profile and self._steady_state_profile is None:
            logger.warning(
                "Steady-state collection refused: unknown profile %r",
                steady_state_profile,
            )
        if self._steady_state_profile is not None and not (
            self._steady_state_profile.supported
        ):
            logger.warning(
                "Steady-state collection refused: profile %r is not supported",
                steady_state_profile,
            )
            self._steady_state_profile = None
        # PHASE_START supplies the bucket size. The primary basis's TPOT
        # trigger needs the collector registered before any event can arrive,
        # so the size is latched later.
        self._collector = (
            SuperPassCollector() if self._steady_state_profile is not None else None
        )

        self._table = MetricsTable(self._registry)
        self._register_triggers(streaming)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register_metrics(
        self,
        streaming: bool,
        sig_figs: int,
        n_histogram_buckets: int,
    ) -> None:
        """Register all counters and series on the registry.

        Series are registered as ``sample_latency_ns``, primary ``isl`` and
        ``osl``, ``ttft_ns``, ``chunk_delta_ns``, primary ``tpot_ns``, then the
        other basis's ``TOKEN_SERIES``. Snapshot series follow registration
        order, so a snapshot lists the primary basis's series before the other
        basis's.

        Must stay in lockstep with ``_register_triggers``: recording into an
        unregistered series raises inside ``process()``. Both take the token
        series names from ``self._series_names``.
        """
        registry = self._registry
        primary = self._token_count_source
        for key in MetricCounterKey:
            registry.register_counter(key.value)

        def register_timing(key: MetricSeriesKey, *, tail_latency: bool) -> None:
            registry.register_series(
                key.value,
                hdr_low=_NS_HDR_LOW,
                hdr_high=_NS_HDR_HIGH,
                tail_latency=tail_latency,
                sig_figs=sig_figs,
                n_histogram_buckets=n_histogram_buckets,
            )

        def register_tokens(
            basis: TokenCountSource, specs: tuple[TokenSeriesSpec, ...]
        ) -> None:
            names = self._series_names[basis]
            for spec in specs:
                name = getattr(names, spec.key.value)
                if name is None:
                    continue
                registry.register_series(
                    name,
                    hdr_low=spec.hdr_low,
                    hdr_high=spec.hdr_high,
                    tail_latency=spec.tail_latency and basis is primary,
                    sig_figs=sig_figs,
                    n_histogram_buckets=n_histogram_buckets,
                    dtype=spec.dtype,
                )

        register_timing(MetricSeriesKey.SAMPLE_LATENCY_NS, tail_latency=True)
        register_tokens(
            primary, tuple(spec for spec in TOKEN_SERIES if not spec.streaming_only)
        )
        if streaming:
            register_timing(MetricSeriesKey.TTFT_NS, tail_latency=True)
            register_timing(MetricSeriesKey.CHUNK_DELTA_NS, tail_latency=False)
        register_tokens(
            primary, tuple(spec for spec in TOKEN_SERIES if spec.streaming_only)
        )
        register_tokens(primary.other, TOKEN_SERIES)

    def _register_triggers(self, streaming: bool) -> None:
        """Register metric triggers on the table.

        Both bases always record: the tokenizer triggers into the client
        names, and one ``ServerUsageTrigger`` into the server names. Only the
        primary basis's TPOT trigger gets the steady-state collector.
        Streaming-only triggers (TTFT, chunk_delta, client TPOT) are
        registered only when ``streaming=True``; the ISL trigger only when ISL
        is collected.
        """
        table = self._table
        registry = self._registry
        queue = self._token_queue
        primary = self._token_count_source
        client = self._series_names[TokenCountSource.CLIENT]
        server = self._series_names[TokenCountSource.SERVER]

        if client.isl is not None:
            table.add_trigger(
                SampleField.ISSUED_NS,
                IslTrigger(registry, queue, metric_name=client.isl),
            )
        table.add_trigger(SampleField.COMPLETE_NS, SampleLatencyTrigger(registry))
        table.add_trigger(
            SampleField.COMPLETE_NS,
            OslTrigger(registry, queue, metric_name=client.osl),
        )
        table.add_trigger(
            SampleField.COMPLETE_NS,
            ServerUsageTrigger(
                registry,
                names=server,
                collector=(
                    self._collector if primary is TokenCountSource.SERVER else None
                ),
                primary=primary,
            ),
        )

        if streaming:
            table.add_trigger(SampleField.RECV_FIRST_NS, TtftTrigger(registry))
            table.add_trigger(SampleField.LAST_RECV_NS, ChunkDeltaTrigger(registry))
        if client.tpot_ns is not None:
            table.add_trigger(
                SampleField.COMPLETE_NS,
                TpotTrigger(
                    registry,
                    queue,
                    metric_name=client.tpot_ns,
                    collector=(
                        self._collector if primary is TokenCountSource.CLIENT else None
                    ),
                ),
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
                    # Only performance phases define the bucket size. Warmup
                    # announces first with its own dataset size, and the
                    # standalone parser follows the same rule.
                    #
                    # announce_phase also refuses to resize a series once
                    # bucket collection has started.
                    if (
                        collector is not None
                        and isinstance(record.data, PhaseData)
                        and record.data.phase_type is PhaseType.PERFORMANCE
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
                    # A row can outlive the tracking window.
                    # For ISSUED after STOP_PERFORMANCE_TRACKING, set_field
                    # returns early but keeps the in-flight row. Bucketing that issue
                    # would use last_issue_ns outside the window and change reported
                    # system TPS. The event-log parser skips it too.
                    if collector is not None and table.is_tracking:
                        # A row's superpass_index marks its bucket slot. If it already
                        # has one, this issue is a retry.
                        if row.superpass_index < 0:
                            row.superpass_index = collector.assign(ts)
                        else:
                            collector.reissue(row.superpass_index, ts)
            elif ev == SampleEventType.RECV_FIRST:
                # Read the row before set_field overwrites recv_first_ns. A
                # retried sample re-emits RECV_FIRST and must not add a second
                # TTFT.
                row = table.get_row(uuid) if collector is not None else None
                first_chunk = row is not None and row.recv_first_ns is None
                table.set_field(uuid, SampleField.RECV_FIRST_NS, ts, record)
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
                if (
                    collector is not None
                    and row is not None
                    and row.superpass_index >= 0
                    and row.issued_ns is not None
                ):
                    collector.on_recv_first(
                        row.superpass_index,
                        row.issued_ns,
                        ts,
                        record.turn,
                        first=first_chunk,
                    )
            elif ev == SampleEventType.RECV_NON_FIRST:
                table.set_field(uuid, SampleField.LAST_RECV_NS, ts, record)
            elif ev == SampleEventType.COMPLETE:
                # Read the row before set_field, which drops it from the table.
                row = table.get_row(uuid)
                is_tracked = row is not None
                table.set_field(uuid, SampleField.COMPLETE_NS, ts, record)
                if (
                    collector is not None
                    and row is not None
                    and row.superpass_index >= 0
                    and row.issued_ns is not None
                ):
                    collector.on_complete(
                        row.superpass_index, row.issued_ns, row.recv_first_ns, ts
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

    async def _steady_state_verdict(self, n_pending: int) -> SteadyState | None:
        """Return the steady-state verdict for the terminal snapshot.

        Returns None when collection is disabled or the series does not describe
        the completed run. An interrupted run is truncated. Otherwise the gate
        follows the primary basis, which feeds the collector's TPOT and OSL. On
        the server basis a tracked sample without usable usage leaves them
        incomplete, while a pending tokenizer drain only affects the client
        series and is ignored. On the client basis a pending drain leaves them
        incomplete, while missing server usage is ignored.

        This runs before the snapshot is built. The single atomic final write
        then carries both the metrics and the verdict, so the analysis sits in
        front of the file Report is built from.

        The analysis carries no deadline of its own. It is pure CPU over the
        super-pass rollups the collector already built, and its cost tracks the
        super-pass count: measured 2.9s at 100 super-passes (438,800 samples,
        the longest shape this runs at) and 19s at 245. A deadline in front of
        the final snapshot write would be a knob guarding a cost that does not
        reach it.

        The compute runs in ``asyncio.to_thread`` so the event loop stays
        responsive while it works: a SIGTERM arriving mid-analysis still
        reaches its handler and writes the INTERRUPTED snapshot, rather than
        waiting on a blocked loop.
        """
        collector, profile = self._collector, self._steady_state_profile
        if collector is None or profile is None:
            return None
        if self._token_count_source is TokenCountSource.SERVER:
            incomplete_counts = (
                self._registry.counter_value(
                    MetricCounterKey.TRACKED_SAMPLES_MISSING_USAGE.value
                )
                > 0
            )
        else:
            incomplete_counts = n_pending > 0
        if self._interrupted or incomplete_counts or not collector.superpass_size:
            logger.info("Steady state not computed: the run is not described by it")
            return None
        try:
            return await asyncio.to_thread(
                compute_steady_state_metrics,
                collector.series(),
                superpass_size=collector.superpass_size,
                cov_bounds=profile.cov_bounds,
                warmup_driver=profile.warmup_driver,
            )
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
