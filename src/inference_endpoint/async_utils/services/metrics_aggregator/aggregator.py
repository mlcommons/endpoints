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

"""MetricsAggregatorService: real-time metrics from EventRecord stream."""

from __future__ import annotations

import asyncio
import logging

from inference_endpoint.async_utils.transport.zmq.pubsub import (
    ZmqEventRecordSubscriber,
)
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import PromptData, TextModelOutput

from .emitter import MetricEmitter
from .metrics_table import MetricsTable, SampleRow
from .token_metrics import TokenizePool

logger = logging.getLogger(__name__)

# SampleEventTypes that correspond to tracked timestamps.
# The enum value (e.g. "issued") is NOT used as a field name lookup —
# dispatch is explicit in process().
_TRACKED_SAMPLE_EVENTS = frozenset(
    {
        SampleEventType.ISSUED,
        SampleEventType.COMPLETE,
        SampleEventType.RECV_FIRST,
        SampleEventType.RECV_NON_FIRST,
        SampleEventType.CLIENT_SEND,
        SampleEventType.CLIENT_RESP_DONE,
    }
)


class MetricsAggregatorService(ZmqEventRecordSubscriber):
    """Subscribes to EventRecords and computes per-sample metrics in real time.

    Metrics are emitted to a MetricEmitter as soon as the triggering event arrives:
      - ttft_ns:             RECV_FIRST.timestamp - ISSUED.timestamp
      - sample_latency_ns:   COMPLETE.timestamp - ISSUED.timestamp
      - request_duration_ns: CLIENT_RESP_DONE.timestamp - CLIENT_SEND.timestamp
      - chunk_delta_ns:      each RECV_NON_FIRST.timestamp - previous recv timestamp
      - isl:                 len(token_ids) or token_count(prompt_text) — on ISSUED
      - osl:                 token_count(full_output) — computed on COMPLETE
      - tpot_ns:             (COMPLETE.timestamp - RECV_FIRST.timestamp) / (osl - first_chunk_tokens)
                             for streaming responses where osl > first_chunk_tokens

    A sample is only tracked if it is ISSUED while is_tracking is True.
    is_tracking is toggled by START/STOP_PERFORMANCE_TRACKING session events.
    """

    def __init__(
        self,
        *args,
        emitter: MetricEmitter,
        tokenize_pool: TokenizePool | None = None,
        shutdown_event: asyncio.Event | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._emitter = emitter
        self._tokenize_pool = tokenize_pool
        self._shutdown_event = shutdown_event

        self._table = MetricsTable()
        self._is_tracking = False
        self._session_started_ns: int | None = None
        self._shutdown_received = False

    async def process(self, records: list[EventRecord]) -> None:
        saw_shutdown = False

        for record in records:
            if self._shutdown_received:
                break

            ev = record.event_type

            # --- Session-level events ---
            if ev == SessionEventType.STARTED:
                self._session_started_ns = record.timestamp_ns
                continue
            if ev == SessionEventType.START_PERFORMANCE_TRACKING:
                self._is_tracking = True
                continue
            if ev == SessionEventType.STOP_PERFORMANCE_TRACKING:
                self._is_tracking = False
                continue
            if ev == SessionEventType.ENDED:
                self._shutdown_received = True
                saw_shutdown = True
                continue

            # --- Sample-level events ---
            if not isinstance(ev, SampleEventType) or ev not in _TRACKED_SAMPLE_EVENTS:
                continue

            sample_uuid = record.sample_uuid
            if not sample_uuid:
                continue

            if ev == SampleEventType.ISSUED:
                self._on_issued(record)
            elif ev == SampleEventType.RECV_FIRST:
                self._on_recv_first(record)
            elif ev == SampleEventType.RECV_NON_FIRST:
                self._on_recv_non_first(record)
            elif ev == SampleEventType.CLIENT_SEND:
                self._on_client_send(record)
            elif ev == SampleEventType.CLIENT_RESP_DONE:
                self._on_client_resp_done(record)
            elif ev == SampleEventType.COMPLETE:
                await self._on_complete(record)

        if saw_shutdown:
            self._finalize()

    # --- Per-event handlers ---

    def _on_issued(self, record: EventRecord) -> None:
        if not self._is_tracking:
            return

        row = self._table.create_row(record.sample_uuid)
        row.issued_ns = record.timestamp_ns

        if isinstance(record.data, PromptData):
            if record.data.token_ids is not None:
                # SGLang path: ISL is len(token_ids), no tokenization needed.
                self._emitter.emit(row.sample_uuid, "isl", len(record.data.token_ids))
            elif record.data.text is not None:
                # OpenAI path: tokenize the prompt text to compute ISL.
                row.prompt_text = record.data.text
                if self._tokenize_pool is not None:
                    self._schedule_isl(record.sample_uuid, record.data.text)

    def _on_recv_first(self, record: EventRecord) -> None:
        row = self._get_tracked_row(record.sample_uuid)
        if row is None:
            return

        row.recv_first_ns = record.timestamp_ns
        row.last_recv_ns = record.timestamp_ns

        ttft = row.ttft_ns()
        if ttft is not None:
            self._emitter.emit(row.sample_uuid, "ttft_ns", ttft)

        if isinstance(record.data, TextModelOutput):
            text = str(record.data)
            row.first_chunk_text = text
            row.output_chunks.append(text)

    def _on_recv_non_first(self, record: EventRecord) -> None:
        row = self._get_tracked_row(record.sample_uuid)
        if row is None:
            return

        if row.last_recv_ns is not None:
            delta = record.timestamp_ns - row.last_recv_ns
            self._emitter.emit(row.sample_uuid, "chunk_delta_ns", delta)

        row.last_recv_ns = record.timestamp_ns

        if isinstance(record.data, TextModelOutput):
            row.output_chunks.append(str(record.data))

    def _on_client_send(self, record: EventRecord) -> None:
        row = self._get_tracked_row(record.sample_uuid)
        if row is None:
            return
        row.client_send_ns = record.timestamp_ns

    def _on_client_resp_done(self, record: EventRecord) -> None:
        row = self._get_tracked_row(record.sample_uuid)
        if row is None:
            return

        row.client_resp_done_ns = record.timestamp_ns
        duration = row.request_duration_ns()
        if duration is not None:
            self._emitter.emit(row.sample_uuid, "request_duration_ns", duration)

    async def _on_complete(self, record: EventRecord) -> None:
        row = self._get_tracked_row(record.sample_uuid)
        if row is None:
            return

        row.complete_ns = record.timestamp_ns

        latency = row.sample_latency_ns()
        if latency is not None:
            self._emitter.emit(row.sample_uuid, "sample_latency_ns", latency)

        await self._compute_token_metrics(record, row)
        self._table.remove_row(row.sample_uuid)

    # --- Helpers ---

    def _get_tracked_row(self, sample_uuid: str) -> SampleRow | None:
        return self._table.get_row(sample_uuid)

    def _schedule_isl(self, sample_uuid: str, prompt_text: str) -> None:
        """Schedule ISL computation as a fire-and-forget async task.

        If the emitter is closed before the task completes (e.g. session ENDED
        arrives while tokenization is in progress), the emit is silently dropped
        by the emitter's closed-file guard.
        """
        if self.loop is None or self._tokenize_pool is None:
            return

        async def _compute() -> None:
            try:
                assert self._tokenize_pool is not None and self.loop is not None
                count = await self._tokenize_pool.token_count_async(
                    prompt_text, self.loop
                )
                self._emitter.emit(sample_uuid, "isl", count)
            except Exception:
                logger.exception("ISL tokenization failed for sample %s", sample_uuid)

        self.loop.create_task(_compute())

    async def _compute_token_metrics(self, record: EventRecord, row: SampleRow) -> None:
        output_text = row.output_text()
        if not output_text:
            output_text = self._extract_complete_output(record)

        if not output_text or self._tokenize_pool is None or self.loop is None:
            return

        try:
            osl = await self._tokenize_pool.token_count_async(output_text, self.loop)
            self._emitter.emit(row.sample_uuid, "osl", osl)

            # TPOT: time per output token after the first chunk.
            # The first chunk may contain multiple tokens, so we tokenize it
            # to get the correct denominator: (osl - first_chunk_tokens).
            if row.recv_first_ns is not None and row.first_chunk_text:
                first_chunk_tokens = await self._tokenize_pool.token_count_async(
                    row.first_chunk_text, self.loop
                )
                tokens_after_first = osl - first_chunk_tokens
                if tokens_after_first > 0:
                    generation_ns = record.timestamp_ns - row.recv_first_ns
                    tpot_ns = generation_ns / tokens_after_first
                    self._emitter.emit(row.sample_uuid, "tpot_ns", tpot_ns)
        except Exception:
            logger.exception(
                "Output tokenization failed for sample %s", row.sample_uuid
            )

    @staticmethod
    def _extract_complete_output(record: EventRecord) -> str:
        if isinstance(record.data, TextModelOutput):
            return str(record.data)
        return ""

    def _finalize(self) -> None:
        self._emitter.flush()
        self.close()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        elif self.loop is not None and self.loop.is_running():
            self.loop.stop()

    def close(self) -> None:
        self._emitter.close()
        super().close()
