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

"""Wire schema and codec for metrics snapshots published over pub/sub.

The aggregator subprocess publishes ``MetricsSnapshot`` messages at a fixed
cadence. Each snapshot carries a ``SessionState`` (``LIVE`` during the run,
``DRAINING`` between ``ENDED`` and the final publish, ``COMPLETE`` for the
last snapshot). The snapshot is the only public wire format between the
aggregator and any consumer (main process, future TUI).
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Final

import msgspec
import msgspec.msgpack
from inference_endpoint.core.record import TOPIC_FRAME_SIZE


class SessionState(str, Enum):
    """The aggregator's session state at the time a snapshot was emitted.

    LIVE      → run in progress; tick task publishing live HDR-derived stats.
    DRAINING  → ``SessionEventType.ENDED`` has been received; the aggregator
                is awaiting the in-flight async tokenize tasks (bounded by
                the 30 s drain timeout). Tick task continues at this stage,
                still HDR-derived; no new events will arrive.
    COMPLETE  → the ``MetricsPublisher.publish_final()`` snapshot. Percentiles
                and histograms are exact (computed from raw values). This
                is always the last snapshot of the run.

    Drain timeout is detected as ``state == COMPLETE and n_pending_tasks > 0``.
    """

    LIVE = "live"
    DRAINING = "draining"
    COMPLETE = "complete"


class CounterStat(
    msgspec.Struct,
    tag="counter",
    frozen=True,
    array_like=True,
):  # type: ignore[call-arg]
    """A single counter value (e.g. ``total_samples_issued``)."""

    name: str
    value: int | float


class SeriesStat(
    msgspec.Struct,
    tag="series",
    frozen=True,
    array_like=True,
):  # type: ignore[call-arg]
    """Aggregated statistics for a single series (e.g. ``ttft_ns``).

    For LIVE/DRAINING snapshots, ``percentiles`` and ``histogram`` come from
    a live HDR Histogram. For COMPLETE snapshots they are computed exactly
    from the full in-memory raw values.

    Histogram bucket edges are **dynamic per snapshot**: log-spaced over the
    observed ``[min, max]`` of the data so far. The bucket count is fixed
    at construction (default 30); the edges auto-zoom each frame so no
    buckets are wasted on empty range. Empty series (no recordings) emit
    ``histogram=[]``.

    Consumers MUST re-render from ``(lo, hi, count)`` triples each frame
    and MUST NOT track bucket-by-index across snapshots — bucket ``i`` is
    not guaranteed to span the same range in consecutive snapshots.
    """

    name: str
    count: int
    total: int | float
    min: int | float  # noqa: A003 — wire field name; collides with builtin only here.
    max: int | float  # noqa: A003 — wire field name; collides with builtin only here.
    sum_sq: int | float
    percentiles: dict[str, float]
    histogram: list[tuple[tuple[float, float], int]]


# Tagged union: msgspec dispatches on the ``tag`` literal at decode time.
MetricStat = CounterStat | SeriesStat


class MetricsSnapshot(
    msgspec.Struct,
    frozen=True,
    array_like=True,
):  # type: ignore[call-arg]
    """A single point-in-time view of all aggregator metrics.

    Fields:
        counter:          Monotonic emit count, incremented by the producing
                          ``MetricsRegistry`` on every ``build_snapshot()``
                          call. Resets only on aggregator restart. Consumers
                          can use it to detect dropped/out-of-order delivery
                          or producer restarts. Diagnostic only — not used
                          for ordering on the wire. Unrelated to the
                          ``CounterStat`` metric kind in ``metrics``.
        timestamp_ns:     ``time.monotonic_ns()`` from the aggregator process
                          at snapshot composition time. Producer-local; not
                          comparable across processes.
        state:            ``SessionState`` enum — ``LIVE``, ``DRAINING``, or
                          ``COMPLETE``. See the enum docstring. ``COMPLETE``
                          marks the last snapshot of the run; for
                          ``COMPLETE`` snapshots, percentiles and histograms
                          are exact, otherwise HDR-derived.
        n_pending_tasks:  Count of in-flight async tokenize tasks at snapshot
                          composition time. ``> 0`` during normal load (ISL/
                          OSL/TPOT post-processing in flight) and during the
                          drain phase. **Drain timeout is detected as**
                          ``state == COMPLETE and n_pending_tasks > 0``: the
                          aggregator gave up draining; some async-only series
                          are missing samples that were still being tokenized.
        metrics:          Tagged union of ``CounterStat`` and ``SeriesStat``,
                          ordered counters-first then series, registration
                          order within each.
    """

    counter: int
    timestamp_ns: int
    state: SessionState
    n_pending_tasks: int
    metrics: list[MetricStat]


# 4-byte topic to match TOPIC_FRAME_SIZE-prefix protocol used by the
# pub/sub layer. The topic is null-padded to TOPIC_FRAME_SIZE on the wire.
METRICS_SNAPSHOT_TOPIC: Final[bytes] = b"MET\x00".ljust(TOPIC_FRAME_SIZE, b"\x00")


class MetricsSnapshotCodec:
    """``MessageCodec[MetricsSnapshot]`` — binds pub/sub layer to msgpack.

    Implements the structural ``MessageCodec`` Protocol from
    ``inference_endpoint.async_utils.transport.protocol`` without importing
    it (avoids a transport→service back-import). Mirrors the pattern in
    ``EventRecordCodec``.
    """

    __slots__ = ()

    _ENCODER: ClassVar = msgspec.msgpack.Encoder()
    _DECODER: ClassVar = msgspec.msgpack.Decoder(type=MetricsSnapshot)

    def encode(self, item: MetricsSnapshot) -> tuple[bytes, bytes]:
        return METRICS_SNAPSHOT_TOPIC, self._ENCODER.encode(item)

    def decode(self, payload: bytes) -> MetricsSnapshot:
        return self._DECODER.decode(payload)

    def on_decode_error(self, payload: bytes, exc: Exception) -> MetricsSnapshot | None:
        # Only swallow genuine wire-format failures. Anything else is a bug
        # in the decode path and should propagate.
        if not isinstance(exc, msgspec.DecodeError):
            raise exc
        # A malformed metrics frame is always safe to drop: snapshots are
        # idempotent and the next live tick or final replaces it.
        return None
