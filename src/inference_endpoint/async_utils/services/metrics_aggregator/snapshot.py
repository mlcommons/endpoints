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

import math
from enum import Enum
from typing import ClassVar, Final

import msgspec
import msgspec.msgpack
from inference_endpoint.core.record import TOPIC_FRAME_SIZE


class SessionState(str, Enum):
    """The aggregator's session state at the time a snapshot was emitted.

    INITIALIZE  → aggregator has been constructed but no ``STARTED`` event
                  has arrived yet. The tick task is not running, so consumers
                  should not see a snapshot in this state on the wire today;
                  it exists so the in-process state machine has a well-defined
                  starting point (and so future setup-phase ticks have a
                  state to carry).
    LIVE        → run in progress; tick task publishing live HDR-derived stats.
    DRAINING    → ``SessionEventType.ENDED`` has been received; the aggregator
                  is tokenizing the buffered samples (bounded by the
                  ``--drain-timeout`` budget, default 60 s). Tick task
                  continues at this stage, still HDR-derived; no new events
                  will arrive.
    COMPLETE    → terminal clean state. The ``publish_final()`` snapshot
                  written from the ``ENDED`` path. Percentiles and histograms
                  are exact (computed from raw values).
    INTERRUPTED → terminal interrupted state. The ``publish_final()`` snapshot
                  written from a signal handler (SIGTERM / SIGINT) before
                  ``ENDED`` arrived. Stats are best-effort partial captures of
                  whatever the aggregator had at signal time — drain didn't
                  complete and raw values may be missing late samples.
                  Distinguishes "user killed the run" from "clean shutdown";
                  Report renders this with a clear interrupted indicator.

    Transitions are forward-only:
        INITIALIZE → LIVE → DRAINING → {COMPLETE | INTERRUPTED}
    No state ever moves backward, and the terminal states (COMPLETE,
    INTERRUPTED) are not re-entrant (``MetricsPublisher._finalized``
    enforces a single publish_final call).

    Drain timeout is detected as ``state == COMPLETE and n_pending_tasks > 0``.
    Interrupted-run is detected as ``state == INTERRUPTED`` directly.
    """

    INITIALIZE = "initialize"
    LIVE = "live"
    DRAINING = "draining"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"


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
        state:            ``SessionState`` enum — ``INITIALIZE``, ``LIVE``,
                          ``DRAINING``, ``COMPLETE``, or ``INTERRUPTED``. See
                          the enum docstring. Terminal states (``COMPLETE``,
                          ``INTERRUPTED``) mark the last snapshot of the run;
                          for ``COMPLETE`` snapshots percentiles and
                          histograms are exact, otherwise HDR-derived.
        n_pending_tasks:  Count of buffered tokenizations not yet recorded at
                          snapshot composition time. ``> 0`` during normal
                          load (ISL/OSL/TPOT buffered between publish-tick
                          flushes) and during the drain phase. **An
                          incomplete drain is detected as** ``state ==
                          COMPLETE and n_pending_tasks > 0``: the end-of-run
                          flush timed out or failed; the token-derived series
                          are missing those samples.
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


# ---------------------------------------------------------------------------
# Dict form of a snapshot.
#
# This is the shape used by:
# - the persisted ``final_snapshot.json`` file (writer in ``publisher.py``)
# - ``Report.from_snapshot`` as its canonical input
#
# The wire ``MetricsSnapshot`` Struct uses ``array_like=True`` for compact
# msgpack on the pub/sub hot path — that encoding is positional, which is
# wrong for both file storage (unreadable JSON arrays) and for consumer
# code that wants to read fields by name. ``snapshot_to_dict`` is the
# one-way bridge from the wire form to the consumer form.
#
# There is intentionally no inverse: consumers operate on the dict
# directly with ``dict.get(key, default)``. Decoding a dict back into an
# ``array_like=True`` Struct is ergonomically painful (msgspec's decoders
# follow the Struct's array_like flag), and the consumer doesn't need it.
# ---------------------------------------------------------------------------


def _scrub_nonfinite(v):
    """Map non-finite floats (``NaN`` / ``±Inf``) to ``None``.

    The dict form is consumed by ``json.dumps(..., allow_nan=False)``,
    which rejects non-finite floats so the producer-side bug surfaces
    loudly rather than silently writing ``NaN`` / ``Infinity`` literals
    that ``jq``, Go's ``encoding/json``, and any strict-JSON consumer
    refuse to parse. Mapping non-finite to ``None`` keeps the JSON
    strict and self-describes the gap to the consumer.
    """
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def snapshot_to_dict(snap: MetricsSnapshot) -> dict:
    """Convert a wire ``MetricsSnapshot`` to its dict form.

    Manual mapping is the source of truth for the dict schema. When
    adding a field to ``MetricsSnapshot`` (or ``CounterStat`` /
    ``SeriesStat``), update this function so the field appears in both
    the persisted JSON file and the input to ``Report.from_snapshot``.
    """
    return {
        "counter": snap.counter,
        "timestamp_ns": snap.timestamp_ns,
        "state": snap.state.value,
        "n_pending_tasks": snap.n_pending_tasks,
        "metrics": [_metric_to_dict(m) for m in snap.metrics],
    }


def _metric_to_dict(m: MetricStat) -> dict:
    if isinstance(m, CounterStat):
        return {"type": "counter", "name": m.name, "value": _scrub_nonfinite(m.value)}
    return {
        "type": "series",
        "name": m.name,
        "count": m.count,
        "total": _scrub_nonfinite(m.total),
        "min": _scrub_nonfinite(m.min),
        "max": _scrub_nonfinite(m.max),
        "sum_sq": _scrub_nonfinite(m.sum_sq),
        "percentiles": {k: _scrub_nonfinite(v) for k, v in m.percentiles.items()},
        # Histogram tuples → JSON arrays. Consumers reading the dict can
        # iterate the two-element ranges directly without coercion.
        # Bucket edges are floats from log-spacing — scrub for safety.
        "histogram": [
            [[_scrub_nonfinite(rng[0]), _scrub_nonfinite(rng[1])], c]
            for rng, c in m.histogram
        ],
    }


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
