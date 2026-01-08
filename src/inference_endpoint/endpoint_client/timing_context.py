# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Timing context and printers for request overhead measurement."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import orjson

from inference_endpoint.utils.logging import TRACE

logger = logging.getLogger(__name__)

# Unit conversion constant: nanoseconds to milliseconds
_NS_PER_MS: float = 1_000_000.0


# =============================================================================
# Request Timing Context
# =============================================================================


class RequestTimingContext(dict):
    """
    Dict subclass for request timing timestamps.
    Collects nanosecond timestamps and computes overhead metrics.

    Required timestamps for pre-overhead:
        t_recv, t_encode, t_prepare, t_conn_start, t_conn_end, t_http

    Required timestamps for post-overhead:
        t_recv, t_http, t_task_awake, t_headers, t_first_chunk, t_response, t_zmq_sent

    Example:
        >>> timing = RequestTimingContext(id="req-123")
        >>> timing["t_recv"] = time.monotonic_ns()
        >>> timing["t_http"] = time.monotonic_ns()
        >>> metrics = timing.compute_pre_overheads()
        >>> printer.write({"query_id": timing.id, "phase": "pre", "metrics": metrics})
    """

    __slots__ = ("id",)

    def __init__(self, id: str):  # noqa: A002 - shadowing builtin intentional
        super().__init__()
        self.id = id

    def compute_pre_overheads(self) -> dict[str, float]:
        """Compute pre-send overhead metrics.

        Returns:
            Dict of metric name to value (durations in milliseconds).
            Includes pool state (pool_idle, pool_acquired, pool_waiters) if captured.
        """
        t_recv = self["t_recv"]
        t_encode = self["t_encode"]
        t_prepare = self["t_prepare"]
        t_conn_start = self["t_conn_start"]
        t_conn_end = self["t_conn_end"]
        t_http = self["t_http"]

        return {
            "recv_to_bytes": (t_encode - t_recv) / _NS_PER_MS,
            "bytes_to_http_payload": (t_prepare - t_encode) / _NS_PER_MS,
            "tcp_conn_pool": (t_conn_end - t_conn_start) / _NS_PER_MS,
            "http_payload_send": (t_http - t_conn_end) / _NS_PER_MS,
            "pre_overhead": (t_http - t_recv) / _NS_PER_MS,
        }

    def compute_post_overheads(self) -> dict[str, float]:
        """Compute post-receive overhead metrics.

        Returns:
            Dict of metric name to value (durations in ms, raw timestamps in ns).
        """
        t_recv = self["t_recv"]
        t_http = self["t_http"]
        t_task_awake = self["t_task_awake"]
        t_headers = self["t_headers"]
        t_first_chunk = self["t_first_chunk"]
        t_response = self["t_response"]
        t_zmq_sent = self["t_zmq_sent"]

        headers_to_first = (t_first_chunk - t_headers) / _NS_PER_MS
        first_to_last = (t_response - t_first_chunk) / _NS_PER_MS

        return {
            "task_overhead": (t_task_awake - t_http) / _NS_PER_MS,
            "http_to_headers": (t_headers - t_http) / _NS_PER_MS,
            "headers_to_first_chunk": headers_to_first,
            "first_to_last_chunk": first_to_last,
            "headers_to_first": headers_to_first,
            "first_to_last": first_to_last,
            "in_flight_time": (t_response - t_http) / _NS_PER_MS,
            "query_result_sent": (t_zmq_sent - t_response) / _NS_PER_MS,
            "post_overhead": (t_zmq_sent - t_response) / _NS_PER_MS,
            "end_to_end": (t_zmq_sent - t_recv) / _NS_PER_MS,
            "t_recv": float(t_recv),
            "t_zmq_sent": float(t_zmq_sent),
        }


# =============================================================================
# Printers
# =============================================================================


class DisabledPrinter:
    """No-op printer that discards all timing data."""

    __slots__ = ()

    def write(self, data: dict) -> None:
        """No-op."""
        pass

    def flush(self) -> None:
        """No-op."""
        pass


class BufferPrinter:
    """
    Buffered JSONL file writer - stores dicts, serializes on flush.

    Accumulates data in memory during the run to minimize per-write overhead.
    Serialization happens once during flush for efficiency.

    Example:
        >>> printer = BufferPrinter(Path("/tmp/data.jsonl"))
        >>> printer.write({"query_id": "q1", "phase": "pre", "metrics": {...}})
        >>> printer.flush()  # Serializes and writes all entries
    """

    __slots__ = ("path", "_entries", "_flushed")

    def __init__(self, path: Path):
        self.path = path
        self._entries: list[dict] = []
        self._flushed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: dict) -> None:
        """Buffer a dict entry."""
        self._entries.append(data)

    def flush(self) -> None:
        """Serialize and write all buffered entries to file."""
        if self._flushed or not self._entries:
            return

        try:
            with open(self.path, "wb") as f:
                f.write(b"\n".join(orjson.dumps(e) for e in self._entries))
                f.write(b"\n")
            logger.debug(f"Wrote {len(self._entries)} entries to {self.path}")
        except Exception as e:
            logger.error(f"Failed to write to {self.path}: {e}")
        finally:
            self._entries.clear()
            self._flushed = True

    def __len__(self) -> int:
        """Return number of buffered entries."""
        return len(self._entries)


class LogPrinter:
    """
    Log-based printer with configurable formatter.

    Emits data via Python logging.

    Example:
        >>> def fmt_timing(d): return f"[{d['query_id']}] {d['phase']}: ..."
        >>> printer = LogPrinter(formatter=fmt_timing)
        >>> printer.write({"query_id": "q1", "phase": "pre", "metrics": {...}})
    """

    __slots__ = ("_level", "_formatter")

    def __init__(
        self, level: int = TRACE, formatter: Callable[[dict], str] | None = None
    ):
        self._level = level
        self._formatter = formatter

    def write(self, data: dict) -> None:
        """Log formatted data."""
        if not logger.isEnabledFor(self._level):
            return
        msg = self._formatter(data) if self._formatter else str(data)
        logger.log(self._level, msg)

    def flush(self) -> None:
        """No-op for log printer."""
        pass


# =============================================================================
# Log Formatters
# =============================================================================


def format_timing_log(data: dict) -> str:
    """Format timing entry for logging."""
    query_id = data["query_id"]
    phase = data["phase"]
    metrics = data["metrics"]

    parts = []
    for name, value in metrics.items():
        if name.startswith("t_"):
            parts.append(f"{name}={int(value)}")
        elif isinstance(value, float):
            parts.append(f"d_{name}={value:.4f}ms")
        else:
            parts.append(f"{name}={value}")

    return f"[{query_id}] timing_{phase}: {', '.join(parts)}"
