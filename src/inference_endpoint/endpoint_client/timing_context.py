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

"""Timing context for request overhead measurement."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import orjson

from inference_endpoint.utils.logging import TRACE

logger = logging.getLogger(__name__)

# Unit conversion constant: nanoseconds to milliseconds
_NS_PER_MS: float = 1_000_000.0

# Type alias for timing printer callable
TimingPrinter = Callable[[str, str, dict[str, float]], None]


class LoggingTimingPrinter:
    """
    Logging-based timing printer.

    Emits timing metrics via Python logging. Checks if the log level is enabled
    at construction time to short-circuit __call__ when logging is disabled.

    Args:
        level: Log level to use. Default is TRACE (requires -vvv).

    Example:
        >>> printer = LoggingTimingPrinter()  # Uses TRACE level
        >>> printer("query-1", "pre", {"overhead": 1.5})
    """

    __slots__ = ("_level", "_enabled")

    def __init__(self, level: int = TRACE):
        self._level = level
        self._enabled = logger.isEnabledFor(level)

    def __call__(self, query_id: str, phase: str, metrics: dict[str, float]) -> None:
        """Emit timing metrics via logging."""
        if not self._enabled:
            return

        duration_parts = []
        timestamp_parts = []
        for name, value in metrics.items():
            if name.startswith("t_"):
                # Raw timestamps in nanoseconds (for IPC delay calculation)
                timestamp_parts.append(f"{name}={int(value)}")
            else:
                # Duration metrics in milliseconds
                duration_parts.append(f"d_{name}={value:.4f}ms")

        log_parts = duration_parts + timestamp_parts
        logger.log(self._level, f"[{query_id}] timing_{phase}: {', '.join(log_parts)}")


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
        >>> printer(timing.id, "pre", metrics)
    """

    __slots__ = ("id",)

    def __init__(self, id: str):  # noqa: A002 - shadowing builtin intentional
        super().__init__()
        self.id = id

    def compute_pre_overheads(self) -> dict[str, float]:
        """Compute pre-send overhead metrics.

        Returns:
            Dict of metric name to value (durations in milliseconds).
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


class MemoryBufferPrinter:
    """
    In-memory timing printer that buffers entries and flushes to file.

    Accumulates timing data in memory during the run to avoid per-request I/O.
    Automatically flushes on exit, interrupt (SIGINT), or termination (SIGTERM).

    Directories are created during initialization to fail fast on permission errors.

    Each line is a JSON object with:
        - query_id: The request ID
        - worker_id: The worker process ID
        - phase: "pre" or "post"
        - metrics: Dict of timing metrics (durations in ms, timestamps in ns)

    Example:
        >>> printer = MemoryBufferPrinter(Path("/tmp/timing.jsonl"), worker_id=0)
        >>> printer("query-1", "pre", {"overhead": 1.5})
        >>> printer("query-1", "post", {"end_to_end": 10.0})
        >>> # Flushes automatically on exit, or call printer.flush() manually
    """

    __slots__ = ("path", "worker_id", "_entries", "_flushed")

    def __init__(self, path: Path, worker_id: int):
        self.path = path
        self.worker_id = worker_id
        self._entries: list[bytes] = []
        self._flushed = False

        # Create directories during init to fail fast on permission errors
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, query_id: str, phase: str, metrics: dict[str, float]) -> None:
        """Buffer a timing entry."""
        entry = {
            "query_id": query_id,
            "worker_id": self.worker_id,
            "phase": phase,
            "metrics": metrics,
        }
        self._entries.append(orjson.dumps(entry))

    def flush(self) -> None:
        """Write all buffered entries to file and clear buffer."""
        if not self._entries or self._flushed:
            return

        try:
            with open(self.path, "wb") as f:
                f.write(b"\n".join(self._entries))
                f.write(b"\n")
            logger.debug(f"Wrote {len(self._entries)} timing entries to {self.path}")
        except Exception as e:
            # Log error but don't raise - we're likely in cleanup
            logger.error(f"Failed to write timing entries: {e}")
        finally:
            self._entries.clear()
            self._flushed = True

    def __len__(self) -> int:
        """Return number of buffered entries."""
        return len(self._entries)

    @property
    def buffer_size_bytes(self) -> int:
        """Return total size of buffered entries in bytes."""
        return sum(len(entry) for entry in self._entries)
