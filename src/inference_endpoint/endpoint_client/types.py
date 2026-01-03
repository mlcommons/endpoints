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

"""Type definitions for the endpoint client module."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from aiohttp.client_reqrep import ClientRequest, ClientResponse

from inference_endpoint.utils.logging import TRACE

logger = logging.getLogger(__name__)


# =============================================================================
# Timing Printer
# =============================================================================

TimingPrinter = Callable[[str, str, dict[str, float]], None]


# TODO(vir): replace / add prometheus printer
def logging_timing_printer(
    query_id: str, phase: str, metrics: dict[str, float]
) -> None:
    """Emit timing metrics via TRACE level logging (requires -vvv)."""
    if not logger.isEnabledFor(TRACE):
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
    logger.log(TRACE, f"[{query_id}] timing_{phase}: {', '.join(log_parts)}")


# Active printer - swap to change output destination
timing_printer: TimingPrinter = logging_timing_printer


# =============================================================================
# PreparedRequest
# =============================================================================


@dataclass(slots=True)
class PreparedRequest:
    """
    Encapsulates a pre-built HTTP request.

    This class tracks the lifecycle of an HTTP request through the worker,
    capturing timestamps at key stages for performance analysis.

    Attributes:
        query_id: Unique identifier for this request.
        client_request: Pre-built aiohttp ClientRequest ready to send.
        timing_ctx: Dictionary of nanosecond timestamps for performance tracking.
        process: Async callable to process the response (bound method).
        response: The HTTP response once received (initially None).
        connection: The aiohttp TCP connection, set after connect (initially None).

    Timing Context Keys (all in nanoseconds from time.monotonic_ns()):
        PRE-SEND (logged in timing_pre):
            t_recv       - Query received from ZMQ
            t_prepare    - Query encoded, ClientRequest built
            t_conn_start - connector.connect() called
            t_conn_end   - Connection acquired from pool
            t_http       - POST request.send() completed

        POST-SEND (logged in timing_post):
            t_task_created - Asyncio task created
            t_task_awake   - Asyncio task started executing
            t_headers      - Response headers received
            t_first_chunk  - First SSE chunk (streaming only)
            t_response     - Response completed
            t_zmq_sent     - Result sent via ZMQ

    Computed Metrics:
        PRE-SEND:
            recv_to_prepare = t_prepare - t_recv
            pool_acquire    = t_conn_end - t_conn_start (waiting for connection)
            http_send       = t_http - t_conn_end (sending POST request)
            pre_overhead    = t_http - t_recv (total)

        POST-SEND:
            task_overhead   = t_task_awake - t_task_created
            http_to_headers = t_headers - t_http
            ... (streaming metrics)
            post_overhead   = t_zmq_sent - t_response
            end_to_end      = t_zmq_sent - t_recv
    """

    query_id: str
    client_request: ClientRequest
    timing_ctx: dict[str, int]
    process: Callable[[], Any] | None = None
    response: ClientResponse | None = field(default=None, repr=False)
    connection: Any | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        query_id: str,
        client_request: ClientRequest,
        timing_ctx: dict[str, int],
        handler: Callable[[PreparedRequest], Any],
    ) -> PreparedRequest:
        """
        Factory method to create a fully-initialized PreparedRequest.

        Args:
            query_id: Unique identifier for tracking this request.
            client_request: Pre-built aiohttp ClientRequest.
            timing_ctx: Dict for tracking timestamps (caller should set t_recv).
            handler: Async callable to process the response (will be bound to this instance).
        """
        prepared = cls(
            query_id=query_id,
            client_request=client_request,
            timing_ctx=timing_ctx,
        )
        prepared.process = partial(handler, prepared)
        return prepared

    def log_timing_pre(self) -> None:
        """
        Emit pre-send timing metrics.

        Call this after timing_ctx["t_http"] is set, just after the HTTP request is sent.
        """
        ctx = self.timing_ctx
        t_recv = ctx["t_recv"]
        t_prepare = ctx["t_prepare"]
        t_conn_start = ctx.get("t_conn_start")
        t_conn_end = ctx.get("t_conn_end")
        t_http = ctx["t_http"]

        metrics = {
            "recv_to_prepare": (t_prepare - t_recv) / 1_000_000.0,
        }

        # Time waiting for connection from pool
        if t_conn_start is not None and t_conn_end is not None:
            metrics["pool_acquire"] = (t_conn_end - t_conn_start) / 1_000_000.0

        # Time to send HTTP POST (after connection acquired)
        if t_conn_end is not None:
            metrics["http_send"] = (t_http - t_conn_end) / 1_000_000.0

        metrics["pre_overhead"] = (t_http - t_recv) / 1_000_000.0

        timing_printer(self.query_id, "pre", metrics)

    def log_timing_post(self) -> None:
        """
        Emit post-receive timing metrics.

        Call this after timing_ctx["t_zmq_sent"] is set, once the response has been
        fully processed and sent via ZMQ.
        """
        ctx = self.timing_ctx
        t_recv = ctx["t_recv"]
        t_http = ctx["t_http"]
        t_headers = ctx["t_headers"]
        t_first_chunk = ctx.get("t_first_chunk")
        t_response = ctx["t_response"]
        t_zmq_sent = ctx["t_zmq_sent"]

        # Asyncio task scheduling overhead
        t_task_created = ctx.get("t_task_created")
        t_task_awake = ctx.get("t_task_awake")

        metrics: dict[str, float] = {}

        if t_task_created is not None and t_task_awake is not None:
            metrics["task_overhead"] = (t_task_awake - t_task_created) / 1_000_000.0

        metrics["http_to_headers"] = (t_headers - t_http) / 1_000_000.0

        if t_first_chunk is not None:
            metrics["headers_to_first"] = (t_first_chunk - t_headers) / 1_000_000.0
            metrics["first_to_last"] = (t_response - t_first_chunk) / 1_000_000.0

        metrics["response_to_zmq"] = (t_zmq_sent - t_response) / 1_000_000.0
        metrics["post_overhead"] = (t_zmq_sent - t_response) / 1_000_000.0
        metrics["end_to_end"] = (t_zmq_sent - t_recv) / 1_000_000.0

        # Raw timestamps for IPC delay calculation
        metrics["t_recv"] = float(t_recv)
        metrics["t_zmq_sent"] = float(t_zmq_sent)

        timing_printer(self.query_id, "post", metrics)
