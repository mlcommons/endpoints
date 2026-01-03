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
from typing import Any

from aiohttp.client_reqrep import ClientRequest, ClientResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Timing Printer
# =============================================================================

TimingPrinter = Callable[[str, str, dict[str, float]], None]


# TODO(vir): replace / add prometheus printer
def logging_timing_printer(
    query_id: str, phase: str, metrics: dict[str, float]
) -> None:
    """Emit timing metrics via logging.debug."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    parts = [f"d_{name}={value:.1f}us" for name, value in metrics.items()]
    logger.debug(f"[{query_id}] timing_{phase}: {', '.join(parts)}")


# Active printer - swap to change output destination
timing_printer: TimingPrinter = logging_timing_printer


# =============================================================================
# PreparedRequest
# =============================================================================


class PreparedRequest:
    """
    Encapsulates a pre-built HTTP request with timing instrumentation.

    This class tracks the lifecycle of an HTTP request through the worker,
    capturing timestamps at key stages for performance analysis.

    Attributes:
        query_id: Unique identifier for this request.
        client_request: Pre-built aiohttp ClientRequest ready to send.
        timing_ctx: Dictionary of nanosecond timestamps (see below).
        process: Async callable to process the response (bound method).
        response: The HTTP response once received (initially None).

    Timing Context (timing_ctx):
        The timing_ctx dict accumulates timestamps through the request lifecycle.
        All timestamps are in nanoseconds from time.perf_counter_ns().

        Required keys for log_timing_pre():
            t_recv      - Query received from ZMQ
            t_prepare   - Query encoded, ClientRequest built
            t_http      - Just before request.send()

        Required keys for log_timing_post():
            t_recv      - (same as above)
            t_http      - (same as above)
            t_headers   - Response headers received
            t_response  - Full response body read
            t_zmq_sent  - Result sent via ZMQ

        Optional keys:
            t_first_chunk - First SSE chunk received (streaming only)

        Any additional keys in timing_ctx are included in the output.

    Computed Metrics (in microseconds):
        PRE-SEND (log_timing_pre):
            recv_to_prepare  = t_prepare - t_recv
            prepare_to_http  = t_http - t_prepare
            pre_overhead     = t_http - t_recv  (total client overhead before send)

        POST-RECV (log_timing_post):
            http_to_headers  = t_headers - t_http
            headers_to_first = t_first_chunk - t_headers  (if streaming)
            first_to_last    = t_response - t_first_chunk (if streaming)
            response_to_zmq  = t_zmq_sent - t_response
            post_overhead    = t_zmq_sent - t_response  (total client overhead after recv)
            end_to_end       = t_zmq_sent - t_recv

    Example:
        >>> prepared = PreparedRequest(
        ...     query_id="abc-123",
        ...     client_request=client_req,
        ...     timing_ctx={"t_recv": time.perf_counter_ns()},
        ...     process=None,
        ... )
        >>> prepared.timing_ctx["t_prepare"] = time.perf_counter_ns()
        >>> # ... build request ...
        >>> prepared.timing_ctx["t_http"] = time.perf_counter_ns()
        >>> prepared.log_timing_pre()  # Emits pre-send metrics
    """

    __slots__ = ("query_id", "client_request", "timing_ctx", "process", "response")

    def __init__(
        self,
        query_id: str,
        client_request: ClientRequest,
        timing_ctx: dict[str, int],
        process: Callable[[], Any] | None,
    ) -> None:
        """
        Initialize a PreparedRequest.

        Args:
            query_id: Unique identifier for tracking this request.
            client_request: Pre-built aiohttp ClientRequest.
            timing_ctx: Dict to accumulate timestamps. Caller should set t_recv.
            process: Async callable to handle response (set after init).
        """
        self.query_id = query_id
        self.client_request = client_request
        self.timing_ctx = timing_ctx
        self.process = process
        self.response: ClientResponse | None = None

    def __repr__(self) -> str:
        return (
            f"PreparedRequest(query_id={self.query_id!r}, "
            f"timing_keys={list(self.timing_ctx.keys())}, "
            f"has_response={self.response is not None})"
        )

    def log_timing_pre(self) -> None:
        """
        Emit pre-send timing metrics.

        Call this after t_http is set, just before the HTTP request is sent.

        Required timing_ctx keys: t_recv, t_prepare, t_http
        """
        ctx = self.timing_ctx
        t_recv = ctx["t_recv"]
        t_prepare = ctx["t_prepare"]
        t_http = ctx["t_http"]

        metrics = {
            "recv_to_prepare": (t_prepare - t_recv) / 1000.0,
            "prepare_to_http": (t_http - t_prepare) / 1000.0,
            "pre_overhead": (t_http - t_recv) / 1000.0,
        }

        timing_printer(self.query_id, "pre", metrics)

    def log_timing_post(self) -> None:
        """
        Emit post-receive timing metrics.

        Call this after t_zmq_sent is set, once the response has been
        fully processed and sent via ZMQ.

        Required timing_ctx keys: t_recv, t_http, t_headers, t_response, t_zmq_sent
        Optional: t_first_chunk (for streaming responses)
        """
        ctx = self.timing_ctx
        t_recv = ctx["t_recv"]
        t_http = ctx["t_http"]
        t_headers = ctx["t_headers"]
        t_first_chunk = ctx.get("t_first_chunk")
        t_response = ctx["t_response"]
        t_zmq_sent = ctx["t_zmq_sent"]

        metrics: dict[str, float] = {
            "http_to_headers": (t_headers - t_http) / 1000.0,
        }

        if t_first_chunk is not None:
            metrics["headers_to_first"] = (t_first_chunk - t_headers) / 1000.0
            metrics["first_to_last"] = (t_response - t_first_chunk) / 1000.0

        metrics["response_to_zmq"] = (t_zmq_sent - t_response) / 1000.0
        metrics["post_overhead"] = (t_zmq_sent - t_response) / 1000.0
        metrics["end_to_end"] = (t_zmq_sent - t_recv) / 1000.0

        timing_printer(self.query_id, "post", metrics)
