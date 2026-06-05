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

"""Metrics aggregator service: EventRecord subscriber for real-time metrics."""

import argparse
import asyncio
import logging
import signal
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.ready_check import send_ready_signal
from inference_endpoint.utils.logging import setup_logging

from .aggregator import MetricCounterKey, MetricsAggregatorService
from .metrics_table import MetricsTable
from .publisher import MetricsPublisher
from .registry import MetricsRegistry
from .snapshot import MetricsSnapshotCodec
from .token_metrics import TokenizePool

logger = logging.getLogger(__name__)


def _make_sigterm_handler(
    *,
    loop: asyncio.AbstractEventLoop,
    registry: MetricsRegistry,
    publisher: MetricsPublisher,
    table: MetricsTable,
    shutdown_event: asyncio.Event,
) -> tuple[Callable[[], None], set[asyncio.Task]]:
    """Build the SIGTERM handler that writes the INTERRUPTED final snapshot.

    Returns ``(handler, pending_tasks)``. ``pending_tasks`` is the
    strong-reference container that keeps spawned finalize tasks alive
    while they run: asyncio tracks tasks only by weakref, so a task
    whose only reference is the local variable inside the handler can
    be garbage-collected mid-execution (per Python's asyncio docs).
    Each spawned task self-removes from the set via
    ``add_done_callback`` once it completes.

    Exposed at module level (rather than nested in ``main()``) so the
    GC-safety contract is unit-testable without driving the whole
    subprocess lifecycle.
    """
    pending_tasks: set[asyncio.Task] = set()

    async def _signal_finalize() -> None:
        try:
            # Mirror the ENDED-driven path: refresh tracked_duration_ns
            # from the table BEFORE publish_final, otherwise an
            # interrupted run whose STOP_PERFORMANCE_TRACKING never
            # fired would report duration_ns=0 and QPS=N/A in the final
            # report even after processing many tracked samples.
            registry.set_counter(
                MetricCounterKey.TRACKED_DURATION_NS.value,
                table.total_tracked_duration_ns,
            )
            await publisher.publish_final(
                registry,
                n_pending_tasks=table.in_flight_tasks_count,
                interrupted=True,
            )
        except Exception:  # noqa: BLE001 — best-effort.
            logger.exception(
                "metrics aggregator: SIGTERM-triggered publish_final failed"
            )
        shutdown_event.set()

    def _on_sigterm() -> None:
        logger.warning(
            "metrics aggregator received SIGTERM; " "writing INTERRUPTED final snapshot"
        )
        task = loop.create_task(_signal_finalize())
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    return _on_sigterm, pending_tasks


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metrics aggregator service - subscribes to EventRecords and computes real-time metrics"
    )
    parser.add_argument(
        "--socket-dir",
        type=str,
        required=True,
        help="Directory containing ZMQ IPC sockets (must already exist)",
    )
    parser.add_argument(
        "--socket-name",
        type=str,
        required=True,
        help="EventRecord PUB socket name within socket-dir to subscribe to",
    )
    parser.add_argument(
        "--metrics-socket",
        type=str,
        required=True,
        help="IPC socket name (within socket-dir) for the metrics PUB output",
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        required=True,
        help="Directory for the final-snapshot disk fallback (created if missing)",
    )
    parser.add_argument(
        "--publish-interval",
        type=float,
        default=0.25,
        help="Live snapshot publish interval in seconds (default: 0.25, i.e. 4 Hz)",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=60.0,
        help=(
            "Wall-clock budget (seconds) to wait for in-flight async tokenize "
            "tasks to finish after ENDED before the aggregator cancels them "
            "and emits the final snapshot with n_pending_tasks > 0 "
            "(default: 60.0; 0 = wait indefinitely). Increase for long-context "
            "/ low-worker-count tokenize workloads."
        ),
    )
    parser.add_argument(
        "--hdr-sig-figs",
        type=int,
        default=3,
        help="HDR Histogram significant figures (default: 3)",
    )
    parser.add_argument(
        "--n-histogram-buckets",
        type=int,
        default=30,
        help="Number of dense histogram buckets per series (default: 30)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name for ISL/OSL/TPOT (e.g. 'gpt2'). If not set, token metrics are disabled.",
    )
    parser.add_argument(
        "--tokenizer-workers",
        type=int,
        default=2,
        help="Number of tokenizer worker threads (default: 2)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Enable streaming metrics (TTFT, chunk_delta, TPOT). Off by default.",
    )
    parser.add_argument(
        "--readiness-path",
        type=str,
        default=None,
        help="ZMQ socket path to signal readiness (optional)",
    )
    parser.add_argument(
        "--readiness-id",
        type=int,
        default=0,
        help="Identity to send in the readiness signal",
    )
    args = parser.parse_args()
    setup_logging(level="INFO")

    # The parent owns directory setup — `commands/benchmark/execute.py`
    # creates `<report_dir>/metrics/` and validates it before launching
    # this subprocess. Validate here as a fail-fast contract check so a
    # misbehaving launcher (or a manual invocation) surfaces a clear
    # error in this subprocess's stderr instead of crashing later on
    # the atomic-write path.
    metrics_output_dir: Path = args.metrics_output_dir
    if not metrics_output_dir.is_dir():
        raise SystemExit(
            f"FATAL: --metrics-output-dir {metrics_output_dir!s} does not "
            "exist or is not a directory. The parent process is responsible "
            "for creating it before launching the aggregator subprocess."
        )

    shutdown_event = asyncio.Event()
    loop = LoopManager().default_loop

    # Using ternary operator causes errors in MyPy object type coalescing
    # (coalesces to 'object' not 'AbstractContextManager[TokenizePool | None]')
    pool_cm: AbstractContextManager[TokenizePool | None]
    if args.tokenizer:
        pool_cm = TokenizePool(args.tokenizer, n_workers=args.tokenizer_workers)
    else:
        pool_cm = nullcontext()

    with (
        pool_cm as pool,
        ManagedZMQContext.scoped(socket_dir=args.socket_dir) as zmq_ctx,
    ):
        registry = MetricsRegistry()
        publisher = MetricsPublisher(
            MetricsSnapshotCodec(),
            zmq_ctx,
            args.metrics_socket,
            loop,
            final_snapshot_path=metrics_output_dir / "final_snapshot.json",
        )
        try:
            aggregator = MetricsAggregatorService(
                args.socket_name,
                zmq_ctx,
                loop,
                topics=None,
                registry=registry,
                publisher=publisher,
                publish_interval_s=args.publish_interval,
                sig_figs=args.hdr_sig_figs,
                n_histogram_buckets=args.n_histogram_buckets,
                tokenize_pool=pool,
                streaming=args.streaming,
                shutdown_event=shutdown_event,
                drain_timeout_s=None if args.drain_timeout == 0 else args.drain_timeout,
            )
            aggregator.start()

            # SIGTERM only — the parent's ServiceLauncher.kill_all uses
            # SIGTERM to kill the aggregator child before an ENDED event
            # arrives; without this handler that path leaves the Report
            # consumer with no final_snapshot file. The signal-triggered
            # snapshot is tagged INTERRUPTED so Report can distinguish
            # "parent killed the run" from a clean shutdown.
            # publish_final is idempotent (see
            # MetricsPublisher._finalized), so racing with the
            # ENDED-driven call is safe.
            #
            # SIGINT is deliberately NOT handled in the same way. On an
            # interactive ^C, the OS sends SIGINT to the whole
            # foreground process group — parent + child both receive
            # it. If we finalized eagerly here, the aggregator would
            # write final_snapshot.json from whatever state it had at
            # signal time, then exit; samples that completed during the
            # parent's own graceful shutdown window would never reach
            # the file (the parent eventually emits ENDED on its events
            # channel, but `_finalized=True` makes that a no-op). The
            # parent's clean-shutdown path is what we want to drive the
            # aggregator's finalize — so we install a no-op handler for
            # SIGINT here, which prevents Python's default
            # KeyboardInterrupt and lets the parent control the lifecycle.
            on_sigterm, _sigterm_tasks = _make_sigterm_handler(
                loop=loop,
                registry=registry,
                publisher=publisher,
                table=aggregator._table,
                shutdown_event=shutdown_event,
            )
            loop.add_signal_handler(signal.SIGTERM, on_sigterm)
            # No-op SIGINT handler: silence the default KeyboardInterrupt
            # and let the parent's ENDED-driven path drive shutdown.
            loop.add_signal_handler(
                signal.SIGINT,
                lambda: logger.info(
                    "metrics aggregator received SIGINT — ignoring "
                    "(parent's ENDED path is authoritative)"
                ),
            )

            if args.readiness_path:
                await send_ready_signal(zmq_ctx, args.readiness_path, args.readiness_id)

            await shutdown_event.wait()
        finally:
            # aclose() awaits the tick task before closing the underlying
            # transport, avoiding cancelled-tick-vs-socket-close races.
            await publisher.aclose()


if __name__ == "__main__":
    # Surface startup / bind / tokenizer-load failures with structured
    # context. Without this wrap, the parent's ServiceLauncher only sees
    # the non-zero exit code and a raw traceback — no diagnostic context
    # to correlate against the parent's logs. The except/raise pattern
    # preserves the original exit code (1) and traceback while emitting
    # the structured logger.exception line before the interpreter prints
    # the trace.
    try:
        LoopManager().default_loop.run_until_complete(main())
    except SystemExit:
        # argparse / explicit sys.exit — already user-facing, don't dress up.
        raise
    except Exception as e:
        # Catch Exception (not BaseException) so KeyboardInterrupt /
        # SystemExit propagate untouched — those are control-flow
        # signals, not crashes, and labeling them as "crashed" would
        # mislead operators. The exception type goes first in the log
        # message so it's grep-able without scrolling through the
        # traceback.
        logger.exception("metrics aggregator subprocess crashed (%s)", type(e).__name__)
        raise
