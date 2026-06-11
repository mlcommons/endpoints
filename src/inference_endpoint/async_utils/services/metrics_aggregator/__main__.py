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
from .token_metrics import BatchTokenizer, TokenBatchQueue

logger = logging.getLogger(__name__)


def _make_sigterm_handler(
    *,
    loop: asyncio.AbstractEventLoop,
    registry: MetricsRegistry,
    publisher: MetricsPublisher,
    table: MetricsTable,
    token_queue: TokenBatchQueue | None,
    shutdown_event: asyncio.Event,
) -> tuple[Callable[[], None], set[asyncio.Task]]:
    """Build the SIGTERM handler that writes the INTERRUPTED final snapshot.

    Returns ``(handler, pending_tasks)``: asyncio holds tasks only by
    weakref, so the handler's finalize task must live in this
    strong-reference set until done. Module-level so the GC-safety
    contract is unit-testable.
    """
    pending_tasks: set[asyncio.Task] = set()

    async def _signal_finalize() -> None:
        try:
            # Refresh tracked_duration_ns before publish_final (mirrors the
            # ENDED path) — otherwise an interrupted run whose
            # STOP_PERFORMANCE_TRACKING never fired reports QPS=N/A.
            registry.set_counter(
                MetricCounterKey.TRACKED_DURATION_NS.value,
                table.total_tracked_duration_ns,
            )
            await publisher.publish_final(
                registry,
                n_pending_tasks=token_queue.pending if token_queue is not None else 0,
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
        required=True,
        help=(
            "Wall-clock budget (seconds) to finish tokenizing buffered samples "
            "after ENDED (0 = wait indefinitely). The benchmark forwards "
            "--metrics-drain-timeout; the default lives in config/schema.py."
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
        required=True,
        help=(
            "In-process tokenizer threads for live (mid-run) ISL/OSL/TPOT "
            "(0 = defer everything to the end-of-run drain, which always uses "
            "the auto-sized sharded pool). The benchmark forwards "
            "--metrics-tokenizer-workers; the default lives in config/schema.py."
        ),
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

    if args.tokenizer_workers < 0:
        raise SystemExit("FATAL: --tokenizer-workers must be >= 0")

    # The parent (commands/benchmark/execute.py) owns directory creation;
    # fail fast here so a bad launcher errors now, not on the atomic write.
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
    # (coalesces to 'object' not 'AbstractContextManager[BatchTokenizer | None]')
    tokenizer_cm: AbstractContextManager[BatchTokenizer | None]
    if args.tokenizer:
        try:
            tokenizer_cm = BatchTokenizer(
                args.tokenizer, live_workers=args.tokenizer_workers
            )
        except RuntimeError as exc:
            # An environment that cannot shard is a launch failure, not a
            # silent slow path that cannot keep up with completions.
            raise SystemExit(f"FATAL: {exc}") from exc
    else:
        tokenizer_cm = nullcontext()

    with (
        tokenizer_cm as tokenizer,
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
                tokenizer=tokenizer,
                live_flush_interval_s=(
                    args.publish_interval if args.tokenizer_workers > 0 else None
                ),
                streaming=args.streaming,
                shutdown_event=shutdown_event,
                drain_timeout_s=None if args.drain_timeout == 0 else args.drain_timeout,
            )
            aggregator.start()

            # SIGTERM (ServiceLauncher.kill_all) must still produce a final
            # snapshot, tagged INTERRUPTED; publish_final is idempotent, so
            # racing the ENDED-driven call is safe. SIGINT (^C hits the whole
            # process group) is a no-op: finalizing eagerly at signal time
            # would freeze the snapshot before the parent's graceful-shutdown
            # samples land — the parent's ENDED drives finalize instead.
            on_sigterm, _sigterm_tasks = _make_sigterm_handler(
                loop=loop,
                registry=registry,
                publisher=publisher,
                table=aggregator.table,
                token_queue=aggregator.token_queue,
                shutdown_event=shutdown_event,
            )
            loop.add_signal_handler(signal.SIGTERM, on_sigterm)
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
    try:
        LoopManager().default_loop.run_until_complete(main())
    except SystemExit:
        # argparse / explicit sys.exit — already user-facing, don't dress up.
        raise
    except Exception as e:
        # Structured log line so the crash is grep-able against the parent's
        # logs; KeyboardInterrupt/SystemExit propagate untouched.
        logger.exception("metrics aggregator subprocess crashed (%s)", type(e).__name__)
        raise
