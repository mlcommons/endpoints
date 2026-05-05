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
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.ready_check import send_ready_signal
from inference_endpoint.utils.logging import setup_logging

from .aggregator import MetricsAggregatorService
from .publisher import MetricsPublisher
from .registry import MetricsRegistry
from .snapshot import MetricsSnapshotCodec
from .token_metrics import TokenizePool


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
        "--refresh-hz",
        type=float,
        default=4.0,
        help="Live snapshot publish rate (default: 4.0)",
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

    metrics_output_dir: Path = args.metrics_output_dir
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    shutdown_event = asyncio.Event()
    loop = LoopManager().default_loop

    # Using ternary operator causes errors in MyPy object type coalescing
    # (coalesces to 'object' not 'AbstractContextManager[TokenizePool | None]')
    if args.tokenizer:
        pool_cm: AbstractContextManager[TokenizePool | None] = TokenizePool(
            args.tokenizer, n_workers=args.tokenizer_workers
        )
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
            fallback_path=metrics_output_dir / "final_snapshot.msgpack",
        )
        try:
            aggregator = MetricsAggregatorService(
                args.socket_name,
                zmq_ctx,
                loop,
                topics=None,
                registry=registry,
                publisher=publisher,
                refresh_hz=args.refresh_hz,
                sig_figs=args.hdr_sig_figs,
                n_histogram_buckets=args.n_histogram_buckets,
                tokenize_pool=pool,
                streaming=args.streaming,
                shutdown_event=shutdown_event,
            )
            aggregator.start()

            if args.readiness_path:
                await send_ready_signal(zmq_ctx, args.readiness_path, args.readiness_id)

            await shutdown_event.wait()
        finally:
            publisher.close()


if __name__ == "__main__":
    LoopManager().default_loop.run_until_complete(main())
