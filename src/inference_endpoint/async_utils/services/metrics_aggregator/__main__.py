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
from pathlib import Path

from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext

from .aggregator import MetricsAggregatorService
from .emitter import JsonlMetricEmitter
from .token_metrics import TokenizePool


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metrics aggregator service - subscribes to EventRecords and computes real-time metrics"
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory for metrics output (JSONL file)",
    )
    parser.add_argument(
        "--socket-address",
        type=str,
        required=True,
        help="ZMQ socket address to connect to",
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
    args = parser.parse_args()

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_dir / "metrics"

    shutdown_event = asyncio.Event()
    loop = LoopManager().default_loop

    pool = None
    if args.tokenizer:
        pool = TokenizePool(args.tokenizer, n_workers=args.tokenizer_workers)

    try:
        with ManagedZMQContext.scoped(socket_dir=args.metrics_dir.parent) as zmq_ctx:
            emitter = JsonlMetricEmitter(metrics_file, flush_interval=100)
            aggregator = MetricsAggregatorService(
                args.socket_address,
                zmq_ctx,
                loop,
                topics=None,
                emitter=emitter,
                tokenize_pool=pool,
                shutdown_event=shutdown_event,
            )
            loop.call_soon_threadsafe(aggregator.start)
            await shutdown_event.wait()
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    LoopManager().default_loop.run_until_complete(main())
