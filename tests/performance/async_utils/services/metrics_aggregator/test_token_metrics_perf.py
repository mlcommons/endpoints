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

"""Throughput benchmarks for the metrics-tokenizer lanes.

Drives :class:`TokenBatchQueue` end-to-end (enqueue → flush → recorder
callback) over both production tokenization lanes:

* **Live lane** — ``flush_live_once`` in a loop, exactly as the mid-run
  flush cadence does: each flush takes at most ``_LIVE_FLUSH_MAX_ITEMS``
  and encodes on the small in-process thread pool (rayon capped to
  ``live_workers`` cores).

* **Drain lane** — ``flush_remaining(None)``, as the end-of-run drain
  does: the whole buffer fans out across every pinned shard process.

Uses the checked-in char-level tokenizer (``tests/assets/tokenizers/char``,
fast Rust backend, no network), so the numbers track the queue/flush/shard
machinery rather than BPE merge cost — a real model tokenizer shifts the
absolute texts/s but not the machinery regressions these guard.

Reports texts/s per lane (asserting only correctness: every enqueued item
recorded, nothing left pending); rows land in the shared
``record_result`` summary table. Run::

    pytest -vs -m performance --no-cov \
        tests/performance/async_utils/services/metrics_aggregator/test_token_metrics_perf.py
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
    TokenBatchQueue,
)

TOKENIZER = str(Path(__file__).resolve().parents[4] / "assets" / "tokenizers" / "char")

# Matches the CLI default for --tokenizer-workers (the live thread lane).
LIVE_WORKERS = 2

# Sized so each lane runs long enough (seconds, not milliseconds) for a
# stable texts/s number on a dev box: the live lane is bounded to
# _LIVE_FLUSH_MAX_ITEMS per flush on a 2-core rayon pool; the drain lane
# spans every shard.
N_LIVE = 32_768
N_DRAIN = 524_288

_WORDS = (
    "benchmark",
    "endpoint",
    "tokenizer",
    "throughput",
    "latency",
    "sample",
    "metric",
    "stream",
    "poisson",
    "drain",
    "flush",
    "shard",
    "queue",
    "token",
    "batch",
    "worker",
)


def _make_corpus(n: int, seed: int = 42) -> list[str]:
    """Deterministic synthetic texts, 8-128 words each (~600 chars avg)."""
    rng = random.Random(seed)
    return [" ".join(rng.choices(_WORDS, k=rng.randint(8, 128))) for _ in range(n)]


class _Recorder:
    """Counting ``on_count`` callback shared by both lanes."""

    def __init__(self) -> None:
        self.items = 0
        self.tokens = 0

    def __call__(self, count: int) -> None:
        self.items += 1
        self.tokens += count


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.asyncio
async def test_live_lane_throughput(record_result):
    """Mid-run live lane: bounded flushes on the in-process thread pool."""
    texts = _make_corpus(N_LIVE)
    loop = asyncio.get_running_loop()
    rec = _Recorder()
    with BatchTokenizer(TOKENIZER, live_workers=LIVE_WORKERS, n_workers=0) as tok:
        queue = TokenBatchQueue(tok, loop)
        for t in texts:
            queue.enqueue_text(t, rec)
        t0 = time.perf_counter()
        while queue.pending:
            await queue.flush_live_once()
        elapsed = time.perf_counter() - t0

    assert rec.items == N_LIVE
    assert queue.pending == 0
    texts_per_s = N_LIVE / elapsed
    record_result(
        f"tok live lane ({LIVE_WORKERS} thr)",
        qps=texts_per_s,
        total=N_LIVE,
        elapsed=elapsed,
        failed=queue.pending,
    )
    print(
        f"\n  tok live lane  workers={LIVE_WORKERS}: "
        f"texts/s={texts_per_s:>9,.0f}  tokens/s={rec.tokens / elapsed:>12,.0f}  "
        f"total={N_LIVE:,}  elapsed={elapsed:.2f}s"
    )


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.asyncio
async def test_drain_lane_throughput(record_result):
    """End-of-run drain: full buffer fanned out across every pinned shard."""
    texts = _make_corpus(N_DRAIN)
    loop = asyncio.get_running_loop()
    rec = _Recorder()
    with BatchTokenizer(TOKENIZER, live_workers=LIVE_WORKERS) as tok:
        n_shards = len(tok._procs)
        queue = TokenBatchQueue(tok, loop)
        for t in texts:
            queue.enqueue_text(t, rec)
        t0 = time.perf_counter()
        pending = await queue.flush_remaining(None)
        elapsed = time.perf_counter() - t0

    assert pending == 0
    assert rec.items == N_DRAIN
    texts_per_s = N_DRAIN / elapsed
    record_result(
        f"tok drain lane ({n_shards} shards)",
        qps=texts_per_s,
        total=N_DRAIN,
        elapsed=elapsed,
        failed=pending,
    )
    print(
        f"\n  tok drain lane  shards={n_shards}: "
        f"texts/s={texts_per_s:>9,.0f}  tokens/s={rec.tokens / elapsed:>12,.0f}  "
        f"total={N_DRAIN:,}  elapsed={elapsed:.2f}s"
    )
