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

"""Apples-to-apples benchmark of OUTPUT-tokenization strategies for the
metrics-aggregator drain (OSL / TPOT).

Why this exists: at the end of a run the aggregator tokenizes every sample's
output to derive OSL/TPOT. The live impl fires one asyncio task per sample,
each awaiting ``loop.run_in_executor(thread_pool, len(tok.tokenize(text)))``
(see ``metrics_aggregator/metrics_table.py::AsyncTokenTrigger.fire`` +
``token_metrics.py::TokenizePool.token_count_async``). This script reproduces
that exact pattern standalone and pits it against a single batched
``tokenizer(texts)`` call (the batched strategy from the prior ISL ablation)
so the cost of the current design — and the win from replacing it — is measured
on identical inputs. Measured (Qwen2.5-0.5B, 48-core, 12 workers): encode_batch
is ~4.6x the current per-sample async pattern on short outputs, ~2.0x on the
realistic right-skewed OSL distribution (mean ~3.8k tok) — and, more
importantly, removes the per-sample asyncio-task backlog (1 task/sample) that
drives the drain timeout. The single batched Rust call beats thread-sharding
(the HF fast tokenizer already parallelises a batch internally).

Strategies (all plain ``tokenize``, no chat template — matches the OSL/TPOT
text path taken when the output has no tool_calls):

  current_async   EXACT live drain pattern: per-sample loop.create_task ->
                  TokenizePool.token_count_async -> run_in_executor, gathered.
  sync_loop       Serial ``len(tok.tokenize(t))`` — isolates raw tokenize cost
                  from asyncio/thread-pool overhead.
  batch           One ``tokenizer(texts)`` Rust call over all texts.
  thread_batch    Shard texts across ``--workers`` threads, each batch-tokenizes
                  its shard (GIL released inside the Rust call).

Usage:
    uv run python scripts/bench_drain_tokenize.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --n-samples 20000 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    TokenizePool,
)
from transformers import AutoTokenizer

_WORDS = (
    "the quick brown fox jumps over the lazy dog inference benchmark "
    "tokenization latency throughput performance model weights attention "
    "transformer layer norm softmax gradient embedding sequence decode "
).split()


# Measured OSL token-length distribution (max_new_tokens=20000 cap; heavily
# right-skewed: median 2153, mean 3824). Piecewise-linear inverse-CDF from the
# measured percentiles so generated lengths match the real drain workload.
_OSL_PCTL: tuple[tuple[float, int], ...] = (
    (0, 177),
    (1, 303),
    (5, 463),
    (10, 578),
    (25, 951),
    (50, 2153),
    (75, 4977),
    (80, 6001),
    (90, 9564),
    (95, 13510),
    (97, 16422),
    (99, 20000),
    (100, 20000),
)


def _sample_osl(rng: random.Random) -> int:
    p = rng.random() * 100.0
    for (p0, v0), (p1, v1) in zip(_OSL_PCTL, _OSL_PCTL[1:], strict=False):
        if p <= p1:
            frac = (p - p0) / (p1 - p0) if p1 > p0 else 0.0
            return int(v0 + frac * (v1 - v0))
    return _OSL_PCTL[-1][1]


def _make_outputs(
    n: int, profile: str, min_words: int, max_words: int, seed: int = 42
) -> list[str]:
    """Synthetic model-output texts (plain text, the OSL/TPOT common case).

    profile='mlperf' draws word counts from the measured OSL distribution
    (token≈word for these common words); 'uniform' uses [min_words, max_words].
    """
    rng = random.Random(seed)
    if profile == "mlperf":
        lengths = [_sample_osl(rng) for _ in range(n)]
    else:
        lengths = [rng.randint(min_words, max_words) for _ in range(n)]
    return [" ".join(rng.choices(_WORDS, k=length)) for length in lengths]


def _result(name: str, secs: float, n: int, total_tokens: int) -> dict[str, Any]:
    return {
        "strategy": name,
        "wall_s": round(secs, 4),
        "samples_per_s": round(n / secs) if secs else 0,
        "tokens_per_s": round(total_tokens / secs) if secs else 0,
    }


def bench_sync_loop(texts: list[str], tok: Any) -> tuple[float, int]:
    t0 = time.perf_counter()
    total = 0
    for t in texts:
        total += len(tok.tokenize(t))
    return time.perf_counter() - t0, total


def bench_batch(texts: list[str], tok: Any) -> tuple[float, int]:
    t0 = time.perf_counter()
    enc = tok(texts, add_special_tokens=False, return_attention_mask=False)
    total = sum(len(ids) for ids in enc["input_ids"])
    return time.perf_counter() - t0, total


def bench_encode_batch(texts: list[str], tok: Any) -> tuple[float, int]:
    """Raw Rust ``encode_batch`` on the backend tokenizer — skips the
    BatchEncoding/padding wrapper that ``tokenizer(...)`` builds. We only need
    counts, so this is the leanest count-only path."""
    backend = tok.backend_tokenizer
    # encode_batch_fast (tokenizers>=0.20) skips offset computation; fall back
    # to encode_batch where unavailable.
    fn = getattr(backend, "encode_batch_fast", None) or backend.encode_batch
    t0 = time.perf_counter()
    encs = fn(texts, add_special_tokens=False)
    total = sum(len(e.ids) for e in encs)
    return time.perf_counter() - t0, total


def bench_batch_chunked(
    texts: list[str], tok: Any, chunk: int = 50_000
) -> tuple[float, int]:
    """Chunked batches — bounds peak memory for very large drains while still
    feeding the Rust parallel path large slices."""
    t0 = time.perf_counter()
    total = 0
    for i in range(0, len(texts), chunk):
        enc = tok(
            texts[i : i + chunk],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        total += sum(len(ids) for ids in enc["input_ids"])
    return time.perf_counter() - t0, total


def bench_thread_batch(
    texts: list[str], tokenizer_name: str, workers: int
) -> tuple[float, int]:
    # Each worker loads its own tokenizer (thread-local, like TokenizePool) and
    # batch-tokenizes a contiguous shard.
    shards: list[list[str]] = [texts[i::workers] for i in range(workers)]
    tls = threading.local()

    def _work_tls(shard: list[str]) -> int:
        tok = getattr(tls, "tok", None)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            tls.tok = tok
        if not shard:
            return 0
        enc = tok(shard, add_special_tokens=False, return_attention_mask=False)
        return sum(len(ids) for ids in enc["input_ids"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Warm tokenizers on every thread before timing.
        list(ex.map(lambda _: _work_tls([]), range(workers)))
        t0 = time.perf_counter()
        total = sum(ex.map(_work_tls, shards))
    return time.perf_counter() - t0, total


async def bench_current_async(
    texts: list[str], pool: TokenizePool
) -> tuple[float, int]:
    """EXACT live drain pattern: one asyncio task per sample, each awaiting
    pool.token_count_async (-> loop.run_in_executor), then gathered."""
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    tasks = [loop.create_task(pool.token_count_async(t, loop)) for t in texts]
    counts = await asyncio.gather(*tasks)
    return time.perf_counter() - t0, sum(counts)


def _run_current_async(texts: list[str], pool: TokenizePool) -> tuple[float, int]:
    try:
        import uvloop  # the aggregator runs on uvloop; match it.

        runner = uvloop.run
    except ImportError:
        runner = asyncio.run
    return runner(bench_current_async(texts, pool))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--workers",
        type=int,
        default=max(2, (os.cpu_count() or 16) // 4),
        help="TokenizePool / thread_batch worker count (aggregator default).",
    )
    ap.add_argument("--osl-profile", choices=("mlperf", "uniform"), default="mlperf")
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model}")
    AutoTokenizer.from_pretrained(args.model)  # warm cache before timing
    tok = AutoTokenizer.from_pretrained(args.model)

    print(
        f"Generating {args.n_samples} synthetic outputs (profile={args.osl_profile})..."
    )
    texts = _make_outputs(
        args.n_samples, args.osl_profile, args.min_words, args.max_words
    )
    avg_words = sum(t.count(" ") + 1 for t in texts) / len(texts)
    print(
        f"profile={args.osl_profile} | avg {avg_words:.0f} words/output "
        f"| workers={args.workers}\n"
    )

    pool = TokenizePool(args.model, n_workers=args.workers)
    results: list[dict[str, Any]] = []
    try:
        strategies = [
            ("current_async", lambda: _run_current_async(texts, pool)),
            ("sync_loop", lambda: bench_sync_loop(texts, tok)),
            ("batch", lambda: bench_batch(texts, tok)),
            ("batch_chunked", lambda: bench_batch_chunked(texts, tok)),
            ("encode_batch", lambda: bench_encode_batch(texts, tok)),
            (
                "thread_batch",
                lambda: bench_thread_batch(texts, args.model, args.workers),
            ),
        ]
        for name, fn in strategies:
            best_secs = float("inf")
            total_tokens = 0
            for _ in range(args.runs):
                secs, total_tokens = fn()
                best_secs = min(best_secs, secs)
            r = _result(name, best_secs, args.n_samples, total_tokens)
            results.append(r)
            print(
                f"{name:<16} {r['wall_s']:>9.4f}s  "
                f"{r['samples_per_s']:>12,} samples/s  "
                f"{r['tokens_per_s']:>14,} tok/s"
            )
    finally:
        pool.close()

    base = next(r for r in results if r["strategy"] == "current_async")
    print("\nspeedup vs current_async (best wall):")
    for r in results:
        if r["strategy"] != "current_async" and r["samples_per_s"]:
            print(
                f"  {r['strategy']:<16} {r['samples_per_s'] / base['samples_per_s']:>6.1f}x"
            )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
