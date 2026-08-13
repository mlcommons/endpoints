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
callback) over the production tokenization paths:

* **Live text lane** — ``flush_live_once`` in a loop, exactly as the mid-run
  flush cadence does: each flush takes at most ``_LIVE_FLUSH_MAX_ITEMS``
  per input kind and encodes on the small in-process thread pool (rayon
  capped to ``live_workers`` cores).

* **Drain text lane** — ``flush_remaining(None)``, as the end-of-run drain
  does: the whole ``TextInput`` buffer fans out across every pinned shard
  process.

* **Message lane** — structured assistant outputs (``MessageInput`` with
  content + reasoning + a tool call) rendered per item via
  ``apply_chat_template`` on the in-process thread pool. This is the OSL/TPOT
  path for tool-call outputs and never touches the shard pool, so its
  throughput bounds how fast a structured backlog drains.

* **Prompt lane** — complete chat inputs (``PromptInput`` with message
  history + tools + generation prompt), the structured-ISL path; same
  per-item chat-template rendering.

Uses the checked-in char-level tokenizers (``tests/assets/tokenizers/char``
for text, ``char_chat`` — same Rust backend plus a minimal ChatML-style
chat template — for the structured lanes; no network). Numbers track the
queue/flush/shard/template machinery rather than BPE merge cost — a real
model tokenizer shifts the absolute texts/s but not the machinery
regressions these guard.

Reports items/s per lane (asserting only correctness: every enqueued item
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
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    PromptInput,
    TextInput,
    TokenizationInput,
)

_TOKENIZER_ROOT = Path(__file__).resolve().parents[4] / "assets" / "tokenizers"
TOKENIZER = str(_TOKENIZER_ROOT / "char")
CHAT_TOKENIZER = str(_TOKENIZER_ROOT / "char_chat")

# Matches the CLI default for --tokenizer-workers (the live thread lane).
LIVE_WORKERS = 2

# Sized so each lane runs long enough (seconds, not milliseconds) for a
# stable items/s number on a dev box: the live lane is bounded to
# _LIVE_FLUSH_MAX_ITEMS per flush on a 2-core rayon pool; the drain lane
# spans every shard; the structured lanes render one chat template per item
# on the thread pool.
N_LIVE = 32_768
N_DRAIN = 524_288
N_STRUCTURED = 16_384

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

_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
)


def _sentence(rng: random.Random, lo: int, hi: int) -> str:
    return " ".join(rng.choices(_WORDS, k=rng.randint(lo, hi)))


def _make_texts(n: int, seed: int = 42) -> list[str]:
    """Deterministic synthetic texts, 8-128 words each (~600 chars avg)."""
    rng = random.Random(seed)
    return [_sentence(rng, 8, 128) for _ in range(n)]


def _make_messages(n: int, seed: int = 43) -> list[MessageInput]:
    """Structured assistant outputs: content + reasoning + one tool call."""
    rng = random.Random(seed)
    return [
        MessageInput(
            content=_sentence(rng, 8, 64),
            reasoning=_sentence(rng, 16, 96),
            tool_calls=(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": _sentence(rng, 1, 3)},
                    },
                },
            ),
        )
        for i in range(n)
    ]


def _make_prompts(n: int, seed: int = 44) -> list[PromptInput]:
    """Complete chat inputs: 4-message history + tools + generation prompt."""
    rng = random.Random(seed)
    return [
        PromptInput(
            messages=(
                {"role": "system", "content": _sentence(rng, 8, 32)},
                {"role": "user", "content": _sentence(rng, 16, 96)},
                {"role": "assistant", "content": _sentence(rng, 16, 64)},
                {"role": "user", "content": _sentence(rng, 16, 96)},
            ),
            tools=_TOOLS,
            chat_template_kwargs=None,
            chat_template=None,
        )
        for _ in range(n)
    ]


class _Recorder:
    """Counting ``on_count`` callback shared by all lanes."""

    def __init__(self) -> None:
        self.items = 0
        self.tokens = 0

    def __call__(self, count: int) -> None:
        self.items += 1
        self.tokens += count


def _report(
    record_result, label: str, n_items: int, elapsed: float, rec: _Recorder
) -> None:
    items_per_s = n_items / elapsed
    record_result(label, qps=items_per_s, total=n_items, elapsed=elapsed, failed=0)
    print(
        f"\n  {label}: items/s={items_per_s:>9,.0f}  "
        f"tokens/s={rec.tokens / elapsed:>12,.0f}  "
        f"total={n_items:,}  elapsed={elapsed:.2f}s"
    )


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.asyncio
async def test_live_text_lane_throughput(record_result):
    """Mid-run live lane: bounded text flushes on the in-process thread pool."""
    items: list[TokenizationInput] = [TextInput(t) for t in _make_texts(N_LIVE)]
    loop = asyncio.get_running_loop()
    rec = _Recorder()
    with BatchTokenizer(TOKENIZER, live_workers=LIVE_WORKERS, n_workers=0) as tok:
        queue = TokenBatchQueue(tok, loop)
        for item in items:
            queue.enqueue(item, rec)
        t0 = time.perf_counter()
        while queue.pending:
            await queue.flush_live_once()
        elapsed = time.perf_counter() - t0

    assert rec.items == N_LIVE
    assert queue.pending == 0
    _report(record_result, f"tok live text ({LIVE_WORKERS} thr)", N_LIVE, elapsed, rec)


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.asyncio
async def test_drain_text_lane_throughput(record_result):
    """End-of-run drain: full text buffer fanned out across every shard."""
    items: list[TokenizationInput] = [TextInput(t) for t in _make_texts(N_DRAIN)]
    loop = asyncio.get_running_loop()
    rec = _Recorder()
    with BatchTokenizer(TOKENIZER, live_workers=LIVE_WORKERS) as tok:
        n_shards = len(tok._procs)
        queue = TokenBatchQueue(tok, loop)
        for item in items:
            queue.enqueue(item, rec)
        t0 = time.perf_counter()
        pending = await queue.flush_remaining(None)
        elapsed = time.perf_counter() - t0

    assert pending == 0
    assert rec.items == N_DRAIN
    _report(record_result, f"tok drain text ({n_shards} shards)", N_DRAIN, elapsed, rec)


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.asyncio
async def test_message_lane_throughput(record_result):
    """Structured assistant outputs (OSL/TPOT): per-item chat-template render.

    Uses the ``char_chat`` tokenizer so ``apply_chat_template`` really renders
    (the plain ``char`` tokenizer has no template and would exercise only the
    whitespace fallback). Skips shard setup — structured items never touch the
    shard pool.
    """
    items = _make_messages(N_STRUCTURED)
    loop = asyncio.get_running_loop()
    rec = _Recorder()
    with BatchTokenizer(CHAT_TOKENIZER, live_workers=LIVE_WORKERS, n_workers=0) as tok:
        queue = TokenBatchQueue(tok, loop)
        for item in items:
            queue.enqueue(item, rec)
        t0 = time.perf_counter()
        pending = await queue.flush_remaining(None)
        elapsed = time.perf_counter() - t0

    assert pending == 0
    assert rec.items == N_STRUCTURED
    assert rec.tokens > 0, "chat-template render produced no tokens"
    _report(record_result, "tok msg chat-template", N_STRUCTURED, elapsed, rec)


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.asyncio
async def test_prompt_lane_throughput(record_result):
    """Complete chat prompts (structured ISL): per-item chat-template render."""
    items = _make_prompts(N_STRUCTURED)
    loop = asyncio.get_running_loop()
    rec = _Recorder()
    with BatchTokenizer(CHAT_TOKENIZER, live_workers=LIVE_WORKERS, n_workers=0) as tok:
        queue = TokenBatchQueue(tok, loop)
        for item in items:
            queue.enqueue(item, rec)
        t0 = time.perf_counter()
        pending = await queue.flush_remaining(None)
        elapsed = time.perf_counter() - t0

    assert pending == 0
    assert rec.items == N_STRUCTURED
    assert rec.tokens > 0, "chat-template render produced no tokens"
    _report(record_result, "tok prompt chat-template", N_STRUCTURED, elapsed, rec)
