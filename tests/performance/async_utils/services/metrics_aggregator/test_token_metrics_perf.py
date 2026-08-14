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

"""Throughput matrix for the metrics-tokenizer paths.

One parametrized test drives :class:`TokenBatchQueue` end-to-end (enqueue →
flush → recorder callback) over every input kind × flush lane combination,
so a single run gives the full token-throughput overview:

* ``text`` — the pre-existing batched plain-text path. The live lane encodes
  on the small in-process thread pool (≤1024 items/flush, rayon capped to
  ``live_workers``); the drain lane fans the whole buffer out across every
  pinned shard process.

* ``msg`` — a structured assistant *output* (content + reasoning + tool
  call), the chat-template OSL/TPOT path added with structured token
  counting (#441). One ``apply_chat_template`` render per item on the
  thread pool; never touches the shard pool, so live ≈ drain.

* ``prompt`` — a complete structured *input* (message history + tools +
  generation prompt), the chat-template ISL path added with #441. Same
  per-item rendering; the heavier render of the two (whole history +
  framing).

Uses the checked-in char-level tokenizers (``tests/assets/tokenizers/char``
for text; ``char_chat`` — same Rust backend plus a minimal ChatML-style chat
template — for the structured kinds; no network). Numbers track the
queue/flush/shard/template machinery rather than BPE merge cost — a real
model tokenizer shifts absolute items/s but not the machinery regressions
these guard, and real agentic content scales the chat-template kinds
linearly with rendered length.

Reports items/s per lane (asserting only correctness: every enqueued item
recorded, nothing left pending); rows land in the shared ``record_result``
summary table. Run::

    pytest -vs -m performance --no-cov \
        tests/performance/async_utils/services/metrics_aggregator/test_token_metrics_perf.py
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import msgspec
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

# Per-(kind, lane) corpus sizes, sized so each cell runs whole seconds for a
# stable items/s number on a dev box. Only the text drain gets the huge
# corpus (it spans every shard); the chat-template kinds run per item on the
# thread pool at ~1-2k items/s, so 16k items is already ~10s per cell.
_N = {
    ("text", "live"): 32_768,
    ("text", "drain"): 524_288,
    ("msg", "live"): 16_384,
    ("msg", "drain"): 16_384,
    ("prompt", "live"): 16_384,
    ("prompt", "drain"): 16_384,
}

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


def _make_items(kind: str, n: int) -> list[TokenizationInput]:
    """Deterministic synthetic inputs for one kind (fixed seed per kind)."""
    if kind == "text":
        rng = random.Random(42)
        return [TextInput(_sentence(rng, 8, 128)) for _ in range(n)]
    if kind == "msg":
        rng = random.Random(43)
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
    rng = random.Random(44)
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


def _payload_bytes(item: TokenizationInput) -> int:
    """UTF-8 bytes of the text payload an item carries to the tokenizer.

    For structured kinds this counts the enqueued content (messages, tool
    JSON), not the template framing the render adds around it — it answers
    "how much workload text did this lane chew through".
    """
    if isinstance(item, TextInput):
        return len(item.text.encode())
    if isinstance(item, MessageInput):
        return (
            len(item.content.encode())
            + len((item.reasoning or "").encode())
            + (len(msgspec.json.encode(item.tool_calls)) if item.tool_calls else 0)
        )
    if isinstance(item, PromptInput):
        return sum(len(str(m.get("content", "")).encode()) for m in item.messages) + (
            len(msgspec.json.encode(item.tools)) if item.tools else 0
        )
    return len(item.token_ids)  # TokenIdsInput: pre-tokenized, no text payload


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("lane", ["live", "drain"])
@pytest.mark.parametrize("kind", ["text", "msg", "prompt"])
@pytest.mark.asyncio
async def test_tokenizer_lane_throughput(kind, lane, record_result):
    """One (input kind, flush lane) cell of the tokenizer throughput matrix."""
    n = _N[(kind, lane)]
    items = _make_items(kind, n)
    # Only the text drain uses the shard pool; every other cell runs
    # in-process (n_workers=0 skips the shard spawn). Structured kinds need
    # the chat-template tokenizer so apply_chat_template really renders.
    shards = kind == "text" and lane == "drain"
    tokenizer_name = TOKENIZER if kind == "text" else CHAT_TOKENIZER
    loop = asyncio.get_running_loop()
    rec = _Recorder()
    with BatchTokenizer(
        tokenizer_name,
        live_workers=LIVE_WORKERS,
        n_workers=-1 if shards else 0,
    ) as tok:
        detail = f"{len(tok._procs)} shards" if shards else f"{LIVE_WORKERS} thr"
        queue = TokenBatchQueue(tok, loop)
        for item in items:
            queue.enqueue(item, rec)
        t0 = time.perf_counter()
        if lane == "live":
            while queue.pending:
                await queue.flush_live_once()
            pending = queue.pending
        else:
            pending = await queue.flush_remaining(None)
        elapsed = time.perf_counter() - t0

    assert pending == 0
    assert rec.items == n
    assert rec.tokens > 0
    items_per_s = n / elapsed
    data_mb = sum(_payload_bytes(item) for item in items) / 1e6
    label = f"tok {kind} {lane} ({detail})"
    record_result(
        label,
        qps=items_per_s,
        total=n,
        data_mb=data_mb,
        elapsed=elapsed,
        failed=pending,
    )
    print(
        f"\n  {label}: items/s={items_per_s:>9,.0f}  "
        f"tokens/s={rec.tokens / elapsed:>12,.0f}  "
        f"data={data_mb:,.1f}MB ({data_mb / elapsed:,.1f}MB/s)  "
        f"total={n:,}  elapsed={elapsed:.2f}s"
    )
