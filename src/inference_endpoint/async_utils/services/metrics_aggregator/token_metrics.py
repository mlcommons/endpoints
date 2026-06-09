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

"""Tokenization for ISL/OSL/TPOT metrics.

``BatchTokenizer`` tokenizes whole batches of text at once. A single BPE rayon
pool saturates ~8 CPU cores (memory-bound), so to use the whole machine it
shards each batch across worker *processes*, one pinned to each block of
``CORES_PER_WORKER`` cores (their rayon pools stay NUMA-local). The aggregator
buffers per-sample text as COMPLETE events arrive and calls ``count_texts`` once
per flush (publish tick + drain) — so batching, not a per-request coalescer,
keeps tokenization ahead of completions. Falls back to a single in-process
thread when there is no fast Rust backend or fewer than two core blocks fit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol, cast

import msgspec
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

# A single rayon pool peaks at ~8 cores for BPE (memory-bound; more threads
# oversubscribe and, on multi-socket Grace, cross the NUMA boundary). Sharding
# across processes pinned to disjoint 8-core blocks is how the whole machine is
# used. Measured on GB200: ~16k texts/s at 18 blocks vs ~1.5k single-process.
CORES_PER_WORKER = 8

# Minimal user message used to satisfy chat templates that reject assistant-only
# message lists. Its token count is subtracted so only the assistant payload is
# measured.
_PREFIX_USER_MSG: dict[str, str] = {"role": "user", "content": ""}

logger = logging.getLogger(__name__)


def _normalize_tool_calls_for_template(
    tool_calls: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure ``function.arguments`` is a dict, not the OpenAI-wire JSON string.

    Hermes-style chat templates iterate ``arguments`` as a mapping; a string
    payload raises and forces the fallback path, inflating token counts.
    """
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                normalized.append(tc)
                continue
            if isinstance(parsed, dict):
                new_tc = dict(tc)
                new_tc["function"] = {**fn, "arguments": parsed}
                normalized.append(new_tc)
                continue
        normalized.append(tc)
    return normalized


# ---------------------------------------------------------------------------
# Process-worker entry points (module-level so ProcessPoolExecutor can pickle
# them by name). Each worker holds one raw tokenizers backend, pinned to a
# fixed core block.
# ---------------------------------------------------------------------------

_WORKER_BACKEND: Any = None


def _init_worker(tokenizer_name: str, core_set: list[int]) -> None:
    """Pin this worker to ``core_set``, then load the raw tokenizers backend.

    Affinity is set before the first encode so the Rust rayon pool sizes itself
    to the pinned core count (num_cpus respects sched_getaffinity on Linux).
    """
    if core_set:
        try:
            os.sched_setaffinity(0, set(core_set))
        except (OSError, AttributeError):
            logger.debug("could not pin tokenizer worker to %s", core_set)
    transformers_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    global _WORKER_BACKEND
    _WORKER_BACKEND = getattr(tok, "backend_tokenizer", None)
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.encode("warmup", add_special_tokens=False)


def _worker_encode_lengths(texts: list[str]) -> list[int]:
    """Per-text token counts for a shard, in one rayon-parallel call."""
    backend = _WORKER_BACKEND
    if backend is None:
        raise RuntimeError("tokenizer worker backend unavailable")
    encode_batch = getattr(backend, "encode_batch_fast", None) or backend.encode_batch
    return [len(e.ids) for e in encode_batch(texts, add_special_tokens=False)]


def _worker_ready(_: int) -> bool:
    """Warmup probe: returns once the worker's backend is loaded."""
    return _WORKER_BACKEND is not None


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _even_chunks(items: list[str], n: int) -> list[list[str]]:
    """Split ``items`` into at most ``n`` near-equal contiguous chunks."""
    if n <= 1 or len(items) <= 1:
        return [items]
    size = (len(items) + n - 1) // n
    return [items[i : i + size] for i in range(0, len(items), size)]


class BatchTokenizer:
    """Counts tokens for batches of text, sharded across pinned CPU cores.

    ``count_texts`` / ``count_texts_async`` tokenize a whole list in one shot.
    The sync ``token_count`` and chat-template ``token_count_message`` paths run
    on a small in-process thread pool — they are rare (single ISL probes, tool
    calls) relative to the batched OSL/ISL/TPOT flush.
    """

    def __init__(
        self,
        tokenizer_name: str,
        *,
        cores_per_worker: int = CORES_PER_WORKER,
    ) -> None:
        self._tokenizer_name = tokenizer_name
        self._fallback_warned: set[str] = set()
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._prefix_len = 0
        self._baseline = 0
        # In-process thread for the sync + chat-template paths.
        self._thread: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tok-thread"
        )
        self._load_tokenizer()  # also computes the chat-template baseline
        # Process shards for the batched text path (or empty -> in-process).
        self._procs: list[ProcessPoolExecutor] = []
        self._setup_shards(cores_per_worker)

    # -- setup --------------------------------------------------------------

    def _load_tokenizer(self) -> None:
        tok = AutoTokenizer.from_pretrained(self._tokenizer_name)
        self._tokenizer = tok
        # Baseline = tokens from a [user, empty-assistant] pair minus the [user]
        # prefix alone, so the assistant frame is subtracted from message counts.
        try:
            prefix = cast(
                str,
                tok.apply_chat_template(
                    [_PREFIX_USER_MSG], tokenize=False, add_generation_prompt=False
                ),
            )
            self._prefix_len = len(tok.tokenize(prefix))
            with_assistant = cast(
                str,
                tok.apply_chat_template(
                    [_PREFIX_USER_MSG, {"role": "assistant", "content": ""}],
                    tokenize=False,
                    add_generation_prompt=False,
                ),
            )
            self._baseline = len(tok.tokenize(with_assistant)) - self._prefix_len
        except Exception:
            self._prefix_len = 0
            self._baseline = 0
            logger.exception(
                "Failed to compute chat-template baseline for %s; tool-call "
                "token counts may be over-estimated",
                self._tokenizer_name,
            )

    def _setup_shards(self, cores_per_worker: int) -> None:
        """Spawn one pinned single-worker process per core block.

        No-op (leaving the batch path in-process) when the tokenizer has no fast
        Rust backend, affinity is unavailable, or fewer than two blocks fit — a
        single shard is no faster than the in-process backend.
        """
        if cores_per_worker <= 0:
            return
        if getattr(self._tokenizer, "backend_tokenizer", None) is None:
            return
        try:
            available = sorted(os.sched_getaffinity(0))
        except (OSError, AttributeError):
            return
        n = len(available) // cores_per_worker
        if n < 2:
            return
        ctx = multiprocessing.get_context("spawn")
        procs: list[ProcessPoolExecutor] = []
        try:
            for i in range(n):
                block = available[i * cores_per_worker : (i + 1) * cores_per_worker]
                ex = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_init_worker,
                    initargs=(self._tokenizer_name, block),
                )
                procs.append(ex)
            # Force spawn + pin + tokenizer-load now (not on the first batch).
            # Submit to every shard first so the loads run in parallel, then
            # await — waiting on each before submitting the next would
            # serialize P tokenizer loads and can exceed the launch budget.
            ready = [ex.submit(_worker_ready, 0) for ex in procs]
            for f in ready:
                f.result()
        except Exception:
            for ex in procs:
                ex.shutdown(wait=False)
            logger.exception(
                "tokenizer shard setup failed; using in-process tokenization"
            )
            return
        self._procs = procs
        logger.info(
            "BatchTokenizer: %d shards x %d cores", len(procs), cores_per_worker
        )

    # -- batched text path --------------------------------------------------

    def _encode_lengths_inproc(self, texts: list[str]) -> list[int]:
        tok = self._tokenizer
        backend = getattr(tok, "backend_tokenizer", None)
        if backend is not None:
            encode_batch = getattr(backend, "encode_batch_fast", None)
            if encode_batch is None:
                encode_batch = backend.encode_batch
            return [len(e.ids) for e in encode_batch(texts, add_special_tokens=False)]
        return [len(tok.tokenize(t)) for t in texts]  # type: ignore[union-attr]

    def count_texts(self, texts: list[str]) -> list[int]:
        """Per-text token counts for a whole batch (blocking)."""
        if not texts:
            return []
        if not self._procs:
            return self._encode_lengths_inproc(texts)
        chunks = _even_chunks(texts, len(self._procs))
        futures = [
            self._procs[i].submit(_worker_encode_lengths, chunk)
            for i, chunk in enumerate(chunks)
        ]
        out: list[int] = []
        for f in futures:
            out.extend(f.result())
        return out

    async def count_texts_async(
        self, texts: list[str], loop: asyncio.AbstractEventLoop
    ) -> list[int]:
        """Per-text token counts for a whole batch without blocking the loop."""
        if not texts:
            return []
        if not self._procs:
            return await loop.run_in_executor(
                self._thread, self._encode_lengths_inproc, texts
            )
        chunks = _even_chunks(texts, len(self._procs))
        futures = [
            asyncio.wrap_future(self._procs[i].submit(_worker_encode_lengths, chunk))
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*futures)
        out: list[int] = []
        for r in results:
            out.extend(r)
        return out

    # -- sync + chat-template paths (in-process thread) ---------------------

    def _token_count_text(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text))  # type: ignore[union-attr]

    def _token_count_message(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
    ) -> int:
        tok = self._tokenizer
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = _normalize_tool_calls_for_template(tool_calls)
        try:
            rendered = tok.apply_chat_template(  # type: ignore[union-attr]
                [_PREFIX_USER_MSG, msg], tokenize=False, add_generation_prompt=False
            )
            full = len(tok.tokenize(rendered))  # type: ignore[union-attr]
            return max(0, full - self._prefix_len - self._baseline)
        except Exception as exc:
            key = f"{self._tokenizer_name}:{type(exc).__name__}"
            if key not in self._fallback_warned:
                self._fallback_warned.add(key)
                logger.exception(
                    "apply_chat_template failed for %s (%s); falling back to "
                    "whitespace tokenization. Tool-call OSL/TPOT may diverge.",
                    self._tokenizer_name,
                    type(exc).__name__,
                )
            tool_calls_json = (
                msgspec.json.encode(list(tool_calls)).decode() if tool_calls else None
            )
            parts = [
                p for p in (content or None, reasoning or None, tool_calls_json) if p
            ]
            return self._token_count_text("\n".join(parts))

    def token_count(self, text: str) -> int:
        """Token count for a single string (blocking)."""
        if self._thread is None:
            raise RuntimeError("BatchTokenizer is closed")
        return self._thread.submit(self._token_count_text, text).result()

    def token_count_message(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
    ) -> int:
        """Token count for an assistant message via the chat template (blocking)."""
        if self._thread is None:
            raise RuntimeError("BatchTokenizer is closed")
        return self._thread.submit(
            self._token_count_message, content, reasoning, tool_calls
        ).result()

    async def token_count_message_async(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
        loop: asyncio.AbstractEventLoop,
    ) -> int:
        """Chat-template message token count without blocking the loop."""
        if self._thread is None:
            raise RuntimeError("BatchTokenizer is closed")
        return await loop.run_in_executor(
            self._thread, self._token_count_message, content, reasoning, tool_calls
        )

    def close(self) -> None:
        """Shut down all workers. Idempotent."""
        for ex in self._procs:
            ex.shutdown(wait=False)
        self._procs = []
        if self._thread is not None:
            self._thread.shutdown(wait=True)
            self._thread = None

    def __enter__(self) -> BatchTokenizer:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


# Type alias for the (content, reasoning, tool_calls) tuple a message trigger
# enqueues for chat-template tokenization.
MessageParts = tuple[str, str | None, "tuple[dict[str, Any], ...] | None"]


class TokenCounter(Protocol):
    """The async tokenization surface ``TokenBatchQueue`` depends on.

    ``BatchTokenizer`` satisfies this structurally; tests pass lightweight
    stubs. Declared as a Protocol so the queue is decoupled from the concrete
    tokenizer and test doubles type-check without inheritance.
    """

    async def count_texts_async(
        self, texts: list[str], loop: asyncio.AbstractEventLoop, /
    ) -> list[int]: ...

    async def token_count_message_async(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
        loop: asyncio.AbstractEventLoop,
        /,
    ) -> int: ...


class TokenBatchQueue:
    """Buffers per-sample tokenization work and clears it in batches.

    Triggers call ``enqueue_text`` / ``enqueue_message`` at event time with a
    ``on_count`` callback that records the resulting metric. The aggregator
    drains the buffer with ``flush`` (once per publish tick, so live ISL/OSL/
    TPOT stay current) and with ``flush_remaining`` at end-of-run. Holding the
    work until a flush lets the whole buffer go through ``BatchTokenizer`` in
    one sharded call, instead of one event-loop task per completion — the latter
    is what fell behind and stretched the drain on high-completion-rate runs.

    ``pending`` counts enqueued-but-not-yet-recorded items; it is the
    ``n_pending_tasks`` surfaced on the snapshot, and a non-zero value in the
    final snapshot means the end-of-run flush did not finish within the drain
    budget.
    """

    def __init__(
        self, tokenizer: TokenCounter, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._tokenizer = tokenizer
        self._loop = loop
        self._text: list[tuple[str, Callable[[int], None]]] = []
        self._msg: list[tuple[MessageParts, Callable[[int], None]]] = []
        self._inflight = 0
        # Serializes flushes so a periodic tick flush and the end-of-run flush
        # never record the same item twice or race on the pending count.
        self._lock = asyncio.Lock()

    @property
    def pending(self) -> int:
        """Enqueued items not yet tokenized-and-recorded."""
        return self._inflight

    def enqueue_text(self, text: str, on_count: Callable[[int], None]) -> None:
        self._inflight += 1
        self._text.append((text, on_count))

    def enqueue_message(
        self, parts: MessageParts, on_count: Callable[[int], None]
    ) -> None:
        self._inflight += 1
        self._msg.append((parts, on_count))

    async def flush(self) -> None:
        """Tokenize everything buffered so far and run each ``on_count``.

        Items are detached from the buffer up front so concurrent enqueues land
        in the next flush. ``_inflight`` is decremented only after a callback
        runs, so a cancellation (drain timeout) leaves it reflecting exactly the
        items that were not recorded.
        """
        async with self._lock:
            if not (self._text or self._msg):
                return
            text_items, self._text = self._text, []
            msg_items, self._msg = self._msg, []
            if text_items:
                counts = await self._tokenizer.count_texts_async(
                    [t for t, _ in text_items], self._loop
                )
                for (_, on_count), count in zip(text_items, counts, strict=True):
                    try:
                        on_count(count)
                    finally:
                        self._inflight -= 1
            for (content, reasoning, tool_calls), on_count in msg_items:
                count = await self._tokenizer.token_count_message_async(
                    content, reasoning, tool_calls, self._loop
                )
                try:
                    on_count(count)
                finally:
                    self._inflight -= 1

    async def flush_remaining(self, timeout: float | None) -> int:
        """End-of-run flush, bounded by ``timeout`` seconds.

        Returns the number of items still un-tokenized — non-zero only if the
        budget was exhausted (``timeout`` reached). ``None`` waits indefinitely.
        """
        if self._inflight == 0:
            return 0
        try:
            if timeout is None:
                await self.flush()
            else:
                await asyncio.wait_for(self.flush(), timeout)
        except TimeoutError:
            logger.warning(
                "tokenizer drain timed out after %.1fs; %d items not counted",
                timeout,
                self._inflight,
            )
        return self._inflight
