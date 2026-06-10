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

``BatchTokenizer`` tokenizes whole batches at once, sharded across worker
processes each pinned to a block of ``CORES_PER_WORKER`` cores (a single BPE
rayon pool is memory-bound and saturates ~8 cores). The aggregator buffers
per-sample text and flushes the batch once per publish tick and at drain.
Sharding requires the fast (Rust) tokenizers backend and Linux CPU affinity;
an environment that cannot shard is a startup error, never a silent slow
path — ``--tokenizer-workers 0`` is the only (explicit) in-process mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import signal
import time
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

# Budget for the parallel shard warmup (spawn + transformers import +
# tokenizer load per worker). A hung load (e.g. a stuck network filesystem)
# must degrade to the in-process path, not wedge service startup.
_SHARD_WARMUP_TIMEOUT_S = 120.0

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
    # Ctrl-C sends SIGINT to the whole foreground process group; the parent
    # drives worker shutdown, so a worker dying mid-drain would break the pool
    # and lose the buffered tokenizations it was counting.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
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


def _encode_batch_lengths(backend: Any, texts: list[str]) -> list[int]:
    """Per-text token counts via the raw tokenizers backend, one rayon call."""
    encode_batch = getattr(backend, "encode_batch_fast", None) or backend.encode_batch
    return [len(e.ids) for e in encode_batch(texts, add_special_tokens=False)]


def _worker_encode_lengths(texts: list[str]) -> list[int]:
    """Per-text token counts for a shard, in one rayon-parallel call."""
    backend = _WORKER_BACKEND
    if backend is None:
        raise RuntimeError("tokenizer worker backend unavailable")
    return _encode_batch_lengths(backend, texts)


def _worker_ready(_: int) -> bool:
    """Warmup probe: returns once the worker's backend is loaded."""
    return _WORKER_BACKEND is not None


def _terminate_procs(procs: list[ProcessPoolExecutor]) -> None:
    """Best-effort immediate stop: cancel queued work and SIGTERM workers.

    ``shutdown(wait=False)`` alone leaves an in-flight encode running, and the
    non-daemon worker would still be joined at interpreter exit — so a drain
    timeout could stall process shutdown until the chunk finished.
    """
    for ex in procs:
        ex.shutdown(wait=False, cancel_futures=True)
        workers = getattr(ex, "_processes", None) or {}  # CPython impl detail.
        for p in workers.values():
            try:
                p.terminate()
            except Exception:  # noqa: BLE001 — already-dead workers are fine.
                pass


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

    ``count_texts_async`` tokenizes a whole list in one sharded call. The
    chat-template ``token_count_message_async`` path runs on a small in-process
    thread — rare (tool calls) relative to the batched OSL/ISL/TPOT flush.
    """

    def __init__(
        self,
        tokenizer_name: str,
        *,
        cores_per_worker: int = CORES_PER_WORKER,
        n_workers: int = -1,
    ) -> None:
        self._tokenizer_name = tokenizer_name
        self._fallback_warned: set[str] = set()
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._prefix_len = 0
        self._baseline = 0
        # In-process thread for the chat-template path.
        self._thread: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tok-thread"
        )
        self._load_tokenizer()  # also computes the chat-template baseline
        # Process shards for the batched text path. Empty only when
        # in-process mode was explicitly requested (n_workers=0).
        self._procs: list[ProcessPoolExecutor] = []
        self._setup_shards(cores_per_worker, n_workers)

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

    def _setup_shards(self, cores_per_worker: int, n_workers: int) -> None:
        """Spawn one pinned single-worker process per core block.

        ``n_workers == 0`` explicitly selects in-process tokenization. Auto
        (``< 0``) fits one shard per ``cores_per_worker`` block of this
        process's affinity mask, always at least one; an explicit count is
        clamped to that capacity. An environment that cannot shard — no fast
        Rust backend, no CPU affinity, a warmup that fails or exceeds its
        budget — raises instead of silently degrading to a slow path that
        cannot keep up with completions.
        """
        if cores_per_worker <= 0 or n_workers == 0:
            logger.info("BatchTokenizer: in-process tokenization (explicit)")
            return
        if getattr(self._tokenizer, "backend_tokenizer", None) is None:
            raise RuntimeError(
                f"tokenizer {self._tokenizer_name!r} has no fast (Rust) "
                "backend; token metrics require one to keep up with "
                "completions. Pass --tokenizer-workers 0 to explicitly run "
                "single-threaded in-process tokenization."
            )
        try:
            available = sorted(os.sched_getaffinity(0))
        except (OSError, AttributeError) as exc:
            raise RuntimeError(
                "CPU affinity is unavailable; tokenizer sharding requires "
                "Linux. Pass --tokenizer-workers 0 to explicitly run "
                "in-process tokenization."
            ) from exc
        capacity = max(1, len(available) // cores_per_worker)
        n = capacity if n_workers < 0 else min(n_workers, capacity)
        t0 = time.perf_counter()
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
            # The wait is bounded: one hung load must not wedge startup.
            ready = [ex.submit(_worker_ready, 0) for ex in procs]
            deadline = time.monotonic() + _SHARD_WARMUP_TIMEOUT_S
            for f in ready:
                f.result(timeout=max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            _terminate_procs(procs)
            raise RuntimeError(
                "tokenizer shard warmup failed; refusing to fall back to a "
                "slow path. Fix the environment (or pass --tokenizer-workers "
                "0 to explicitly run in-process)."
            ) from exc
        self._procs = procs
        logger.info(
            "BatchTokenizer: %d shards x %d cores (setup %.1fs)",
            len(procs),
            cores_per_worker,
            time.perf_counter() - t0,
        )

    # -- batched text path --------------------------------------------------

    def _encode_lengths_inproc(self, texts: list[str]) -> list[int]:
        tok = self._tokenizer
        backend = getattr(tok, "backend_tokenizer", None)
        if backend is not None:
            return _encode_batch_lengths(backend, texts)
        return [len(tok.tokenize(t)) for t in texts]  # type: ignore[union-attr]

    async def count_texts_async(
        self, texts: list[str], loop: asyncio.AbstractEventLoop
    ) -> list[int]:
        """Per-text token counts for a whole batch without blocking the loop.

        A worker-shard failure propagates and is treated as an incomplete drain.
        """
        if not texts:
            return []
        if self._procs:
            chunks = _even_chunks(texts, len(self._procs))
            futures = [
                asyncio.wrap_future(ex.submit(_worker_encode_lengths, chunk))
                for ex, chunk in zip(self._procs, chunks, strict=False)
            ]
            results = await asyncio.gather(*futures)
            return [n for r in results for n in r]
        if self._thread is None:
            raise RuntimeError("BatchTokenizer is closed")
        return await loop.run_in_executor(
            self._thread, self._encode_lengths_inproc, texts
        )

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
        """Shut down all workers. Idempotent.

        Shards are stopped without waiting (a hung worker must not block
        aggregator shutdown) and terminated so an in-flight encode cannot
        stall interpreter exit after a drain timeout.
        """
        _terminate_procs(self._procs)
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
MessageParts = tuple[str, str | None, tuple[dict[str, Any], ...] | None]


class TokenCounter(Protocol):
    """The async tokenization surface ``TokenBatchQueue`` depends on.

    ``BatchTokenizer`` satisfies this structurally; tests pass lightweight
    stubs. Declared as a Protocol so the queue is decoupled from the concrete
    tokenizer and test doubles type-check without inheritance.
    """

    async def count_texts_async(
        self, texts: list[str], loop: asyncio.AbstractEventLoop, /
    ) -> list[int]:
        """Per-text token counts for a whole batch."""
        raise NotImplementedError

    async def token_count_message_async(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
        loop: asyncio.AbstractEventLoop,
        /,
    ) -> int:
        """Chat-template token count for one assistant message."""
        raise NotImplementedError


class TokenBatchQueue:
    """Buffers per-sample tokenization work and clears it in batches.

    Triggers call ``enqueue_text`` / ``enqueue_message`` at event time with an
    ``on_count`` callback that records the resulting metric. The aggregator
    flushes the buffer with ``flush`` once per publish tick (so live ISL/OSL/
    TPOT stay current) and with ``flush_remaining`` at end-of-run, sending the
    whole batch through ``BatchTokenizer`` in one sharded call.

    ``pending`` counts enqueued-but-not-yet-recorded items; it is the
    ``n_pending_tasks`` on the snapshot. A non-zero value in the final snapshot
    means the end-of-run flush did not finish within the drain budget or failed.
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
        runs, so a cancellation (drain timeout) or a tokenizer error leaves it
        reflecting exactly the items that were not recorded — those surface as
        ``pending`` (an incomplete drain), not as silently dropped samples.
        """
        async with self._lock:
            if not (self._text or self._msg):
                return
            text_items, self._text = self._text, []
            msg_items, self._msg = self._msg, []
            # The text and message phases fail independently — they run on
            # separate executors, so a dead text shard must not drop message
            # items that would still succeed (and vice versa). The first
            # failure is re-raised after both phases so callers still see it.
            failure: Exception | None = None
            if text_items:
                try:
                    counts = await self._tokenizer.count_texts_async(
                        [t for t, _ in text_items], self._loop
                    )
                except Exception as exc:  # noqa: BLE001 — isolate phases.
                    failure = exc
                else:
                    for (_, on_count), count in zip(text_items, counts, strict=True):
                        self._record(on_count, count)
            for (content, reasoning, tool_calls), on_count in msg_items:
                try:
                    count = await self._tokenizer.token_count_message_async(
                        content, reasoning, tool_calls, self._loop
                    )
                except Exception as exc:  # noqa: BLE001 — isolate items.
                    failure = failure or exc
                    continue
                self._record(on_count, count)
            if failure is not None:
                raise failure

    def _record(self, on_count: Callable[[int], None], count: int) -> None:
        """Run one recorder callback; a raising recorder must not poison the
        rest of the batch, and the item still counts as recorded."""
        try:
            on_count(count)
        except Exception:  # noqa: BLE001 — per-item isolation.
            logger.exception("token metric recorder failed")
        finally:
            self._inflight -= 1

    async def flush_remaining(self, timeout: float | None) -> int:
        """End-of-run flush, bounded by ``timeout`` seconds.

        Returns the number of items still un-tokenized — non-zero if the budget
        was exhausted (``timeout`` reached) or tokenization failed. ``None``
        waits indefinitely. Never raises: a failure here must not stop the
        aggregator from publishing the (incomplete) final snapshot.
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
        except Exception:  # noqa: BLE001 — drain must not block finalize.
            logger.exception(
                "tokenizer drain failed; %d items not counted", self._inflight
            )
        return self._inflight
