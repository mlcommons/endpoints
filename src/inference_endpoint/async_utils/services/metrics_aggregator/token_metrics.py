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
per-sample text. The sharded pool is the drain-phase accelerator and is
auto-sized (one shard per core block); live mid-run flushes run on a small
in-process thread pool (``--tokenizer-workers``, default 2) owned by the
queue's live loop. A tokenizer without a fast (Rust) backend is a startup
error, never a silent slow path. Platforms without CPU affinity (e.g. macOS)
shard unpinned at full speed; only cache/NUMA locality is lost.
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
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol, cast

import msgspec
from inference_endpoint.endpoint_client.cpu_affinity import (
    cgroup_clamped_cpus,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging as transformers_logging

# A single rayon pool peaks at ~8 cores for BPE (memory-bound; more threads
# oversubscribe and, on multi-socket Grace, cross the NUMA boundary). Sharding
# across processes pinned to disjoint 8-core blocks is how the whole machine is
# used. Measured on GB200: ~16k texts/s at 18 blocks vs ~1.5k single-process.
CORES_PER_WORKER = 8

# Budget for the parallel shard warmup (spawn + transformers import +
# tokenizer load per worker). A hung load (e.g. a stuck network filesystem)
# must become a bounded startup error, not wedge service startup — and the
# error must fire before the parent's 30 s service-launch budget kills the
# subprocess, so the diagnostic wins the race.
_SHARD_WARMUP_TIMEOUT_S = 25.0

# Per-flush ceiling for the LIVE lane. Bounds three things at once: how long
# the queue lock is held mid-run, how much work an unstoppable in-flight
# thread encode can hold after a drain-start cancellation, and how much the
# drain re-encodes for items the cancelled flush gave back. The drain has no
# ceiling — it always takes the whole buffer.
_LIVE_FLUSH_MAX_ITEMS = 1024

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
        # Size the rayon pool to the block explicitly: the parent process caps
        # its own pool for the live lane, and spawn children inherit that env —
        # without the override every shard would run at the live-lane width.
        os.environ["RAYON_NUM_THREADS"] = str(len(core_set))
        try:
            os.sched_setaffinity(0, set(core_set))
        except (OSError, AttributeError):
            # No pinning (e.g. macOS): the rayon cap above still keeps
            # unpinned shards from oversubscribing each other.
            logger.debug("could not pin tokenizer worker to %s", core_set)
    transformers_logging.set_verbosity_error()
    # trust_remote_code is required: supported tokenizers (e.g. DeepSeek-R1)
    # ship custom code. The tokenizer name is operator-supplied (not a new
    # trust boundary vs. the pre-PR main-process load), but sharding now
    # executes that code in each worker process.
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
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
    """Best-effort immediate stop: SIGTERM shard workers, then close executors.

    ``shutdown(wait=False)`` alone leaves an in-flight (or init-hung) encode
    running, and the executor's atexit handler still *joins* the non-daemon
    worker — so a drain timeout would stall interpreter exit until the chunk
    finished. SIGTERM the workers first so that join returns promptly.

    The aggregator subprocess's only multiprocessing children are these shard
    workers (ZMQ runs on threads), so ``active_children()`` — public API — is
    exactly that set, and unlike ``ProcessPoolExecutor._processes`` it also
    includes a worker still hung in its initializer.
    """
    for child in multiprocessing.active_children():
        try:
            child.terminate()
        except (OSError, ValueError):  # already dead / not yet started
            pass
    for ex in procs:
        ex.shutdown(wait=False, cancel_futures=True)


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
        live_workers: int,
        cores_per_worker: int = CORES_PER_WORKER,
        n_workers: int = -1,
    ) -> None:
        self._tokenizer_name = tokenizer_name
        # The live lane runs in-process: cap this process's rayon pool so a
        # mid-run batched encode uses ~live_workers cores, not the whole
        # machine. Must be set before the first encode initializes the pool;
        # setdefault lets an operator-exported RAYON_NUM_THREADS win.
        os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, live_workers)))
        self._fallback_warned: set[str] = set()
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._prefix_len = 0
        self._baseline = 0
        # In-process threads: the live token-metric lane plus the
        # chat-template path.
        self._thread: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max(1, live_workers), thread_name_prefix="tok-thread"
        )
        self._load_tokenizer()  # also computes the chat-template baseline
        # Process shards for the batched text path. Empty only when
        # in-process mode was explicitly requested (n_workers=0 or
        # cores_per_worker<=0; ctor overrides used primarily by tests —
        # production wiring passes live_workers only and shards auto-size).
        self._procs: list[ProcessPoolExecutor] = []
        self._setup_shards(cores_per_worker, n_workers)

    # -- setup --------------------------------------------------------------

    def _load_tokenizer(self) -> None:
        try:
            tok = AutoTokenizer.from_pretrained(
                self._tokenizer_name, trust_remote_code=True
            )
        except Exception:
            # Some checkpoints (e.g. DeepSeek-V4) fail AutoTokenizer's config
            # detection; fall back to loading the fast tokenizer directly.
            tok = PreTrainedTokenizerFast.from_pretrained(self._tokenizer_name)
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
        process's affinity mask (or the online CPU count when the platform
        has no affinity API — shards then run unpinned), always at least one;
        an explicit count is clamped to that capacity. An environment that
        cannot shard — no fast Rust backend, a warmup that fails or exceeds
        its budget — raises instead of silently degrading to a slow path
        that cannot keep up with completions.
        """
        if cores_per_worker <= 0 or n_workers == 0:
            logger.info("BatchTokenizer: in-process tokenization (explicit)")
            return
        if getattr(self._tokenizer, "backend_tokenizer", None) is None:
            raise RuntimeError(
                f"tokenizer {self._tokenizer_name!r} has no fast (Rust) "
                "backend; token metrics require one to keep up with "
                "completions. Use a fast tokenizer, or disable token metrics."
            )
        # The full allowed CPU universe (cgroup-clamped) drives the shard block
        # math. cgroup_clamped_cpus owns the probe-and-restore of this process's
        # mask, so the aggregator's event loop, publisher, and live tokenizer
        # threads stay exactly where the parent placed them (the loadgen mask on
        # a pinned Linux run); only the drain-phase shard processes, each pinned
        # to its own block, span the machine.
        available = cgroup_clamped_cpus()
        if available is None:
            # No affinity API (e.g. macOS): shard unpinned — the OS scheduler
            # spreads the workers, only cache/NUMA locality is lost. Workers cap
            # their rayon pools to the block size instead (_init_worker).
            available = list(range(os.cpu_count() or 1))
            logger.info("BatchTokenizer: CPU affinity unavailable; sharding unpinned")
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
                "slow path that cannot keep up with completions. Fix the "
                "environment (see the chained error)."
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
        self,
        texts: list[str],
        loop: asyncio.AbstractEventLoop,
        /,
        *,
        live: bool = False,
    ) -> list[int]:
        """Per-text token counts for a whole batch without blocking the loop.

        ``live=True`` is the mid-run lane: it never touches the shard
        processes — it runs on this process's small thread pool with a rayon
        pool capped to ``live_workers`` cores. The default (drain) path fans
        out across every shard; a worker-shard failure propagates and is
        treated as an incomplete drain.
        """
        if not texts:
            return []
        if self._procs and not live:
            return await self._fan_out(self._procs, texts)
        if self._thread is None:
            raise RuntimeError("BatchTokenizer is closed")
        return await loop.run_in_executor(
            self._thread, self._encode_lengths_inproc, texts
        )

    @staticmethod
    async def _fan_out(procs: list[ProcessPoolExecutor], texts: list[str]) -> list[int]:
        chunks = _even_chunks(texts, len(procs))
        futures = [
            asyncio.wrap_future(ex.submit(_worker_encode_lengths, chunk))
            for ex, chunk in zip(procs, chunks, strict=False)
        ]
        results = await asyncio.gather(*futures)
        return list(chain.from_iterable(results))

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
        /,
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
        stall interpreter exit after a drain timeout. The live thread pool
        drops queued encodes (``cancel_futures``); only a single in-flight
        encode (bounded by ``_LIVE_FLUSH_MAX_ITEMS``) is waited on.
        """
        _terminate_procs(self._procs)
        self._procs = []
        if self._thread is not None:
            self._thread.shutdown(wait=True, cancel_futures=True)
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
        self,
        texts: list[str],
        loop: asyncio.AbstractEventLoop,
        /,
        *,
        live: bool = False,
    ) -> list[int]:
        """Per-text token counts (``live=True`` = the bounded mid-run lane)."""
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
    ``on_count`` callback that records the resulting metric. The queue owns
    its own flush cadence: ``start_live`` begins a periodic flush through the
    tokenizer's bounded live lane (so live ISL/OSL/TPOT stay current without
    touching the benchmark's cores), and ``flush_remaining`` drains everything
    left at end-of-run through every shard.

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
        self._live_task: asyncio.Task | None = None
        # Serializes flushes so the periodic live flush and the end-of-run
        # flush never record the same item twice or race on the pending count.
        self._lock = asyncio.Lock()

    def start_live(self, interval_s: float) -> None:
        """Begin the periodic live flush (idempotent).

        Failures are logged once and never interrupt the loop — unflushed
        items stay visible as ``pending`` and the end-of-run drain picks
        them up.
        """
        if self._live_task is not None:
            return
        self._live_task = self._loop.create_task(self._live_flush_loop(interval_s))

    async def _live_flush_loop(self, interval_s: float) -> None:
        failure_logged = False
        while True:
            await asyncio.sleep(interval_s)
            try:
                await self.flush_live_once()
            except Exception:  # noqa: BLE001 — keep live metrics flowing.
                if not failure_logged:
                    failure_logged = True
                    logger.exception(
                        "live token flush failed; retrying each interval "
                        "(further failures logged at debug)"
                    )
                else:
                    logger.debug("live token flush failed again")

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

    async def flush_live_once(self) -> None:
        """One bounded mid-run flush (live lane).

        Routes text batches through the tokenizer's bounded live lane instead
        of the full shard pool, takes at most ``_LIVE_FLUSH_MAX_ITEMS`` per
        kind (bounding lock-hold time and the unstoppable in-flight encode a
        drain-start cancellation leaves behind), and re-queues items on failure
        or cancellation so a mid-run hiccup never loses samples — the
        end-of-run drain retries them.
        """
        await self._flush(live=True)

    async def drain_all(self) -> None:
        """End-of-run drain: tokenize the whole buffer through the shard pool.

        Failures are terminal: un-recorded items stay counted in ``pending``
        (``_inflight`` is decremented only after a callback runs) and surface
        as an incomplete drain, not as silently dropped samples.
        """
        await self._flush(live=False)

    async def _flush(self, live: bool) -> None:
        """Tokenize everything buffered so far and run each ``on_count``.

        Items are detached from the buffer up front so concurrent enqueues land
        in the next flush. Callers use ``flush_live_once`` / ``drain_all``.
        """
        async with self._lock:
            if not (self._text or self._msg):
                return
            if live:
                cap = _LIVE_FLUSH_MAX_ITEMS
                text_items = self._text[:cap]
                del self._text[:cap]  # in-place: O(cap), not O(backlog).
                msg_items = self._msg[:cap]
                del self._msg[:cap]
            else:
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
                        [t for t, _ in text_items], self._loop, live=live
                    )
                except asyncio.CancelledError:
                    if live:
                        self._text[:0] = text_items
                        self._msg[:0] = msg_items
                    raise
                except Exception as exc:  # noqa: BLE001 — isolate phases.
                    failure = exc
                    if live:
                        # A live hiccup must not lose samples: give the items
                        # back so the end-of-run drain (full pool) retries.
                        # Drain failures are terminal and stay pending-only.
                        self._text[:0] = text_items
                else:
                    if len(counts) != len(text_items):
                        # Tokenizer contract violation (wrong-length result).
                        # Treat it like any other text-phase failure so the
                        # message phase still runs: re-queue live items, leave
                        # drain items pending (terminal), re-raise after both.
                        failure = RuntimeError(
                            f"tokenizer returned {len(counts)} counts for "
                            f"{len(text_items)} texts"
                        )
                        if live:
                            self._text[:0] = text_items
                    else:
                        for (_, on_count), count in zip(
                            text_items, counts, strict=False
                        ):
                            self._record(on_count, count)
            for i, ((content, reasoning, tool_calls), on_count) in enumerate(msg_items):
                try:
                    count = await self._tokenizer.token_count_message_async(
                        content, reasoning, tool_calls, self._loop
                    )
                except asyncio.CancelledError:
                    if live:
                        self._msg[:0] = msg_items[i:]
                    raise
                except Exception as exc:  # noqa: BLE001 — isolate items.
                    failure = failure or exc
                    if live:
                        self._msg.append(((content, reasoning, tool_calls), on_count))
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

        Stops the live flush loop, then drains through the full shard pool.
        Returns the number of items still un-tokenized — non-zero if the budget
        was exhausted (``timeout`` reached) or tokenization failed. ``None``
        waits indefinitely. Never raises on a tokenizer failure: it must not
        stop the aggregator from publishing the (incomplete) final snapshot.
        The one exception that propagates is ``CancelledError`` (e.g. the
        outer ``process()`` task being cancelled on shutdown) — cancellation
        is control flow, not a drain failure, and is re-raised so teardown
        proceeds.
        """
        if self._live_task is not None:
            self._live_task.cancel()
            await asyncio.gather(self._live_task, return_exceptions=True)
            self._live_task = None
        if self._inflight == 0:
            return 0
        try:
            if timeout is None:
                await self.drain_all()
            else:
                await asyncio.wait_for(self.drain_all(), timeout)
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
