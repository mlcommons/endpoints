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

"""Tokenization utilities for metrics aggregation."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import threading
from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

# Minimal user message used to satisfy chat templates that reject assistant-only
# message lists. Its token count is subtracted so only the assistant payload is
# measured.
_PREFIX_USER_MSG: dict[str, str] = {"role": "user", "content": ""}

# Coalescing batch defaults for the async text path. Concurrent token_count_async
# calls (one per COMPLETE event) are buffered and tokenized in one
# backend.encode_batch_fast call, which parallelizes across the batch via the
# Rust tokenizer's rayon pool. Batch size adapts to load — whatever accumulates
# while batches encode becomes the next batch — and is capped so a huge
# offline-drain backlog chunks into bounded batches.
_DEFAULT_MAX_BATCH_SIZE = 512
_DEFAULT_MAX_BATCH_DELAY_S = 0.01

# A single rayon pool saturates at ~8 threads for BPE tokenization (memory-bound;
# more threads oversubscribe and, on multi-socket Grace, cross the NUMA boundary
# into the first-touch heap). To use the whole machine we shard across worker
# *processes*, each pinned to its own block of this many cores so its rayon pool
# stays NUMA-local. Measured on GB200: 1 process ~1.5k items/s, 18 processes
# (×8 cores) ~25k items/s.
_DEFAULT_CORES_PER_WORKER = 8

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
# them by qualified name). Each worker process holds one raw tokenizers backend
# in a process global, pinned to a fixed core block.
# ---------------------------------------------------------------------------

_WORKER_BACKEND: Any = None


def _init_proc_worker(tokenizer_name: str, core_set: list[int]) -> None:
    """ProcessPoolExecutor initializer: pin affinity, then load the backend.

    Affinity is set before the first encode so the Rust rayon pool sizes itself
    to ``core_set`` (num_cpus respects sched_getaffinity on Linux) and stays on
    one NUMA node.
    """
    if core_set:
        try:
            os.sched_setaffinity(0, set(core_set))
        except (OSError, AttributeError):
            # Affinity not settable (non-Linux, restricted cpuset): rayon falls
            # back to all visible cores; correctness is unaffected.
            logger.debug("could not pin tokenizer worker to %s", core_set)
    transformers_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    global _WORKER_BACKEND
    _WORKER_BACKEND = getattr(tok, "backend_tokenizer", None)
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.encode("warmup", add_special_tokens=False)


def _proc_encode_batch_lengths(texts: list[str]) -> list[int]:
    """Process-worker task: per-text token counts via encode_batch_fast."""
    backend = _WORKER_BACKEND
    if backend is None:
        raise RuntimeError("tokenizer worker backend unavailable")
    encode_batch = getattr(backend, "encode_batch_fast", None)
    if encode_batch is None:
        encode_batch = backend.encode_batch
    return [len(e.ids) for e in encode_batch(texts, add_special_tokens=False)]


def _proc_ping() -> bool:
    """Warmup probe: returns True once the worker's backend is loaded."""
    return _WORKER_BACKEND is not None


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TokenizePool:
    """Tokenizes text for ISL/OSL/TPOT metrics, scaling across CPU cores.

    The async text path (``token_count_async`` — the hot path, one call per
    COMPLETE event) coalesces concurrent calls into batches and runs each batch
    through ``tokenizers.Tokenizer.encode_batch_fast``. Batches are fanned out
    across one or more *lanes*:

    - **Process lanes** (default when a fast tokenizer + multiple core blocks are
      available): ``available_cores // cores_per_worker`` worker processes, each
      pinned to its own core block. A single rayon pool tops out at ~8 cores for
      BPE, so sharding across pinned processes is the only way to use the whole
      machine (and each process's heap stays NUMA-local). Up to one batch per
      process runs concurrently.
    - **Thread lane** (fallback for slow tokenizers, single-core boxes, or
      ``cores_per_worker <= 0``): one in-flight batch on a worker thread, using a
      thread-local tokenizer. The Rust backend releases the GIL during encode.

    The sync ``token_count`` and the chat-template ``token_count_message*`` paths
    (used only when tool calls / reasoning are present) always run on the thread
    pool; they are rare relative to plain-text OSL.

    Thread-safety: the coalescing buffer and lane bookkeeping are touched only
    from the aggregator's single event-loop thread (``token_count_async`` and its
    flush callbacks all run there), so no lock is needed.
    """

    def __init__(
        self,
        tokenizer_name: str,
        n_workers: int,
        *,
        cores_per_worker: int = _DEFAULT_CORES_PER_WORKER,
        max_processes: int | None = None,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        max_batch_delay_s: float = _DEFAULT_MAX_BATCH_DELAY_S,
    ) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        self._tokenizer_name = tokenizer_name
        self._n_workers = n_workers
        self._thread_local = threading.local()
        self._fallback_warned: set[str] = set()
        self._max_batch_size = max_batch_size
        self._max_batch_delay_s = max_batch_delay_s

        # Thread executor: sync token_count, the message (tool-call) path, and the
        # async fallback / process-failure retry path.
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="TokenizePool",
        )
        # Pre-load a tokenizer on every worker thread so the first real call does
        # not pay the AutoTokenizer.from_pretrained cost. Submitting n_workers
        # tasks hits every thread because from_pretrained blocks long enough that
        # no thread finishes before all tasks are submitted. (Not a hard
        # guarantee — a mock tokenizer in tests must block ~100ms to simulate it;
        # an un-warmed thread merely pays the load cost on its first real call.)
        warmup = [
            self._executor.submit(self._get_thread_tokenizer) for _ in range(n_workers)
        ]
        try:
            for f in warmup:
                f.result()
        except Exception:
            self._executor.shutdown(wait=False)
            self._executor = None
            raise

        # Process lanes for the async batch path.
        self._proc_executors: list[ProcessPoolExecutor] = []
        self._setup_process_lanes(cores_per_worker, max_processes)

        # Lane bookkeeping for the async coalescer. Process mode dispatches the
        # picklable module function to the pinned worker pools; thread mode runs
        # the bound batch method on the shared thread executor.
        if self._proc_executors:
            self._lane_executors: list[Executor] = list(self._proc_executors)
            self._lane_fn: Any = _proc_encode_batch_lengths
            self._lane_is_process = True
        else:
            self._lane_executors = [self._executor]
            self._lane_fn = self._encode_batch_lengths
            self._lane_is_process = False
        self._free_lanes: deque[int] = deque(range(len(self._lane_executors)))
        self._pending_texts: list[str] = []
        self._pending_futs: list[asyncio.Future[int]] = []
        self._flush_handle: asyncio.TimerHandle | None = None

    # -- process-lane setup -------------------------------------------------

    def _setup_process_lanes(
        self, cores_per_worker: int, max_processes: int | None
    ) -> None:
        """Spawn and warm one pinned single-worker process per core block.

        No-op (leaving the pool in thread mode) when the tokenizer has no fast
        Rust backend, affinity is unavailable, or fewer than two core blocks fit
        — a single process is no faster than the thread lane, so it is not worth
        the spawn cost and memory.
        """
        if cores_per_worker <= 0:
            return
        if getattr(self._get_thread_tokenizer(), "backend_tokenizer", None) is None:
            # Slow tokenizer: encode_batch_fast unavailable, stay on threads.
            return
        try:
            available = sorted(os.sched_getaffinity(0))
        except (OSError, AttributeError):
            return
        n_proc = len(available) // cores_per_worker
        if max_processes is not None:
            n_proc = min(n_proc, max_processes)
        if n_proc < 2:
            return

        ctx = multiprocessing.get_context("spawn")
        executors: list[ProcessPoolExecutor] = []
        try:
            for i in range(n_proc):
                core_set = available[i * cores_per_worker : (i + 1) * cores_per_worker]
                ex = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_init_proc_worker,
                    initargs=(self._tokenizer_name, core_set),
                )
                executors.append(ex)
            # Force each worker to spawn + run its initializer (pin + load) now,
            # so the first real batch does not stall on tokenizer load.
            for ex in executors:
                ex.submit(_proc_ping).result()
        except Exception:
            for ex in executors:
                ex.shutdown(wait=False)
            logger.exception(
                "tokenizer process-lane setup failed; falling back to thread pool"
            )
            return
        self._proc_executors = executors
        logger.info(
            "TokenizePool: %d process lanes x %d cores (async batch tokenization)",
            n_proc,
            cores_per_worker,
        )

    # -- thread-local tokenizer + per-item workers --------------------------

    def _get_thread_tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer for the current thread, loading it if needed."""
        if getattr(self._thread_local, "tokenizer", None) is None:
            self._thread_local.tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name
            )
            # Baseline = tokens contributed by a [user, empty-assistant] pair minus
            # the [user] prefix alone. Some templates (Qwen3-Coder, etc.) reject
            # assistant-only message lists, so a user prefix is required; we
            # subtract it out so the baseline reflects only the assistant frame.
            try:
                tok = self._thread_local.tokenizer
                prefix_rendered = tok.apply_chat_template(
                    [_PREFIX_USER_MSG],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prefix_len = len(tok.tokenize(prefix_rendered))
                with_empty_assistant_rendered = tok.apply_chat_template(
                    [_PREFIX_USER_MSG, {"role": "assistant", "content": ""}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                with_empty_assistant_len = len(
                    tok.tokenize(with_empty_assistant_rendered)
                )
                self._thread_local.prefix_len = prefix_len
                self._thread_local.baseline = with_empty_assistant_len - prefix_len
            except Exception:
                self._thread_local.prefix_len = 0
                self._thread_local.baseline = 0
                logger.exception(
                    "Failed to compute chat-template baseline for %s; tool-call token counts may be over-estimated",
                    self._tokenizer_name,
                )
        return self._thread_local.tokenizer

    def _token_count_worker(self, text: str) -> int:
        """Worker entry: return the number of tokens in text."""
        tokenizer = self._get_thread_tokenizer()
        return len(tokenizer.tokenize(text))

    def _encode_batch_lengths(self, texts: list[str]) -> list[int]:
        """Thread-lane batch worker: per-text token counts in one call.

        Uses the raw ``tokenizers.Tokenizer`` backend's ``encode_batch_fast``
        (parallel across the batch on the rayon pool, no transformers wrapper).
        ``len(encode(...).ids)`` equals ``len(tokenize(...))`` for counting, so
        OSL/ISL/TPOT counts match the per-text path. Falls back to per-text
        tokenize for slow tokenizers with no fast backend.
        """
        tokenizer = self._get_thread_tokenizer()
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            encode_batch = getattr(backend, "encode_batch_fast", None)
            if encode_batch is None:
                encode_batch = backend.encode_batch
            return [len(e.ids) for e in encode_batch(texts, add_special_tokens=False)]
        return [len(tokenizer.tokenize(t)) for t in texts]

    def _token_count_message_worker(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
    ) -> int:
        """Worker entry: tokenize a full assistant message using apply_chat_template.

        Falls back to whitespace-split tokenization if apply_chat_template raises
        (e.g. the template does not support tool_calls or reasoning fields).
        """
        tokenizer = self._get_thread_tokenizer()
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = _normalize_tool_calls_for_template(tool_calls)
        try:
            rendered = tokenizer.apply_chat_template(
                [_PREFIX_USER_MSG, msg],
                tokenize=False,
                add_generation_prompt=False,
            )
            full = len(tokenizer.tokenize(rendered))
            prefix_len = getattr(self._thread_local, "prefix_len", 0)
            baseline = getattr(self._thread_local, "baseline", 0)
            return max(0, full - prefix_len - baseline)
        except Exception as exc:
            key = f"{self._tokenizer_name}:{type(exc).__name__}"
            if key not in self._fallback_warned:
                self._fallback_warned.add(key)
                logger.exception(
                    "apply_chat_template failed for %s (%s); falling back to "
                    "whitespace tokenization. Tool-call OSL/TPOT may diverge "
                    "from server-side counts for this run.",
                    self._tokenizer_name,
                    type(exc).__name__,
                )
            tool_calls_json = (
                msgspec.json.encode(list(tool_calls)).decode() if tool_calls else None
            )
            parts = [
                p for p in (content or None, reasoning or None, tool_calls_json) if p
            ]
            fallback_text = "\n".join(parts)
            return self._token_count_worker(fallback_text)

    # -- sync + message API (thread pool) -----------------------------------

    def token_count(self, text: str) -> int:
        """Return the number of tokens in the input string (blocking)."""
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        future = self._executor.submit(self._token_count_worker, text)
        return future.result()

    def token_count_message(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
    ) -> int:
        """Return the token count for an assistant message (blocking)."""
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        future = self._executor.submit(
            self._token_count_message_worker, content, reasoning, tool_calls
        )
        return future.result()

    async def token_count_message_async(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
        loop: asyncio.AbstractEventLoop,
    ) -> int:
        """Return the token count for an assistant message without blocking the event loop."""
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        return await loop.run_in_executor(
            self._executor,
            self._token_count_message_worker,
            content,
            reasoning,
            tool_calls,
        )

    # -- async text path (coalesced batch tokenization) ---------------------

    async def token_count_async(
        self, text: str, loop: asyncio.AbstractEventLoop
    ) -> int:
        """Return the number of tokens without blocking the event loop.

        Coalesces concurrent calls into batches dispatched across the lanes
        (pinned worker processes, or the thread lane as fallback). The caller
        awaits a future resolved when its batch completes. ``loop`` must be the
        single event loop that drives this pool.
        """
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        fut: asyncio.Future[int] = loop.create_future()
        self._pending_texts.append(text)
        self._pending_futs.append(fut)
        self._pump(loop)
        return await fut

    def _pump(self, loop: asyncio.AbstractEventLoop, *, force: bool = False) -> None:
        """Dispatch batches to free lanes; arm a timer for a partial remainder.

        Dispatches a full batch as soon as one is buffered and a lane is free
        (up to all lanes concurrently). A partial batch is only dispatched when
        ``force`` (timer fired), bounding small-batch overhead while keeping
        tail latency at ``max_batch_delay_s``."""
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        while self._free_lanes and self._pending_texts:
            if not force and len(self._pending_texts) < self._max_batch_size:
                break
            self._dispatch_one(loop)
        if self._pending_texts and self._free_lanes and self._flush_handle is None:
            self._flush_handle = loop.call_later(
                self._max_batch_delay_s, self._flush_partial, loop
            )

    def _flush_partial(self, loop: asyncio.AbstractEventLoop) -> None:
        """Timer callback: dispatch the buffered remainder even if under-size."""
        self._pump(loop, force=True)

    def _dispatch_one(self, loop: asyncio.AbstractEventLoop) -> None:
        """Slice up to ``max_batch_size`` pending texts onto one free lane."""
        lane = self._free_lanes.popleft()
        n = min(len(self._pending_texts), self._max_batch_size)
        texts = self._pending_texts[:n]
        futs = self._pending_futs[:n]
        self._pending_texts = self._pending_texts[n:]
        self._pending_futs = self._pending_futs[n:]
        loop.create_task(self._run_batch(lane, texts, futs, loop))

    async def _run_batch(
        self,
        lane: int,
        texts: list[str],
        futs: list[asyncio.Future[int]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            lengths = await loop.run_in_executor(
                self._lane_executors[lane], self._lane_fn, texts
            )
            self._resolve(futs, lengths)
        except Exception as exc:
            # A process lane can die (worker crash / OOM); retry the batch on the
            # thread lane so samples are never silently dropped.
            if self._lane_is_process and self._executor is not None:
                try:
                    lengths = await loop.run_in_executor(
                        self._executor, self._encode_batch_lengths, texts
                    )
                    self._resolve(futs, lengths)
                except Exception as retry_exc:
                    self._fail(futs, retry_exc)
            else:
                self._fail(futs, exc)
        finally:
            self._free_lanes.append(lane)
            self._pump(loop)

    @staticmethod
    def _resolve(futs: list[asyncio.Future[int]], lengths: list[int]) -> None:
        for f, n in zip(futs, lengths, strict=False):
            if not f.done():
                f.set_result(n)

    @staticmethod
    def _fail(futs: list[asyncio.Future[int]], exc: BaseException) -> None:
        for f in futs:
            if not f.done():
                f.set_exception(exc)

    def close(self) -> None:
        """Shut down all lanes. Idempotent."""
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        self._fail(self._pending_futs, RuntimeError("TokenizePool closed"))
        self._pending_texts = []
        self._pending_futs = []
        for ex in self._proc_executors:
            ex.shutdown(wait=False)
        self._proc_executors = []
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> TokenizePool:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
