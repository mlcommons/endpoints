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
backend pool is memory-bound and saturates ~8 cores). The aggregator buffers
per-sample text. The sharded pool is the drain-phase accelerator and is
auto-sized (one shard per core block); live mid-run flushes run on a small
in-process thread pool (``--tokenizer-workers``, default 4) owned by the
queue's live loop. Plain text counting uses a Hugging Face fast tokenizer's
Rust backend. Structured chat counting uses the full tokenizer's
``apply_chat_template`` path and does not require that backend. Platforms
without CPU affinity (e.g. macOS) shard unpinned at full speed; only cache/NUMA
locality is lost.
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
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    PromptInput,
    TextInput,
    TokenIdsInput,
    TokenizationInput,
)
from inference_endpoint.endpoint_client.cpu_affinity import (
    cgroup_clamped_cpus,
)
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

# CORES_PER_WORKER bounds each Rayon BPE pool: larger pools oversubscribe this
# memory-bound workload and can cross NUMA boundaries. Multiple processes pinned
# to disjoint CORES_PER_WORKER-sized blocks scale across the allowed CPU set.
CORES_PER_WORKER = 8

# DRAIN_RESERVED_CPUS keeps part of the CPU set out of the metrics-drain shard
# pool so the parent, aggregator event loop, and host remain responsive.
DRAIN_RESERVED_CPUS = 2

# _SHARD_WARMUP_TIMEOUT_S bounds parallel shard setup (spawn, imports, and
# tokenizer load) so a hung worker cannot wedge this service indefinitely.
# The parent's configurable service_ready_timeout_s independently bounds the
# overall service launch and may expire first.
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


def _normalize_prompt_messages_for_template(
    messages: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Normalize historical tool calls without mutating the event payload."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list | tuple) or not tool_calls:
            normalized.append(message)
            continue
        normalized.append(
            {
                **message,
                "tool_calls": _normalize_tool_calls_for_template(tool_calls),
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Process-worker entry points (module-level so ProcessPoolExecutor can pickle
# them by name). Each worker holds one raw tokenizers backend, pinned to a
# fixed core block.
# ---------------------------------------------------------------------------

_WORKER_TEXT_BACKEND: Any = None


def load_reference_tokenizer(tokenizer_name: str) -> Any:
    """Load the run's reference tokenizer.

    ``trust_remote_code`` is required for tokenizers that ship custom code (e.g.
    DeepSeek-R1). The single construction point shared by the perf-window
    tokenizer, the sharded length-counting workers, and finalize-side accuracy
    OSL, so every OSL number in a run comes from the same tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def load_reference_backend(tokenizer_name: str) -> Any | None:
    """Load the optional fast backend used only for plain-text counting."""
    tokenizer = load_reference_tokenizer(tokenizer_name)
    return getattr(tokenizer, "backend_tokenizer", None)


def _init_worker(tokenizer_name: str, core_set: list[int]) -> None:
    """Pin this worker to ``core_set``, then load its token-counting path.

    Affinity is set before the first encode so the Hugging Face rayon pool sizes
    itself to the pinned core count (num_cpus respects sched_getaffinity on
    Linux).
    """
    # Ctrl-C sends SIGINT to the whole foreground process group; the parent
    # drives worker shutdown, so a worker dying mid-drain would break the pool
    # and lose the buffered tokenizations it was counting.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if core_set:
        # Size the Hugging Face rayon pool to the block explicitly: the parent
        # process caps its own pool for the live lane, and spawn children inherit
        # that env — without the override every shard would run at the live-lane
        # width.
        os.environ["RAYON_NUM_THREADS"] = str(len(core_set))
        try:
            os.sched_setaffinity(0, set(core_set))
        except (OSError, AttributeError):
            # No pinning (e.g. macOS): the rayon cap above still keeps
            # unpinned shards from oversubscribing each other.
            logger.debug("could not pin tokenizer worker to %s", core_set)
    transformers_logging.set_verbosity_error()
    global _WORKER_TEXT_BACKEND
    _WORKER_TEXT_BACKEND = load_reference_backend(tokenizer_name)
    if _WORKER_TEXT_BACKEND is not None:
        _WORKER_TEXT_BACKEND.encode("warmup", add_special_tokens=False)


def encode_lengths(backend: Any, texts: list[str]) -> list[int]:
    """Per-text token counts via one bounded backend batch call."""
    encode_batch = getattr(backend, "encode_batch_fast", None) or backend.encode_batch
    encoded = encode_batch(texts, add_special_tokens=False)
    return [len(item.ids) for item in encoded]


def _worker_encode_lengths(texts: list[str]) -> list[int]:
    """Per-text token counts for a shard, in one rayon-parallel call."""
    backend = _WORKER_TEXT_BACKEND
    if backend is None:
        raise RuntimeError("tokenizer worker backend unavailable")
    return encode_lengths(backend, texts)


def _worker_ready(_: int) -> bool:
    """Warmup probe: returns once the worker's backend is loaded."""
    return _WORKER_TEXT_BACKEND is not None


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

    ``count_batch_async`` explicitly routes token IDs, text, assistant
    messages, and complete prompts to their corresponding counting path.
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
        # The live lane runs in-process: cap the Hugging Face rayon pool before
        # its first encode. setdefault lets an operator-exported HF cap win.
        os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, live_workers)))
        self._fallback_warned: set[str] = set()
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._text_backend: Any | None = None
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
        transformers_logging.set_verbosity_error()
        tok = load_reference_tokenizer(self._tokenizer_name)
        self._tokenizer = tok
        self._text_backend = getattr(tok, "backend_tokenizer", None)
        # Baseline = tokens from a [user, empty-assistant] pair minus the [user]
        # prefix alone, so the assistant frame is subtracted from message counts.
        try:
            prefix = cast(
                list[int],
                tok.apply_chat_template(
                    [_PREFIX_USER_MSG],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False,
                ),
            )
            self._prefix_len = len(prefix)
            with_assistant_tokens = cast(
                list[int],
                tok.apply_chat_template(
                    [_PREFIX_USER_MSG, {"role": "assistant", "content": ""}],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False,
                ),
            )
            self._baseline = len(with_assistant_tokens) - self._prefix_len
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
        (``< 0``) fits one shard per ``cores_per_worker`` block after reserving
        ``DRAIN_RESERVED_CPUS`` from this process's affinity mask (or the online
        CPU count when the platform has no affinity API — shards then run unpinned).
        The final block may be partial; at least one CPU remains usable. An
        explicit count is clamped to that capacity. A tokenizer without a fast
        text backend skips shard creation; structured chat tokenization remains
        available. A shard warmup failure or timeout raises at startup.
        """
        if cores_per_worker <= 0 or n_workers == 0:
            logger.info("BatchTokenizer: in-process tokenization (explicit)")
            return
        if self._text_backend is None:
            logger.info(
                "BatchTokenizer: no fast backend for %s; using the tokenizer "
                "wrapper for in-process plain-text tokenization",
                self._tokenizer_name,
            )
            return
        # Reserve DRAIN_RESERVED_CPUS from the cgroup-clamped CPU universe for
        # the parent, aggregator event loop, and system responsiveness.
        # cgroup_clamped_cpus owns the probe-and-restore of this process's mask;
        # only drain-phase shard processes span the usable CPUs.
        available = cgroup_clamped_cpus()
        if available is None:
            # No affinity API (e.g. macOS): shard unpinned — the OS scheduler
            # spreads the workers, only cache/NUMA locality is lost. Workers cap
            # their rayon pools to the block size instead (_init_worker).
            available = list(range(os.cpu_count() or 1))
            logger.info("BatchTokenizer: CPU affinity unavailable; sharding unpinned")
        usable = available[: max(1, len(available) - DRAIN_RESERVED_CPUS)]
        blocks = [
            usable[start : start + cores_per_worker]
            for start in range(0, len(usable), cores_per_worker)
        ]
        capacity = len(blocks)
        n = capacity if n_workers < 0 else min(n_workers, capacity)
        t0 = time.perf_counter()
        ctx = multiprocessing.get_context("spawn")
        procs: list[ProcessPoolExecutor] = []
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for block in blocks[:n]:
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
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        self._procs = procs
        logger.info(
            "BatchTokenizer: %d shards across %d CPUs (setup %.1fs)",
            len(procs),
            sum(len(block) for block in blocks[:n]),
            time.perf_counter() - t0,
        )

    # -- batched text path --------------------------------------------------

    def _encode_lengths_inproc(self, texts: list[str]) -> list[int]:
        backend = self._text_backend
        if backend is not None:
            return encode_lengths(backend, texts)
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("BatchTokenizer is closed")
        return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]

    async def _count_texts_async(
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

    def _warn_template_fallback(self, exc: Exception, impact: str) -> None:
        """Log a chat-template fallback once per tokenizer and error type."""
        key = f"{self._tokenizer_name}:{type(exc).__name__}"
        if key in self._fallback_warned:
            return
        self._fallback_warned.add(key)
        logger.exception(
            "apply_chat_template failed for %s (%s); falling back to "
            "whitespace tokenization. %s",
            self._tokenizer_name,
            type(exc).__name__,
            impact,
        )

    def _token_count_message(
        self,
        content: str,
        reasoning: str | None,
        tool_calls: tuple[dict[str, Any], ...] | None,
    ) -> int:
        """Count one assistant output without surrounding chat-template framing.

        Render the structured assistant content, reasoning, and tool calls with
        a minimal user prefix, then subtract both that prefix and the empty
        assistant frame. The result is the assistant payload count used for
        OSL and TPOT.
        """
        tok = self._tokenizer
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = _normalize_tool_calls_for_template(tool_calls)
        try:
            encoded = tok.apply_chat_template(  # type: ignore[union-attr]
                [_PREFIX_USER_MSG, msg],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            return max(0, len(encoded) - self._prefix_len - self._baseline)
        except Exception as exc:
            self._warn_template_fallback(exc, "Tool-call OSL/TPOT may diverge.")
            tool_calls_json = (
                msgspec.json.encode(list(tool_calls)).decode() if tool_calls else None
            )
            parts = [
                p for p in (content or None, reasoning or None, tool_calls_json) if p
            ]
            return self._token_count_text("\n".join(parts))

    def _token_count_prompt(
        self,
        messages: tuple[dict[str, Any], ...],
        tools: tuple[dict[str, Any], ...] | None,
        chat_template_kwargs: dict[str, Any] | None,
        chat_template: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> int:
        """Count a complete structured input prompt for ISL.

        Render the full message history and optional tools using the selected
        chat template and model-specific keyword arguments. Unlike assistant
        output counting, this keeps all conversation framing and appends the
        generation prompt because those tokens are part of the server input.
        """
        kwargs = dict(chat_template_kwargs or {})
        kwargs.update(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if tools is not None:
            kwargs["tools"] = list(tools)
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        prompt_messages = _normalize_prompt_messages_for_template(messages)
        try:
            encoded = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                prompt_messages, **kwargs
            )
            return len(encoded)
        except Exception as exc:
            self._warn_template_fallback(exc, "Structured ISL may diverge.")
            prompt = {"messages": prompt_messages}
            if tools is not None:
                prompt["tools"] = list(tools)
            return self._token_count_text(msgspec.json.encode(prompt).decode())

    async def _count_indexed_texts_async(
        self,
        indexed_texts: list[tuple[int, str]],
        loop: asyncio.AbstractEventLoop,
        *,
        live: bool,
    ) -> list[tuple[int, int | Exception]]:
        """Count one text batch and pair each outcome with its input index."""
        texts = [text for _, text in indexed_texts]
        try:
            counts = await self._count_texts_async(texts, loop, live=live)
        except Exception as exc:  # noqa: BLE001 - isolate this input kind.
            return [(index, exc) for index, _ in indexed_texts]

        if len(counts) != len(indexed_texts):
            length_error = RuntimeError(
                f"tokenizer returned {len(counts)} counts for "
                f"{len(indexed_texts)} texts"
            )
            return [(index, length_error) for index, _ in indexed_texts]

        return [
            (index, count)
            for (index, _), count in zip(indexed_texts, counts, strict=True)
        ]

    async def count_batch_async(
        self,
        inputs: list[TokenizationInput],
        loop: asyncio.AbstractEventLoop,
        /,
        *,
        live: bool = False,
    ) -> list[int | Exception]:
        """Count a mixed batch while preserving input order."""
        outcomes: list[int | Exception | None] = [None] * len(inputs)
        indexed_texts: list[tuple[int, str]] = []
        structured: list[tuple[int, MessageInput | PromptInput]] = []

        for index, item in enumerate(inputs):
            match item:
                case TokenIdsInput(token_ids=token_ids):
                    outcomes[index] = len(token_ids)
                case TextInput(text=text):
                    indexed_texts.append((index, text))
                case MessageInput() | PromptInput():
                    structured.append((index, item))

        if indexed_texts:
            text_outcomes = await self._count_indexed_texts_async(
                indexed_texts, loop, live=live
            )
            for index, outcome in text_outcomes:
                outcomes[index] = outcome

        for index, item in structured:
            if self._thread is None:
                outcomes[index] = RuntimeError("BatchTokenizer is closed")
                continue
            try:
                match item:
                    case MessageInput(content, reasoning, tool_calls):
                        outcomes[index] = await loop.run_in_executor(
                            self._thread,
                            self._token_count_message,
                            content,
                            reasoning,
                            tool_calls,
                        )
                    case PromptInput(
                        messages,
                        tools,
                        chat_template_kwargs,
                        chat_template,
                        tool_choice,
                    ):
                        outcomes[index] = await loop.run_in_executor(
                            self._thread,
                            self._token_count_prompt,
                            messages,
                            tools,
                            chat_template_kwargs,
                            chat_template,
                            tool_choice,
                        )
            except Exception as exc:  # noqa: BLE001 - isolate this input.
                outcomes[index] = exc

        if any(outcome is None for outcome in outcomes):
            raise AssertionError("unhandled TokenizationInput variant")
        return cast(list[int | Exception], outcomes)

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


class TokenCounter(Protocol):
    """The async tokenization surface ``TokenBatchQueue`` depends on.

    ``BatchTokenizer`` satisfies this structurally; tests pass lightweight
    stubs. Declared as a Protocol so the queue is decoupled from the concrete
    tokenizer and test doubles type-check without inheritance.
    """

    async def count_batch_async(
        self,
        inputs: list[TokenizationInput],
        loop: asyncio.AbstractEventLoop,
        /,
        *,
        live: bool = False,
    ) -> list[int | Exception]:
        """Return one count or error per input, in input order."""
        ...


class TokenBatchQueue:
    """Buffers per-sample tokenization work and clears it in batches.

    Triggers enqueue plain text, assistant messages, or complete chat prompts
    at event time with a callback that records the resulting metric. The queue owns
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
        self._items: list[tuple[TokenizationInput, Callable[[int], None]]] = []
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

    def enqueue(self, item: TokenizationInput, on_count: Callable[[int], None]) -> None:
        self._inflight += 1
        self._items.append((item, on_count))

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
            if not self._items:
                return
            if live:
                selected: list[tuple[TokenizationInput, Callable[[int], None]]] = []
                remaining: list[tuple[TokenizationInput, Callable[[int], None]]] = []
                selected_by_type: dict[type, int] = {}
                for queued in self._items:
                    item_type = type(queued[0])
                    count = selected_by_type.get(item_type, 0)
                    if count < _LIVE_FLUSH_MAX_ITEMS:
                        selected.append(queued)
                        selected_by_type[item_type] = count + 1
                    else:
                        remaining.append(queued)
                items = selected
                self._items = remaining
            else:
                items, self._items = self._items, []

            try:
                outcomes = await self._tokenizer.count_batch_async(
                    [item for item, _ in items], self._loop, live=live
                )
            except asyncio.CancelledError:
                if live:
                    self._items[:0] = items
                raise
            except Exception:
                if live:
                    self._items[:0] = items
                raise

            if len(outcomes) != len(items):
                if live:
                    self._items[:0] = items
                raise RuntimeError(
                    f"tokenizer returned {len(outcomes)} outcomes for "
                    f"{len(items)} inputs"
                )

            failure: Exception | None = None
            retry: list[tuple[TokenizationInput, Callable[[int], None]]] = []
            for queued, outcome in zip(items, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    failure = failure or outcome
                    retry.append(queued)
                else:
                    self._record(queued[1], outcome)
            if live and retry:
                self._items[:0] = retry
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
