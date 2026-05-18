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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import msgspec
from transformers import AutoTokenizer

# Minimal user message used to satisfy chat templates that reject assistant-only
# message lists. Its token count is subtracted so only the assistant payload is
# measured.
_PREFIX_USER_MSG: dict[str, str] = {"role": "user", "content": ""}


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


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TokenizePool:
    """A pool of worker threads, each with its own HuggingFace AutoTokenizer.

    Uses multi-threading (not multiprocessing) because HuggingFace tokenizers
    use a Rust backend that releases the GIL during tokenization, so threads
    can run tokenization in parallel without GIL contention. Multiprocessing
    would add process spawn overhead and per-process tokenizer memory and
    IPC latency.

    Thread-safety notes:
    - The ThreadPoolExecutor itself is thread-safe (submit/shutdown are synchronized).
    - Each worker thread has its own tokenizer via thread-local storage, so there
      is no shared mutable state during tokenization.
    - The blocking `token_count()` method is safe to call from multiple threads
      concurrently.
    - In an async context, use `token_count_async` to avoid blocking the event loop.
    """

    def __init__(self, tokenizer_name: str, n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        self._tokenizer_name = tokenizer_name
        self._n_workers = n_workers
        self._thread_local = threading.local()
        self._fallback_warned: set[str] = set()
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="TokenizePool",
        )
        # Pre-load a tokenizer on every worker thread so the first real
        # token_count call doesn't pay the AutoTokenizer.from_pretrained cost.
        # Submitting n_workers tasks is guaranteed to hit every thread because
        # AutoTokenizer.from_pretrained blocks long enough that no thread
        # completes before all tasks are submitted.
        # **IMPORTANT**: This is not a guarantee - for instance when using a mock
        # object in tests for the tokenizer, the mock object *must* block in the 100ms
        # range to simulate proper .from_pretrained behavior.
        # It is not super impactful if a thread is not pre-initialized - it will just
        # have to pay the cost of .from_pretrained on the first pool.token_count call
        # for that thread.
        futures = [
            self._executor.submit(self._get_thread_tokenizer) for _ in range(n_workers)
        ]
        for f in futures:
            f.result()

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

    async def token_count_async(
        self, text: str, loop: asyncio.AbstractEventLoop
    ) -> int:
        """Return the number of tokens without blocking the event loop.

        Submits directly to the TokenizePool's executor so tokenization runs
        on a thread with a pre-loaded thread-local tokenizer instance.
        """
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        return await loop.run_in_executor(
            self._executor, self._token_count_worker, text
        )

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

    def close(self) -> None:
        """Shut down the worker pool. Idempotent."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> TokenizePool:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
