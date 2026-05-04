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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


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
            try:
                self._thread_local.tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer_name
                )
            except Exception:
                # AutoTokenizer loads config.json to detect the model type; for
                # models with unknown model_type (e.g. deepseek_v4 in older
                # transformers) or missing rope config fields, this fails.
                # Fall back to PreTrainedTokenizerFast which reads only
                # tokenizer.json / tokenizer_config.json and skips model config.
                self._thread_local.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    self._tokenizer_name
                )
        return self._thread_local.tokenizer

    def _token_count_worker(self, text: str) -> int:
        """Worker entry: return the number of tokens in text."""
        tokenizer = self._get_thread_tokenizer()
        return len(tokenizer.tokenize(text))

    def token_count(self, text: str) -> int:
        """Return the number of tokens in the input string (blocking)."""
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        future = self._executor.submit(self._token_count_worker, text)
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

    def close(self) -> None:
        """Shut down the worker pool. Idempotent."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> TokenizePool:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
