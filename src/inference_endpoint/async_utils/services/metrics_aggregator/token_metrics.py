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

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# Create a thread-local storage for the tokenizer so each thread contains its own instance.
_thread_local = threading.local()


def _get_thread_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    """Return the tokenizer for the current thread, loading it if needed."""
    if not hasattr(_thread_local, "tokenizer") or _thread_local.tokenizer is None:
        _thread_local.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return _thread_local.tokenizer


def _tokenize_worker(tokenizer_name: str, text: str) -> list[str]:
    """Worker entry: load tokenizer for this thread and tokenize."""
    tokenizer = _get_thread_tokenizer(tokenizer_name)
    return tokenizer.tokenize(text)


def _token_count_worker(tokenizer_name: str, text: str) -> int:
    """Worker entry: return the number of tokens in text."""
    tokenizer = _get_thread_tokenizer(tokenizer_name)
    return len(tokenizer.encode(text))


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
    - The blocking `tokenize()` / `token_count()` methods are safe to call from
      multiple threads concurrently.
    - In an async context, use the `_async` variants to avoid blocking the event loop.
      These use `loop.run_in_executor(None, ...)` to offload to the default executor,
      which then submits to the TokenizePool's own ThreadPoolExecutor.
    """

    def __init__(self, tokenizer_name: str, n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        self._tokenizer_name = tokenizer_name
        self._n_workers = n_workers
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="TokenizePool",
        )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize the input string via the worker pool (blocking)."""
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        future = self._executor.submit(_tokenize_worker, self._tokenizer_name, text)
        return future.result()

    def token_count(self, text: str) -> int:
        """Return the number of tokens in the input string (blocking)."""
        if self._executor is None:
            raise RuntimeError("TokenizePool is closed")
        future = self._executor.submit(_token_count_worker, self._tokenizer_name, text)
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
            self._executor, _token_count_worker, self._tokenizer_name, text
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
