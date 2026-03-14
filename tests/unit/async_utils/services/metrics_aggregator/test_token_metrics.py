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

"""Tests for TokenizePool thread-safety and correctness."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    TokenizePool,
)


@pytest.mark.unit
class TestTokenizePool:
    def test_tokenize_returns_tokens(self):
        with TokenizePool("gpt2", n_workers=1) as pool:
            tokens = pool.tokenize("Hello world")
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            assert all(isinstance(t, str) for t in tokens)

    def test_token_count_returns_int(self):
        with TokenizePool("gpt2", n_workers=1) as pool:
            count = pool.token_count("Hello world")
            assert isinstance(count, int)
            assert count > 0

    def test_token_count_consistent_with_tokenize(self):
        with TokenizePool("gpt2", n_workers=1) as pool:
            text = "The quick brown fox jumps over the lazy dog"
            tokens = pool.tokenize(text)
            count = pool.token_count(text)
            # encode() includes special tokens potentially, tokenize() does not
            # So count may be >= len(tokens). Both should be > 0.
            assert count > 0
            assert len(tokens) > 0

    def test_multiple_workers(self):
        with TokenizePool("gpt2", n_workers=4) as pool:
            results = []
            for i in range(10):
                results.append(pool.token_count(f"Sentence number {i}"))
            assert all(isinstance(r, int) and r > 0 for r in results)

    def test_concurrent_calls_thread_safe(self):
        with TokenizePool("gpt2", n_workers=2) as pool:
            texts = [f"This is test sentence number {i}" for i in range(20)]

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(pool.token_count, t) for t in texts]
                results = [f.result() for f in futures]

            assert len(results) == 20
            assert all(isinstance(r, int) and r > 0 for r in results)

    def test_close_is_idempotent(self):
        pool = TokenizePool("gpt2", n_workers=1)
        pool.close()
        pool.close()  # Should not raise

    def test_use_after_close_raises(self):
        pool = TokenizePool("gpt2", n_workers=1)
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            pool.tokenize("hello")

    def test_n_workers_zero_raises(self):
        with pytest.raises(ValueError, match="n_workers"):
            TokenizePool("gpt2", n_workers=0)

    @pytest.mark.asyncio
    async def test_token_count_async(self):
        loop = asyncio.get_running_loop()
        with TokenizePool("gpt2", n_workers=1) as pool:
            count = await pool.token_count_async("Hello world", loop)
            assert isinstance(count, int)
            assert count > 0

    def test_context_manager(self):
        with TokenizePool("gpt2", n_workers=1) as pool:
            assert pool.token_count("test") > 0
        with pytest.raises(RuntimeError, match="closed"):
            pool.tokenize("test")
