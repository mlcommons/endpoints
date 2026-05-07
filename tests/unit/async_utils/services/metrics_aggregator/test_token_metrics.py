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
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    TokenizePool,
)

_MOCK_TARGET = "inference_endpoint.async_utils.services.metrics_aggregator.token_metrics.AutoTokenizer"


class _FakeTokenizer:
    """Deterministic tokenizer that splits on whitespace."""

    def __init__(self, load_delay: float = 0.1):
        # Simulate the blocking cost of from_pretrained so that
        # pre-initialization in __init__ saturates all worker threads.
        time.sleep(load_delay)

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    @classmethod
    def from_pretrained(cls, name: str) -> "_FakeTokenizer":
        return cls()


@pytest.mark.unit
class TestTokenizePool:
    def test_token_count_returns_int(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with TokenizePool("fake", n_workers=1) as pool:
                count = pool.token_count("Hello world")
                assert count == 2

    def test_multiple_workers(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with TokenizePool("fake", n_workers=4) as pool:
                results = []
                for i in range(10):
                    results.append(pool.token_count(f"Sentence number {i}"))
                assert all(isinstance(r, int) and r > 0 for r in results)

    def test_concurrent_calls_thread_safe(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with TokenizePool("fake", n_workers=2) as pool:
                texts = [f"word{i} word{i+1}" for i in range(20)]

                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(pool.token_count, t) for t in texts]
                    results = [f.result() for f in futures]

                assert len(results) == 20
                assert all(r == 2 for r in results)

    def test_close_is_idempotent(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            pool = TokenizePool("fake", n_workers=1)
            pool.close()
            pool.close()  # Should not raise

    def test_use_after_close_raises(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            pool = TokenizePool("fake", n_workers=1)
            pool.close()
            with pytest.raises(RuntimeError, match="closed"):
                pool.token_count("hello")

    def test_n_workers_zero_raises(self):
        with pytest.raises(ValueError, match="n_workers"):
            TokenizePool("fake", n_workers=0)

    @pytest.mark.asyncio
    async def test_token_count_async(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with TokenizePool("fake", n_workers=1) as pool:
                count = await pool.token_count_async("Hello world foo", loop)
                assert count == 3

    def test_context_manager(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with TokenizePool("fake", n_workers=1) as pool:
                assert pool.token_count("a b c") == 3
            with pytest.raises(RuntimeError, match="closed"):
                pool.token_count("test")


class _FakeTokenizerWithTemplate(_FakeTokenizer):
    """Tokenizer that supports apply_chat_template for tool-call testing."""

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False
    ) -> list[int]:
        parts = []
        for msg in messages:
            parts.append(msg.get("content") or "")
            if msg.get("reasoning_content"):
                parts.append(msg["reasoning_content"])
            if msg.get("tool_calls"):
                import msgspec

                parts.append(msgspec.json.encode(msg["tool_calls"]).decode())
        combined = " ".join(p for p in parts if p)
        # Add 2 wrapper tokens to simulate template overhead
        return [0, 0] + list(range(len(combined.split())))


@pytest.mark.unit
class TestTokenizePoolMessageTokenization:
    def test_token_count_message_subtracts_baseline(self):
        """token_count_message returns full_tokens - baseline."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            with TokenizePool("fake", n_workers=1) as pool:
                # "hello world" -> 2 content words + 2 wrapper = 4; baseline = 0 + 2 = 2; net = 2
                count = pool.token_count_message("hello world", None, None)
                assert count == 2

    def test_token_count_message_includes_tool_calls(self):
        """token_count_message includes tool-call JSON tokens."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            with TokenizePool("fake", n_workers=1) as pool:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                count_without = pool.token_count_message("hello", None, None)
                count_with = pool.token_count_message("hello", None, tool_calls)
                assert count_with > count_without

    def test_token_count_message_fallback_on_exception(self):
        """Falls back to whitespace split when apply_chat_template raises."""

        class _BadTemplateTokenizer(_FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("template does not support tool_calls")

        with patch(_MOCK_TARGET, _BadTemplateTokenizer):
            with TokenizePool("fake", n_workers=1) as pool:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                # Should not raise; falls back to whitespace tokenizer
                count = pool.token_count_message("hello world", None, tool_calls)
                assert count > 0

    @pytest.mark.asyncio
    async def test_token_count_message_async(self):
        """token_count_message_async returns count without blocking event loop."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            loop = asyncio.get_running_loop()
            with TokenizePool("fake", n_workers=1) as pool:
                count = await pool.token_count_message_async(
                    "hello world", None, None, loop
                )
                assert count == 2
