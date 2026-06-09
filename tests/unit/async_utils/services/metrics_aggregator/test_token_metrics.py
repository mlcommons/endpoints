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

"""Tests for BatchTokenizer and TokenBatchQueue."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
    TokenBatchQueue,
)

_MOCK_TARGET = "inference_endpoint.async_utils.services.metrics_aggregator.token_metrics.AutoTokenizer"


class _FakeTokenizer:
    """Deterministic tokenizer that splits on whitespace.

    Has no ``backend_tokenizer``, so BatchTokenizer keeps the batch path
    in-process (no subprocess shards) and ``count_texts`` falls back to
    ``tokenize`` per text — which is what these tests assert against.
    """

    def __init__(self, load_delay: float = 0.0):
        time.sleep(load_delay)

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    @classmethod
    def from_pretrained(cls, name: str) -> "_FakeTokenizer":
        return cls()


@pytest.mark.unit
class TestBatchTokenizer:
    def test_token_count_returns_int(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with BatchTokenizer("fake") as tok:
                assert tok.token_count("Hello world") == 2

    def test_count_texts_batch(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with BatchTokenizer("fake") as tok:
                assert tok.count_texts(["a b", "c d e", "x"]) == [2, 3, 1]

    def test_count_texts_empty(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with BatchTokenizer("fake") as tok:
                assert tok.count_texts([]) == []

    def test_concurrent_token_count_thread_safe(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with BatchTokenizer("fake") as tok:
                texts = [f"word{i} word{i + 1}" for i in range(20)]
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(tok.token_count, t) for t in texts]
                    results = [f.result() for f in futures]
                assert results == [2] * 20

    def test_close_is_idempotent(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            tok = BatchTokenizer("fake")
            tok.close()
            tok.close()  # must not raise

    def test_use_after_close_raises(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            tok = BatchTokenizer("fake")
            tok.close()
            with pytest.raises(RuntimeError, match="closed"):
                tok.token_count("hello")

    @pytest.mark.asyncio
    async def test_count_texts_async(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                counts = await tok.count_texts_async(["Hello world foo", "a"], loop)
                assert counts == [3, 1]

    def test_context_manager(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with BatchTokenizer("fake") as tok:
                assert tok.token_count("a b c") == 3
            with pytest.raises(RuntimeError, match="closed"):
                tok.token_count("test")


class _FakeTokenizerWithTemplate(_FakeTokenizer):
    """Tokenizer that supports apply_chat_template for tool-call testing."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        # Simulate 2 wrapper tokens for the template frame.
        parts = ["WRAPPER", "WRAPPER"]
        for msg in messages:
            content = msg.get("content")
            if content:
                parts.append(content)
            if msg.get("reasoning_content"):
                parts.append(msg["reasoning_content"])
            if msg.get("tool_calls"):
                import msgspec

                parts.append(msgspec.json.encode(msg["tool_calls"]).decode())
        rendered = " ".join(parts)
        if tokenize:
            return list(range(len(rendered.split())))
        return rendered


@pytest.mark.unit
class TestBatchTokenizerMessageTokenization:
    def test_token_count_message_subtracts_baseline(self):
        """token_count_message returns full_tokens - baseline."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            with BatchTokenizer("fake") as tok:
                # "hello world" -> 2 content + 2 wrapper = 4; baseline = 0, prefix = 2
                assert tok.token_count_message("hello world", None, None) == 2

    def test_token_count_message_includes_tool_calls(self):
        """token_count_message includes tool-call JSON tokens."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            with BatchTokenizer("fake") as tok:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                without = tok.token_count_message("hello", None, None)
                with_calls = tok.token_count_message("hello", None, tool_calls)
                assert with_calls > without

    def test_token_count_message_fallback_on_exception(self):
        """Falls back to whitespace split when apply_chat_template raises."""

        class _BadTemplateTokenizer(_FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("template does not support tool_calls")

        with patch(_MOCK_TARGET, _BadTemplateTokenizer):
            with BatchTokenizer("fake") as tok:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                # Must not raise; falls back to whitespace tokenizer.
                assert tok.token_count_message("hello world", None, tool_calls) > 0

    @pytest.mark.asyncio
    async def test_token_count_message_async(self):
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                count = await tok.token_count_message_async(
                    "hello world", None, None, loop
                )
                assert count == 2


class _CapturingTokenizer:
    """Minimal tokenizer stub for queue tests: whitespace counts, no procs."""

    async def count_texts_async(self, texts, _loop):
        return [len(t.split()) for t in texts]

    async def token_count_message_async(self, content, reasoning, tool_calls, _loop):
        parts = [p for p in (content, reasoning) if p]
        return len(" ".join(parts).split()) + (len(tool_calls) if tool_calls else 0)


@pytest.mark.unit
@pytest.mark.asyncio
class TestTokenBatchQueue:
    async def test_flush_records_text_via_callback(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue_text("a b c", recorded.append)
        queue.enqueue_text("d e", recorded.append)
        assert queue.pending == 2
        await queue.flush()
        assert sorted(recorded) == [2, 3]
        assert queue.pending == 0

    async def test_flush_records_message_via_callback(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue_message(("hello world", None, None), recorded.append)
        await queue.flush()
        assert recorded == [2]

    async def test_flush_empty_is_noop(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        await queue.flush()
        assert queue.pending == 0

    async def test_flush_remaining_clean_returns_zero(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue_text("a b", recorded.append)
        assert await queue.flush_remaining(timeout=5.0) == 0
        assert recorded == [2]

    async def test_flush_remaining_timeout_reports_pending(self):
        """A tokenizer slower than the budget leaves items pending."""

        class _BlockingTokenizer:
            async def count_texts_async(self, texts, _loop):
                await asyncio.sleep(10.0)
                return [0] * len(texts)

            async def token_count_message_async(self, *args):
                return 0

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_BlockingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue_text("never counted", recorded.append)
        n_pending = await queue.flush_remaining(timeout=0.05)
        assert n_pending == 1
        assert recorded == []
