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
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import patch

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator import (
    token_metrics as token_metrics_module,
)
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
    TokenBatchQueue,
    _encode_batch_lengths,
    _even_chunks,
    _worker_encode_lengths,
)

_MOCK_TARGET = "inference_endpoint.async_utils.services.metrics_aggregator.token_metrics.AutoTokenizer"


class _FakeTokenizer:
    """Deterministic tokenizer that splits on whitespace.

    Has no ``backend_tokenizer``, so BatchTokenizer keeps the batch path
    in-process (no subprocess shards) and counts via ``tokenize`` per text.
    """

    def __init__(self, load_delay: float = 0.0):
        time.sleep(load_delay)

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    @classmethod
    def from_pretrained(cls, name: str) -> "_FakeTokenizer":
        return cls()


class _FakeProc:
    """Stands in for a ProcessPoolExecutor shard; whitespace-counts its chunk."""

    def submit(self, _fn, chunk):
        fut: Future = Future()
        fut.set_result([len(t.split()) for t in chunk])
        return fut

    def shutdown(self, wait=False):
        pass


class _BrokenProc:
    """A shard whose work resolves to BrokenProcessPool (worker died)."""

    def submit(self, _fn, _chunk):
        fut: Future = Future()
        fut.set_exception(BrokenProcessPool("worker died"))
        return fut

    def shutdown(self, wait=False):
        pass


@pytest.mark.unit
class TestBatchTokenizer:
    @pytest.mark.asyncio
    async def test_count_texts_async(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                counts = await tok.count_texts_async(["Hello world foo", "a"], loop)
                assert counts == [3, 1]

    @pytest.mark.asyncio
    async def test_count_texts_async_empty(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                assert await tok.count_texts_async([], loop) == []

    @pytest.mark.asyncio
    async def test_count_texts_async_sharded(self):
        """With shards present, chunks are reassembled in original order."""
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                tok._procs = [_FakeProc(), _FakeProc()]
                counts = await tok.count_texts_async(["a", "b b", "c c c", "d"], loop)
                assert counts == [1, 2, 3, 1]

    @pytest.mark.asyncio
    async def test_count_texts_async_shard_failure_propagates(self):
        """A dead shard surfaces as an error, not a silent in-process fallback."""
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                tok._procs = [_BrokenProc()]
                with pytest.raises(BrokenProcessPool):
                    await tok.count_texts_async(["a b"], loop)

    def test_close_is_idempotent(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            tok = BatchTokenizer("fake")
            tok.close()
            tok.close()  # must not raise

    @pytest.mark.asyncio
    async def test_use_after_close_raises(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            tok = BatchTokenizer("fake")
            tok.close()
            with pytest.raises(RuntimeError, match="closed"):
                await tok.count_texts_async(["hello"], loop)


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
    @pytest.mark.asyncio
    async def test_token_count_message_subtracts_baseline(self):
        """token_count_message_async returns full_tokens - baseline."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                # "hello world" -> 2 content + 2 wrapper = 4; baseline = 0, prefix = 2
                count = await tok.token_count_message_async(
                    "hello world", None, None, loop
                )
                assert count == 2

    @pytest.mark.asyncio
    async def test_token_count_message_includes_tool_calls(self):
        """Tool-call JSON tokens are included in the count."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                without = await tok.token_count_message_async("hello", None, None, loop)
                with_calls = await tok.token_count_message_async(
                    "hello", None, tool_calls, loop
                )
                assert with_calls > without

    @pytest.mark.asyncio
    async def test_token_count_message_fallback_on_exception(self):
        """Falls back to whitespace split when apply_chat_template raises."""

        class _BadTemplateTokenizer(_FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("template does not support tool_calls")

        with patch(_MOCK_TARGET, _BadTemplateTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake") as tok:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                # Must not raise; falls back to whitespace tokenizer.
                count = await tok.token_count_message_async(
                    "hello world", None, tool_calls, loop
                )
                assert count > 0


class _Encoding:
    def __init__(self, n: int):
        self.ids = list(range(n))


class _FastBackend:
    """Raw-tokenizers backend stub with the fast batch entry point."""

    def encode_batch_fast(self, texts, add_special_tokens=False):
        return [_Encoding(len(t.split())) for t in texts]


class _SlowBackend:
    """Raw-tokenizers backend stub without encode_batch_fast."""

    def encode_batch(self, texts, add_special_tokens=False):
        return [_Encoding(len(t.split())) for t in texts]


@pytest.mark.unit
class TestEncodeHelpers:
    def test_encode_batch_lengths_prefers_fast(self):
        assert _encode_batch_lengths(_FastBackend(), ["a b", "c"]) == [2, 1]

    def test_encode_batch_lengths_falls_back_to_encode_batch(self):
        assert _encode_batch_lengths(_SlowBackend(), ["a b c", "d"]) == [3, 1]

    def test_worker_encode_lengths_raises_without_backend(self, monkeypatch):
        monkeypatch.setattr(token_metrics_module, "_WORKER_BACKEND", None)
        with pytest.raises(RuntimeError, match="backend unavailable"):
            _worker_encode_lengths(["a"])

    def test_worker_encode_lengths_uses_backend(self, monkeypatch):
        monkeypatch.setattr(token_metrics_module, "_WORKER_BACKEND", _FastBackend())
        assert _worker_encode_lengths(["a b", "c d e"]) == [2, 3]


@pytest.mark.unit
class TestEvenChunks:
    def test_splits_into_near_equal_chunks(self):
        assert _even_chunks(["a", "b", "c", "d", "e"], 2) == [
            ["a", "b", "c"],
            ["d", "e"],
        ]

    def test_single_chunk_when_n_le_one(self):
        assert _even_chunks(["a", "b"], 1) == [["a", "b"]]

    def test_single_item_input(self):
        assert _even_chunks(["only"], 4) == [["only"]]

    def test_preserves_order_and_bounds_chunk_count(self):
        items = [str(i) for i in range(10)]
        chunks = _even_chunks(items, 3)
        assert [x for c in chunks for x in c] == items
        assert len(chunks) <= 3


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

    async def test_flush_remaining_failure_reports_pending(self):
        """A tokenizer error leaves items pending and never raises."""

        class _FailingTokenizer:
            async def count_texts_async(self, texts, _loop):
                raise RuntimeError("tokenizer boom")

            async def token_count_message_async(self, *args):
                raise RuntimeError("tokenizer boom")

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_FailingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue_text("x y", recorded.append)
        assert await queue.flush_remaining(timeout=5.0) == 1
        assert recorded == []

    async def test_flush_text_failure_does_not_drop_message_items(self):
        """The message phase runs (and records) even when the text batch fails."""

        class _TextFailingTokenizer:
            async def count_texts_async(self, texts, _loop):
                raise RuntimeError("text shard died")

            async def token_count_message_async(
                self, content, reasoning, tool_calls, _loop
            ):
                return len(content.split())

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_TextFailingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue_text("never counted", recorded.append)
        queue.enqueue_message(("hello world", None, None), recorded.append)
        with pytest.raises(RuntimeError, match="text shard died"):
            await queue.flush()
        assert recorded == [2], "message item must survive the text failure"
        assert queue.pending == 1, "only the text item remains pending"

    async def test_flush_recorder_failure_does_not_poison_batch(self):
        """One raising on_count is logged; the rest of the batch still records."""
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []

        def bad_recorder(count: int) -> None:
            raise ValueError("recorder bug")

        queue.enqueue_text("a b", bad_recorder)
        queue.enqueue_text("c d e", recorded.append)
        await queue.flush()
        assert recorded == [3]
        assert queue.pending == 0, "a raising recorder still counts as recorded"
