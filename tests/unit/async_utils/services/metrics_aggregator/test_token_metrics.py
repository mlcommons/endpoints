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
import threading
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


class _FakeEncoding:
    """Mimics tokenizers.Encoding: exposes .ids of the right length."""

    def __init__(self, n: int):
        self.ids = [0] * n


class _FakeBackend:
    """Mimics tokenizers.Tokenizer: encode_batch_fast over a whitespace split.

    Class-level spies record how many batch calls happened and the largest
    batch seen, so tests can prove concurrent calls coalesced into few calls.
    """

    batch_calls = 0
    max_batch_seen = 0
    _lock = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.batch_calls = 0
            cls.max_batch_seen = 0

    def encode_batch_fast(self, texts, add_special_tokens=False):
        with _FakeBackend._lock:
            _FakeBackend.batch_calls += 1
            _FakeBackend.max_batch_seen = max(_FakeBackend.max_batch_seen, len(texts))
        return [_FakeEncoding(len(t.split())) for t in texts]


class _FakeTokenizerWithBackend(_FakeTokenizer):
    """Fast-tokenizer stand-in exposing a backend_tokenizer for the batch path."""

    def __init__(self, load_delay: float = 0.05):
        super().__init__(load_delay)
        self.backend_tokenizer = _FakeBackend()


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeProcExec:
    """Stand-in for ProcessPoolExecutor that records pinning without spawning.

    ``initargs`` of every constructed executor land in the class-level
    ``created`` list so tests can assert lane count and per-worker core sets.
    """

    created: list = []

    @classmethod
    def reset(cls) -> None:
        cls.created = []

    def __init__(self, *, max_workers, mp_context, initializer, initargs):
        _FakeProcExec.created.append(initargs)

    def submit(self, fn, *args):
        return _FakeFuture(True)

    def shutdown(self, wait=True):
        pass


_PROC_EXEC_TARGET = (
    "inference_endpoint.async_utils.services.metrics_aggregator."
    "token_metrics.ProcessPoolExecutor"
)


@pytest.mark.unit
class TestTokenizePoolBatchPath:
    # cores_per_worker=0 forces the thread lane so these exercise the coalescing
    # logic without spawning real worker processes (the fake backend can't be
    # loaded in a spawned subprocess). Process sharding is validated on cluster.
    @pytest.mark.asyncio
    async def test_async_uses_backend_and_counts_correctly(self):
        _FakeBackend.reset()
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            loop = asyncio.get_running_loop()
            with TokenizePool("fake", n_workers=1, cores_per_worker=0) as pool:
                count = await pool.token_count_async("a b c d", loop)
                assert count == 4
                assert _FakeBackend.batch_calls >= 1

    @pytest.mark.asyncio
    async def test_concurrent_calls_coalesce_into_few_batches(self):
        """50 concurrent awaits return correct per-text counts in few batch calls."""
        _FakeBackend.reset()
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            loop = asyncio.get_running_loop()
            with TokenizePool("fake", n_workers=1, cores_per_worker=0) as pool:
                texts = [" ".join(["w"] * (i + 1)) for i in range(50)]
                counts = await asyncio.gather(
                    *(pool.token_count_async(t, loop) for t in texts)
                )
                # Each text i has i+1 whitespace tokens.
                assert counts == [i + 1 for i in range(50)]
                # Coalescing: far fewer encode calls than texts.
                assert _FakeBackend.batch_calls < 50
                assert _FakeBackend.max_batch_seen > 1

    @pytest.mark.asyncio
    async def test_batch_slice_capped_at_max_batch_size(self):
        """No single encode batch exceeds max_batch_size, even under a backlog."""
        _FakeBackend.reset()
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            loop = asyncio.get_running_loop()
            with TokenizePool(
                "fake", n_workers=1, cores_per_worker=0, max_batch_size=4
            ) as pool:
                texts = [f"t{i}" for i in range(30)]
                counts = await asyncio.gather(
                    *(pool.token_count_async(t, loop) for t in texts)
                )
                assert counts == [1] * 30
                assert _FakeBackend.max_batch_seen <= 4
                assert _FakeBackend.batch_calls >= 8  # 30 items / 4 per batch

    @pytest.mark.asyncio
    async def test_fallback_when_no_backend(self):
        """Tokenizers without a backend_tokenizer fall back to per-text tokenize."""
        _FakeBackend.reset()
        with patch(_MOCK_TARGET, _FakeTokenizer):  # no backend_tokenizer attr
            loop = asyncio.get_running_loop()
            with TokenizePool("fake", n_workers=1, cores_per_worker=0) as pool:
                count = await pool.token_count_async("one two three", loop)
                assert count == 3
                assert _FakeBackend.batch_calls == 0  # backend never used


@pytest.mark.unit
class TestProcessLaneSetup:
    """Process-lane selection logic (without spawning real workers)."""

    def test_disabled_when_cores_per_worker_zero(self):
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            with TokenizePool("fake", n_workers=1, cores_per_worker=0) as pool:
                assert pool._proc_executors == []
                assert pool._lane_is_process is False

    def test_disabled_for_slow_tokenizer(self):
        # No backend_tokenizer => no encode_batch_fast => stay on threads even
        # with a large core budget.
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with patch("os.sched_getaffinity", return_value=set(range(64))):
                with TokenizePool("fake", n_workers=1, cores_per_worker=8) as pool:
                    assert pool._proc_executors == []
                    assert pool._lane_is_process is False

    def test_disabled_when_fewer_than_two_blocks(self):
        # 8 available cores / 8 per worker = 1 block -> single process not worth
        # it, fall back to the thread lane.
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            with patch("os.sched_getaffinity", return_value=set(range(8))):
                with TokenizePool("fake", n_workers=1, cores_per_worker=8) as pool:
                    assert pool._proc_executors == []
                    assert pool._lane_is_process is False

    def test_spawns_one_process_per_core_block(self):
        # 32 cores / 8 per worker = 4 pinned process lanes; assert count + pinning.
        _FakeProcExec.reset()
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            with (
                patch("os.sched_getaffinity", return_value=set(range(32))),
                patch(_PROC_EXEC_TARGET, _FakeProcExec),
            ):
                pool = TokenizePool("fake", n_workers=1, cores_per_worker=8)
                try:
                    assert pool._lane_is_process is True
                    assert len(pool._proc_executors) == 4
                    core_sets = [ia[1] for ia in _FakeProcExec.created]
                    assert core_sets == [
                        list(range(0, 8)),
                        list(range(8, 16)),
                        list(range(16, 24)),
                        list(range(24, 32)),
                    ]
                finally:
                    pool.close()

    def test_max_processes_caps_lane_count(self):
        _FakeProcExec.reset()
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            with (
                patch("os.sched_getaffinity", return_value=set(range(144))),
                patch(_PROC_EXEC_TARGET, _FakeProcExec),
            ):
                pool = TokenizePool(
                    "fake", n_workers=1, cores_per_worker=8, max_processes=4
                )
                try:
                    assert len(pool._proc_executors) == 4  # capped from 18
                finally:
                    pool.close()
