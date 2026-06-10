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

    def shutdown(self, wait=False, cancel_futures=False):
        pass


class _BrokenProc:
    """A shard whose work resolves to BrokenProcessPool (worker died)."""

    def submit(self, _fn, _chunk):
        fut: Future = Future()
        fut.set_exception(BrokenProcessPool("worker died"))
        return fut

    def shutdown(self, wait=False, cancel_futures=False):
        pass


@pytest.mark.unit
class TestBatchTokenizer:
    @pytest.mark.asyncio
    async def test_count_texts_async(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0) as tok:
                counts = await tok.count_texts_async(["Hello world foo", "a"], loop)
                assert counts == [3, 1]

    @pytest.mark.asyncio
    async def test_count_texts_async_empty(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0) as tok:
                assert await tok.count_texts_async([], loop) == []

    @pytest.mark.asyncio
    async def test_count_texts_async_sharded(self):
        """With shards present, chunks are reassembled in original order."""
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0) as tok:
                tok._procs = [_FakeProc(), _FakeProc()]
                counts = await tok.count_texts_async(["a", "b b", "c c c", "d"], loop)
                assert counts == [1, 2, 3, 1]

    @pytest.mark.asyncio
    async def test_count_texts_async_shard_failure_propagates(self):
        """A dead shard surfaces as an error, not a silent in-process fallback."""
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0) as tok:
                tok._procs = [_BrokenProc()]
                with pytest.raises(BrokenProcessPool):
                    await tok.count_texts_async(["a b"], loop)

    def test_close_is_idempotent(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            tok = BatchTokenizer("fake", n_workers=0)
            tok.close()
            tok.close()  # must not raise

    @pytest.mark.asyncio
    async def test_use_after_close_raises(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            tok = BatchTokenizer("fake", n_workers=0)
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
            with BatchTokenizer("fake", n_workers=0) as tok:
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
            with BatchTokenizer("fake", n_workers=0) as tok:
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
            with BatchTokenizer("fake", n_workers=0) as tok:
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


class _FakeTokenizerWithBackend(_FakeTokenizer):
    """Fast-backend fake: lets ``_setup_shards`` proceed past the backend guard."""

    backend_tokenizer = _FastBackend()


class _SpawnlessExecutor:
    """Stands in for ProcessPoolExecutor: records ctor args, instant warmup."""

    def __init__(self, max_workers, mp_context=None, initializer=None, initargs=()):
        self.initargs = initargs

    def submit(self, fn, *args):
        fut: Future = Future()
        fut.set_result(True)
        return fut

    def shutdown(self, wait=False, cancel_futures=False):
        pass


@pytest.mark.unit
class TestSetupShardsDecisions:
    """Pins the BatchTokenizer(n_workers=...) shard contract: -1 auto / N
    clamped / 0 explicit in-process (auto-sized in production — the CLI's
    --tokenizer-workers maps to the live thread lane, not to shards).

    An environment that cannot shard is a startup error — never a silent
    in-process fallback.
    """

    def _make(self, monkeypatch, cpus, n_workers, executor=_SpawnlessExecutor):
        monkeypatch.setattr(token_metrics_module, "ProcessPoolExecutor", executor)
        # Patch the probe + the restore so no real affinity syscalls run.
        monkeypatch.setattr(
            token_metrics_module,
            "expand_to_all_online_cpus",
            lambda: set(range(cpus)),
        )
        monkeypatch.setattr(
            token_metrics_module.os, "sched_getaffinity", lambda pid: {0, 1}
        )
        self.restored: list[set] = []
        monkeypatch.setattr(
            token_metrics_module.os,
            "sched_setaffinity",
            lambda pid, mask: self.restored.append(set(mask)),
        )
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            return BatchTokenizer("fake", n_workers=n_workers)

    @pytest.mark.parametrize(
        "cpus, n_workers, expected_shards",
        [
            (16, -1, 2),  # auto: one shard per 8-core block
            (10, -1, 1),  # auto: always at least one shard
            (6, -1, 1),  # auto: even below one full block
            (48, 3, 3),  # explicit count under capacity
            (16, 10, 2),  # explicit count clamped to capacity
            (16, 1, 1),  # explicit single shard honored
            (16, 0, 0),  # 0 = explicit in-process mode
        ],
    )
    def test_shard_count(self, monkeypatch, cpus, n_workers, expected_shards):
        with self._make(monkeypatch, cpus, n_workers) as tok:
            assert len(tok._procs) == expected_shards

    def test_blocks_are_disjoint_consecutive_core_sets(self, monkeypatch):
        with self._make(monkeypatch, 16, -1) as tok:
            blocks = [set(ex.initargs[1]) for ex in tok._procs]
            assert blocks == [set(range(0, 8)), set(range(8, 16))]

    def test_probe_restores_the_inherited_mask(self, monkeypatch):
        """The aggregator keeps the mask its parent gave it; only the probe
        widens, and only the shard children pin elsewhere."""
        with self._make(monkeypatch, 16, -1):
            pass
        assert self.restored == [{0, 1}]

    def test_no_fast_backend_is_a_startup_error(self, monkeypatch):
        monkeypatch.setattr(
            token_metrics_module, "ProcessPoolExecutor", _SpawnlessExecutor
        )
        with patch(_MOCK_TARGET, _FakeTokenizer):  # no backend_tokenizer
            with pytest.raises(RuntimeError, match="fast"):
                BatchTokenizer("fake")

    def test_affinity_unavailable_shards_unpinned(self, monkeypatch):
        """No affinity API (e.g. macOS): shard from the CPU count, unpinned."""
        monkeypatch.setattr(
            token_metrics_module, "ProcessPoolExecutor", _SpawnlessExecutor
        )

        def _unsupported():
            raise RuntimeError("affinity requires Linux")

        monkeypatch.setattr(
            token_metrics_module, "expand_to_all_online_cpus", _unsupported
        )

        def _raise(pid):
            raise AttributeError("no sched_getaffinity")

        monkeypatch.setattr(token_metrics_module.os, "sched_getaffinity", _raise)
        monkeypatch.setattr(token_metrics_module.os, "cpu_count", lambda: 16)
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            with BatchTokenizer("fake") as tok:
                assert len(tok._procs) == 2

    def test_warmup_failure_is_a_startup_error(self, monkeypatch):
        class _BrokenWarmup(_SpawnlessExecutor):
            def submit(self, fn, *args):
                fut: Future = Future()
                fut.set_exception(RuntimeError("spawn died"))
                return fut

        with pytest.raises(RuntimeError, match="warmup"):
            self._make(monkeypatch, 16, -1, executor=_BrokenWarmup)


class _RecordingProc(_FakeProc):
    """_FakeProc that records the chunks submitted to it."""

    def __init__(self):
        self.chunks = []

    def submit(self, _fn, chunk):
        self.chunks.append(list(chunk))
        return super().submit(_fn, chunk)


@pytest.mark.unit
class TestLiveLane:
    @pytest.mark.asyncio
    async def test_live_never_touches_the_shard_pool(self):
        """Mid-run flushes run in-process; the shards are drain-only."""
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=1) as tok:
                procs = [_RecordingProc(), _RecordingProc(), _RecordingProc()]
                tok._procs = procs
                counts = await tok.count_texts_live_async(["a b", "c"], loop)
                assert counts == [2, 1]
                assert all(p.chunks == [] for p in procs)

    @pytest.mark.asyncio
    async def test_drain_uses_every_shard(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=1) as tok:
                procs = [_RecordingProc(), _RecordingProc()]
                tok._procs = procs
                await tok.count_texts_async(["a", "b", "c", "d"], loop)
                assert all(p.chunks for p in procs)


@pytest.mark.unit
@pytest.mark.asyncio
class TestQueueLiveLoop:
    async def test_start_live_flushes_periodically(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue_text("a b c", recorded.append)
        queue.start_live(0.01)
        queue.start_live(0.01)  # idempotent
        await asyncio.sleep(0.05)
        assert recorded == [3]
        assert queue.pending == 0
        await queue.flush_remaining(timeout=1.0)

    async def test_live_loop_survives_tokenizer_failure(self):
        class _FailingLive(_CapturingTokenizer):
            async def count_texts_live_async(self, texts, _loop):
                raise RuntimeError("live lane boom")

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_FailingLive(), loop)
        recorded: list[int] = []
        queue.enqueue_text("a b", recorded.append)
        queue.start_live(0.01)
        await asyncio.sleep(0.05)
        assert recorded == []
        assert queue.pending == 1, "failed live flush must keep items pending"
        assert queue._live_task is not None and not queue._live_task.done()
        # The end-of-run drain (full pool) still recovers the items.
        assert await queue.flush_remaining(timeout=1.0) == 0
        assert recorded == [2]

    async def test_flush_remaining_stops_live_loop(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        queue.start_live(0.01)
        task = queue._live_task
        await queue.flush_remaining(timeout=1.0)
        assert queue._live_task is None
        assert task is not None and task.cancelled()


@pytest.mark.unit
class TestRayonCaps:
    def test_ctor_caps_rayon_to_live_workers(self, monkeypatch):
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with BatchTokenizer("fake", n_workers=0, live_workers=3):
                assert token_metrics_module.os.environ["RAYON_NUM_THREADS"] == "3"

    def test_ctor_respects_operator_exported_cap(self, monkeypatch):
        monkeypatch.setenv("RAYON_NUM_THREADS", "7")
        with patch(_MOCK_TARGET, _FakeTokenizer):
            with BatchTokenizer("fake", n_workers=0, live_workers=3):
                assert token_metrics_module.os.environ["RAYON_NUM_THREADS"] == "7"

    def test_init_worker_overrides_inherited_cap_with_block_size(self, monkeypatch):
        """Spawn children inherit the parent's live cap; each shard must
        re-size its rayon pool to its own core block."""
        monkeypatch.setenv("RAYON_NUM_THREADS", "2")

        def _no_affinity(pid, mask):
            raise AttributeError("no sched_setaffinity")

        monkeypatch.setattr(token_metrics_module.os, "sched_setaffinity", _no_affinity)
        with patch(_MOCK_TARGET, _FakeTokenizer):
            token_metrics_module._init_worker("fake", [0, 1, 2, 3, 4, 5, 6, 7])
        assert token_metrics_module.os.environ["RAYON_NUM_THREADS"] == "8"


@pytest.mark.unit
@pytest.mark.asyncio
class TestLiveFlushBounds:
    async def test_live_flush_takes_at_most_the_cap(self, monkeypatch):
        monkeypatch.setattr(token_metrics_module, "_LIVE_FLUSH_MAX_ITEMS", 3)
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        for i in range(5):
            queue.enqueue_text(f"t{i}", recorded.append)
        await queue.flush(live=True)
        assert len(recorded) == 3
        assert queue.pending == 2
        # The drain takes everything that remains.
        assert await queue.flush_remaining(timeout=1.0) == 0
        assert len(recorded) == 5

    async def test_live_cancellation_requeues_texts(self):
        class _Hanging(_CapturingTokenizer):
            async def count_texts_live_async(self, texts, _loop):
                await asyncio.sleep(30)
                return [0] * len(texts)

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_Hanging(), loop)
        recorded: list[int] = []
        queue.enqueue_text("a b", recorded.append)
        task = loop.create_task(queue.flush(live=True))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert queue.pending == 1
        assert len(queue._text) == 1, "cancelled live flush must give items back"
        assert await queue.flush_remaining(timeout=1.0) == 0
        assert recorded == [2]

    async def test_live_message_failure_requeues_message(self):
        class _MsgFailing(_CapturingTokenizer):
            async def token_count_message_async(self, *args):
                raise RuntimeError("template boom")

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_MsgFailing(), loop)
        recorded: list[int] = []
        queue.enqueue_message(("hello world", None, None), recorded.append)
        with pytest.raises(RuntimeError, match="template boom"):
            await queue.flush(live=True)
        assert queue.pending == 1
        assert len(queue._msg) == 1, "failed live message must be re-queued"


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

    async def count_texts_live_async(self, texts, _loop):
        return await self.count_texts_async(texts, _loop)

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

            async def count_texts_live_async(self, texts, _loop):
                return await self.count_texts_async(texts, _loop)

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

            async def count_texts_live_async(self, texts, _loop):
                return await self.count_texts_async(texts, _loop)

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

            async def count_texts_live_async(self, texts, _loop):
                return await self.count_texts_async(texts, _loop)

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
