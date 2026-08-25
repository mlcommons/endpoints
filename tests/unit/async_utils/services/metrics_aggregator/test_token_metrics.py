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
import multiprocessing
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import wait as wait_for_process
from unittest.mock import patch

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator import (
    token_metrics as token_metrics_module,
)
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
    TokenBatchQueue,
    _even_chunks,
    _terminate_procs,
    _worker_encode_lengths,
    encode_lengths,
)
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    PromptInput,
    TextInput,
    TokenIdsInput,
)

_MOCK_TARGET = "inference_endpoint.async_utils.services.metrics_aggregator.token_metrics.AutoTokenizer"


@pytest.mark.unit
def test_tokenization_inputs_make_the_four_paths_explicit():
    assert TokenIdsInput((1, 2)).token_ids == (1, 2)
    assert TextInput("hello").text == "hello"
    assert MessageInput("answer", "thought", None).reasoning == "thought"
    assert PromptInput(({"role": "user", "content": "hi"},), None, None, None).messages


class _FakeTokenizer:
    """Deterministic tokenizer that splits on whitespace.

    Has no ``backend_tokenizer`` and therefore supports only the structured
    chat-template path when a subclass supplies ``apply_chat_template``.
    """

    def __init__(self, load_delay: float = 0.0):
        time.sleep(load_delay)

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))

    @classmethod
    def from_pretrained(cls, name: str, **kwargs: object) -> "_FakeTokenizer":
        assert kwargs == {"trust_remote_code": True}
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
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                counts = await tok._count_texts_async(["Hello world foo", "a"], loop)
                assert counts == [3, 1]

    @pytest.mark.asyncio
    async def test_count_texts_async_empty(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                assert await tok._count_texts_async([], loop) == []

    @pytest.mark.asyncio
    async def test_plain_text_falls_back_to_tokenizer_encode_without_backend(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                counts = await tok._count_texts_async(
                    ["Hello world", "one two three"], loop
                )
                assert counts == [2, 3]

    @pytest.mark.asyncio
    async def test_count_texts_async_sharded(self):
        """With shards present, chunks are reassembled in original order."""
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                tok._procs = [_FakeProc(), _FakeProc()]
                counts = await tok._count_texts_async(["a", "b b", "c c c", "d"], loop)
                assert counts == [1, 2, 3, 1]

    @pytest.mark.asyncio
    async def test_count_texts_async_shard_failure_propagates(self):
        """A dead shard surfaces as an error, not a silent in-process fallback."""
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                tok._procs = [_BrokenProc()]
                with pytest.raises(BrokenProcessPool):
                    await tok._count_texts_async(["a b"], loop)

    def test_close_is_idempotent(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            tok = BatchTokenizer("fake", n_workers=0, live_workers=2)
            tok.close()
            tok.close()  # must not raise

    @pytest.mark.asyncio
    async def test_use_after_close_raises(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            tok = BatchTokenizer("fake", n_workers=0, live_workers=2)
            tok.close()
            with pytest.raises(RuntimeError, match="closed"):
                await tok._count_texts_async(["hello"], loop)


class _FakeTokenizerWithTemplate(_FakeTokenizer):
    """Tokenizer that supports apply_chat_template for tool-call testing."""

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=True,
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
        if tools:
            parts.extend(tool["function"]["name"] for tool in tools)
        if add_generation_prompt:
            parts.append("GENERATION")
        rendered = " ".join(parts)
        if tokenize:
            token_ids = list(range(len(rendered.split())))
            if return_dict:
                return {
                    "input_ids": token_ids,
                    "attention_mask": [1] * len(token_ids),
                }
            return token_ids
        return rendered


@pytest.mark.unit
class TestBatchTokenizerMessageTokenization:
    def test_chat_template_requests_token_ids_not_batch_encoding(self):
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                count = tok._token_count_prompt(
                    ({"role": "user", "content": "one two three four"},),
                    None,
                    None,
                    None,
                    None,
                )

                assert count == 7

    @pytest.mark.asyncio
    async def test_token_count_prompt_preserves_messages_tools_and_generation_prompt(
        self,
    ):
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                messages = (
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "reasoning_content": "reasoning",
                        "content": None,
                        "tool_calls": (
                            {
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            },
                        ),
                    },
                    {"role": "tool", "content": "result"},
                )
                tools = (
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}},
                    },
                )

                count = (
                    await tok.count_batch_async(
                        [PromptInput(messages, tools, None, None)], loop
                    )
                )[0]

                # Wrapper + question + reasoning + tool call + tool result +
                # declared tool + generation prompt.
                assert count == 8

    @pytest.mark.asyncio
    async def test_token_count_message_subtracts_baseline(self):
        """Structured message counting returns full tokens minus baseline."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                # "hello world" -> 2 content + 2 wrapper = 4; baseline = 0, prefix = 2
                count = (
                    await tok.count_batch_async(
                        [MessageInput("hello world", None, None)], loop
                    )
                )[0]
                assert count == 2

    @pytest.mark.asyncio
    async def test_token_count_message_includes_tool_calls(self):
        """Tool-call JSON tokens are included in the count."""
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                without = (
                    await tok.count_batch_async(
                        [MessageInput("hello", None, None)], loop
                    )
                )[0]
                with_calls = (
                    await tok.count_batch_async(
                        [MessageInput("hello", None, tool_calls)], loop
                    )
                )[0]
                assert with_calls > without

    @pytest.mark.asyncio
    async def test_token_count_message_fallback_on_exception(self):
        """Falls back to whitespace split when apply_chat_template raises."""

        class _BadTemplateTokenizer(_FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("template does not support tool_calls")

        with patch(_MOCK_TARGET, _BadTemplateTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                tool_calls = (
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                )
                # Must not raise; falls back to whitespace tokenizer.
                count = (
                    await tok.count_batch_async(
                        [MessageInput("hello world", None, tool_calls)], loop
                    )
                )[0]
                assert count > 0

    def test_token_count_prompt_falls_back_on_template_error(self):
        class _TextOnlyTemplateTokenizer(_FakeTokenizerWithTemplate):
            def apply_chat_template(self, messages, **kwargs):
                if any(
                    isinstance(message.get("content"), list) for message in messages
                ):
                    raise TypeError("text-only template")
                return super().apply_chat_template(messages, **kwargs)

        with patch(_MOCK_TARGET, _TextOnlyTemplateTokenizer):
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                count = tok._token_count_prompt(
                    (
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe this image"},
                                {"type": "image_url", "image_url": {"url": "x"}},
                            ],
                        },
                    ),
                    None,
                    None,
                    None,
                    None,
                )

                assert count > 0

    def test_token_count_prompt_normalizes_tools_and_forwards_template_kwargs(self):
        class _RecordingTokenizer(_FakeTokenizerWithTemplate):
            last_messages = None
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                type(self).last_messages = messages
                type(self).last_kwargs = dict(kwargs)
                kwargs.pop("enable_thinking", None)
                return super().apply_chat_template(messages, **kwargs)

        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
        )
        with patch(_MOCK_TARGET, _RecordingTokenizer):
            with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
                tok._token_count_prompt(
                    messages,
                    None,
                    {"enable_thinking": False},
                    "custom template",
                    "auto",
                )

        normalized_call = _RecordingTokenizer.last_messages[0]["tool_calls"][0]
        assert normalized_call["function"]["arguments"] == {"city": "SF"}
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
            '{"city": "SF"}'
        )
        assert _RecordingTokenizer.last_kwargs["enable_thinking"] is False
        assert _RecordingTokenizer.last_kwargs["chat_template"] == "custom template"
        assert _RecordingTokenizer.last_kwargs["tool_choice"] == "auto"
        assert _RecordingTokenizer.last_kwargs["return_dict"] is False


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
    def test_encode_lengths_prefers_fast(self):
        assert encode_lengths(_FastBackend(), ["a b", "c"]) == [2, 1]

    def test_encode_lengths_falls_back_to_encode_batch(self):
        assert encode_lengths(_SlowBackend(), ["a b c", "d"]) == [3, 1]

    def test_load_reference_backend_passes_trust_remote_code(self, monkeypatch):
        # trust_remote_code must be forwarded so tokenizers that ship custom code
        # (e.g. DeepSeek-R1) load instead of silently disabling OSL.
        captured: dict = {}

        class _FakeTok:
            backend_tokenizer = "BACKEND"

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                captured.update(name=name, kwargs=kwargs)
                return _FakeTok()

        monkeypatch.setattr(token_metrics_module, "AutoTokenizer", _FakeAutoTokenizer)
        assert token_metrics_module.load_reference_backend("m") == "BACKEND"
        assert captured["name"] == "m"
        assert captured["kwargs"].get("trust_remote_code") is True

    def test_worker_encode_lengths_raises_without_backend(self, monkeypatch):
        monkeypatch.setattr(token_metrics_module, "_WORKER_TEXT_BACKEND", None)
        with pytest.raises(RuntimeError, match="backend unavailable"):
            _worker_encode_lengths(["a"])

    def test_worker_encode_lengths_uses_backend(self, monkeypatch):
        monkeypatch.setattr(
            token_metrics_module, "_WORKER_TEXT_BACKEND", _FastBackend()
        )
        assert _worker_encode_lengths(["a b", "c d e"]) == [2, 3]


class _FakeTokenizerWithBackend(_FakeTokenizer):
    """Fast-backend fake: lets ``_setup_shards`` proceed past the backend guard."""

    backend_tokenizer = _FastBackend()


class _FakeTokenizerWithTemplateAndBackend(_FakeTokenizerWithTemplate):
    backend_tokenizer = _FastBackend()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_count_batch_routes_all_four_inputs_and_preserves_order():
    with patch(_MOCK_TARGET, _FakeTokenizerWithTemplateAndBackend):
        loop = asyncio.get_running_loop()
        with BatchTokenizer("fake", n_workers=0, live_workers=2) as tok:
            outcomes = await tok.count_batch_async(
                [
                    TokenIdsInput((1, 2, 3)),
                    TextInput("plain text"),
                    MessageInput("answer here", None, None),
                    PromptInput(
                        ({"role": "user", "content": "ask now"},),
                        None,
                        None,
                        None,
                    ),
                ],
                loop,
            )

    assert outcomes == [3, 2, 2, 5]


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

    A fast text backend is optional. If present, shard warmup failures are
    startup errors rather than silent in-process fallbacks.
    """

    def _make(self, monkeypatch, cpus, n_workers, executor=_SpawnlessExecutor):
        monkeypatch.setattr(token_metrics_module, "ProcessPoolExecutor", executor)
        # cgroup_clamped_cpus owns the probe-and-restore (tested in
        # test_cpu_affinity); here we just feed _setup_shards a CPU universe.
        monkeypatch.setattr(
            token_metrics_module, "cgroup_clamped_cpus", lambda: list(range(cpus))
        )
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            return BatchTokenizer("fake", n_workers=n_workers, live_workers=2)

    @pytest.mark.parametrize(
        "cpus, n_workers, expected_shards",
        [
            (16, -1, 2),  # auto: one shard per 8-core block
            (10, -1, 1),  # auto: always at least one shard
            (6, -1, 1),  # auto: even below one full block
            (24, -1, 3),  # auto: 22 usable cores, including a partial block
            (48, 3, 3),  # explicit count under capacity
            (16, 10, 2),  # explicit count clamped to capacity
            (16, 1, 1),  # explicit single shard honored
            (16, 0, 0),  # 0 = explicit in-process mode
        ],
    )
    def test_shard_count(self, monkeypatch, cpus, n_workers, expected_shards):
        with self._make(monkeypatch, cpus, n_workers) as tok:
            assert len(tok._procs) == expected_shards

    def test_blocks_reserve_two_cpus_and_use_partial_final_block(self, monkeypatch):
        with self._make(monkeypatch, 16, -1) as tok:
            blocks = [set(ex.initargs[1]) for ex in tok._procs]
            assert blocks == [set(range(0, 8)), set(range(8, 14))]

    def test_worker_spawn_ignores_sigint_and_restores_parent_handler(self, monkeypatch):
        dispositions = []

        class _SignalCapturingExecutor(_SpawnlessExecutor):
            def submit(self, fn, *args):
                dispositions.append(signal.getsignal(signal.SIGINT))
                return super().submit(fn, *args)

        previous = signal.getsignal(signal.SIGINT)
        with self._make(monkeypatch, 16, -1, executor=_SignalCapturingExecutor):
            pass

        assert dispositions
        assert all(handler is signal.SIG_IGN for handler in dispositions)
        assert signal.getsignal(signal.SIGINT) is previous

    def test_structured_tokenization_does_not_require_a_text_backend(self, monkeypatch):
        monkeypatch.setattr(
            token_metrics_module, "ProcessPoolExecutor", _SpawnlessExecutor
        )
        with patch(_MOCK_TARGET, _FakeTokenizerWithTemplate):
            with BatchTokenizer("fake", live_workers=2) as tok:
                assert tok._procs == []
                assert (
                    tok._token_count_prompt(
                        ({"role": "user", "content": "one two"},),
                        None,
                        None,
                        None,
                        None,
                    )
                    == 5
                )

    def test_affinity_unavailable_shards_unpinned(self, monkeypatch):
        """No affinity API (e.g. macOS): shard from the CPU count, unpinned."""
        monkeypatch.setattr(
            token_metrics_module, "ProcessPoolExecutor", _SpawnlessExecutor
        )
        monkeypatch.setattr(token_metrics_module, "cgroup_clamped_cpus", lambda: None)
        monkeypatch.setattr(token_metrics_module.os, "cpu_count", lambda: 16)
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            with BatchTokenizer("fake", live_workers=2) as tok:
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
        with patch(_MOCK_TARGET, _FakeTokenizerWithBackend):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=1) as tok:
                procs = [_RecordingProc(), _RecordingProc(), _RecordingProc()]
                tok._procs = procs
                counts = await tok._count_texts_async(["a b", "c"], loop, live=True)
                assert counts == [2, 1]
                assert all(p.chunks == [] for p in procs)

    @pytest.mark.asyncio
    async def test_drain_uses_every_shard(self):
        with patch(_MOCK_TARGET, _FakeTokenizer):
            loop = asyncio.get_running_loop()
            with BatchTokenizer("fake", n_workers=0, live_workers=1) as tok:
                procs = [_RecordingProc(), _RecordingProc()]
                tok._procs = procs
                await tok._count_texts_async(["a", "b", "c", "d"], loop)
                assert all(p.chunks for p in procs)


@pytest.mark.unit
@pytest.mark.asyncio
class TestQueueLiveLoop:
    async def test_start_live_flushes_periodically(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("a b c"), recorded.append)
        queue.start_live(0.01)
        queue.start_live(0.01)  # idempotent
        await asyncio.sleep(0.05)
        assert recorded == [3]
        assert queue.pending == 0
        await queue.flush_remaining(timeout=1.0)

    async def test_live_loop_survives_tokenizer_failure(self):
        class _FailingLive(_CapturingTokenizer):
            async def count_batch_async(self, inputs, _loop, live=False):
                if live:
                    raise RuntimeError("live lane boom")
                return await super().count_batch_async(inputs, _loop)

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_FailingLive(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("a b"), recorded.append)
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
            queue.enqueue(TextInput(f"t{i}"), recorded.append)
        await queue.flush_live_once()
        assert len(recorded) == 3
        assert queue.pending == 2
        # The drain takes everything that remains.
        assert await queue.flush_remaining(timeout=1.0) == 0
        assert len(recorded) == 5

    async def test_live_cancellation_requeues_texts(self):
        class _Hanging(_CapturingTokenizer):
            async def count_batch_async(self, inputs, _loop, live=False):
                if live:
                    await asyncio.sleep(30)
                return await super().count_batch_async(inputs, _loop)

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_Hanging(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("a b"), recorded.append)
        task = loop.create_task(queue.flush_live_once())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert queue.pending == 1
        assert len(queue._items) == 1, "cancelled live flush must give items back"
        assert await queue.flush_remaining(timeout=1.0) == 0
        assert recorded == [2]

    async def test_live_cancellation_requeues_messages_too(self):
        """A cancel landing in the text encode must give back BOTH kinds."""

        class _Hanging(_CapturingTokenizer):
            async def count_batch_async(self, inputs, _loop, live=False):
                if live:
                    await asyncio.sleep(30)
                return await super().count_batch_async(inputs, _loop)

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_Hanging(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("a b"), recorded.append)
        queue.enqueue(MessageInput("hello world", None, None), recorded.append)
        task = loop.create_task(queue.flush_live_once())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert queue.pending == 2
        assert len(queue._items) == 2, "all detached items must be re-queued"
        assert await queue.flush_remaining(timeout=1.0) == 0
        assert sorted(recorded) == [2, 2]

    async def test_live_message_failure_requeues_message(self):
        class _MsgFailing(_CapturingTokenizer):
            async def count_batch_async(self, inputs, _loop, live=False):
                return [RuntimeError("template boom") for _ in inputs]

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_MsgFailing(), loop)
        recorded: list[int] = []
        queue.enqueue(MessageInput("hello world", None, None), recorded.append)
        with pytest.raises(RuntimeError, match="template boom"):
            await queue.flush_live_once()
        assert queue.pending == 1
        assert len(queue._items) == 1, "failed live message must be re-queued"


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

    async def count_batch_async(self, inputs, _loop, live=False):
        outcomes = []
        for item in inputs:
            if isinstance(item, TokenIdsInput):
                outcomes.append(len(item.token_ids))
            elif isinstance(item, TextInput):
                outcomes.append(len(item.text.split()))
            elif isinstance(item, MessageInput):
                parts = [p for p in (item.content, item.reasoning) if p]
                outcomes.append(
                    len(" ".join(parts).split())
                    + (len(item.tool_calls) if item.tool_calls else 0)
                )
            elif isinstance(item, PromptInput):
                outcomes.append(len(item.messages))
        return outcomes


@pytest.mark.unit
@pytest.mark.asyncio
class TestTokenBatchQueue:
    async def test_flush_records_text_via_callback(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("a b c"), recorded.append)
        queue.enqueue(TextInput("d e"), recorded.append)
        assert queue.pending == 2
        await queue.drain_all()
        assert sorted(recorded) == [2, 3]
        assert queue.pending == 0

    async def test_flush_records_message_via_callback(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue(MessageInput("hello world", None, None), recorded.append)
        await queue.drain_all()
        assert recorded == [2]

    async def test_flush_wrong_length_text_result_isolated(self):
        """A wrong-length tokenizer result must not drop the message phase or
        miscount _inflight: text items stay pending (drain-terminal), the
        message still records, and the failure is raised after both phases."""

        class _WrongLength:
            async def count_batch_async(self, inputs, _loop, live=False):
                error = RuntimeError("tokenizer returned 1 counts for 2 texts")
                return [error, error, 7]

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_WrongLength(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("a b"), recorded.append)
        queue.enqueue(TextInput("c d"), recorded.append)
        queue.enqueue(MessageInput("hi", None, None), recorded.append)
        assert queue.pending == 3

        with pytest.raises(RuntimeError, match="counts for"):
            await queue.drain_all()

        assert recorded == [7], "message phase must still record"
        assert queue.pending == 2, "text items left pending, not dropped or miscounted"

    async def test_flush_empty_is_noop(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        await queue.drain_all()
        assert queue.pending == 0

    async def test_flush_remaining_clean_returns_zero(self):
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("a b"), recorded.append)
        assert await queue.flush_remaining(timeout=5.0) == 0
        assert recorded == [2]

    async def test_flush_remaining_timeout_reports_pending(self):
        """A tokenizer slower than the budget leaves items pending."""

        class _BlockingTokenizer:
            async def count_batch_async(self, inputs, _loop, live=False):
                await asyncio.sleep(10.0)
                return [0] * len(inputs)

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_BlockingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("never counted"), recorded.append)
        n_pending = await queue.flush_remaining(timeout=0.05)
        assert n_pending == 1
        assert recorded == []

    async def test_flush_remaining_failure_reports_pending(self):
        """A tokenizer error leaves items pending and never raises."""

        class _FailingTokenizer:
            async def count_batch_async(self, inputs, _loop, live=False):
                raise RuntimeError("tokenizer boom")

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_FailingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("x y"), recorded.append)
        assert await queue.flush_remaining(timeout=5.0) == 1
        assert recorded == []

    async def test_flush_text_failure_does_not_drop_message_items(self):
        """The message phase runs (and records) even when the text batch fails."""

        class _TextFailingTokenizer:
            async def count_batch_async(self, inputs, _loop, live=False):
                return [
                    RuntimeError("text shard died")
                    if isinstance(item, TextInput)
                    else len(item.content.split())
                    for item in inputs
                ]

        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_TextFailingTokenizer(), loop)
        recorded: list[int] = []
        queue.enqueue(TextInput("never counted"), recorded.append)
        queue.enqueue(MessageInput("hello world", None, None), recorded.append)
        with pytest.raises(RuntimeError, match="text shard died"):
            await queue.drain_all()
        assert recorded == [2], "message item must survive the text failure"
        assert queue.pending == 1, "only the text item remains pending"

    async def test_flush_recorder_failure_does_not_poison_batch(self):
        """One raising on_count is logged; the rest of the batch still records."""
        loop = asyncio.get_running_loop()
        queue = TokenBatchQueue(_CapturingTokenizer(), loop)
        recorded: list[int] = []

        def bad_recorder(count: int) -> None:
            raise ValueError("recorder bug")

        queue.enqueue(TextInput("a b"), bad_recorder)
        queue.enqueue(TextInput("c d e"), recorded.append)
        await queue.drain_all()
        assert recorded == [3]
        assert queue.pending == 0, "a raising recorder still counts as recorded"


@pytest.mark.unit
def test_terminate_procs_kills_running_workers():
    """``_terminate_procs`` must SIGTERM live workers.

    Regression guard: ``ProcessPoolExecutor.shutdown()`` clears ``_processes``
    to ``None``, so the worker handles must be snapshotted BEFORE shutdown — a
    blocked worker is only killed if terminate() actually runs.
    """
    ex = ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    )
    procs = []
    manager_thread = None
    try:
        ex.submit(time.sleep, 0).result(timeout=30)
        future = ex.submit(time.sleep, 30)
        manager_thread = getattr(ex, "_executor_manager_thread", None)
        deadline = time.monotonic() + 5
        while (
            not future.running() and not future.done() and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if future.done():
            future.result()
        assert future.running(), "worker task did not start"
        procs = list((getattr(ex, "_processes", None) or {}).values())
        assert procs, "worker did not spawn"
        assert not wait_for_process(
            [p.sentinel for p in procs], timeout=0
        ), "worker exited before termination"

        _terminate_procs([ex])

        for p in procs:
            # The executor manager may reap the child concurrently; sentinel
            # readiness observes exit without racing its return-code update.
            assert wait_for_process(
                [p.sentinel], timeout=5
            ), "worker was not terminated"
    finally:
        cleanup_procs = procs or list((getattr(ex, "_processes", None) or {}).values())
        for p in cleanup_procs:
            if not wait_for_process([p.sentinel], timeout=0):
                try:
                    p.kill()
                except (OSError, ValueError):
                    pass
        ex.shutdown(wait=False, cancel_futures=True)
        if manager_thread is not None:
            manager_thread.join(5)
