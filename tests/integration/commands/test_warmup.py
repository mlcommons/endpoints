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

"""Integration tests for warmup phase behaviour.

Covers two properties:

Salt
  When ``salt=True`` the server receives distinct prompts for warmup vs perf
  (preventing KV-cache reuse).  When ``salt=False`` the prompts are identical.

Drain
  When ``drain=True`` all warmup responses complete before perf requests start
  (zero concurrent overlap at the server).  When ``drain=False`` the perf
  phase starts immediately, so the server handles both warmup and perf
  requests simultaneously.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

import pytest
from aiohttp import web
from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import Dataset as ConfigDataset
from inference_endpoint.config.schema import (
    DatasetType,
    EndpointConfig,
    LoadPattern,
    LoadPatternType,
    ModelParams,
    OfflineBenchmarkConfig,
    OfflineSettings,
    RuntimeConfig,
    StreamingMode,
    TestMode,
    WarmupConfig,
)
from inference_endpoint.core.types import QueryResult, TextModelOutput
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.openai.openai_adapter import OpenAIAdapter

# ── helpers ──────────────────────────────────────────────────────────────────

_SALT_RE = re.compile(r"^\[([0-9a-f]{16})\] (.+)$")

_MINIMAL_CLIENT = HTTPClientConfig(
    num_workers=1, warmup_connections=0, max_connections=10
)


def _echo_response(prompt: str) -> dict:
    """Build a minimal valid OpenAI chat-completion response."""
    req_id = str(uuid.uuid4())
    result = QueryResult(id=req_id, response_output=TextModelOutput(output=prompt))
    body = OpenAIAdapter.to_endpoint_response(result).model_dump(mode="json")
    body["id"] = req_id
    return body


async def _user_prompt(request: web.Request) -> str:
    """Extract the user message content from an OpenAI chat-completion request."""
    body = await request.json()
    for msg in body.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return str(content) if content is not None else ""
    return ""


def _offline_config(
    endpoint_url: str,
    dataset_path: str | Path,
    warmup: WarmupConfig,
    n_perf_samples: int = 5,
) -> OfflineBenchmarkConfig:
    return OfflineBenchmarkConfig(
        endpoint_config=EndpointConfig(endpoints=[endpoint_url]),
        model_params=ModelParams(name="test-model", streaming=StreamingMode.OFF),
        datasets=[ConfigDataset(path=str(dataset_path), type=DatasetType.PERFORMANCE)],
        settings=OfflineSettings(
            runtime=RuntimeConfig(min_duration_ms=0, n_samples_to_issue=n_perf_samples),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
            client=_MINIMAL_CLIENT,
            warmup=warmup,
        ),
    )


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def single_prompt_dataset(tmp_path: Path) -> Path:
    """JSONL with one sample: ``{"prompt": "hello world"}``."""
    f = tmp_path / "single.jsonl"
    f.write_text(json.dumps({"prompt": "hello world"}) + "\n")
    return f


@pytest.fixture
def multi_prompt_dataset(tmp_path: Path) -> Path:
    """JSONL with five distinct samples."""
    f = tmp_path / "multi.jsonl"
    f.write_text(
        "\n".join(json.dumps({"prompt": f"prompt_{i}"}) for i in range(5)) + "\n"
    )
    return f


# ── salt tests ────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestWarmupSalt:
    """Server-side prompt observations when salt is on vs off."""

    def test_salt_enabled_warmup_prompts_differ_from_perf(
        self,
        mock_http_echo_server_factory,
        single_prompt_dataset: Path,
    ):
        """With salt=True the server sees a distinct prompt for warmup vs perf.

        The warmup request carries a ``[<16-hex>] `` prefix; the perf request
        carries the raw prompt.  The base content after stripping the salt must
        match the perf prompt.
        """
        received: list[str] = []

        async def handler(req: web.Request) -> web.Response:
            prompt = await _user_prompt(req)
            received.append(prompt)
            return web.json_response(_echo_response(prompt))

        server = mock_http_echo_server_factory(handler)
        config = _offline_config(
            server.url,
            single_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=1, salt=True, drain=True),
            n_perf_samples=1,
        )
        run_benchmark(config, TestMode.PERF)

        assert (
            len(received) == 2
        ), f"Expected 2 requests (1 warmup + 1 perf), got: {received}"

        warmup_prompts = [p for p in received if _SALT_RE.match(p)]
        perf_prompts = [p for p in received if not _SALT_RE.match(p)]
        assert len(warmup_prompts) == 1, "Expected exactly 1 salted (warmup) prompt"
        assert len(perf_prompts) == 1, "Expected exactly 1 unsalted (perf) prompt"

        # Base content after stripping salt must equal the perf prompt
        m = _SALT_RE.match(warmup_prompts[0])
        assert m is not None
        assert (
            m.group(2) == perf_prompts[0]
        ), f"Stripped warmup prompt {m.group(2)!r} != perf prompt {perf_prompts[0]!r}"

    def test_salt_disabled_server_sees_identical_prompts(
        self,
        mock_http_echo_server_factory,
        single_prompt_dataset: Path,
    ):
        """With salt=False warmup and perf send the exact same prompt.

        A KV cache on the real server would reuse the cached result, defeating
        the purpose of warming up unique sequences.
        """
        received: list[str] = []

        async def handler(req: web.Request) -> web.Response:
            prompt = await _user_prompt(req)
            received.append(prompt)
            return web.json_response(_echo_response(prompt))

        server = mock_http_echo_server_factory(handler)
        config = _offline_config(
            server.url,
            single_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=1, salt=False, drain=True),
            n_perf_samples=1,
        )
        run_benchmark(config, TestMode.PERF)

        assert len(received) == 2, f"Expected 2 requests, got: {received}"
        assert (
            received[0] == received[1]
        ), f"Expected identical prompts with salt=False, got: {received}"
        assert not any(
            _SALT_RE.match(p) for p in received
        ), "No prompt should have a salt prefix when salt=False"

    def test_salt_count_matches_n_requests(
        self,
        mock_http_echo_server_factory,
        multi_prompt_dataset: Path,
    ):
        """Exactly ``n_requests`` salted prompts reach the server."""
        received: list[str] = []

        async def handler(req: web.Request) -> web.Response:
            prompt = await _user_prompt(req)
            received.append(prompt)
            return web.json_response(_echo_response(prompt))

        server = mock_http_echo_server_factory(handler)
        config = _offline_config(
            server.url,
            multi_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=3, salt=True, drain=True),
            n_perf_samples=5,
        )
        run_benchmark(config, TestMode.PERF)

        warmup_count = sum(1 for p in received if _SALT_RE.match(p))
        assert (
            warmup_count == 3
        ), f"Expected 3 warmup (salted) prompts, got {warmup_count} from: {received}"

    def test_each_salted_warmup_prompt_is_unique(
        self,
        mock_http_echo_server_factory,
        multi_prompt_dataset: Path,
    ):
        """Every warmup request has a distinct salt even when the same sample is reused.

        With n_requests (10) > dataset size (5) samples are cycled; without
        unique salts the same raw text would appear twice, allowing cache reuse.
        """
        received: list[str] = []

        async def handler(req: web.Request) -> web.Response:
            prompt = await _user_prompt(req)
            received.append(prompt)
            return web.json_response(_echo_response(prompt))

        server = mock_http_echo_server_factory(handler)
        config = _offline_config(
            server.url,
            multi_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=10, salt=True, drain=True),
            n_perf_samples=1,
        )
        run_benchmark(config, TestMode.PERF)

        warmup_prompts = [p for p in received if _SALT_RE.match(p)]
        assert len(warmup_prompts) == 10
        assert (
            len(set(warmup_prompts)) == 10
        ), "Expected all salted warmup prompts to be unique (distinct salt per request)"


# ── drain tests ───────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestWarmupDrain:
    """Concurrency overlap between warmup and perf at the server.

    Strategy
    --------
    * Use a slow server (``_DELAY`` seconds per response) so that warmup
      requests stay in-flight long enough for the perf phase to start.
    * Use ``salt=True`` to identify warmup vs perf requests at the server.
    * Track concurrent in-flight counts inside the server handler.  Since all
      handlers share a single asyncio event loop, plain Python ints are safe
      (no actual concurrent mutation — only cooperative interleaving at
      ``await`` points).
    * ``run_benchmark`` is synchronous and blocks until the perf phase drains,
      guaranteeing all overlap events have been recorded before we assert.
    """

    _DELAY = 0.15  # seconds; long enough for perf to start while warmup is pending

    def _make_handler(self, delay: float):
        """Return ``(handler_coro, state)``.

        ``state`` keys after ``run_benchmark`` returns:

        * ``overlap``        – ``True`` if warmup and perf were in-flight simultaneously
        * ``max_concurrent`` – peak total in-flight request count
        """
        state: dict = {
            "warmup_inflight": 0,
            "perf_inflight": 0,
            "overlap": False,
            "max_concurrent": 0,
        }

        async def handler(req: web.Request) -> web.Response:
            prompt = await _user_prompt(req)
            is_warmup = bool(_SALT_RE.match(prompt))

            if is_warmup:
                state["warmup_inflight"] += 1
            else:
                state["perf_inflight"] += 1

            total = state["warmup_inflight"] + state["perf_inflight"]
            if total > state["max_concurrent"]:
                state["max_concurrent"] = total

            if state["warmup_inflight"] > 0 and state["perf_inflight"] > 0:
                state["overlap"] = True

            await asyncio.sleep(delay)

            if is_warmup:
                state["warmup_inflight"] -= 1
            else:
                state["perf_inflight"] -= 1

            return web.json_response(_echo_response(prompt))

        return handler, state

    def test_drain_true_no_overlap(
        self,
        mock_http_echo_server_factory,
        multi_prompt_dataset: Path,
    ):
        """With ``drain=True`` the server never handles warmup and perf simultaneously.

        Timeline: all warmup responses arrive → perf phase starts → no overlap.
        """
        handler, state = self._make_handler(self._DELAY)
        server = mock_http_echo_server_factory(handler)

        config = _offline_config(
            server.url,
            multi_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=5, salt=True, drain=True),
            n_perf_samples=5,
        )
        run_benchmark(config, TestMode.PERF)

        assert not state[
            "overlap"
        ], "With drain=True warmup and perf must not be in-flight at the same time"

    def test_drain_false_overlap_observed(
        self,
        mock_http_echo_server_factory,
        multi_prompt_dataset: Path,
    ):
        """With ``drain=False`` perf requests start before warmup responses arrive.

        Timeline: warmup issues 5 requests at MAX_THROUGHPUT → immediately
        perf phase starts → both sets in-flight → overlap detected.
        """
        handler, state = self._make_handler(self._DELAY)
        server = mock_http_echo_server_factory(handler)

        config = _offline_config(
            server.url,
            multi_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=5, salt=True, drain=False),
            n_perf_samples=5,
        )
        run_benchmark(config, TestMode.PERF)

        assert state[
            "overlap"
        ], "With drain=False perf requests should start while warmup is in-flight"

    def test_drain_true_max_concurrent_bounded_by_phase_size(
        self,
        mock_http_echo_server_factory,
        multi_prompt_dataset: Path,
    ):
        """With ``drain=True`` at most one phase worth of requests is in-flight.

        Both warmup and perf have 5 requests; phases never overlap so the peak
        concurrent count is at most 5.
        """
        handler, state = self._make_handler(self._DELAY)
        server = mock_http_echo_server_factory(handler)

        config = _offline_config(
            server.url,
            multi_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=5, salt=True, drain=True),
            n_perf_samples=5,
        )
        run_benchmark(config, TestMode.PERF)

        assert state["max_concurrent"] <= 5, (
            f"With drain=True max concurrent {state['max_concurrent']} should be ≤ 5 "
            "(one phase at a time)"
        )

    def test_drain_false_max_concurrent_exceeds_single_phase_size(
        self,
        mock_http_echo_server_factory,
        multi_prompt_dataset: Path,
    ):
        """With ``drain=False`` the server concurrently handles requests from both phases.

        5 warmup requests stay in-flight while 5 perf requests arrive, so the
        server reaches a peak of more than 5 simultaneous requests.
        """
        handler, state = self._make_handler(self._DELAY)
        server = mock_http_echo_server_factory(handler)

        config = _offline_config(
            server.url,
            multi_prompt_dataset,
            warmup=WarmupConfig(enabled=True, n_requests=5, salt=True, drain=False),
            n_perf_samples=5,
        )
        run_benchmark(config, TestMode.PERF)

        assert state["max_concurrent"] > 5, (
            f"With drain=False peak concurrent {state['max_concurrent']} should be > 5 "
            "(warmup + perf requests overlap)"
        )
