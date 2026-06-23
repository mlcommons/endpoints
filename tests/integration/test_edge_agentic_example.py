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

"""CI smoke for the Edge-Agentic performance example.

Guards the things most likely to silently rot between releases without needing a
real model or the ``bfcl`` extra (the accuracy path's dependency, deliberately
excluded from CI):

  1. The combined example config still parses and keeps the locked submission
     params (deterministic decoding, single-stream) for both its perf and
     accuracy datasets.
  2. The committed performance dataset still loads through the production
     ``DataLoaderFactory`` path (catches dataset-schema drift).
  3. The perf pipeline runs end-to-end against the in-repo ``EchoServer`` on a
     real recorded conversation (production row schema: user/assistant/tool with
     tool_calls + tool_results) with 0 dropped turns — the run-validity invariant
     the compliance checker gates on.

Deliberately uses only core deps + EchoServer, so it runs in the standard CI
``test`` job (``-m "not slow and not performance"``).
"""

import asyncio
import collections
import json
import random
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    AgenticInferenceConfig,
    BenchmarkConfig,
    LoadPattern,
    LoadPatternType,
    ScorerMethod,
)
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.agentic_inference_dataset import (
    AgenticInferenceDataset,
)
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator.agentic_inference_strategy import (
    AgenticInferenceStrategy,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    EventPublisher,
    PhaseConfig,
    PhaseType,
)
from inference_endpoint.testing.echo_server import EchoServer

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "10_Edge_Agentic_Example"
FULL_RUN_CONFIG = "online_edge_full_run.yaml"
PERF_DATASET = EXAMPLE_DIR / "agentic_coding_2.5h.jsonl"


def _perf_dataset_cfg(cfg):
    return next(d for d in cfg.datasets if d.type.value == "performance")


def _acc_dataset_cfg(cfg):
    return next(d for d in cfg.datasets if d.type.value == "accuracy")


class _NoOpPublisher(EventPublisher):
    def publish(self, event_record) -> None:  # type: ignore[override]
        pass

    def flush(self) -> None:
        pass


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _smallest_conversation_rows(rows: list[dict]) -> list[dict]:
    counts = collections.Counter(r["conversation_id"] for r in rows)
    smallest = min(counts, key=lambda cid: counts[cid])
    return [r for r in rows if r["conversation_id"] == smallest]


# ---------------------------------------------------------------------------
# 1. Config parse + locked params
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_edge_full_run_config_parses_and_is_locked():
    # Combined perf+accuracy back-to-back run. Parsing does not load datasets,
    # so this validates the multi-dataset schema wiring without the bfcl extra.
    cfg = BenchmarkConfig.from_yaml_file(EXAMPLE_DIR / FULL_RUN_CONFIG)

    assert cfg.model_params.temperature == 0
    assert cfg.model_params.seed == 42
    assert cfg.settings.load_pattern.type == LoadPatternType.AGENTIC_INFERENCE
    assert cfg.settings.load_pattern.target_concurrency == 1
    assert cfg.settings.client.num_workers == 1
    assert cfg.settings.client.max_connections == 1

    perf = _perf_dataset_cfg(cfg)
    acc = _acc_dataset_cfg(cfg)
    assert perf.agentic_inference is not None
    assert perf.accuracy_config is not None
    assert perf.accuracy_config.eval_method == ScorerMethod.AGENTIC_INFERENCE_INLINE
    assert perf.path is not None
    assert (REPO_ROOT / perf.path).exists()
    assert acc.agentic_inference is None
    assert acc.accuracy_config is not None
    assert acc.accuracy_config.eval_method == ScorerMethod.BFCL_V4


# ---------------------------------------------------------------------------
# 2. Production dataset loader on the committed perf dataset
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_edge_perf_dataset_loads_via_factory():
    cfg = BenchmarkConfig.from_yaml_file(EXAMPLE_DIR / FULL_RUN_CONFIG)
    # Resolve the repo-relative dataset path to absolute so the loader is
    # independent of the working directory.
    dataset_cfg = _perf_dataset_cfg(cfg).model_copy(update={"path": str(PERF_DATASET)})

    loader = DataLoaderFactory.create_loader(dataset_cfg)
    loader.load()

    assert isinstance(loader, AgenticInferenceDataset)
    assert loader.conversation_metadata is not None
    assert loader.num_samples() > 0
    conversations = {s.conversation_id for s in loader.conversation_metadata.samples}
    assert len(conversations) == 20


# ---------------------------------------------------------------------------
# 3. End-to-end perf slice against EchoServer (real recorded conversation)
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_server():
    server = EchoServer(port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


async def _run_perf_slice(
    server_url: str, ds: AgenticInferenceDataset
) -> tuple[int, int]:
    """Run one perf phase of the dataset against echo; return (issued, completed)."""
    loop = asyncio.get_running_loop()
    responses: dict = {}

    agentic_cfg = AgenticInferenceConfig(turn_timeout_s=30.0)
    assert ds.conversation_metadata is not None
    strategy = AgenticInferenceStrategy(
        conversation_manager=ConversationManager(),
        dataset_metadata=ds.conversation_metadata,
        agentic_inference_config=agentic_cfg,
        target_concurrency=1,
    )

    def on_complete(result: QueryResult) -> None:
        strategy.on_sample_complete(result)
        responses[result.id] = result.get_response_output_string()

    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(server_url.rstrip("/") + "/", "v1/chat/completions")],
        warmup_connections=0,
        num_workers=1,
        max_connections=1,
        min_required_connections=0,
        worker_initialization_timeout=120.0,
    )
    http_client = await HTTPEndpointClient.create(http_config, loop)
    issuer = HttpClientSampleIssuer(http_client)
    try:
        session = BenchmarkSession(
            issuer=issuer,
            event_publisher=_NoOpPublisher(),
            loop=loop,
            on_sample_complete=on_complete,
        )
        rt = RuntimeSettings(
            metrics.Throughput(1000),
            [metrics.Throughput(1000)],
            min_duration_ms=0,
            max_duration_ms=60_000,
            n_samples_from_dataset=ds.num_samples(),
            n_samples_to_issue=ds.num_samples(),
            min_sample_count=1,
            rng_sched=random.Random(42),
            rng_sample_index=random.Random(42),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        phase = PhaseConfig("perf", rt, ds, PhaseType.PERFORMANCE, strategy=strategy)
        result = await asyncio.wait_for(session.run([phase]), timeout=60.0)
        return result.perf_results[0].issued_count, len(responses)
    finally:
        await http_client.shutdown_async()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edge_perf_smoke_zero_dropped_turns(echo_server):
    rows = _smallest_conversation_rows(_read_jsonl(PERF_DATASET))
    ds = AgenticInferenceDataset(dataframe=pd.DataFrame(rows))
    ds.load()
    expected_turns = len(ds.conversation_metadata.samples)
    assert expected_turns > 0

    issued, completed = await _run_perf_slice(echo_server.url, ds)

    # Run validity: every issued turn completed (0 dropped).
    assert issued == expected_turns
    assert completed == issued
