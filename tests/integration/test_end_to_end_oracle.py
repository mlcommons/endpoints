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

"""End-to-end oracle test: verify responses match expected dataset outputs.

Uses the async BenchmarkSession to issue all samples to a mock oracle server,
then checks each response against the expected ground-truth output.
"""

import asyncio
import random
from urllib.parse import urljoin

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.core.record import EventRecord
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnRemap,
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import (
    HttpClientSampleIssuer,
)
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    PhaseConfig,
)


class _NoOpPublisher:
    """Minimal EventPublisher that discards all events."""

    def publish(self, event_record: EventRecord) -> None:
        pass

    def flush(self) -> None:
        pass


async def _run_oracle_test(url: str, dataloader: Dataset, rt_settings: RuntimeSettings):
    """Run benchmark session against an oracle server and verify responses."""
    loop = asyncio.get_running_loop()
    n_samples = dataloader.num_samples()

    # Collect responses via callback
    responses: dict[str, str] = {}

    def on_complete(result: QueryResult) -> None:
        responses[result.id] = result.get_response_output_string()

    # Create HTTP client with warmup disabled (test server)
    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(url, "/v1/chat/completions")],
        warmup_connections=0,
        num_workers=2,
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
        phases = [PhaseConfig("performance", rt_settings, dataloader)]
        result = await asyncio.wait_for(session.run(phases), timeout=60.0)
    finally:
        await http_client.shutdown_async()

    # Verify all samples got responses
    assert result.perf_results[0].issued_count == n_samples
    assert len(responses) == n_samples

    # Build expected values from dataset
    expected = {}
    for i in range(n_samples):
        entry = dataloader.load_sample(i)
        expected[i] = entry["output"]

    # Verify each response matches ground truth
    uuid_to_index = result.perf_results[0].uuid_to_index
    for sample_uuid, resp in responses.items():
        sample_index = uuid_to_index[sample_uuid]
        assert resp == expected[sample_index], (
            f"Sample {sample_uuid} (index {sample_index}): "
            f"expected {expected[sample_index][:50]!r}, got {resp[:50]!r}"
        )

    return responses


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_generator_full_run_mock_http_oracle_server(
    mock_http_oracle_server,
    ds_dataset_path,
    hf_model_name,
):
    dummy_dataloader = Dataset.load_from_file(
        ds_dataset_path,
        transforms=[
            ColumnRemap({"text_input": "prompt", "ref_output": "output"}),
            AddStaticColumns({"model": hf_model_name}),
        ],
    )
    dummy_dataloader.load()
    n_samples = dummy_dataloader.num_samples()
    assert n_samples > 0

    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=0,
        max_duration_ms=60_000,
        n_samples_from_dataset=n_samples,
        n_samples_to_issue=n_samples,
        min_sample_count=1,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    await _run_oracle_test(mock_http_oracle_server.url, dummy_dataloader, rt_settings)


async def _run_load_generator_full_run_url(url, dataset_path, hf_model_name):
    dummy_dataloader = Dataset.load_from_file(
        dataset_path,
        transforms=[
            ColumnRemap({"text_input": "prompt", "ref_output": "output"}),
            AddStaticColumns({"model": hf_model_name}),
        ],
    )
    dummy_dataloader.load()
    n_samples = dummy_dataloader.num_samples()
    assert n_samples > 0

    rt_settings = RuntimeSettings(
        metrics.Throughput(50),
        [metrics.Throughput(50)],
        min_duration_ms=0,
        max_duration_ms=60_000,
        n_samples_from_dataset=n_samples,
        n_samples_to_issue=n_samples,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    await _run_oracle_test(url, dummy_dataloader, rt_settings)


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_vllm_docker_server(
    vllm_docker_server,
    ds_dataset_path,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        vllm_docker_server.url, ds_dataset_path, hf_model_name
    )


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_sglang_docker_server(
    sglang_docker_server,
    ds_dataset_path,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        sglang_docker_server.url, ds_dataset_path, hf_model_name
    )


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_trtllm_docker_server(
    trtllm_docker_server,
    ds_dataset_path,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        trtllm_docker_server.url, ds_dataset_path, hf_model_name
    )
