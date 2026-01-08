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

import logging
import random
from urllib.parse import urljoin

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnNameRemap,
)
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator import (
    BenchmarkSession,
    MaxThroughputScheduler,
    SampleEvent,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)

from tests.test_helpers import get_test_socket_path


class DeepSeekR1SampleIssuer(HttpClientSampleIssuer):
    def __init__(self, tmp_path: str, url: str):
        self.http_config = HTTPClientConfig(
            endpoint_url=urljoin(url, "/v1/chat/completions"),
            num_workers=16,
        )
        self.aiohttp_config = AioHttpConfig()
        self.zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_streaming", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_streaming", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "test_streaming", "_ready"
            ),
        )
        super().__init__(
            HTTPEndpointClient(self.http_config, self.aiohttp_config, self.zmq_config)
        )


async def run_benchmark(server_url, dataloader, tmp_path, rt_settings):
    # Step 1. Register the complete hook to store the responses from the server.
    server_responses: {str: str} = {}

    def on_complete_hook(result: QueryResult):
        """Callback to store the responses from the server."""
        server_responses[result.id] = result.response_output

    SampleEventHandler.register_hook(SampleEvent.COMPLETE, on_complete_hook)

    # Step 2. Create the scheduler.
    scheduler = MaxThroughputScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
    )
    logging.info(f"Number of samples to issue: {scheduler.total_samples_to_issue}")

    try:
        # Step 3. Create the sample issuer.
        sample_issuer = DeepSeekR1SampleIssuer(tmp_path, server_url)

        # Step 4. Create the benchmark session.
        sess = BenchmarkSession.start(
            rt_settings,
            dataloader,
            sample_issuer,
            scheduler,
            name="pytest_run_benchmark",
            max_shutdown_timeout_s=3 * 60,
        )

        # Step 5. Wait for the test to end.
        logging.info("Waiting for the test to end...")
        sess.wait_for_test_end()
        # Step 6. Return the sample UUID map and the server responses.
        return sess.sample_uuid_map, server_responses
    finally:
        # Step 7. Shutdown the sample issuer and the HTTP client.
        sample_issuer.shutdown()
        sample_issuer.http_client.shutdown()


"""
Test the load generator full run with a given URL.
"""


async def _run_load_generator_full_run_url(
    url, dataset_path, tmp_path, clean_sample_event_hooks, hf_model_name
):
    dummy_dataloader = Dataset.load_from_file(
        dataset_path,
        transforms=[
            ColumnNameRemap({"text_input": "prompt", "ref_output": "output"}),
            AddStaticColumns({"model": hf_model_name}),
        ],
    )
    dummy_dataloader.load()
    assert dummy_dataloader.num_samples() > 0

    rt_settings = RuntimeSettings(
        metrics.Throughput(50),
        [metrics.Throughput(50)],
        min_duration_ms=1_00,
        max_duration_ms=1_000,
        n_samples_from_dataset=dummy_dataloader.num_samples(),
        n_samples_to_issue=dummy_dataloader.num_samples(),
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    scheduler = MaxThroughputScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
    )
    logging.info(f"Number of samples to issue: {scheduler.total_samples_to_issue}")
    # Now call the benchmark
    sample_uuid_map, response_cache = await run_benchmark(
        url, dummy_dataloader, tmp_path, rt_settings
    )
    num_responses_in_cache = len(response_cache)
    assert (
        num_responses_in_cache == scheduler.total_samples_to_issue
    ), "Number of samples in response cache and number of samples in dataset should be the same"
    vals = {}
    for i in range(dummy_dataloader.num_samples()):
        entry = dummy_dataloader.load_sample(i)
        vals[i] = entry["output"]
    num_samples_in_dataset = len(vals)
    logging.info(f"Number of samples in dataset: {num_samples_in_dataset}")
    logging.info(f"Total samples to issue: {scheduler.total_samples_to_issue}")
    logging.info(f"Request data: {num_responses_in_cache}")

    for sample_uuid, resp in response_cache.items():
        if resp is None:
            logging.error(f"Sample {sample_uuid} has no response")
        else:
            sample_index = sample_uuid_map[sample_uuid].index
            logging.info(
                f"Sample {sample_uuid} should have been response {vals[sample_index][0:30]}, but was response {resp[0:30]}"
            )


@pytest.mark.asyncio
async def test_load_generator_full_run_mock_http_oracle_server(
    mock_http_oracle_server,
    ds_pickle_dataset_path,
    tmp_path,
    clean_sample_event_hooks,
    hf_model_name,
):
    dummy_dataloader = Dataset.load_from_file(
        ds_pickle_dataset_path,
        transforms=[
            ColumnNameRemap({"text_input": "prompt", "ref_output": "output"}),
            AddStaticColumns({"model": hf_model_name}),
        ],
    )
    dummy_dataloader.load()
    assert dummy_dataloader.num_samples() > 0

    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=1_000,
        max_duration_ms=10_000_000,
        n_samples_from_dataset=dummy_dataloader.num_samples(),
        n_samples_to_issue=dummy_dataloader.num_samples(),
        min_sample_count=1,
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    scheduler = MaxThroughputScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
    )
    logging.info(f"Number of samples to issue: {scheduler.total_samples_to_issue}")

    sample_uuid_map, response_cache = await run_benchmark(
        mock_http_oracle_server.url, dummy_dataloader, tmp_path, rt_settings
    )
    num_responses_in_cache = len(response_cache)
    assert (
        num_responses_in_cache == scheduler.total_samples_to_issue
    ), "Number of samples in response cache and number of samples in dataset should be the same"
    vals = {}
    for i in range(dummy_dataloader.num_samples()):
        entry = dummy_dataloader.load_sample(i)
        vals[i] = entry["output"]
    num_samples_in_dataset = len(vals)
    logging.info(f"Number of samples in dataset: {num_samples_in_dataset}")
    logging.info(f"Total samples to issue: {scheduler.total_samples_to_issue}")
    logging.info(f"Request data: {num_responses_in_cache}")
    assert (
        num_samples_in_dataset == scheduler.total_samples_to_issue
    ), "Number of samples in dataset and number of samples in request data should be the same"

    for sample_uuid, resp in response_cache.items():
        sample_index = sample_uuid_map["performance"][sample_uuid]
        logging.info(
            f"Sample {sample_uuid} should have been response {vals[sample_index][0:30]}, but was response {resp[0:30]}"
        )
        assert (
            resp == vals[sample_index]
        ), f"Sample {sample_uuid} should have been response {vals[sample_index][0:30]}, but was response {resp[0:30]}"


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_vllm_docker_server(
    vllm_docker_server,
    ds_pickle_dataset_path,
    tmp_path,
    clean_sample_event_hooks,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        vllm_docker_server.url,
        ds_pickle_dataset_path,
        tmp_path,
        clean_sample_event_hooks,
        hf_model_name,
    )


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_sglang_docker_server(
    sglang_docker_server,
    ds_pickle_dataset_path,
    tmp_path,
    clean_sample_event_hooks,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        sglang_docker_server.url,
        ds_pickle_dataset_path,
        tmp_path,
        clean_sample_event_hooks,
        hf_model_name,
    )


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
async def test_load_generator_full_run_trtllm_docker_server(
    trtllm_docker_server,
    ds_pickle_dataset_path,
    tmp_path,
    clean_sample_event_hooks,
    hf_model_name,
):
    await _run_load_generator_full_run_url(
        trtllm_docker_server.url,
        ds_pickle_dataset_path,
        tmp_path,
        clean_sample_event_hooks,
        hf_model_name,
    )
