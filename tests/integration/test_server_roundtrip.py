# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import aiohttp
import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.dataset_manager.dataloader import (
    DeepSeekR1ChatCompletionDataLoader,
)
from inference_endpoint.openai.openai_adapter import OpenAIAdapter
from inference_endpoint.openai.openai_types_gen import CreateChatCompletionResponse


@pytest.mark.asyncio
async def test_ds_chat_completion_data_loader_with_oracle_server(
    ds_pickle_dataset_path, mock_http_oracle_server
):
    """
    Test the DeepSeekR1ChatCompletionDataLoader by performing a roundtrip request through a mock HTTP Oracle server.

    Validates the end-to-end flow of loading dataset samples, transforming requests to OpenAI format,
    sending requests to a mock server, and verifying the server's responses match expected outputs.

    The test iterates through each sample in the dataset, sends an HTTP POST request to the mock server,
    and checks that the server returns a response matching the sample's reference output.
    """

    def parser(x):
        return {"prompt": x.text_input, "output": x.ref_output}

    ds_chat_completion_data_loader = DeepSeekR1ChatCompletionDataLoader(
        ds_pickle_dataset_path, parser=parser
    )
    ds_chat_completion_data_loader.load()
    assert ds_chat_completion_data_loader.num_samples() == 5
    for i in range(ds_chat_completion_data_loader.num_samples()):
        sample = ds_chat_completion_data_loader.load_sample(i)
        async with aiohttp.ClientSession() as session:
            payload = OpenAIAdapter.to_endpoint_request(
                Query(
                    id="test-chat-completions",
                    data={"prompt": str(sample["prompt"]), "model": "test-model"},
                )
            ).model_dump(mode="json")

            async with session.post(
                f"{mock_http_oracle_server.url}/v1/chat/completions", json=payload
            ) as response:
                assert response.status == 200

                response_data = await response.json()
                assert (
                    OpenAIAdapter.from_endpoint_response(
                        CreateChatCompletionResponse(**response_data)
                    ).response_output
                    == sample["output"]
                )
                logging.debug(
                    f"Sample {i} passed : in:\n {sample['prompt'][0:30]} out:\n {sample['output'][0:30]}"
                )
