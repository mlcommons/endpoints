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

import json

import pandas as pd
import pytest
from inference_endpoint.config.schema import APIType, ModelParams, StreamingMode
from inference_endpoint.core.types import Query
from inference_endpoint.dataset_manager.agentic_inference_dataset import (
    AgenticInferenceDataset,
)
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.random import RandomDataset
from inference_endpoint.dataset_manager.transforms import ColumnFilter
from inference_endpoint.openai.openai_msgspec_adapter import OpenAIMsgspecAdapter

_STREAMING = ModelParams(name="m", streaming=StreamingMode.ON)


def _plain_sample() -> dict:
    # A preset ColumnFilter that keeps the column replaces the adapter's filter.
    ds = Dataset(
        pd.DataFrame([{"prompt": "hi", "include_usage": True}]),
        transforms=[
            ColumnFilter(
                required_columns=["prompt"], optional_columns=["include_usage"]
            )
        ],
    )
    ds.load(api_type=APIType.OPENAI, model_params=_STREAMING)
    return ds.load_sample(0)


def _agentic_sample() -> dict:
    row = {
        "conversation_id": "c1",
        "turn": 1,
        "role": "user",
        "content": "hi",
        "stream": True,
        "include_usage": True,
    }
    ds = AgenticInferenceDataset(pd.DataFrame([row]))
    ds.load(api_type=APIType.OPENAI, model_params=_STREAMING)
    return ds.load_sample(0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "load", [_plain_sample, _agentic_sample], ids=["dataset", "agentic"]
)
def test_dataset_column_named_include_usage_passes_through(load):
    sample = load()
    assert sample["include_usage"] is True
    # The run setting, not the column, decides whether usage is requested.
    body = json.loads(
        OpenAIMsgspecAdapter.encode_query(Query(data=sample), include_usage=False)
    )
    assert "stream_options" not in body and "include_usage" not in body


@pytest.mark.unit
def test_ds_reader(ds_reader):
    ds_reader.load()
    assert ds_reader.num_samples() == 5
    data_item = ds_reader.load_sample(0)
    assert isinstance(data_item, dict)
    print(data_item["dataset"])
    print(data_item["ground_truth"])
    print(data_item["ref_accuracy"])
    assert "dataset" in data_item and data_item["dataset"] == "livecodebench"
    assert "ground_truth" in data_item and data_item["ground_truth"] == "3154"
    assert "ref_accuracy" in data_item and data_item["ref_accuracy"] == 100.0


@pytest.mark.unit
def test_ds_reader_unique_dataset(ds_reader):
    ds_reader.load()
    unique_datasets = set()
    for i in range(ds_reader.num_samples()):
        samples = ds_reader.load_sample(i)
        unique_datasets.add(samples["dataset"])
    assert len(unique_datasets) == 5


@pytest.mark.unit
@pytest.mark.skip(reason="Skipping dataset test for now due to refactoring")
def test_hf_squad_dataset(hf_squad_dataset):
    hf_squad_dataset.load()
    assert hf_squad_dataset.num_samples() == 50
    sample = hf_squad_dataset.load_sample(0)
    assert all(k in sample for k in ["id", "title", "context", "question", "answers"])
    assert sample["title"] == "Egypt"


@pytest.mark.slow
@pytest.mark.parametrize("range_ratio", [0.5, 0.8, 1.0])
def test_random_data_loader(range_ratio):
    num_sequences = 1024
    input_seq_length = 1024
    random_seed = 42
    tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    save_tokenized_data = True
    random_data_loader = RandomDataset.get_dataloader(
        num_sequences=num_sequences,
        input_seq_length=input_seq_length,
        range_ratio=range_ratio,
        random_seed=random_seed,
        tokenizer=tokenizer,
        save_tokenized_data=save_tokenized_data,
    )
    random_data_loader.load()
    assert (
        len(random_data_loader.data) == num_sequences
    ), f"Expected {num_sequences} samples, got {random_data_loader.num_samples()}"
    # Note that the input tokens are only loaded if save_tokenized_data is True, useful for debugging or other purposes
    assert (
        random_data_loader.num_samples() == num_sequences
    ), f"Expected {num_sequences} samples, got {random_data_loader.num_samples()}"
    # Go over the data and check the input tokens and the data length
    for i in range(random_data_loader.num_samples()):
        sample = random_data_loader.load_sample(i)
        assert isinstance(
            sample["prompt"], str
        ), f"Expected string, got {type(random_data_loader.data[i])}"
        # Note that the number of tokens can be smaller than the input_seq_length * range ration due to
        # the decoding-encoding which may coalesce some sequences to newer tokens. We use a 0.8 factor to allow for this.
        # And we allow for a 20% overhead due to the decoding-encoding.
        assert (
            len(sample["input_tokens"]) > input_seq_length * range_ratio * 0.8
            and len(sample["input_tokens"]) <= input_seq_length * 1.2
        ), f"Expected {input_seq_length*range_ratio*0.8} to {input_seq_length*0.2} input tokens, got {len(sample["input_tokens"])}"

        assert (
            len(sample["prompt"]) >= 1024 * range_ratio * 0.5
            and len(sample["prompt"]) <= 7 * 1024
        ), f"Expected length between 1024*range_ratio*0.5 and 1024, got {len(sample["prompt"])}"
