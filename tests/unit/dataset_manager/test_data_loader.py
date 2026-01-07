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


import pytest
from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.dataset import (
    RandomDataGenerator,
)


def test_ds_pickle_reader(ds_pickle_reader):
    ds_pickle_reader.load()
    assert ds_pickle_reader.num_samples() == 5
    data_item = ds_pickle_reader.load_sample(0)
    assert isinstance(data_item, dict)
    print(data_item["dataset"])
    print(data_item["ground_truth"])
    print(data_item["ref_accuracy"])
    assert "dataset" in data_item and data_item["dataset"] == "livecodebench"
    assert "ground_truth" in data_item and data_item["ground_truth"] == "3154"
    assert "ref_accuracy" in data_item and data_item["ref_accuracy"] == 100.0


def test_ds_pickle_reader_unique_dataset(ds_pickle_reader):
    ds_pickle_reader.load()
    unique_datasets = set()
    for i in range(ds_pickle_reader.num_samples()):
        samples = ds_pickle_reader.load_sample(i)
        unique_datasets.add(samples["dataset"])
    assert len(unique_datasets) == 5


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
    datagen = RandomDataGenerator(
        num_sequences=num_sequences,
        input_seq_length=input_seq_length,
        range_ratio=range_ratio,
        random_seed=random_seed,
        tokenizer=tokenizer,
        save_tokenized_data=save_tokenized_data,
    )
    datagen.read()
    random_data_loader = Dataset(datagen.get_dataframe())
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
