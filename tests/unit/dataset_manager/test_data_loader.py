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

from typing import Any

import pytest
from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.dataset import (
    DatasetFormat,
    RandomDataGenerator,
    RowProcessor,
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


def test_custom_parser_pickle_reader(ds_pickle_dataset_path):
    class TestRowProcessor(RowProcessor):
        def __init__(self):
            super().__init__()

        def __call__(self, row: dict[str, Any]) -> Any:
            # custom parser to only return dataset and text_input
            return {"dataset": row["dataset"], "text_input": row["text_input"]}

    data_loader = Dataset.load_from_file(
        ds_pickle_dataset_path, row_processor=TestRowProcessor()
    )
    data_loader.load()
    # check number of samples
    assert data_loader.num_samples() == 5
    # check first sample
    samples = data_loader.load_sample(0)

    # check columns that were not requested are not present
    assert "ref_output" not in samples and "metric" not in samples
    # check columns that were requested are present
    assert "dataset" in samples and "text_input" in samples
    # check order or rows - zeroth row should be livecodebench
    assert samples["dataset"] == "livecodebench"


def test_hf_squad_dataset(hf_squad_dataset):
    hf_squad_dataset.load()
    assert hf_squad_dataset.num_samples() == 50
    sample = hf_squad_dataset.load_sample(0)
    assert all(k in sample for k in ["id", "title", "context", "question", "answers"])
    assert sample["title"] == "Egypt"


def test_custom_parser_hf_squad_dataset(hf_squad_dataset_path):
    def parser(row):
        return {
            "title": row["title"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"],
        }

    dataset = Dataset.load_from_file(
        file_path=hf_squad_dataset_path, row_processor=parser, format=DatasetFormat.HF
    )
    dataset.load()
    assert dataset.num_samples() == 50
    sample = dataset.load_sample(0)
    assert "id" not in sample
    assert all(k in sample for k in ["title", "context", "question", "answers"])
    assert sample["title"] == "Egypt"


@pytest.mark.slow
@pytest.mark.parametrize("range_ratio", [0.5, 0.8, 1.0])
def test_random_data_loader(range_ratio):
    num_sequences = 1024
    input_seq_length = 1024
    random_seed = 42
    tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    save_tokenized_data = True
    random_data_loader = Dataset(
        RandomDataGenerator(
            num_sequences=num_sequences,
            input_seq_length=input_seq_length,
            range_ratio=range_ratio,
            random_seed=random_seed,
            tokenizer=tokenizer,
            save_tokenized_data=save_tokenized_data,
        ).get_dataframe()
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
