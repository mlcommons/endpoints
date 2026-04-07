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

"""Unit tests for Wan22Dataset."""

from pathlib import Path

import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.wan22.dataset import Wan22Dataset


@pytest.mark.unit
class TestWan22Dataset:
    @pytest.fixture
    def prompts_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "prompts.txt"
        p.write_text(
            "a golden retriever running in a field\n"
            "a red sports car on a mountain road\n"
            "\n"
            "cats playing in the snow\n"
        )
        return p

    def test_loads_prompts_skipping_blank_lines(self, prompts_file: Path):
        ds = Wan22Dataset(prompts_path=prompts_file)
        ds.load()
        assert ds.num_samples() == 3

    def test_load_sample_returns_expected_keys(self, prompts_file: Path):
        ds = Wan22Dataset(prompts_path=prompts_file)
        ds.load()
        sample = ds.load_sample(0)
        assert set(sample.keys()) == {
            "prompt",
            "negative_prompt",
            "sample_id",
            "sample_index",
        }

    def test_load_sample_correct_values(self, prompts_file: Path):
        ds = Wan22Dataset(prompts_path=prompts_file)
        ds.load()
        sample = ds.load_sample(1)
        assert sample["prompt"] == "a red sports car on a mountain road"
        assert sample["sample_index"] == 1
        assert sample["sample_id"] == "1"
        assert sample["negative_prompt"] == ""

    def test_negative_prompt_propagated(self, prompts_file: Path):
        ds = Wan22Dataset(prompts_path=prompts_file, negative_prompt="blurry")
        ds.load()
        assert ds.load_sample(0)["negative_prompt"] == "blurry"

    def test_index_wrapping(self, prompts_file: Path):
        ds = Wan22Dataset(prompts_path=prompts_file)
        ds.load()
        assert ds.load_sample(3)["prompt"] == ds.load_sample(0)["prompt"]
        assert ds.load_sample(7)["prompt"] == ds.load_sample(1)["prompt"]

    def test_load_before_load_raises_assertion(self, prompts_file: Path):
        ds = Wan22Dataset(prompts_path=prompts_file)
        with pytest.raises(AssertionError):
            ds.num_samples()

    def test_registered_as_predefined_dataset(self):
        assert "wan22_mlperf" in Dataset.PREDEFINED

    def test_get_dataloader_from_path(self, prompts_file: Path):
        ds = Wan22Dataset.get_dataloader(path=prompts_file)
        ds.load()
        assert ds.num_samples() == 3
        assert ds.load_sample(0)["prompt"] == "a golden retriever running in a field"

    def test_get_dataloader_passes_negative_prompt(self, prompts_file: Path):
        ds = Wan22Dataset.get_dataloader(path=prompts_file, negative_prompt="blurry")
        ds.load()
        assert ds.load_sample(0)["negative_prompt"] == "blurry"

    def test_get_dataloader_requires_path(self):
        with pytest.raises((TypeError, ValueError)):
            Wan22Dataset.get_dataloader()
