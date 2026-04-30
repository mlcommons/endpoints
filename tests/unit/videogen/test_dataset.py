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

"""Unit tests for VideoGenDataset."""

from pathlib import Path

import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.videogen.dataset import _MLPERF_NEGATIVE_PROMPT, VideoGenDataset


@pytest.mark.unit
class TestVideoGenDataset:
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
        ds = VideoGenDataset(prompts_path=prompts_file)
        ds.load()
        assert ds.num_samples() == 3

    def test_load_sample_default_includes_mlperf_negative_prompt(
        self, prompts_file: Path
    ):
        ds = VideoGenDataset(prompts_path=prompts_file)
        ds.load()
        sample = ds.load_sample(0)
        assert set(sample.keys()) == {
            "prompt",
            "negative_prompt",
            "sample_id",
            "sample_index",
        }
        assert sample["negative_prompt"] == _MLPERF_NEGATIVE_PROMPT

    def test_load_sample_correct_values(self, prompts_file: Path):
        ds = VideoGenDataset(prompts_path=prompts_file)
        ds.load()
        sample = ds.load_sample(1)
        assert sample["prompt"] == "a red sports car on a mountain road"
        assert sample["sample_index"] == 1
        assert sample["sample_id"] == "1"
        assert sample["negative_prompt"] == _MLPERF_NEGATIVE_PROMPT

    def test_negative_prompt_override(self, prompts_file: Path):
        ds = VideoGenDataset(prompts_path=prompts_file, negative_prompt="blurry")
        ds.load()
        assert ds.load_sample(0)["negative_prompt"] == "blurry"

    def test_negative_prompt_none_omits_field(self, prompts_file: Path):
        ds = VideoGenDataset(prompts_path=prompts_file, negative_prompt=None)
        ds.load()
        assert "negative_prompt" not in ds.load_sample(0)

    def test_latent_path_propagated(self, prompts_file: Path, tmp_path: Path):
        latent = tmp_path / "fixed_latent.pt"
        ds = VideoGenDataset(prompts_path=prompts_file, latent_path=latent)
        ds.load()
        assert ds.load_sample(0)["latent_path"] == str(latent)

    def test_latent_path_default_omitted(self, prompts_file: Path):
        ds = VideoGenDataset(prompts_path=prompts_file)
        ds.load()
        assert "latent_path" not in ds.load_sample(0)

    def test_index_wrapping(self, prompts_file: Path):
        ds = VideoGenDataset(prompts_path=prompts_file)
        ds.load()
        assert ds.load_sample(3)["prompt"] == ds.load_sample(0)["prompt"]
        assert ds.load_sample(7)["prompt"] == ds.load_sample(1)["prompt"]

    def test_load_before_load_raises_assertion(self, prompts_file: Path):
        ds = VideoGenDataset(prompts_path=prompts_file)
        with pytest.raises(AssertionError):
            ds.num_samples()

    def test_registered_as_predefined_dataset(self):
        assert "wan22_mlperf" in Dataset.PREDEFINED

    def test_get_dataloader_from_path(self, prompts_file: Path):
        ds = VideoGenDataset.get_dataloader(path=prompts_file)
        ds.load()
        assert ds.num_samples() == 3
        assert ds.load_sample(0)["prompt"] == "a golden retriever running in a field"

    def test_get_dataloader_passes_negative_prompt(self, prompts_file: Path):
        ds = VideoGenDataset.get_dataloader(path=prompts_file, negative_prompt="blurry")
        ds.load()
        assert ds.load_sample(0)["negative_prompt"] == "blurry"

    def test_get_dataloader_requires_path(self):
        with pytest.raises((TypeError, ValueError)):
            VideoGenDataset.get_dataloader()
