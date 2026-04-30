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

"""Unit tests: DataLoaderFactory creates VideoGenDataset from --dataset path."""

from pathlib import Path

import pytest

from inference_endpoint.config.schema import Dataset as DatasetConfig
from inference_endpoint.dataset_manager.factory import DataLoaderFactory
from inference_endpoint.videogen.dataset import VideoGenDataset


@pytest.fixture
def prompts_file(tmp_path: Path) -> Path:
    p = tmp_path / "prompts.txt"
    p.write_text(
        "a golden retriever running in a field\n"
        "ocean waves at sunset\n"
    )
    return p


@pytest.mark.unit
class TestFactoryVideoGenDataset:
    def test_factory_creates_wan22_dataset_from_name_and_path(self, prompts_file: Path):
        config = DatasetConfig(name="wan22_mlperf", path=str(prompts_file))
        ds = DataLoaderFactory.create_loader(config)
        ds.load()
        assert isinstance(ds, VideoGenDataset)
        assert ds.num_samples() == 2

    def test_factory_wan22_sample_has_prompt(self, prompts_file: Path):
        config = DatasetConfig(name="wan22_mlperf", path=str(prompts_file))
        ds = DataLoaderFactory.create_loader(config)
        ds.load()
        sample = ds.load_sample(0)
        assert sample["prompt"] == "a golden retriever running in a field"

    def test_factory_wan22_requires_path(self):
        config = DatasetConfig(name="wan22_mlperf")
        with pytest.raises((TypeError, ValueError)):
            DataLoaderFactory.create_loader(config)
