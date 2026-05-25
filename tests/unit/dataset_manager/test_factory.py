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

import inference_endpoint.dataset_manager.factory as factory_module
import pytest
from inference_endpoint.config.schema import Dataset as DatasetConfig
from inference_endpoint.config.schema import MultiTurnConfig
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.exceptions import InputValidationError


@pytest.mark.unit
def test_enable_salt_requires_multi_turn_loader(monkeypatch):
    def fake_load_from_file(*args, **kwargs):
        return Dataset()

    monkeypatch.setattr(
        factory_module.Dataset,
        "load_from_file",
        staticmethod(fake_load_from_file),
    )

    config = DatasetConfig(
        path="data.jsonl",
        format=".jsonl",
        multi_turn=MultiTurnConfig(enable_salt=True),
    )

    with pytest.raises(InputValidationError, match="enable_salt"):
        factory_module.DataLoaderFactory.create_loader(config)
