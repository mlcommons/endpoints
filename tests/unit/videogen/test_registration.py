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

"""Tests that APIType.VIDEOGEN is registered and wired up correctly."""

from importlib import import_module

import pytest
from inference_endpoint.core.types import APIType
from inference_endpoint.endpoint_client.config import ACCUMULATOR_MAP, ADAPTER_MAP


@pytest.mark.unit
def test_api_type_videogen_exists():
    assert APIType.VIDEOGEN == "videogen"


@pytest.mark.unit
def test_api_type_videogen_default_route():
    assert APIType.VIDEOGEN.default_route() == "/v1/videos/generations"


@pytest.mark.unit
def test_videogen_in_adapter_map():
    assert APIType.VIDEOGEN in ADAPTER_MAP
    assert "VideoGenAdapter" in ADAPTER_MAP[APIType.VIDEOGEN]


@pytest.mark.unit
def test_videogen_in_accumulator_map():
    assert APIType.VIDEOGEN in ACCUMULATOR_MAP
    assert "VideoGenAccumulator" in ACCUMULATOR_MAP[APIType.VIDEOGEN]


@pytest.mark.unit
def test_videogen_adapter_loadable():
    path = ADAPTER_MAP[APIType.VIDEOGEN]
    module_path, class_name = path.rsplit(".", 1)
    mod = import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None


@pytest.mark.unit
def test_videogen_accumulator_loadable():
    path = ACCUMULATOR_MAP[APIType.VIDEOGEN]
    module_path, class_name = path.rsplit(".", 1)
    mod = import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None
