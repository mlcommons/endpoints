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

"""Tests that APIType.WAN22 is registered and wired up correctly."""

from importlib import import_module

import pytest
from inference_endpoint.core.types import APIType
from inference_endpoint.endpoint_client.config import ACCUMULATOR_MAP, ADAPTER_MAP


@pytest.mark.unit
def test_api_type_wan22_exists():
    assert APIType.WAN22 == "wan22"


@pytest.mark.unit
def test_api_type_wan22_default_route():
    assert APIType.WAN22.default_route() == "/v1/videos/generations"


@pytest.mark.unit
def test_wan22_in_adapter_map():
    assert APIType.WAN22 in ADAPTER_MAP
    assert "Wan22Adapter" in ADAPTER_MAP[APIType.WAN22]


@pytest.mark.unit
def test_wan22_in_accumulator_map():
    assert APIType.WAN22 in ACCUMULATOR_MAP
    assert "Wan22Accumulator" in ACCUMULATOR_MAP[APIType.WAN22]


@pytest.mark.unit
def test_wan22_adapter_loadable():
    path = ADAPTER_MAP[APIType.WAN22]
    module_path, class_name = path.rsplit(".", 1)
    mod = import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None


@pytest.mark.unit
def test_wan22_accumulator_loadable():
    path = ACCUMULATOR_MAP[APIType.WAN22]
    module_path, class_name = path.rsplit(".", 1)
    mod = import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None
