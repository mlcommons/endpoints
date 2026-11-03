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

import pytest
from inference_endpoint.config.runtime_settings import SampleOrderKind, SampleOrderSpec


@pytest.mark.unit
def test_default_is_without_replacement():
    spec = SampleOrderSpec()
    assert spec.kind == SampleOrderKind.WITHOUT_REPLACEMENT
    assert spec.fixed_index is None


@pytest.mark.unit
def test_without_replacement_factory():
    spec = SampleOrderSpec.without_replacement()
    assert spec.kind == SampleOrderKind.WITHOUT_REPLACEMENT
    assert spec.fixed_index is None


@pytest.mark.unit
def test_with_replacement_factory():
    spec = SampleOrderSpec.with_replacement()
    assert spec.kind == SampleOrderKind.WITH_REPLACEMENT
    assert spec.fixed_index is None


@pytest.mark.unit
def test_single_factory_carries_index():
    spec = SampleOrderSpec.single(3)
    assert spec.kind == SampleOrderKind.SINGLE
    assert spec.fixed_index == 3


@pytest.mark.unit
def test_spec_is_frozen():
    spec = SampleOrderSpec()
    with pytest.raises(AttributeError):
        spec.fixed_index = 5  # type: ignore[misc]
