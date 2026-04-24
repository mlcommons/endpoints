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

"""Verify videogen public namespace exports."""

import pytest
from inference_endpoint.videogen import (
    VideoPathRequest,
    VideoPathResponse,
    VideoPayloadResponse,
    VideoGenAccumulator,
    VideoGenAdapter,
)


@pytest.mark.unit
class TestVideoGenPublicExports:
    def test_all_public_exports_importable(self):
        assert VideoPathRequest is not None
        assert VideoPathResponse is not None
        assert VideoPayloadResponse is not None
        assert VideoGenAdapter is not None
        assert VideoGenAccumulator is not None
