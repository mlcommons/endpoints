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
from inference_endpoint.utils import byte_quantity_to_str


@pytest.mark.parametrize(
    "n_bytes, max_unit, expected",
    [
        (1024, "GB", "1KB"),
        (1024 * 1024, "TB", "1MB"),
        (1024 * 1024, "GB", "1MB"),
        (1024 * 1024 * 1024, "GB", "1GB"),
        (1024 * 1024 * 1024 * 1024, "GB", "1024GB"),
        (1024 * 1024 * 1024 * 1024, "TB", "1TB"),
        (5 * 1024 * 1024, "TB", "5MB"),
        (1024 * 1024, "KB", "1024KB"),
    ],
)
def test_byte_quantity_to_str(n_bytes, max_unit, expected):
    assert byte_quantity_to_str(n_bytes, max_unit=max_unit) == expected
