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
from inference_endpoint.utils.atomic_write import atomic_write_bytes


@pytest.mark.unit
def test_writes_and_overwrites_leaving_no_tmp(tmp_path):
    p = tmp_path / "out.json"
    atomic_write_bytes(p, b'{"a": 1}')
    assert p.read_bytes() == b'{"a": 1}'
    # A second write replaces the content atomically.
    atomic_write_bytes(p, b'{"a": 2}')
    assert p.read_bytes() == b'{"a": 2}'
    # The sibling .tmp is consumed by the rename, not left behind.
    assert not (tmp_path / "out.json.tmp").exists()


@pytest.mark.unit
def test_raises_oserror_when_parent_missing(tmp_path):
    # A missing parent dir surfaces as OSError (the caller wraps it into
    # ExecutionError rather than silently dropping the artifact).
    with pytest.raises(OSError):
        atomic_write_bytes(tmp_path / "absent" / "out.json", b"{}")
