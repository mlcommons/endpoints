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

"""Tests for version/git_sha derivation from the VCS-derived __version__."""

import pytest

from inference_endpoint.utils import version


def _clear_caches():
    version.get_git_sha.cache_clear()
    version.get_version_info.cache_clear()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ver", "expected_sha"),
    [
        ("0.6.dev3+g6eac351", "6eac351"),  # dev build after a tag
        ("0.5.post2+g1234abc.d20260626", "1234abc"),  # dirty working tree (date suffix)
        ("0.5", None),  # clean tagged release -> no local segment, no SHA
        ("0.0.0+unknown", None),  # fresh checkout, not yet built
    ],
)
def test_git_sha_parsed_from_version(monkeypatch, ver, expected_sha):
    """get_git_sha extracts the +g<sha> local segment from the VCS version."""
    monkeypatch.setattr(version, "__version__", ver)
    _clear_caches()
    assert version.get_git_sha() == expected_sha
    _clear_caches()


@pytest.mark.unit
def test_version_info_carries_version_and_git_sha(monkeypatch):
    """get_version_info surfaces the SHA-embedded version plus the parsed git_sha."""
    monkeypatch.setattr(version, "__version__", "0.6.dev3+g6eac351")
    _clear_caches()
    info = version.get_version_info()
    assert info["version"] == "0.6.dev3+g6eac351"
    assert info["git_sha"] == "6eac351"
    _clear_caches()
