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

"""Tests for version utilities."""

import pytest
from inference_endpoint import __version__
from inference_endpoint.utils.version import get_git_sha, get_version_info


@pytest.mark.unit
def test_get_git_sha():
    """Test that get_git_sha returns a string or None."""
    sha = get_git_sha()
    if sha is not None:
        assert isinstance(sha, str)
        assert len(sha) == 7  # Short SHA should be 7 chars
        assert sha.isalnum()  # Should only contain alphanumeric chars


@pytest.mark.unit
def test_get_version_info():
    """Test that get_version_info returns correct structure."""
    info = get_version_info()
    assert isinstance(info, dict)
    assert "version" in info
    assert "git_sha" in info
    assert info["version"] == __version__
    # git_sha can be None if not in a git repo
    if info["git_sha"] is not None:
        assert isinstance(info["git_sha"], str)
        assert len(info["git_sha"]) == 7


@pytest.mark.unit
def test_version_info_cached():
    """Test that get_version_info is properly cached."""
    info1 = get_version_info()
    info2 = get_version_info()
    # Should return the same object due to lru_cache
    assert info1 is info2
