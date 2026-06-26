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

"""Tests for git_sha resolution (must track the endpoints package, not the cwd)."""

import pytest

from inference_endpoint.utils import version


@pytest.mark.unit
def test_git_sha_from_archival_substitution(tmp_path, monkeypatch):
    """A substituted _git_archival.txt (git archive export-subst) -> short 7-char SHA."""
    archival = tmp_path / "_git_archival.txt"
    archival.write_text("node: " + "a" * 40 + "\n")
    monkeypatch.setattr(version, "_GIT_ARCHIVAL", archival)
    version.get_git_sha.cache_clear()
    assert version.get_git_sha() == "a" * 7
    version.get_git_sha.cache_clear()


@pytest.mark.unit
def test_git_sha_placeholder_never_leaks(tmp_path, monkeypatch):
    """An unsubstituted "$Format:%H$" placeholder must never be returned verbatim:
    fall through to git (or None), but never emit the literal placeholder."""
    archival = tmp_path / "_git_archival.txt"
    archival.write_text("node: $Format:%H$\n")
    monkeypatch.setattr(version, "_GIT_ARCHIVAL", archival)
    version.get_git_sha.cache_clear()
    sha = version.get_git_sha()
    version.get_git_sha.cache_clear()
    assert sha is None or "$" not in sha


@pytest.mark.unit
def test_git_sha_anchored_to_package_not_cwd(tmp_path, monkeypatch):
    """With the real archival placeholder and the process cwd pointed at an unrelated
    non-git dir, get_git_sha must resolve via the package's own repo (dev checkout ->
    the endpoints HEAD) or None — it must not pick up the caller's cwd repo."""
    monkeypatch.chdir(tmp_path)
    version.get_git_sha.cache_clear()
    sha = version.get_git_sha()
    version.get_git_sha.cache_clear()
    assert sha is None or (len(sha) == 7 and "$" not in sha)
