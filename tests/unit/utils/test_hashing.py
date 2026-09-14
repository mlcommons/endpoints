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

"""Tests for the content-only file SHA-256 helper."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from inference_endpoint.utils.hashing import sha256_file


@pytest.mark.unit
class TestSha256File:
    """Content-only SHA-256 of a file (used to fingerprint events.jsonl)."""

    def test_hashes_contents_only(self, tmp_path: Path):
        """The digest is over file bytes alone, so the same content under a
        different name (a copy/rename) hashes identically and matches a plain
        hashlib digest of those bytes."""
        content = b'{"event": "COMPLETE"}\n' * 4
        a = tmp_path / "events.jsonl"
        b = tmp_path / "events_renamed.jsonl"
        a.write_bytes(content)
        b.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert sha256_file(a) == expected
        assert sha256_file(b) == expected

    def test_differs_when_content_differs(self, tmp_path: Path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_bytes(b"one")
        b.write_bytes(b"two")
        assert sha256_file(a) != sha256_file(b)

    def test_missing_file_returns_none(self, tmp_path: Path):
        """A run that never produced an event log (e.g. SIGKILL before salvage)
        yields None rather than crashing the caller."""
        assert sha256_file(tmp_path / "nope.jsonl") is None

    def test_hashes_large_file_streaming(self, tmp_path: Path):
        """Larger-than-one-chunk input hashes correctly (streamed read, not a
        single read_bytes), matching a plain hashlib digest."""
        content = b"x" * (5 * 1024 * 1024 + 7)
        path = tmp_path / "big.jsonl"
        path.write_bytes(content)
        assert sha256_file(path) == hashlib.sha256(content).hexdigest()
