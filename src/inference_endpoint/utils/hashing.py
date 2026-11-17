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

"""Content hashing for run artifacts (e.g. fingerprinting events.jsonl)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hex SHA-256 over the bytes of ``path``.

    Streamed (``hashlib.file_digest``), so a large file is not read into memory
    at once. The digest covers file contents only — never the name or metadata —
    so copying or renaming the file leaves it unchanged. Raises ``OSError``
    (including ``FileNotFoundError``) if the file cannot be read.
    """
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
