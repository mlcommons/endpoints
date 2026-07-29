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

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

SECRET_KEY_NAMES = {
    "api-key",
    "api_key",
    "authorization",
    "access-token",
    "access_token",
    "endpoint-api-key",
    "endpoint_api_key",
    "password",
    "secret",
    "secret-key",
    "secret_key",
    "token",
    "x-api-key",
    "x_api_key",
}
SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*(?:bearer|basic)\s+)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|token|password)=)[^&\s]+"),
    re.compile(r"(://[^:/\s]+:)[^@\s/]+(@)"),
)
SAFE_ARTIFACT_NAMES = {
    "preds.json",
    "swe_bench_agent.log",
    "swe_bench_eval.log",
    "swe_bench_results.json",
    "status.json",
}


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    if normalized in SECRET_KEY_NAMES:
        return True
    compact = normalized.replace("-", "_")
    return (
        compact.endswith("_key")
        or compact.endswith("_token")
        or "api_key" in compact
        or "access_token" in compact
        or "secret" in compact
        or "password" in compact
    )


def redact_text(text: str, secret_values: set[str] | None = None) -> str:
    redacted = text
    for secret in secret_values or set():
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    for pattern in SECRET_TEXT_PATTERNS:
        if pattern.pattern.startswith("(://"):
            redacted = pattern.sub(r"\1<redacted>\2", redacted)
        else:
            redacted = pattern.sub(r"\1<redacted>", redacted)
    return redacted


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically publish an artifact without exposing a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def redact_secrets(value: Any, *, secret_values: set[str] | None = None) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, val in value.items():
            if _is_secret_key(key):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_secrets(val, secret_values=secret_values)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item, secret_values=secret_values) for item in value]
    if isinstance(value, str):
        return redact_text(value, secret_values)
    return value


def resolve_artifact(run_dir: Path, name: str) -> Path:
    if name not in SAFE_ARTIFACT_NAMES or "/" in name or "\\" in name:
        raise FileNotFoundError(name)
    path = (run_dir / name).resolve()
    root = run_dir.resolve()
    if path.parent != root or not path.is_file():
        raise FileNotFoundError(name)
    return path
