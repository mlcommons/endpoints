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

"""Audit result type and atomic disk writer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Outcome of an AuditTest.verify() call."""

    test_id: str
    passed: bool
    details: dict[str, Any]


def write_result(result: AuditResult, report_dir: Path) -> None:
    """Atomically write audit_result.json and verify_<TEST>.txt to report_dir.

    Each file is written atomically (tmp → fsync(file) → rename →
    fsync(parent_dir)). The durable JSON record is written *before* the
    validator-facing verify_<TEST>.txt, so a crash between the two can never
    leave a "Performance check pass" signal that a validator would accept
    without the backing audit_result.json record.
    """
    test_upper = result.test_id.upper()
    _atomic_write_text(
        report_dir / "audit_result.json",
        json.dumps(
            {"test": result.test_id, "passed": result.passed, **result.details},
            indent=2,
        )
        + "\n",
    )
    _atomic_write_text(
        report_dir / f"verify_{test_upper}.txt",
        f"Performance check pass: {result.passed}\n",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        with open(tmp) as f:
            os.fsync(f.fileno())
        tmp.rename(path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
