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

"""Shared helpers for E2E command tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

DATASET = str(
    Path(__file__).resolve().parents[2] / "assets" / "datasets" / "ds_samples.jsonl"
)


class StubServer(Protocol):
    """Duck-type for the stub servers (MaxThroughputServer, VariableResponseServer).

    Named ``StubServer`` rather than ``TestServer`` so pytest doesn't try to
    collect it as a test class (any class whose name starts with ``Test`` is
    a collection candidate).
    """

    url: str
    stream: bool


def run_cli(
    extra_args: list[str],
    report_dir: Path,
    server: StubServer,
    *,
    dataset: str = DATASET,
) -> dict:
    """Invoke ``inference-endpoint`` in-process via cyclopts; return results.json.

    Client ``--streaming`` is coupled to the server's response mode: the stub
    server always returns the same pre-compiled bytes (JSON or SSE),
    regardless of what the client sent in the request body. Mismatched modes
    produce ``DecodeError: JSON is malformed`` on every response.

    Env overrides (useful in containers where cpu_affinity is restricted):
        ROOFLINE_NUM_WORKERS  — override --workers (default: auto)
        ROOFLINE_INIT_TIMEOUT — override --client.worker-initialization-timeout
    """
    from inference_endpoint.main import app

    report_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "benchmark",
        *extra_args,
        "--endpoints",
        server.url,
        "--streaming",
        "on" if server.stream else "off",
        "--model",
        "max-tp",
        "--dataset",
        dataset,
        "--report-dir",
        str(report_dir),
    ]
    if nw := os.environ.get("ROOFLINE_NUM_WORKERS"):
        args += ["--workers", nw]
    if to := os.environ.get("ROOFLINE_INIT_TIMEOUT"):
        args += ["--client.worker-initialization-timeout", to]
    try:
        app(args)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    return json.loads((report_dir / "results.json").read_text())
