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

import argparse
from pathlib import Path

from aiohttp import web

from .config import ServiceConfig
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SWE-bench service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--artifact-root", default="swebench_service_artifacts")
    parser.add_argument("--max-concurrent-runs", type=int, default=1)
    parser.add_argument("--subprocess-timeout-s", type=int, default=24 * 60 * 60)
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument("--auth-token")
    auth_group.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--max-stored-runs", type=int, default=100)
    args = parser.parse_args()

    config = ServiceConfig(
        host=args.host,
        port=args.port,
        artifact_root=Path(args.artifact_root),
        max_concurrent_runs=args.max_concurrent_runs,
        subprocess_timeout_s=args.subprocess_timeout_s,
        auth_token=args.auth_token,
        allow_unauthenticated=args.allow_unauthenticated,
        max_stored_runs=args.max_stored_runs,
    )
    web.run_app(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
