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

"""Sysinfo CLI subcommands — from-config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cyclopts
import yaml
from pydantic import ValidationError

from inference_endpoint.config.schema import SysInfoFileConfig
from inference_endpoint.exceptions import InputValidationError
from inference_endpoint.sys_info.capture import capture_system_info

sysinfo_app = cyclopts.App(name="sysinfo", help="Capture MLPerf system information.")


@sysinfo_app.command(name="from-config")
def from_config(
    *,
    config: Annotated[Path, cyclopts.Parameter(name=["--config", "-c"])],
) -> None:
    """Capture multi-node system info from a YAML config file.

    The config file must contain a ``system_info`` key with capture settings
    and an optional ``node_config`` for function-based node groupings.

    Example::

        report_dir: results/my_system_info

        system_info:
          system_name: H100x8_vLLM
          ssh_ids:
            - user@host
          accelerator_backend: cuda
          node_config:
            Prefill:
              - node_name: H100
                no_of_nodes: 4
            Decode:
              - node_name: H100
                no_of_nodes: 8
    """
    try:
        resolved = SysInfoFileConfig.from_yaml_file(config)
    except (yaml.YAMLError, ValidationError, ValueError, FileNotFoundError) as e:
        raise InputValidationError(f"Config error: {e}") from e

    capture_cfg = resolved.system_info
    run_metadata_path = None
    candidate = resolved.report_dir / "run_metadata.json"
    if candidate.exists():
        run_metadata_path = candidate

    output_path = capture_system_info(
        capture_cfg, output_dir=resolved.report_dir, run_metadata_path=run_metadata_path
    )
    print(f"System info written to: {output_path}")
