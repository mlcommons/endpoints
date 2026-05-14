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

"""Sys-info capture via the get-mlperf-multi-node-system-info mlcflow script."""

from __future__ import annotations

import logging
from pathlib import Path

from inference_endpoint.config.schema import SysInfoCaptureConfig
from inference_endpoint.exceptions import ExecutionError, SetupError

logger = logging.getLogger(__name__)

_OUT_FILE_NAME = "mlperf-multi-node-system-info.json"


def capture_system_info(config: SysInfoCaptureConfig) -> Path:
    """Invoke the get-mlperf-multi-node-system-info mlcflow script.

    Returns the Path to the generated combined system info JSON file.
    Raises SetupError if mlcflow is not installed.
    Raises ExecutionError if the script returns a non-zero return code.
    """
    # Optional dependency — only imported when this function is actually called.
    # mlcflow (PyPI) installs its runtime under the 'mlc' module name.
    # mlcflow is not a required dependency of this package; see pyproject.toml [sys-info].
    try:
        import mlc  # noqa: PLC0415
    except ImportError as exc:
        raise SetupError(
            "mlcflow is required for sys_info_capture. "
            "Install it with: pip install mlcflow"
        ) from exc

    tags: list[str] = [
        "get-mlperf-multi-node-system-info",
        f"_{config.accelerator_backend}",
    ]
    if config.exclude_current_system:
        tags.append("_exclude_current_node")
    tags_str = ",".join(tags)

    ssh_ids_str = ",".join(t.to_mlcflow_str() for t in config.parsed_ssh_ids)

    # CM/mlcflow scripts use "yes"/"" string convention for boolean env vars.
    skip_ssh_key_file_value = "yes" if config.skip_ssh_key_file else ""

    Path(config.output_path).mkdir(parents=True, exist_ok=True)

    logger.info("Capturing system info from %d node(s)...", len(config.parsed_ssh_ids))

    result = mlc.access(
        {
            "action": "run",
            "automation": "script",
            "tags": tags_str,
            "ssh_ids": ssh_ids_str,
            "out_dir_path": config.output_path,
            "out_file_name": _OUT_FILE_NAME,
            "skip_ssh_key_file": skip_ssh_key_file_value,
            "quiet": True,
        }
    )

    if result.get("return", 1) != 0:
        raise ExecutionError(
            f"sys_info capture failed (return code {result.get('return')}): "
            f"{result.get('error', 'unknown error')}"
        )

    output_path = Path(
        result.get("new_env", {}).get("MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH")
        or Path(config.output_path) / _OUT_FILE_NAME
    )
    logger.info("System info written to %s", output_path)
    return output_path
