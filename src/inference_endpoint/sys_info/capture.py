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
import os
import tempfile
from pathlib import Path

import yaml

from inference_endpoint.config.schema import SysInfoCaptureConfig
from inference_endpoint.exceptions import ExecutionError, SetupError

logger = logging.getLogger(__name__)

_OUT_FILE_NAME = "system_desc.json"


def _write_node_config_tmp(config: SysInfoCaptureConfig) -> str:
    """Serialise node_config to a temp YAML file readable by customize.py.

    customize.py expects:
        system_info:
          node_config:
            <function>:
              - node_name: ...
                no_of_nodes: ...

    Returns the path of the written temp file.
    """
    assert config.node_config is not None
    data = {
        "system_info": {
            "node_config": {
                func: [entry.model_dump() for entry in entries]
                for func, entries in config.node_config.items()
            }
        }
    }
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mlperf_node_cfg_")
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False)
    except Exception:
        os.unlink(path)
        raise
    return path


def capture_system_info(
    config: SysInfoCaptureConfig,
    output_dir: Path,
    run_metadata_path: Path | None = None,
) -> Path:
    """Invoke the get-mlperf-multi-node-system-info mlcflow script.

    Returns the Path to the generated combined system info JSON file.
    Raises SetupError if mlcflow is not installed.
    Raises ExecutionError if the script returns a non-zero return code.
    """
    # Optional dependency — only imported when this function is actually called.
    # mlc-scripts (PyPI) installs its runtime under the 'mlc' module name.
    try:
        import mlc
    except ImportError as exc:
        raise SetupError(
            "mlc-scripts is required for system_info. "
            "Install it with: pip install -e '.[sysinfo]' (from the repo root)"
        ) from exc

    tags: list[str] = ["get-mlperf-multi-node-system-info"]
    if config.accelerator_backend != "none":
        tags.append(f"_{config.accelerator_backend}")
    if config.exclude_current_system:
        tags.append("_exclude_current_node")
    tags_str = ",".join(tags)

    ssh_ids_str = ",".join(t.to_mlcflow_str() for t in config.parsed_ssh_ids)

    # mlcflow scripts use "yes"/"" string convention for boolean env vars.
    skip_ssh_key_file_value = "yes" if config.skip_ssh_key_file else ""

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Capturing system info from %d node(s)...", len(config.parsed_ssh_ids))

    # mlcflow changes cwd during script execution; resolve both paths to
    # absolute here so postprocess() os.path.exists() checks work correctly.
    mlc_kwargs: dict[str, object] = {
        "action": "run",
        "automation": "script",
        "tags": tags_str,
        "ssh_ids": ssh_ids_str,
        "out_dir_path": str(output_dir.resolve()),
        "out_file_name": _OUT_FILE_NAME,
        "skip_ssh_key_file": skip_ssh_key_file_value,
        "serving_framework_type": config.serving_framework,
        "system_name": config.system_name,
        "quiet": True,
    }
    if config.endpoint_url:
        mlc_kwargs["endpoint_url"] = config.endpoint_url
    if config.serving_node:
        mlc_kwargs["serving_node"] = config.serving_node
    if run_metadata_path is not None:
        mlc_kwargs["run_metadata_path"] = str(run_metadata_path.resolve())

    node_config_tmp: str | None = None
    if config.node_config is not None:
        node_config_tmp = _write_node_config_tmp(config)
        mlc_kwargs["node_config_file"] = node_config_tmp
        logger.debug("Node config written to temp file: %s", node_config_tmp)

    try:
        result = mlc.access(mlc_kwargs)
    finally:
        if node_config_tmp and os.path.exists(node_config_tmp):
            os.unlink(node_config_tmp)

    if result.get("return", 1) != 0:
        raise ExecutionError(
            f"sys_info capture failed (return code {result.get('return')}): "
            f"{result.get('error', 'unknown error')}"
        )

    new_env_path = (result.get("new_env") or {}).get(
        "MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH"
    )
    if not new_env_path:
        raise ExecutionError(
            "sys_info capture returned success but MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH "
            "was not set — no system info was collected"
        )
    output_path = Path(new_env_path)
    logger.info("System info written to %s", output_path)
    return output_path
