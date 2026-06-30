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

"""Unit tests for sys_info capture — config models and capture function."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    SshTarget,
    SysInfoCaptureConfig,
)
from inference_endpoint.exceptions import ExecutionError, SetupError
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_SYS_INFO = {
    "system_name": "TestSystem",
    "accelerator_backend": "cuda",
    "ssh_ids": ["alice@192.168.1.1"],
}


def _make_config(**overrides: object) -> SysInfoCaptureConfig:
    return SysInfoCaptureConfig(**{**_MINIMAL_SYS_INFO, **overrides})


# ---------------------------------------------------------------------------
# 1. SSH string parsing — no port
# ---------------------------------------------------------------------------


class TestSshTargetParsing:
    @pytest.mark.unit
    def test_no_port_defaults_to_22(self) -> None:
        cfg = _make_config(ssh_ids=["alice@192.168.1.1"])
        targets = cfg.parsed_ssh_ids
        assert len(targets) == 1
        t = targets[0]
        assert t.username == "alice"
        assert t.host == "192.168.1.1"
        assert t.port == 22
        assert t.to_mlcflow_str() == "alice@192.168.1.1:22"

    # 2. SSH string parsing — with port
    @pytest.mark.unit
    def test_with_explicit_port(self) -> None:
        cfg = _make_config(ssh_ids=["alice@192.168.1.1:2222"])
        targets = cfg.parsed_ssh_ids
        assert targets[0].port == 2222
        assert targets[0].to_mlcflow_str() == "alice@192.168.1.1:2222"

    # 3. SSH string parsing — invalid entries
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_entry",
        [
            "notvalid",
            "@host",
            "user@host:99999",
        ],
    )
    def test_invalid_ssh_id_raises(self, bad_entry: str) -> None:
        with pytest.raises(ValidationError):
            _make_config(ssh_ids=[bad_entry])

    @pytest.mark.unit
    def test_ssh_target_port_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SshTarget(username="alice", host="192.168.1.1", port=99999)

    @pytest.mark.unit
    def test_empty_ssh_ids_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            _make_config(ssh_ids=[])

    @pytest.mark.unit
    def test_system_name_required(self) -> None:
        with pytest.raises(ValidationError, match="system_name"):
            SysInfoCaptureConfig(accelerator_backend="cuda", ssh_ids=["alice@10.0.0.1"])

    # 4. serving_node validation
    @pytest.mark.unit
    def test_serving_node_valid_no_port(self) -> None:
        cfg = _make_config(serving_node="alice@10.0.0.1")
        assert cfg.serving_node == "alice@10.0.0.1"

    @pytest.mark.unit
    def test_serving_node_valid_with_port(self) -> None:
        cfg = _make_config(serving_node="alice@10.0.0.1:2222")
        assert cfg.serving_node == "alice@10.0.0.1:2222"

    @pytest.mark.unit
    def test_serving_node_port_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="1-65535"):
            _make_config(serving_node="alice@10.0.0.1:99999")

    @pytest.mark.unit
    @pytest.mark.parametrize("port", [0, 65536, 100000])
    def test_serving_node_invalid_ports_raise(self, port: int) -> None:
        with pytest.raises(ValidationError, match="1-65535"):
            _make_config(serving_node=f"alice@10.0.0.1:{port}")

    @pytest.mark.unit
    def test_serving_node_invalid_format_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(serving_node="notvalid")


# ---------------------------------------------------------------------------
# 4. Variation tags — cuda, include current
# ---------------------------------------------------------------------------


class TestVariationTags:
    @pytest.mark.unit
    def test_cuda_include_current(self) -> None:
        cfg = _make_config(accelerator_backend="cuda", exclude_current_system=False)
        tags = _build_tags(cfg)
        assert tags == "get-mlperf-multi-node-system-info,_cuda"

    # 5. Variation tags — rocm, exclude current
    @pytest.mark.unit
    def test_rocm_exclude_current(self) -> None:
        cfg = _make_config(accelerator_backend="rocm", exclude_current_system=True)
        tags = _build_tags(cfg)
        assert tags == "get-mlperf-multi-node-system-info,_rocm,_exclude_current_node"

    # 6. Variation tags — none backend omits the backend tag entirely
    @pytest.mark.unit
    def test_none_backend_omits_tag(self) -> None:
        cfg = _make_config(accelerator_backend="none", exclude_current_system=False)
        tags = _build_tags(cfg)
        assert tags == "get-mlperf-multi-node-system-info"


def _build_tags(cfg: SysInfoCaptureConfig) -> str:
    """Mirror the tag-building logic from capture.py."""
    tags: list[str] = ["get-mlperf-multi-node-system-info"]
    if cfg.accelerator_backend != "none":
        tags.append(f"_{cfg.accelerator_backend}")
    if cfg.exclude_current_system:
        tags.append("_exclude_current_node")
    return ",".join(tags)


# ---------------------------------------------------------------------------
# 6. skip_ssh_key_file encoding
# ---------------------------------------------------------------------------


class TestSkipSshKeyFileEncoding:
    @pytest.mark.unit
    def test_true_encodes_as_yes(self) -> None:
        cfg = _make_config(skip_ssh_key_file=True)
        value = _encode_skip_ssh(cfg)
        assert value == "yes"

    @pytest.mark.unit
    def test_false_encodes_as_empty_string(self) -> None:
        cfg = _make_config(skip_ssh_key_file=False)
        value = _encode_skip_ssh(cfg)
        assert value == ""


def _encode_skip_ssh(cfg: SysInfoCaptureConfig) -> str:
    """Mirror the boolean-encoding logic from capture.py."""
    return "yes" if cfg.skip_ssh_key_file else ""


# ---------------------------------------------------------------------------
# 7–10. capture_system_info function tests
# ---------------------------------------------------------------------------


class TestCaptureSystemInfo:
    @pytest.mark.unit
    def test_mlcflow_not_installed_raises_setup_error(self, tmp_path: Path) -> None:
        cfg = _make_config()
        with patch.dict("sys.modules", {"mlc": None}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            with pytest.raises(SetupError, match=r"pip install -e '\.\[sysinfo\]'"):
                capture_mod.capture_system_info(cfg, output_dir=tmp_path)

    @pytest.mark.unit
    def test_mlcflow_nonzero_return_raises_execution_error(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config()
        mock_mlcflow = MagicMock()
        mock_mlcflow.access.return_value = {
            "return": 1,
            "error": "ssh connection refused",
        }
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            with pytest.raises(ExecutionError, match="ssh connection refused"):
                capture_mod.capture_system_info(cfg, output_dir=tmp_path)

    @pytest.mark.unit
    def test_happy_path_output_path_from_new_env(self, tmp_path: Path) -> None:
        cfg = _make_config()
        mock_mlcflow = MagicMock()
        mock_mlcflow.access.return_value = {
            "return": 0,
            "new_env": {
                "MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH": "/tmp/out.json",
            },
        }
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            result = capture_mod.capture_system_info(cfg, output_dir=tmp_path)
        assert result == Path("/tmp/out.json")

    @pytest.mark.unit
    def test_missing_env_path_raises_execution_error(self, tmp_path: Path) -> None:
        """Return code 0 without MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH raises ExecutionError."""
        cfg = _make_config()
        mock_mlcflow = MagicMock()
        mock_mlcflow.access.return_value = {
            "return": 0,
            "new_env": {},
        }
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            with pytest.raises(
                ExecutionError, match="MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH"
            ):
                capture_mod.capture_system_info(cfg, output_dir=tmp_path)

    @pytest.mark.unit
    def test_unreachable_node_no_node_config_succeeds(self, tmp_path: Path) -> None:
        """mlcflow returning 0 (one node internally unreachable, no node_config) is non-fatal."""
        cfg = _make_config()
        mock_mlcflow = MagicMock()
        mock_mlcflow.access.return_value = {
            "return": 0,
            "new_env": {
                "MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH": str(
                    tmp_path / "system_desc.json"
                ),
            },
        }
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            result = capture_mod.capture_system_info(cfg, output_dir=tmp_path)
        assert result == tmp_path / "system_desc.json"

    @pytest.mark.unit
    def test_mlcflow_access_called_with_correct_args(self, tmp_path: Path) -> None:
        cfg = SysInfoCaptureConfig(
            system_name="TestSystem",
            accelerator_backend="cuda",
            exclude_current_system=True,
            skip_ssh_key_file=True,
            ssh_ids=["alice@10.0.0.1:2222", "bob@10.0.0.2"],
        )
        mock_mlcflow = MagicMock()
        mock_mlcflow.access.return_value = {
            "return": 0,
            "new_env": {"MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH": "/tmp/out.json"},
        }
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            capture_mod.capture_system_info(cfg, output_dir=tmp_path)

        call_args = mock_mlcflow.access.call_args[0][0]
        assert (
            call_args["tags"]
            == "get-mlperf-multi-node-system-info,_cuda,_exclude_current_node"
        )
        assert call_args["ssh_ids"] == "alice@10.0.0.1:2222,bob@10.0.0.2:22"
        assert call_args["skip_ssh_key_file"] == "yes"
        assert call_args["out_dir_path"] == str(tmp_path.resolve())
        assert call_args["out_file_name"] == "system_desc.json"
        assert call_args["action"] == "run"
        assert call_args["automation"] == "script"
        assert call_args["system_name"] == "TestSystem"


# ---------------------------------------------------------------------------
# 11. YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    @pytest.mark.unit
    def test_system_info_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent(
            """\
            type: offline
            model_params:
              name: test-model
            endpoint_config:
              endpoints:
                - http://localhost:8000
            datasets:
              - path: dummy.jsonl
            system_info:
              system_name: TestSystem
              accelerator_backend: cuda
              ssh_ids:
                - alice@192.168.1.1
                - bob@192.168.1.2:2222
              exclude_current_system: true
              skip_ssh_key_file: false
            """
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = BenchmarkConfig.from_yaml_file(config_path)

        assert config.system_info is not None
        sic = config.system_info
        assert sic.accelerator_backend == "cuda"
        assert sic.exclude_current_system is True
        assert sic.skip_ssh_key_file is False
        assert len(sic.ssh_ids) == 2

        targets = sic.parsed_ssh_ids
        assert targets[0].to_mlcflow_str() == "alice@192.168.1.1:22"
        assert targets[1].to_mlcflow_str() == "bob@192.168.1.2:2222"

    # 12. Backward compatibility
    @pytest.mark.unit
    def test_yaml_without_system_info_is_none(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent(
            """\
            type: offline
            model_params:
              name: test-model
            endpoint_config:
              endpoints:
                - http://localhost:8000
            datasets:
              - path: dummy.jsonl
            """
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = BenchmarkConfig.from_yaml_file(config_path)
        assert config.system_info is None
