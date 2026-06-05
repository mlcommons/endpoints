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

"""Unit tests for the sysinfo from-config command, NodeEntry, and node_config."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from inference_endpoint.config.schema import (
    NodeEntry,
    SysInfoCaptureConfig,
    SysInfoFileConfig,
)
from inference_endpoint.exceptions import InputValidationError
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_SYSTEM_INFO = {
    "accelerator_backend": "cuda",
    "ssh_ids": ["user@10.0.0.1"],
}


def _make_capture_config(**overrides: object) -> SysInfoCaptureConfig:
    return SysInfoCaptureConfig(**{**_BASE_SYSTEM_INFO, **overrides})


# ---------------------------------------------------------------------------
# NodeEntry model
# ---------------------------------------------------------------------------


class TestNodeEntry:
    @pytest.mark.unit
    def test_valid_entry(self) -> None:
        e = NodeEntry(node_name="H100", no_of_nodes=4)
        assert e.node_name == "H100"
        assert e.no_of_nodes == 4

    @pytest.mark.unit
    def test_default_no_of_nodes_is_one(self) -> None:
        e = NodeEntry(node_name="GB300")
        assert e.no_of_nodes == 1

    @pytest.mark.unit
    def test_zero_nodes_raises(self) -> None:
        with pytest.raises(ValidationError):
            NodeEntry(node_name="H100", no_of_nodes=0)

    @pytest.mark.unit
    def test_negative_nodes_raises(self) -> None:
        with pytest.raises(ValidationError):
            NodeEntry(node_name="H100", no_of_nodes=-1)

    @pytest.mark.unit
    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NodeEntry(node_name="H100", no_of_nodes=1, unknown_field="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SysInfoCaptureConfig with node_config
# ---------------------------------------------------------------------------


class TestSysInfoCaptureConfigNodeConfig:
    @pytest.mark.unit
    def test_node_config_none_by_default(self) -> None:
        cfg = _make_capture_config()
        assert cfg.node_config is None

    @pytest.mark.unit
    def test_node_config_single_function(self) -> None:
        cfg = _make_capture_config(
            node_config={
                "Prefill": [{"node_name": "H100", "no_of_nodes": 4}],
            }
        )
        assert cfg.node_config is not None
        assert len(cfg.node_config["Prefill"]) == 1
        assert cfg.node_config["Prefill"][0].node_name == "H100"
        assert cfg.node_config["Prefill"][0].no_of_nodes == 4

    @pytest.mark.unit
    def test_node_config_multi_function_multi_node(self) -> None:
        cfg = _make_capture_config(
            node_config={
                "Decode": [
                    {"node_name": "GB300", "no_of_nodes": 12},
                    {"node_name": "H100", "no_of_nodes": 15},
                ],
                "Prefill": [
                    {"node_name": "GB300", "no_of_nodes": 20},
                    {"node_name": "H100", "no_of_nodes": 8},
                ],
            }
        )
        assert cfg.node_config is not None
        decode = cfg.node_config["Decode"]
        assert decode[0].node_name == "GB300"
        assert decode[0].no_of_nodes == 12
        assert decode[1].node_name == "H100"
        assert decode[1].no_of_nodes == 15

        prefill = cfg.node_config["Prefill"]
        assert prefill[0].no_of_nodes == 20
        assert prefill[1].no_of_nodes == 8

    @pytest.mark.unit
    def test_node_config_invalid_node_entry_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_capture_config(
                node_config={
                    "Prefill": [{"node_name": "H100", "no_of_nodes": -5}],
                }
            )


# ---------------------------------------------------------------------------
# SysInfoFileConfig YAML loading
# ---------------------------------------------------------------------------


class TestSysInfoFileConfig:
    @pytest.mark.unit
    def test_minimal_config_no_node_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                report_dir: {tmp_path}/results/
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                """
            )
        )
        cfg = SysInfoFileConfig.from_yaml_file(config_path)
        assert cfg.system_info.accelerator_backend == "cuda"
        assert cfg.report_dir == tmp_path / "results"
        assert cfg.system_info.node_config is None

    @pytest.mark.unit
    def test_missing_report_dir_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                """\
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                """
            )
        )
        with pytest.raises(ValidationError):
            SysInfoFileConfig.from_yaml_file(config_path)

    @pytest.mark.unit
    def test_report_dir_parsed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                """\
                report_dir: results/my_system/
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                """
            )
        )
        cfg = SysInfoFileConfig.from_yaml_file(config_path)
        from pathlib import Path as _Path

        assert cfg.report_dir == _Path("results/my_system/")

    @pytest.mark.unit
    def test_with_node_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                report_dir: {tmp_path}/results/
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                  node_config:
                    Prefill:
                      - node_name: H100
                        no_of_nodes: 1
                    Decode:
                      - node_name: H100
                        no_of_nodes: 1
                """
            )
        )
        cfg = SysInfoFileConfig.from_yaml_file(config_path)
        nc = cfg.system_info.node_config
        assert nc is not None
        assert "Prefill" in nc
        assert "Decode" in nc
        assert nc["Prefill"][0].node_name == "H100"
        assert nc["Decode"][0].no_of_nodes == 1

    @pytest.mark.unit
    def test_extra_top_level_keys_ignored(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                report_dir: {tmp_path}/results/
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                some_other_section:
                  foo: bar
                """
            )
        )
        cfg = SysInfoFileConfig.from_yaml_file(config_path)
        assert cfg.system_info.accelerator_backend == "cuda"

    @pytest.mark.unit
    def test_endpoint_config_not_propagated_to_endpoint_url(
        self, tmp_path: Path
    ) -> None:
        """endpoint_config in a shared YAML does not populate system_info.endpoint_url.

        The load target (endpoint_config.endpoints[0]) is not necessarily the
        serving node, so endpoint_url is never inferred automatically.  Users
        must set system_info.endpoint_url explicitly when they want the HTTP
        serving-framework probe to run.
        """
        config_path = tmp_path / "shared.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                report_dir: {tmp_path}/results/
                endpoint_config:
                  endpoints:
                    - http://10.0.0.1:8000
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                """
            )
        )
        cfg = SysInfoFileConfig.from_yaml_file(config_path)
        # endpoint_config is ignored (extra="ignore"); endpoint_url stays None.
        assert cfg.system_info.endpoint_url is None

    @pytest.mark.unit
    def test_endpoint_url_explicit_in_system_info_is_preserved(
        self, tmp_path: Path
    ) -> None:
        """An explicit system_info.endpoint_url is honoured by SysInfoFileConfig."""
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                report_dir: {tmp_path}/results/
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                  endpoint_url: http://10.0.0.2:8000
                """
            )
        )
        cfg = SysInfoFileConfig.from_yaml_file(config_path)
        assert cfg.system_info.endpoint_url == "http://10.0.0.2:8000"

    @pytest.mark.unit
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SysInfoFileConfig.from_yaml_file(tmp_path / "nonexistent.yaml")

    @pytest.mark.unit
    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("- not a mapping\n- item2\n")
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            SysInfoFileConfig.from_yaml_file(config_path)

    @pytest.mark.unit
    def test_missing_system_info_key_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text("other_key: value\n")
        with pytest.raises(ValidationError):
            SysInfoFileConfig.from_yaml_file(config_path)


# ---------------------------------------------------------------------------
# capture_system_info with node_config — temp file creation and cleanup
# ---------------------------------------------------------------------------


class TestCaptureWithNodeConfig:
    @pytest.mark.unit
    def test_node_config_file_passed_to_mlcflow(self, tmp_path: Path) -> None:
        """When node_config is set, mlcflow.access must receive node_config_file."""
        cfg = _make_capture_config(
            node_config={
                "Prefill": [{"node_name": "H100", "no_of_nodes": 1}],
            },
        )
        mock_mlcflow = MagicMock()
        mock_mlcflow.access.return_value = {
            "return": 0,
            "new_env": {
                "MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH": str(tmp_path / "out.json")
            },
        }
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            capture_mod.capture_system_info(cfg, output_dir=tmp_path)

        call_args = mock_mlcflow.access.call_args[0][0]
        assert "node_config_file" in call_args

    @pytest.mark.unit
    def test_node_config_temp_file_content(self, tmp_path: Path) -> None:
        """The temp file must contain system_info.node_config in the expected structure."""
        cfg = _make_capture_config(
            node_config={
                "Decode": [
                    {"node_name": "GB300", "no_of_nodes": 12},
                    {"node_name": "H100", "no_of_nodes": 15},
                ],
            },
        )
        captured_path: list[str] = []
        captured_content: list[dict] = []

        def fake_access(kwargs: dict) -> dict:
            path = kwargs.get("node_config_file", "")
            captured_path.append(str(path))
            if path:
                with open(path) as f:
                    captured_content.append(yaml.safe_load(f))
            return {
                "return": 0,
                "new_env": {
                    "MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH": str(tmp_path / "out.json")
                },
            }

        mock_mlcflow = MagicMock()
        mock_mlcflow.access.side_effect = fake_access
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            capture_mod.capture_system_info(cfg, output_dir=tmp_path)

        assert len(captured_content) == 1
        data = captured_content[0]
        decode_nodes = data["system_info"]["node_config"]["Decode"]
        assert decode_nodes[0]["node_name"] == "GB300"
        assert decode_nodes[0]["no_of_nodes"] == 12
        assert decode_nodes[1]["node_name"] == "H100"
        assert decode_nodes[1]["no_of_nodes"] == 15

    @pytest.mark.unit
    def test_temp_file_deleted_after_success(self, tmp_path: Path) -> None:
        """The temp node_config file must be cleaned up after mlcflow returns."""
        cfg = _make_capture_config(
            node_config={"Prefill": [{"node_name": "H100", "no_of_nodes": 1}]},
        )
        recorded_tmp: list[str] = []

        def fake_access(kwargs: dict) -> dict:
            recorded_tmp.append(str(kwargs.get("node_config_file", "")))
            return {
                "return": 0,
                "new_env": {
                    "MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH": str(tmp_path / "out.json")
                },
            }

        mock_mlcflow = MagicMock()
        mock_mlcflow.access.side_effect = fake_access
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            capture_mod.capture_system_info(cfg, output_dir=tmp_path)

        assert recorded_tmp[0], "temp file path was recorded"
        assert not Path(recorded_tmp[0]).exists(), "temp file was deleted after use"

    @pytest.mark.unit
    def test_temp_file_deleted_on_mlcflow_failure(self, tmp_path: Path) -> None:
        """Temp file must be cleaned up even when mlcflow returns an error."""
        from inference_endpoint.exceptions import ExecutionError

        cfg = _make_capture_config(
            node_config={"Prefill": [{"node_name": "H100", "no_of_nodes": 1}]},
        )
        recorded_tmp: list[str] = []

        def fake_access(kwargs: dict) -> dict:
            recorded_tmp.append(str(kwargs.get("node_config_file", "")))
            return {"return": 1, "error": "ssh timeout"}

        mock_mlcflow = MagicMock()
        mock_mlcflow.access.side_effect = fake_access
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            with pytest.raises(ExecutionError):
                capture_mod.capture_system_info(cfg, output_dir=tmp_path)

        assert not Path(recorded_tmp[0]).exists(), "temp file cleaned up on failure"

    @pytest.mark.unit
    def test_no_node_config_file_arg_when_node_config_is_none(
        self, tmp_path: Path
    ) -> None:
        """Without node_config, node_config_file must NOT be in the mlcflow call."""
        cfg = _make_capture_config()
        mock_mlcflow = MagicMock()
        mock_mlcflow.access.return_value = {
            "return": 0,
            "new_env": {
                "MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH": str(tmp_path / "out.json")
            },
        }
        with patch.dict("sys.modules", {"mlc": mock_mlcflow}):
            import importlib

            from inference_endpoint.sys_info import capture as capture_mod

            importlib.reload(capture_mod)
            capture_mod.capture_system_info(cfg, output_dir=tmp_path)

        call_args = mock_mlcflow.access.call_args[0][0]
        assert "node_config_file" not in call_args


# ---------------------------------------------------------------------------
# from-config CLI command
# ---------------------------------------------------------------------------


class TestReportDirResolution:
    @pytest.mark.unit
    def test_report_dir_passed_as_output_dir(self, tmp_path: Path) -> None:
        """When report_dir is set, capture_system_info must receive it as output_dir."""
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                report_dir: {tmp_path}/results/
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                """
            )
        )
        captured_dirs: list[Path] = []

        def fake_capture(
            cfg: SysInfoCaptureConfig,
            output_dir: Path = Path("."),
            run_metadata_path=None,
        ) -> Path:
            captured_dirs.append(output_dir)
            return tmp_path / "out.json"

        with patch(
            "inference_endpoint.commands.sysinfo.cli.capture_system_info",
            side_effect=fake_capture,
        ):
            from inference_endpoint.commands.sysinfo.cli import from_config

            from_config(config=config_path)

        assert captured_dirs[0] == tmp_path / "results"

    @pytest.mark.unit
    def test_no_report_dir_raises_validation_error(self, tmp_path: Path) -> None:
        """Without report_dir, loading the config raises a ValidationError."""
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                """\
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                """
            )
        )
        with pytest.raises(InputValidationError):
            from inference_endpoint.commands.sysinfo.cli import from_config

            from_config(config=config_path)


class TestMissingSshIds:
    @pytest.mark.unit
    def test_missing_ssh_ids_raises_validation_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            f"report_dir: {tmp_path}/results/\nsystem_info:\n  accelerator_backend: cuda\n"
        )
        with pytest.raises(ValidationError):
            SysInfoFileConfig.from_yaml_file(config_path)

    @pytest.mark.unit
    def test_missing_ssh_ids_exclude_current_false_still_raises(
        self, tmp_path: Path
    ) -> None:
        """exclude_current_system=False does not relax the ssh_ids requirement."""
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            f"report_dir: {tmp_path}/results/\nsystem_info:\n  accelerator_backend: cuda\n  exclude_current_system: false\n"
        )
        with pytest.raises(ValidationError):
            SysInfoFileConfig.from_yaml_file(config_path)

    @pytest.mark.unit
    def test_missing_ssh_ids_via_cli_raises_input_validation_error(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            f"report_dir: {tmp_path}/results/\nsystem_info:\n  accelerator_backend: cuda\n"
        )
        with patch(
            "inference_endpoint.commands.sysinfo.cli.capture_system_info"
        ) as mock_cap:
            from inference_endpoint.commands.sysinfo.cli import from_config

            with pytest.raises(InputValidationError):
                from_config(config=config_path)
            mock_cap.assert_not_called()


class TestFromConfigCLI:
    @pytest.mark.unit
    def test_from_config_calls_capture(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sysinfo.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                report_dir: {tmp_path}/results/
                system_info:
                  ssh_ids:
                    - user@10.0.0.1
                  accelerator_backend: cuda
                  node_config:
                    Prefill:
                      - node_name: H100
                        no_of_nodes: 1
                """
            )
        )
        captured_configs: list[SysInfoCaptureConfig] = []

        def fake_capture(
            cfg: SysInfoCaptureConfig,
            output_dir: Path = Path("."),
            run_metadata_path=None,
        ) -> Path:
            captured_configs.append(cfg)
            return tmp_path / "out.json"

        with patch(
            "inference_endpoint.commands.sysinfo.cli.capture_system_info",
            side_effect=fake_capture,
        ):
            from inference_endpoint.commands.sysinfo.cli import from_config

            from_config(config=config_path)

        assert len(captured_configs) == 1
        assert captured_configs[0].accelerator_backend == "cuda"
        assert captured_configs[0].node_config is not None
        assert "Prefill" in captured_configs[0].node_config

    @pytest.mark.unit
    def test_from_config_missing_file_raises_input_validation_error(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "inference_endpoint.commands.sysinfo.cli.capture_system_info"
        ) as mock_cap:
            from inference_endpoint.commands.sysinfo.cli import from_config

            with pytest.raises(InputValidationError):
                from_config(config=tmp_path / "missing.yaml")

            mock_cap.assert_not_called()

    @pytest.mark.unit
    def test_from_config_invalid_yaml_raises_input_validation_error(
        self, tmp_path: Path
    ) -> None:
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("- not\n- a\n- mapping\n")
        with patch(
            "inference_endpoint.commands.sysinfo.cli.capture_system_info"
        ) as mock_cap:
            from inference_endpoint.commands.sysinfo.cli import from_config

            with pytest.raises(InputValidationError):
                from_config(config=bad_path)

            mock_cap.assert_not_called()
