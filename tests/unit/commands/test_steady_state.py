# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-run steady-state detection step.

Pins the gate (which runs earn a detection pass), the command handed to the
detector subprocess, and the best-effort contract: no failure mode here may
propagate out of finalize.
"""

import json
import subprocess
import sys

import pytest
from inference_endpoint.commands.benchmark import steady_state
from inference_endpoint.config.schema import BenchmarkConfig, TestType
from inference_endpoint.metrics import steady_state_diagnostics

pytestmark = pytest.mark.unit

_DETECTOR_MODULE = "inference_endpoint.metrics.steady_state_diagnostics"


def _report_dir(tmp_path):
    (tmp_path / "events.jsonl").write_text("")
    return tmp_path


class TestGate:
    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-oss-120b",
            "openai/gpt-oss-120b",
            "GPT-OSS-120B",
            "deepseek-r1",
            "deepseek-ai/DeepSeek-R1",
            "dsr1-fp4",
        ],
    )
    def test_allowlisted_models_run(self, model_name):
        assert steady_state.should_run(
            model_name=model_name, enabled=True, is_agentic=False
        )

    @pytest.mark.parametrize(
        "model_name",
        ["meta-llama/Llama-3.1-8B-Instruct", "Qwen3-VL-235B", "kimi-k3", ""],
    )
    def test_other_models_are_skipped(self, model_name):
        assert not steady_state.should_run(
            model_name=model_name, enabled=True, is_agentic=False
        )

    def test_deepseek_v4_is_not_matched_by_the_deepseek_r1_entry(self):
        assert not steady_state.should_run(
            model_name="deepseek-v4", enabled=True, is_agentic=False
        )

    def test_agentic_runs_are_skipped(self):
        assert not steady_state.should_run(
            model_name="gpt-oss-120b", enabled=True, is_agentic=True
        )

    def test_disabled_config_skips(self):
        assert not steady_state.should_run(
            model_name="gpt-oss-120b", enabled=False, is_agentic=False
        )


class TestCommand:
    def test_invokes_the_detector_module_on_the_report_dir(self, tmp_path):
        cmd = steady_state.build_command(tmp_path)
        assert cmd == [
            sys.executable,
            "-m",
            _DETECTOR_MODULE,
            str(tmp_path),
            "--json",
            str(tmp_path / "steady_state.json"),
        ]


class TestRun:
    def test_writes_stdout_to_a_sibling_text_report(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="STEADY STATE OK\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        steady_state.run_for_report(
            report_dir,
            model_name="gpt-oss-120b",
            enabled=True,
            is_agentic=False,
            timeout_s=60.0,
        )
        assert (report_dir / "steady_state.txt").read_text() == "STEADY STATE OK\n"

    def test_skipped_run_spawns_nothing(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kwargs):
            raise AssertionError("detector must not be spawned for a skipped run")

        monkeypatch.setattr(subprocess, "run", explode)
        steady_state.run_for_report(
            report_dir,
            model_name="llama-3.1-8b",
            enabled=True,
            is_agentic=False,
            timeout_s=60.0,
        )
        assert not (report_dir / "steady_state.txt").exists()

    def test_missing_events_file_skips(self, tmp_path, monkeypatch):
        def explode(cmd, **kwargs):
            raise AssertionError("detector needs events.jsonl; must not be spawned")

        monkeypatch.setattr(subprocess, "run", explode)
        steady_state.run_for_report(
            tmp_path,
            model_name="gpt-oss-120b",
            enabled=True,
            is_agentic=False,
            timeout_s=60.0,
        )

    def test_detector_failure_does_not_propagate(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

        monkeypatch.setattr(subprocess, "run", fake_run)
        steady_state.run_for_report(
            report_dir,
            model_name="gpt-oss-120b",
            enabled=True,
            is_agentic=False,
            timeout_s=60.0,
        )

    def test_timeout_does_not_propagate(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 60.0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        steady_state.run_for_report(
            report_dir,
            model_name="gpt-oss-120b",
            enabled=True,
            is_agentic=False,
            timeout_s=60.0,
        )

    def test_unexpected_oserror_does_not_propagate(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)

        def fake_run(cmd, **kwargs):
            raise OSError("no interpreter")

        monkeypatch.setattr(subprocess, "run", fake_run)
        steady_state.run_for_report(
            report_dir,
            model_name="gpt-oss-120b",
            enabled=True,
            is_agentic=False,
            timeout_s=60.0,
        )


class TestRunMeta:
    def test_detector_reads_back_the_dataset_size(self, tmp_path):
        steady_state.write_run_meta(tmp_path, dataset_size=6396)

        parsed = steady_state_diagnostics.read_run_config(
            None, str(tmp_path / "run_meta.json")
        )

        assert parsed["dataset_size"] == 6396

    def test_unknown_dataset_size_writes_no_sidecar(self, tmp_path):
        steady_state.write_run_meta(tmp_path, dataset_size=None)

        assert not (tmp_path / "run_meta.json").exists()


def _config(model_name="gpt-oss-120b", **settings):
    return BenchmarkConfig(
        type=TestType.OFFLINE,
        model_params={"name": model_name},
        endpoint_config={"endpoints": ["http://x"]},
        datasets=[{"path": "D"}],
        settings=settings,
    )


def _agentic_config(model_name="gpt-oss-120b"):
    return BenchmarkConfig(
        type=TestType.ONLINE,
        model_params={"name": model_name},
        endpoint_config={"endpoints": ["http://x"]},
        datasets=[{"path": "D", "agentic_inference": {}}],
        settings={
            "load_pattern": {"type": "agentic_inference", "target_concurrency": 8}
        },
    )


class TestRunForContext:
    def test_eligible_run_writes_sidecar_and_spawns_detector(
        self, tmp_path, monkeypatch
    ):
        report_dir = _report_dir(tmp_path)
        spawned = []

        def fake_run(cmd, **kwargs):
            spawned.append((cmd, kwargs.get("timeout")))
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        steady_state.run_for_context(report_dir, _config(), dataset_size=6396)

        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": 6396
        }
        assert len(spawned) == 1
        assert spawned[0][0] == steady_state.build_command(report_dir)

    def test_agentic_run_is_skipped(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kwargs):
            raise AssertionError("agentic profile is unsupported; must not spawn")

        monkeypatch.setattr(subprocess, "run", explode)
        steady_state.run_for_context(report_dir, _agentic_config(), dataset_size=6396)

    def test_opt_out_is_honoured(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kwargs):
            raise AssertionError("steady_state.enabled=false must not spawn")

        monkeypatch.setattr(subprocess, "run", explode)
        steady_state.run_for_context(
            report_dir,
            _config(**{"steady_state": {"enabled": False}}),
            dataset_size=6396,
        )

    def test_configured_timeout_reaches_the_subprocess(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(kwargs.get("timeout"))
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        steady_state.run_for_context(
            report_dir,
            _config(**{"timeouts": {"steady_state_timeout_s": 42.0}}),
            dataset_size=6396,
        )

        assert seen == [42.0]
