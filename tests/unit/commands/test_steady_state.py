# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-run steady-state detection step.

Pins the gate (which runs earn a detection pass), the command handed to the
detector subprocess, and the best-effort contract: no failure mode here may
propagate out of finalize.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from inference_endpoint.commands.benchmark import steady_state
from inference_endpoint.config.schema import BenchmarkConfig, TestType
from inference_endpoint.metrics import steady_state_diagnostics

pytestmark = pytest.mark.unit

_DETECTOR_MODULE = "inference_endpoint.metrics.steady_state_diagnostics"
_TOKENIZER = "openai/gpt-oss-120b"


def _report_dir(tmp_path):
    (tmp_path / "events.jsonl").write_text("")
    return tmp_path


def _gate(**overrides):
    kwargs = {
        "model_name": "gpt-oss-120b",
        "enabled": True,
        "is_agentic": False,
        "tokenizer_name": _TOKENIZER,
    }
    kwargs.update(overrides)
    return steady_state.should_run(**kwargs)


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
        assert _gate(model_name=model_name)

    @pytest.mark.parametrize(
        "model_name",
        ["meta-llama/Llama-3.1-8B-Instruct", "Qwen3-VL-235B", "kimi-k3", ""],
    )
    def test_other_models_are_skipped(self, model_name):
        assert not _gate(model_name=model_name)

    def test_deepseek_v4_is_not_matched_by_the_deepseek_r1_entry(self):
        assert not _gate(model_name="deepseek-v4")

    def test_agentic_runs_are_skipped(self):
        assert not _gate(is_agentic=True)

    def test_disabled_config_skips(self):
        assert not _gate(enabled=False)

    def test_unresolved_tokenizer_skips(self):
        """Without the run's tokenizer the detector would fall back to its own
        registry, whose kimi entries carry trust_remote_code=True."""
        assert not _gate(tokenizer_name=None)

    def test_no_allowlisted_model_resolves_to_a_trust_remote_code_tokenizer(self):
        """The allowlist must never select a registry entry that would execute
        remote code, in case the tokenizer is ever resolved through it."""
        for substring in steady_state._SUPPORTED_MODEL_SUBSTRINGS:
            resolved = steady_state_diagnostics.resolve_tokenizer(substring)
            assert resolved is not None, f"{substring} resolves no tokenizer"
            _tokenizer_id, trust_remote_code = resolved
            assert not trust_remote_code, f"{substring} would trust remote code"


class TestCommand:
    def test_pins_the_tokenizer_resolved_by_the_run(self, tmp_path):
        cmd = steady_state.build_command(tmp_path, tokenizer_name=_TOKENIZER)

        assert cmd == [
            sys.executable,
            "-m",
            _DETECTOR_MODULE,
            str(tmp_path),
            "--json",
            str(tmp_path / "steady_state.json"),
            "--tokenizer",
            _TOKENIZER,
        ]


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


def _run(report_dir, config=None, **overrides):
    kwargs = {"tokenizer_name": _TOKENIZER, "dataset_size": 6396}
    kwargs.update(overrides)
    steady_state.run_for_context(report_dir, config or _config(), **kwargs)


class TestRunForContext:
    def test_eligible_run_writes_sidecar_and_spawns_detector(
        self, tmp_path, monkeypatch
    ):
        report_dir = _report_dir(tmp_path)
        spawned = []

        def fake_run(cmd, **kwargs):
            spawned.append((cmd, kwargs.get("timeout")))
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        _run(report_dir)

        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": 6396
        }
        assert len(spawned) == 1
        assert spawned[0][0] == steady_state.build_command(
            report_dir, tokenizer_name=_TOKENIZER
        )

    def test_writes_stdout_to_a_sibling_text_report(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, stdout="STEADY STATE OK\n", stderr=""
            ),
        )

        _run(report_dir)

        assert (report_dir / "steady_state.txt").read_text() == "STEADY STATE OK\n"

    def test_detector_notes_on_stderr_survive_a_successful_run(
        self, tmp_path, monkeypatch, caplog
    ):
        """The detector prints its reliability caveats (e.g. 'offline: system TPS
        is unreliable') to stderr; dropping them on success would leave an
        untrustworthy number looking authoritative."""
        report_dir = _report_dir(tmp_path)
        note = "[profile: offline] offline: system TPS is unreliable"
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, stdout="headline\n", stderr=note + "\n"
            ),
        )

        with caplog.at_level(logging.INFO):
            _run(report_dir)

        assert note in (report_dir / "steady_state.txt").read_text()
        assert note in caplog.text

    @pytest.mark.parametrize(
        ("label", "kwargs"),
        [
            ("unsupported model", {"config": _config("llama-3.1-8b")}),
            ("agentic", {"config": _agentic_config()}),
            ("disabled", {"config": _config(**{"steady_state": {"enabled": False}})}),
            ("no tokenizer", {"tokenizer_name": None}),
        ],
    )
    def test_skipped_runs_spawn_nothing_and_leave_no_artifacts(
        self, tmp_path, monkeypatch, label, kwargs
    ):
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kw):
            raise AssertionError(f"detector must not be spawned for {label}")

        monkeypatch.setattr(subprocess, "run", explode)
        _run(report_dir, **kwargs)

        assert not (report_dir / "steady_state.txt").exists()
        assert not (
            report_dir / "run_meta.json"
        ).exists(), "a skipped run must not leave a sidecar nothing will read"

    def test_missing_events_file_skips(self, tmp_path, monkeypatch):
        def explode(cmd, **kw):
            raise AssertionError("detector needs events.jsonl; must not be spawned")

        monkeypatch.setattr(subprocess, "run", explode)
        _run(tmp_path)

    def test_unknown_dataset_size_skips(self, tmp_path, monkeypatch):
        """The detector hard-errors without a super-pass size, so spawning it
        would only burn a tokenizer load to reach exit 2."""
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kw):
            raise AssertionError("must not spawn without a dataset size")

        monkeypatch.setattr(subprocess, "run", explode)
        _run(report_dir, dataset_size=None)

    def test_configured_timeout_reaches_the_subprocess(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(kwargs.get("timeout"))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        _run(
            report_dir, config=_config(**{"timeouts": {"steady_state_timeout_s": 42.0}})
        )

        assert seen == [42.0]


class TestBestEffortContract:
    """Nothing in this module may fail a run whose artifacts are already written."""

    def _expect_no_raise(self, report_dir, monkeypatch, failure):
        monkeypatch.setattr(subprocess, "run", failure)
        _run(report_dir)

    def test_detector_failure_is_absorbed_and_logged(
        self, tmp_path, monkeypatch, caplog
    ):
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 2, stdout="", stderr="boom"
            ),
        )

        with caplog.at_level(logging.WARNING):
            _run(report_dir)

        assert not (report_dir / "steady_state.txt").exists()
        assert "boom" in caplog.text

    def test_timeout_is_absorbed_and_logged(self, tmp_path, monkeypatch, caplog):
        report_dir = _report_dir(tmp_path)

        def timeout(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 60.0)

        with caplog.at_level(logging.WARNING):
            self._expect_no_raise(report_dir, monkeypatch, timeout)

        assert "Steady-state detection" in caplog.text

    def test_spawn_oserror_is_absorbed(self, tmp_path, monkeypatch, caplog):
        report_dir = _report_dir(tmp_path)

        def oserror(cmd, **kw):
            raise OSError("no interpreter")

        with caplog.at_level(logging.WARNING):
            self._expect_no_raise(report_dir, monkeypatch, oserror)

        assert "no interpreter" in caplog.text

    def test_keyboard_interrupt_does_not_lose_a_successful_run(
        self, tmp_path, monkeypatch, caplog
    ):
        """Ctrl-C to skip a slow diagnostic must not convert an already-complete
        run into an interrupted one."""
        report_dir = _report_dir(tmp_path)

        def interrupt(cmd, **kw):
            raise KeyboardInterrupt

        with caplog.at_level(logging.WARNING):
            self._expect_no_raise(report_dir, monkeypatch, interrupt)

        assert "cancelled" in caplog.text.lower()

    @pytest.mark.parametrize("artifact", ["run_meta.json", "steady_state.txt"])
    def test_unwritable_report_dir_is_absorbed(
        self, tmp_path, monkeypatch, caplog, artifact
    ):
        """A full disk or read-only mount must not fail a finished run."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0, stdout="out\n", stderr=""
            ),
        )
        real_write_text = Path.write_text

        def failing_write_text(self, *args, **kwargs):
            if self.name == artifact:
                raise OSError(f"no space left on device: {artifact}")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)

        with caplog.at_level(logging.WARNING):
            _run(report_dir)

        assert artifact in caplog.text


class TestRunMeta:
    def test_detector_reads_back_the_dataset_size(self, tmp_path):
        steady_state.write_run_meta(tmp_path, dataset_size=6396)

        parsed = steady_state_diagnostics.read_run_config(
            None, str(tmp_path / "run_meta.json")
        )

        assert parsed["dataset_size"] == 6396
