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
_DATASET_SIZE = 6396


def _report_dir(tmp_path):
    (tmp_path / "events.jsonl").write_text("")
    return tmp_path


def _gate(**overrides):
    kwargs = {
        "model_name": "gpt-oss-120b",
        "enabled": True,
        "load_pattern": "concurrency",
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

    def test_disabled_config_skips(self):
        assert not _gate(enabled=False)

    @pytest.mark.parametrize(
        "load_pattern", ["concurrency", "poisson", "max_throughput"]
    )
    def test_supported_load_patterns_run(self, load_pattern):
        assert _gate(load_pattern=load_pattern)

    def test_unsupported_load_pattern_skips(self):
        """The gate defers to the detector's own Profile.supported, so enabling a
        workload later is one flag in the detector rather than a change here."""
        assert not _gate(load_pattern="agentic_inference")

    def test_gate_tracks_the_detectors_profile_table(self):
        """If the detector ever marks a profile unsupported, the gate follows."""
        for load_pattern in (
            "concurrency",
            "poisson",
            "max_throughput",
            "agentic_inference",
        ):
            profile = steady_state_diagnostics.profile_for_load_pattern(load_pattern)
            assert _gate(load_pattern=load_pattern) is profile.supported

    def test_no_allowlisted_model_would_trust_remote_code(self):
        """The integration pins --tokenizer, but if the detector's registry were
        ever consulted, no allowlisted model may select a trust_remote_code entry."""
        for substring in steady_state._SUPPORTED_MODEL_SUBSTRINGS:
            resolved = steady_state_diagnostics.resolve_tokenizer(substring)
            if resolved is not None:
                _tokenizer_id, trust_remote_code = resolved
                assert not trust_remote_code, f"{substring} would trust remote code"


class TestCommand:
    def test_pins_every_input_the_detector_would_otherwise_guess(self, tmp_path):
        cmd = steady_state.build_command(
            tmp_path, tokenizer_name=_TOKENIZER, dataset_size=_DATASET_SIZE
        )

        assert cmd == [
            sys.executable,
            "-m",
            _DETECTOR_MODULE,
            str(tmp_path),
            "--json",
            str(tmp_path / "steady_state.json"),
            "--tokenizer",
            _TOKENIZER,
            "--dataset-size",
            str(_DATASET_SIZE),
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


def _completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _detect(report_dir, config=None, **overrides):
    kwargs = {"tokenizer_name": _TOKENIZER, "dataset_size": _DATASET_SIZE}
    kwargs.update(overrides)
    return steady_state.detect_steady_state(report_dir, config or _config(), **kwargs)


class TestDetectSteadyState:
    def test_eligible_run_writes_sidecar_and_spawns_detector(
        self, tmp_path, monkeypatch
    ):
        report_dir = _report_dir(tmp_path)
        spawned = []

        def fake_run(cmd, **kwargs):
            spawned.append((cmd, kwargs.get("timeout")))
            return _completed(cmd, stdout="ok\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = _detect(report_dir)

        assert result == report_dir / "steady_state.json"
        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": _DATASET_SIZE
        }
        assert len(spawned) == 1
        assert spawned[0][0] == steady_state.build_command(
            report_dir, tokenizer_name=_TOKENIZER, dataset_size=_DATASET_SIZE
        )

    def test_writes_stdout_to_a_sibling_text_report(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: _completed(cmd, stdout="HEADLINE\n")
        )

        _detect(report_dir)

        assert (report_dir / "steady_state.txt").read_text() == "HEADLINE\n"

    def test_detector_caveats_on_stderr_survive_a_successful_run(
        self, tmp_path, monkeypatch, caplog
    ):
        """The detector prints profile caveats (e.g. 'system TPS is unreliable')
        to stderr; dropping them would leave an untrustworthy number looking
        authoritative."""
        report_dir = _report_dir(tmp_path)
        caveat = "[profile: offline] offline: system TPS is unreliable"
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: _completed(
                cmd, stdout="headline\n", stderr=caveat + "\n"
            ),
        )

        with caplog.at_level(logging.INFO):
            _detect(report_dir)

        assert caveat in (report_dir / "steady_state.txt").read_text()
        assert caveat in caplog.text

    def test_the_childs_wrote_receipt_is_not_reported_as_a_caveat(
        self, tmp_path, monkeypatch, caplog
    ):
        """The detector prints 'wrote <path>' to stderr on every --json run. It is
        bookkeeping, not a caveat, and must not reach the artifact or the log."""
        report_dir = _report_dir(tmp_path)
        receipt = f"\nwrote {report_dir / 'steady_state.json'}\n"
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: _completed(cmd, stdout="headline\n", stderr=receipt),
        )

        with caplog.at_level(logging.INFO):
            _detect(report_dir)

        assert (report_dir / "steady_state.txt").read_text() == "headline\n"
        assert "wrote " not in caplog.text
        assert "notes" not in caplog.text

    def test_a_real_caveat_survives_alongside_the_receipt(
        self, tmp_path, monkeypatch, caplog
    ):
        report_dir = _report_dir(tmp_path)
        caveat = "[profile: offline] system TPS is unreliable"
        stderr = f"{caveat}\n\nwrote {report_dir / 'steady_state.json'}\n"
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: _completed(cmd, stdout="headline\n", stderr=stderr),
        )

        with caplog.at_level(logging.INFO):
            _detect(report_dir)

        body = (report_dir / "steady_state.txt").read_text()
        assert caveat in body
        assert "wrote " not in body

    @pytest.mark.parametrize(
        ("label", "kwargs"),
        [
            ("unsupported model", {"config": _config("llama-3.1-8b")}),
            ("agentic", {"config": _agentic_config()}),
            ("disabled", {"config": _config(**{"steady_state": {"enabled": False}})}),
            ("no tokenizer", {"tokenizer_name": None}),
            ("unknown dataset size", {"dataset_size": None}),
        ],
    )
    def test_skipped_runs_spawn_nothing_and_leave_no_artifacts(
        self, tmp_path, monkeypatch, caplog, label, kwargs
    ):
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kw):
            raise AssertionError(f"detector must not be spawned for {label}")

        monkeypatch.setattr(subprocess, "run", explode)
        with caplog.at_level(logging.INFO):
            result = _detect(report_dir, **kwargs)

        assert result is None
        assert not (report_dir / "steady_state.txt").exists()
        assert not (
            report_dir / "run_meta.json"
        ).exists(), "a skipped run must not leave a sidecar nothing will read"
        assert (
            "steady-state" in caplog.text.lower()
        ), "a skip must say why at default log level"

    def test_missing_events_file_skips(self, tmp_path, monkeypatch):
        def explode(cmd, **kw):
            raise AssertionError("detector needs events.jsonl; must not be spawned")

        monkeypatch.setattr(subprocess, "run", explode)

        assert _detect(tmp_path) is None

    def test_configured_timeout_reaches_the_subprocess(self, tmp_path, monkeypatch):
        report_dir = _report_dir(tmp_path)
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(kwargs.get("timeout"))
            return _completed(cmd)

        monkeypatch.setattr(subprocess, "run", fake_run)
        _detect(
            report_dir, config=_config(**{"timeouts": {"steady_state_timeout_s": 42.0}})
        )

        assert seen == [42.0]


class TestBestEffortContract:
    """Nothing in this module may fail a run whose artifacts are already written."""

    def test_detector_failure_is_absorbed_and_logged(
        self, tmp_path, monkeypatch, caplog
    ):
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: _completed(cmd, 2, stderr="boom")
        )

        with caplog.at_level(logging.WARNING):
            assert _detect(report_dir) is None

        assert not (report_dir / "steady_state.txt").exists()
        assert "boom" in caplog.text

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(subprocess.TimeoutExpired("cmd", 60.0), id="timeout"),
            pytest.param(OSError("no interpreter"), id="spawn-oserror"),
            pytest.param(KeyboardInterrupt(), id="interrupt"),
            pytest.param(TypeError("bad argv"), id="unexpected"),
        ],
    )
    def test_every_spawn_failure_is_absorbed(
        self, tmp_path, monkeypatch, caplog, failure
    ):
        report_dir = _report_dir(tmp_path)

        def raise_it(cmd, **kw):
            raise failure

        monkeypatch.setattr(subprocess, "run", raise_it)

        with caplog.at_level(logging.WARNING):
            assert _detect(report_dir) is None

        assert "steady-state" in caplog.text.lower()

    def test_a_failed_run_leaves_no_partial_json_behind(self, tmp_path, monkeypatch):
        """The child writes its JSON non-atomically, so a kill can truncate it."""
        report_dir = _report_dir(tmp_path)
        stale = report_dir / "steady_state.json"
        stale.write_text('{"truncated": ')

        def fail(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 1.0)

        monkeypatch.setattr(subprocess, "run", fail)
        _detect(report_dir)

        assert not stale.exists(), "a partial verdict must not survive as a result"

    @pytest.mark.parametrize("artifact", ["run_meta.json", "steady_state.txt"])
    def test_unwritable_report_dir_is_absorbed(
        self, tmp_path, monkeypatch, caplog, artifact
    ):
        """A full disk or read-only mount must not fail a finished run."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: _completed(cmd, stdout="out\n")
        )
        real_write_text = Path.write_text

        def failing_write_text(self, *args, **kwargs):
            if self.name == artifact:
                raise OSError(f"no space left on device: {artifact}")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)

        with caplog.at_level(logging.WARNING):
            _detect(report_dir)

        assert artifact in caplog.text


class TestRunMeta:
    def test_detector_reads_back_the_dataset_size(self, tmp_path):
        steady_state.write_run_meta(tmp_path, dataset_size=_DATASET_SIZE)

        parsed = steady_state_diagnostics.read_run_config(
            None, str(tmp_path / "run_meta.json")
        )

        assert parsed["dataset_size"] == _DATASET_SIZE
