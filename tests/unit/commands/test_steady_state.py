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
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    LoadPatternType,
    TestType,
)
from inference_endpoint.metrics import steady_state_diagnostics

pytestmark = pytest.mark.unit

_DETECTOR_MODULE = "inference_endpoint.metrics.steady_state_diagnostics"
_TOKENIZER = "openai/gpt-oss-120b"
_DATASET_SIZE = 6396

# The load patterns the detector has a validated profile for. Stated here rather
# than derived, so adding a pattern forces a decision instead of inheriting one.
_ELIGIBLE_PATTERNS = {
    LoadPatternType.POISSON,
    LoadPatternType.CONCURRENCY,
}


def _report_dir(tmp_path):
    """A report dir that already holds an earlier run's artifacts.

    report_dir is user-settable and reusable, so the interesting case is the
    dirty one: assertions that nothing is left behind are vacuous against a
    directory that never had anything in it.
    """
    (tmp_path / "events.jsonl").write_text("")
    (tmp_path / "steady_state.json").write_text('{"from": "an earlier run"}')
    (tmp_path / "steady_state.txt").write_text("an earlier run\n")
    (tmp_path / "run_meta.json").write_text('{"dataset_size": 11}')
    return tmp_path


def _gate(model_name="gpt-oss-120b", load_pattern=LoadPatternType.CONCURRENCY):
    return steady_state.is_eligible(model_name=model_name, load_pattern=load_pattern)


class TestEligibility:
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
    def test_allowlisted_models_are_eligible(self, model_name):
        assert _gate(model_name=model_name)

    @pytest.mark.parametrize(
        "model_name",
        ["meta-llama/Llama-3.1-8B-Instruct", "Qwen3-VL-235B", "kimi-k3", ""],
    )
    def test_other_models_are_not(self, model_name):
        assert not _gate(model_name=model_name)

    def test_deepseek_v4_is_not_matched_by_the_deepseek_r1_entry(self):
        assert not _gate(model_name="deepseek-v4")

    @pytest.mark.parametrize("load_pattern", list(LoadPatternType))
    def test_every_load_pattern_has_a_decided_eligibility(self, load_pattern):
        """Exhaustive over the enum: a new load pattern must be classified in the
        detector's profile table rather than inheriting a validated profile."""
        assert _gate(load_pattern=load_pattern) is (load_pattern in _ELIGIBLE_PATTERNS)

    def test_an_unclassified_load_pattern_fails_closed(self):
        """Unmapped patterns must resolve to an unsupported profile, not to the
        concurrency one."""
        assert not steady_state_diagnostics.profile_for_load_pattern(
            "no-such-pattern"
        ).supported

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


class TestCaveatFilter:
    def test_prefix_matches_what_the_detector_actually_emits(self):
        """Pinned against the detector's own formatter rather than a copy of the
        string, so a change on either side fails here instead of silently
        publishing bookkeeping as a reliability note."""
        profile = steady_state_diagnostics.profile_for_load_pattern("poisson")
        assert profile.note, "expected the poisson profile to carry a caveat"

        line = steady_state_diagnostics.format_profile_caveat(profile)

        assert line.startswith(steady_state._CAVEAT_PREFIX)

    def test_keeps_only_the_detectors_own_caveats(self):
        caveat = "[profile: offline] system TPS is unreliable"
        stderr = "\n".join(
            [
                "None of PyTorch, TensorFlow >= 2.0 have been found.",
                caveat,
                "",
                "wrote /runs/r1/steady_state.json",
            ]
        )

        assert steady_state._detector_caveats(stderr) == caveat

    def test_no_caveats_means_empty(self):
        assert steady_state._detector_caveats("wrote /runs/r1/steady_state.json") == ""
        assert steady_state._detector_caveats(None) == ""


def _config(model_name="gpt-oss-120b", **settings):
    """A concurrency run: the detector has a validated profile for it."""
    settings.setdefault(
        "load_pattern", {"type": "concurrency", "target_concurrency": 8}
    )
    return BenchmarkConfig(
        type=TestType.ONLINE,
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


def _succeeding(report_dir, stdout="headline\n", stderr=""):
    """A child that behaves like the real one: exit 0 AND a verdict on disk."""

    def run(cmd, **kw):
        (report_dir / "steady_state.json").write_text("{}")
        return _completed(cmd, stdout=stdout, stderr=stderr)

    return run


class TestDetectSteadyState:
    def test_eligible_run_writes_sidecar_and_spawns_detector(
        self, tmp_path, monkeypatch
    ):
        report_dir = _report_dir(tmp_path)
        spawned = []

        def fake_run(cmd, **kwargs):
            spawned.append((cmd, kwargs.get("timeout")))
            (report_dir / "steady_state.json").write_text("{}")
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
            subprocess, "run", _succeeding(report_dir, stdout="HEADLINE\n")
        )

        _detect(report_dir)

        assert (report_dir / "steady_state.txt").read_text() == "HEADLINE\n"

    def test_detector_caveats_survive_a_successful_run(
        self, tmp_path, monkeypatch, caplog
    ):
        """The detector reports caveats (e.g. 'system TPS is unreliable') on
        stderr; dropping them would leave an untrustworthy number looking
        authoritative."""
        report_dir = _report_dir(tmp_path)
        caveat = "[profile: offline] offline: system TPS is unreliable"
        monkeypatch.setattr(
            subprocess, "run", _succeeding(report_dir, stderr=caveat + "\n")
        )

        with caplog.at_level(logging.INFO):
            _detect(report_dir)

        assert caveat in (report_dir / "steady_state.txt").read_text()
        assert caveat in caplog.text

    def test_child_bookkeeping_and_third_party_noise_stay_out_of_the_artifact(
        self, tmp_path, monkeypatch, caplog
    ):
        """stderr also carries the child's 'wrote <path>' receipt and anything
        transformers emits. Neither is something the detector said."""
        report_dir = _report_dir(tmp_path)
        noise = (
            "None of PyTorch, TensorFlow >= 2.0 have been found.\n"
            f"\nwrote {report_dir / 'steady_state.json'}\n"
        )
        monkeypatch.setattr(subprocess, "run", _succeeding(report_dir, stderr=noise))

        with caplog.at_level(logging.INFO):
            _detect(report_dir)

        assert (report_dir / "steady_state.txt").read_text() == "headline\n"
        assert "wrote " not in caplog.text
        assert "PyTorch" not in caplog.text

    @pytest.mark.parametrize(
        ("kwargs", "expected_reason"),
        [
            ({"config": _config("llama-3.1-8b")}, "not a validated workload"),
            ({"config": _agentic_config()}, "not a validated workload"),
            (
                {"config": _config(**{"steady_state": {"enabled": False}})},
                "disabled by configuration",
            ),
            ({"tokenizer_name": None}, "no tokenizer"),
            ({"dataset_size": None}, "dataset size unknown"),
        ],
    )
    def test_skipped_runs_say_why_and_leave_no_artifacts(
        self, tmp_path, monkeypatch, caplog, kwargs, expected_reason
    ):
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kw):
            raise AssertionError(f"detector must not be spawned: {expected_reason}")

        monkeypatch.setattr(subprocess, "run", explode)
        with caplog.at_level(logging.INFO):
            result = _detect(report_dir, **kwargs)

        assert result is None
        assert expected_reason in caplog.text
        assert not (report_dir / "steady_state.txt").exists()
        assert not (
            report_dir / "run_meta.json"
        ).exists(), "a skipped run must not leave a sidecar nothing will read"

    def test_missing_events_file_skips(self, tmp_path, monkeypatch, caplog):
        def explode(cmd, **kw):
            raise AssertionError("detector needs events.jsonl; must not be spawned")

        monkeypatch.setattr(subprocess, "run", explode)

        with caplog.at_level(logging.INFO):
            assert _detect(tmp_path) is None

        assert "events.jsonl" in caplog.text

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

    def test_a_silent_child_is_not_reported_as_success(self, tmp_path, monkeypatch):
        """Exit 0 without a verdict file is not a result."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: _completed(cmd, stdout="headline\n")
        )

        assert _detect(report_dir) is None


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

    @pytest.mark.parametrize(
        ("outcome", "kwargs"),
        [
            pytest.param("failure", {}, id="nonzero-exit"),
            pytest.param("timeout", {}, id="timeout"),
            pytest.param("interrupt", {}, id="interrupt"),
            pytest.param("silent", {}, id="exit-0-no-verdict"),
            pytest.param("skip", {"config": _config("llama-3.1-8b")}, id="ineligible"),
            pytest.param(
                "skip",
                {"config": _config(**{"steady_state": {"enabled": False}})},
                id="disabled",
            ),
            pytest.param("skip", {"tokenizer_name": None}, id="no-tokenizer"),
            pytest.param("skip", {"dataset_size": None}, id="no-dataset-size"),
        ],
    )
    def test_a_stale_verdict_never_outlives_the_run_it_described(
        self, tmp_path, monkeypatch, outcome, kwargs
    ):
        """report_dir is user-settable and reusable: a previous run's verdict must
        not sit beside a newer run's results."""
        report_dir = _report_dir(tmp_path)

        def run(cmd, **kw):
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(cmd, 1.0)
            if outcome == "interrupt":
                raise KeyboardInterrupt
            if outcome == "silent":
                return _completed(cmd, 0, stdout="headline\n")
            return _completed(cmd, 2, stderr="failed")

        monkeypatch.setattr(subprocess, "run", run)

        assert _detect(report_dir, **kwargs) is None
        assert not (report_dir / "steady_state.json").exists()
        assert not (report_dir / "steady_state.txt").exists()

    def test_a_skipped_run_clears_the_stale_sidecar_too(self, tmp_path, monkeypatch):
        """A stale run_meta.json would feed the wrong super-pass size to a later
        by-hand re-run against this directory."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: pytest.fail("must not spawn")
        )

        _detect(report_dir, config=_config("llama-3.1-8b"))

        assert not (report_dir / "run_meta.json").exists()

    def test_a_failed_run_keeps_its_own_fresh_sidecar(self, tmp_path, monkeypatch):
        """The detector ran, so run_meta.json describes THIS run and stays usable
        for a by-hand retry."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: _completed(cmd, 2, stderr="failed")
        )

        _detect(report_dir)

        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": _DATASET_SIZE
        }

    @pytest.mark.parametrize("artifact", ["run_meta.json", "steady_state.txt"])
    def test_unwritable_report_dir_is_absorbed(
        self, tmp_path, monkeypatch, caplog, artifact
    ):
        """A full disk or read-only mount must not fail a finished run."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(subprocess, "run", _succeeding(report_dir, stdout="out\n"))
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
