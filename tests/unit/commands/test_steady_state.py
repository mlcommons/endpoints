# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-run steady-state detection step.

Pins the gate (which runs earn a detection pass), the command handed to the
detector subprocess, and the best-effort contract: no failure mode here may
propagate out of finalize.
"""

import dataclasses
import json
import logging
import math
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


def _gate(load_pattern=LoadPatternType.CONCURRENCY):
    return steady_state.is_eligible(load_pattern=load_pattern)


class TestEligibility:
    @pytest.mark.parametrize(
        "model_name",
        ["gpt-oss-120b", "deepseek_r1-torch-fp4", "meta-llama/Llama-3.1-8B", ""],
    )
    def test_the_model_does_not_decide_eligibility(self, model_name):
        """No allowlist: detection is opt-in per config, so any model that asks
        for it gets it. Only the load pattern can rule a run out."""
        assert _gate()
        assert steady_state.is_eligible(load_pattern=LoadPatternType.CONCURRENCY)

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


class TestCommand:
    def test_pins_every_input_the_detector_would_otherwise_guess(self, tmp_path):
        cmd = steady_state.build_command(
            tmp_path,
            tokenizer_name=_TOKENIZER,
            dataset_size=_DATASET_SIZE,
            load_pattern=LoadPatternType.CONCURRENCY,
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
            "--profile",
            "concurrency",
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

    def test_a_wrapped_note_is_collapsed_to_one_line(self):
        """The consumer matches the prefix per line, so a multi-line caveat would
        lose everything after the first line."""
        profile = steady_state_diagnostics.profile_for_load_pattern("poisson")
        wrapped = dataclasses.replace(profile, note="first line\nsecond line")

        line = steady_state_diagnostics.format_profile_caveat(wrapped)

        assert "\n" not in line
        assert "second line" in line

    def test_keeps_only_the_detectors_own_caveats(self):
        caveat = "[profile: poisson] usually under-saturated"
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
    # Detection is opt-in. These tests are about what happens once it is on.
    settings.setdefault("steady_state", {"enabled": True})
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
            "load_pattern": {"type": "agentic_inference", "target_concurrency": 8},
            "steady_state": {"enabled": True},
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
    def test_eligible_run_writes_run_meta_and_spawns_detector(
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
            report_dir,
            tokenizer_name=_TOKENIZER,
            dataset_size=_DATASET_SIZE,
            load_pattern=LoadPatternType.CONCURRENCY,
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
        caveat = "[profile: poisson] poisson: usually under-saturated"
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
        ("kwargs", "expected_reason", "keeps_run_meta"),
        [
            # A load pattern the detector cannot profile still has good
            # metadata for this run, and its message points at a hand re-run
            # that reads run_meta.json.
            (
                {"config": _agentic_config()},
                "the detector has no profile for",
                True,
            ),
            (
                {"config": _config(**{"steady_state": {"enabled": False}})},
                "disabled by configuration",
                False,
            ),
            ({"tokenizer_name": None}, "no tokenizer", False),
            ({"dataset_size": None}, "dataset size unknown", False),
        ],
    )
    def test_skipped_runs_say_why_and_leave_the_right_artifacts(
        self, tmp_path, monkeypatch, caplog, kwargs, expected_reason, keeps_run_meta
    ):
        report_dir = _report_dir(tmp_path)

        def explode(cmd, **kw):
            # pytest.fail raises BaseException; detect_steady_state's
            # `except Exception` would swallow an AssertionError and the
            # surviving log assertion would still pass.
            pytest.fail(f"detector must not be spawned: {expected_reason}")

        monkeypatch.setattr(subprocess, "run", explode)
        with caplog.at_level(logging.INFO):
            result = _detect(report_dir, **kwargs)

        assert result is None
        assert expected_reason in caplog.text
        assert not (report_dir / "steady_state.txt").exists()
        assert not (report_dir / "steady_state.json").exists()
        if keeps_run_meta:
            assert json.loads((report_dir / "run_meta.json").read_text()) == {
                "dataset_size": _DATASET_SIZE
            }
        else:
            assert not (
                report_dir / "run_meta.json"
            ).exists(), "a skipped run must not leave a run_meta.json nothing will read"

    def test_missing_events_file_skips(self, tmp_path, monkeypatch, caplog):
        def explode(cmd, **kw):
            pytest.fail("detector needs events.jsonl; must not be spawned")

        monkeypatch.setattr(subprocess, "run", explode)

        with caplog.at_level(logging.INFO):
            assert _detect(tmp_path) is None

        assert "events.jsonl" in caplog.text

    def test_a_missing_event_log_is_reported_before_the_workload_verdict(
        self, tmp_path, monkeypatch, caplog
    ):
        """events.jsonl is the actionable blocker, and the 're-run by hand'
        message an unprofiled load pattern emits would be useless without one."""
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: pytest.fail("must not spawn")
        )

        with caplog.at_level(logging.INFO):
            _detect(tmp_path, config=_agentic_config())

        assert "no events.jsonl" in caplog.text
        assert "not a validated workload" not in caplog.text
        assert not (tmp_path / "run_meta.json").exists()

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

    def test_an_interrupt_propagates_instead_of_being_absorbed(
        self, tmp_path, monkeypatch, caplog
    ):
        """A ^C during detection must reach finalize_benchmark.

        Detection runs before the report is written. Swallowing the first ^C
        would carry on into the slowest part of finalize; the user's second one
        then force-kills the process group and the run loses every artifact.
        The half-written verdict still has to go.
        """
        report_dir = _report_dir(tmp_path)

        def run(cmd, **kw):
            (report_dir / "steady_state.json").write_text('{"truncated": ')
            raise KeyboardInterrupt

        monkeypatch.setattr(subprocess, "run", run)

        with caplog.at_level(logging.WARNING), pytest.raises(KeyboardInterrupt):
            _detect(report_dir)

        assert not (report_dir / "steady_state.json").exists()
        assert "interrupt" in caplog.text.lower()

    @pytest.mark.parametrize(
        ("outcome", "kwargs"),
        [
            pytest.param("failure", {}, id="nonzero-exit"),
            pytest.param("timeout", {}, id="timeout"),
            pytest.param("interrupt", {}, id="interrupt"),
            pytest.param("silent", {}, id="exit-0-no-verdict"),
            pytest.param("skip", {"config": _agentic_config()}, id="ineligible"),
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
            # A killed or failing child can leave a half-written verdict: it
            # writes the JSON non-atomically.
            (report_dir / "steady_state.json").write_text('{"truncated": ')
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(cmd, 1.0)
            if outcome == "interrupt":
                raise KeyboardInterrupt
            if outcome == "silent":
                (report_dir / "steady_state.json").unlink()
                return _completed(cmd, 0, stdout="headline\n")
            return _completed(cmd, 2, stderr="failed")

        monkeypatch.setattr(subprocess, "run", run)

        # An interrupt propagates (finalize_benchmark needs it to mark the run
        # invalid); every other outcome returns None. The verdict goes either way.
        if outcome == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                _detect(report_dir, **kwargs)
        else:
            assert _detect(report_dir, **kwargs) is None
        assert not (report_dir / "steady_state.json").exists()
        assert not (report_dir / "steady_state.txt").exists()

    def test_a_skipped_run_clears_stale_run_meta_too(self, tmp_path, monkeypatch):
        """A stale run_meta.json would feed the wrong super-pass size to a later
        by-hand re-run against this directory."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: pytest.fail("must not spawn")
        )

        _detect(report_dir, tokenizer_name=None)

        assert not (report_dir / "run_meta.json").exists()

    def test_an_unvalidated_workload_still_gets_fresh_run_meta(
        self, tmp_path, monkeypatch
    ):
        """Its caveat tells the user to hand-run the detector, which reads
        run_meta.json. This run's metadata must replace the stale one."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: pytest.fail("must not spawn")
        )

        _detect(report_dir, config=_agentic_config())

        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": _DATASET_SIZE
        }

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(None, id="eligible-pre-spawn"),
            pytest.param(_agentic_config(), id="unvalidated-workload"),
        ],
    )
    def test_an_unclearable_stale_verdict_aborts_rather_than_republishing(
        self, tmp_path, monkeypatch, config
    ):
        """Success is an existence test, so a verdict we could not delete would
        be reported as this run's result. The unvalidated-workload path is the
        sharper case: it would otherwise write a fresh run_meta.json next to the
        stale steady_state.json, making that verdict look like this run's."""
        report_dir = _report_dir(tmp_path)
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: pytest.fail("must not spawn")
        )
        real_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if self.name == "steady_state.json":
                raise OSError("read-only file system")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", failing_unlink)

        assert _detect(report_dir, config=config) is None
        # Either cleared outright, or left as the earlier run's. What must not
        # happen is this run's run_meta.json being published beside a stale
        # steady_state.json, making that verdict look like this run's.
        run_meta = report_dir / "run_meta.json"
        published = run_meta.exists() and json.loads(run_meta.read_text()) == {
            "dataset_size": _DATASET_SIZE
        }
        assert not published

    @pytest.mark.parametrize(
        "outcome",
        [
            pytest.param("failure", id="nonzero-exit"),
            pytest.param("timeout", id="timeout"),
            pytest.param("interrupt", id="interrupt"),
        ],
    )
    def test_a_failed_run_keeps_its_own_fresh_run_meta(
        self, tmp_path, monkeypatch, outcome
    ):
        """The detector ran, so run_meta.json describes THIS run and stays usable
        for a by-hand retry -- on every failure path, not just a non-zero exit."""
        report_dir = _report_dir(tmp_path)

        def run(cmd, **kw):
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(cmd, 1.0)
            if outcome == "interrupt":
                raise KeyboardInterrupt
            return _completed(cmd, 2, stderr="failed")

        monkeypatch.setattr(subprocess, "run", run)

        if outcome == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                _detect(report_dir)
        else:
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
        assert not (
            report_dir / artifact
        ).exists(), "a failed write must not leave the previous run's file in place"


class TestDatasetSize:
    """num_samples is implemented by Dataset subclasses; a raise must not fail
    a run."""

    def test_reports_the_loaded_sample_count(self):
        class _Dataset:
            def num_samples(self):
                return 4242

        assert steady_state.dataset_size_of(_Dataset()) == 4242

    def test_no_dataset_is_no_size(self):
        assert steady_state.dataset_size_of(None) is None

    def test_a_raising_dataset_is_absorbed(self, caplog):
        class _Broken:
            def num_samples(self):
                raise RuntimeError("dataset went away")

        with caplog.at_level(logging.WARNING):
            assert steady_state.dataset_size_of(_Broken()) is None

        assert "dataset size unavailable" in caplog.text


class TestDiscardArtifacts:
    def test_clears_everything_this_step_owns(self, tmp_path):
        report_dir = _report_dir(tmp_path)

        steady_state.discard_artifacts(report_dir)

        assert not (report_dir / "steady_state.json").exists()
        assert not (report_dir / "steady_state.txt").exists()
        assert not (report_dir / "run_meta.json").exists()
        assert (report_dir / "events.jsonl").exists(), "must not touch the event log"


_HEADLINE_CONTEXT_KEYS = ("superpass_size", "n_super_passes", "n_post_warmup")


class TestVerdictHeadline:
    """The headline lifted out of steady_state.json and onto the Report.

    This is the trust boundary: everything downstream formats these values into
    the run's primary artifacts.
    """

    @staticmethod
    def _write(tmp_path, blob):
        path = tmp_path / "steady_state.json"
        path.write_text(json.dumps(blob))
        return path

    def test_lifts_the_headline_and_its_sizing_context(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "superpass_size": 4388,
                "n_super_passes": 12,
                "n_post_warmup": 11,
                "steady_state": {"found": True, "window": {"n_samples": 17552}},
            },
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got["found"] is True
        assert got["superpass_size"] == 4388
        assert got["n_post_warmup"] == 11

    def test_leaves_the_bulk_diagnostics_behind(self, tmp_path):
        """trajectories/cov/drift/per_super_pass would dwarf result_summary.json."""
        path = self._write(
            tmp_path,
            {
                "steady_state": {"found": True},
                "trajectories": {"tpot": [1, 2, 3]},
                "cov": {"tpot": {}},
                "drift": {"tpot": {}},
                "per_super_pass": [{"i": 0}],
            },
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert not {"trajectories", "cov", "drift", "per_super_pass"} & set(got)

    def test_missing_sizing_context_is_omitted_not_defaulted(self, tmp_path):
        """Absent context keys stay absent rather than arriving as a zero that
        reads like a measurement."""
        path = self._write(tmp_path, {"steady_state": {"found": False}})

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got["found"] is False
        assert not set(_HEADLINE_CONTEXT_KEYS) & set(got)

    def test_an_absent_file_is_absorbed(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            assert steady_state.verdict_headline(tmp_path / "nope.json") is None

        assert "could not read" in caplog.text

    def test_truncated_json_is_absorbed(self, tmp_path, caplog):
        path = tmp_path / "steady_state.json"
        path.write_text('{"steady_state": {"found": tru')

        with caplog.at_level(logging.WARNING):
            assert steady_state.verdict_headline(path) is None

        assert "could not read" in caplog.text

    @pytest.mark.parametrize("raw", ["null", "[1, 2]", '"hi"', "3"])
    def test_a_non_object_verdict_file_is_absorbed(self, tmp_path, raw, caplog):
        """Valid JSON, wrong top-level type. json.loads succeeds, so this lands
        past the OSError/ValueError arm and would otherwise raise AttributeError
        out of a finished run."""
        path = tmp_path / "steady_state.json"
        path.write_text(raw)

        with caplog.at_level(logging.WARNING):
            assert steady_state.verdict_headline(path) is None

        assert "no verdict" in caplog.text

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tps", {"per_user": "fast", "system": 1.0}),
            ("tps", {"per_user": True, "system": 1.0}),
            ("tps", {"per_user": float("nan"), "system": float("inf")}),
            ("ttft", {"p50": None, "p90": "slow"}),
            ("window", {"sp_lo": "a", "sp_hi": 5, "n_samples": None}),
            ("drifting_up", "tpot"),
            ("drifting_up", [None, 1, "tpot"]),
            ("short_window", "yes"),
        ],
    )
    def test_unusable_values_are_dropped_at_the_boundary(self, tmp_path, field, value):
        """This is the only place the detector's JSON is checked. Everything
        downstream formats these values into the run's primary artifacts, so a
        value that cannot be rendered must not get past here."""
        path = tmp_path / "steady_state.json"
        path.write_text(json.dumps({"steady_state": {"found": True, field: value}}))

        got = steady_state.verdict_headline(path)

        assert got is not None
        if field == "drifting_up":
            # Only real metric names survive; a bare string must not be
            # exploded into characters downstream.
            assert got["drifting_up"] == (["tpot"] if isinstance(value, list) else [])
        else:
            block = got.get(field)
            for key, raw in value.items() if isinstance(value, dict) else []:
                usable = isinstance(raw, int | float) and not isinstance(raw, bool)
                usable = usable and math.isfinite(raw)
                assert (block or {}).get(key) is None or usable

    def test_a_healthy_verdict_passes_through_intact(self, tmp_path):
        headline = {
            "found": True,
            "reason": None,
            "window": {"sp_lo": 1, "sp_hi": 5, "n_samples": 17552},
            "tps": {"per_user": 302.3, "system": 40960.9},
            "ttft": {"p50": 86.26, "p90": 156.1},
            "tpot": {"p50": 3.29, "p90": 3.44},
            "short_window": {"is_short": False},
            "drifting_up": ["tpot"],
        }
        path = tmp_path / "steady_state.json"
        path.write_text(json.dumps({"superpass_size": 4388, "steady_state": headline}))

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got["superpass_size"] == 4388
        assert got["window"]["n_samples"] == 17552
        assert got["tps"]["per_user"] == 302.3
        assert got["drifting_up"] == ["tpot"]
        assert got["short_window"]["is_short"] is False

    @pytest.mark.parametrize("blob", [{}, {"steady_state": None}, {"steady_state": []}])
    def test_a_blob_without_a_verdict_is_not_one(self, tmp_path, blob, caplog):
        """The agentic/NATL --json output has this shape: real JSON, no verdict."""
        with caplog.at_level(logging.WARNING):
            assert steady_state.verdict_headline(self._write(tmp_path, blob)) is None

        assert "no verdict" in caplog.text


class TestRunMeta:
    def test_detector_reads_back_the_dataset_size(self, tmp_path):
        steady_state.write_run_meta(tmp_path, dataset_size=_DATASET_SIZE)

        parsed = steady_state_diagnostics.read_run_config(
            None, str(tmp_path / "run_meta.json")
        )

        assert parsed["dataset_size"] == _DATASET_SIZE
