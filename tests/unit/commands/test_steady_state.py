# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run's side of steady-state detection.

Pins the gate (which runs the aggregator is asked to collect for), the
artifacts the step owns in a reusable report dir, and the narrowing that turns
the detector's JSON into the headline the Report carries.
"""

import json
import logging
from pathlib import Path

import msgspec.structs
import pytest
from inference_endpoint.commands.benchmark import steady_state
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    LoadPatternType,
    TestType,
)
from inference_endpoint.metrics import steady_state_diagnostics
from inference_endpoint.metrics.report import LevelShift

pytestmark = pytest.mark.unit

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


def _agentic_config(model_name="gpt-oss-120b", *, enabled=True):
    """An agentic run: the detector has no profile for this load pattern."""
    return BenchmarkConfig(
        type=TestType.ONLINE,
        model_params={"name": model_name},
        endpoint_config={"endpoints": ["http://x"]},
        datasets=[{"path": "D", "agentic_inference": {}}],
        settings={
            "load_pattern": {"type": "agentic_inference", "target_concurrency": 8},
            "steady_state": {"enabled": enabled},
        },
    )


def _plan(report_dir, config=None, **overrides):
    kwargs = {
        "tokenizer_name": _TOKENIZER,
        "dataset_size": _DATASET_SIZE,
        "accuracy_only": False,
    }
    kwargs.update(overrides)
    return steady_state.collection_plan(config or _config(), report_dir, **kwargs)


# Every way a run can fail to earn collection, with the reason it must say and
# whether it still owes a hand re-run the super-pass size.
_SKIPS = [
    pytest.param(
        {"config": _config(**{"steady_state": {"enabled": False}})},
        "disabled by configuration",
        False,
        id="disabled",
    ),
    pytest.param(
        {"accuracy_only": True},
        "no performance phase",
        False,
        id="accuracy-only",
    ),
    pytest.param({"tokenizer_name": None}, "no tokenizer", False, id="no-tokenizer"),
    pytest.param(
        {"dataset_size": None}, "dataset size unknown", False, id="no-dataset-size"
    ),
    pytest.param(
        {"config": _agentic_config()},
        "the detector has no profile for",
        True,
        id="unprofiled-load-pattern",
    ),
]


class TestCollectionPlan:
    """What the aggregator is asked to do, decided before the run starts.

    The aggregator rolls super-passes up as the run happens, so this cannot be
    deferred to finalize: a run that was not asked to collect has no verdict to
    publish afterwards.
    """

    def test_an_opted_in_run_collects_one_super_pass_per_pass_over_the_dataset(
        self, tmp_path
    ):
        """The size and the destination are both pinned here so the aggregator
        never guesses either."""
        report_dir = _report_dir(tmp_path)

        assert _plan(report_dir) == steady_state.SteadyStatePlan(
            superpass_size=_DATASET_SIZE,
            verdict_path=report_dir / "steady_state.json",
            profile="concurrency",
        )

    def test_a_collected_run_publishes_its_own_super_pass_size(self, tmp_path):
        """run_meta.json is how a by-hand re-run of the standalone detector
        resolves the size with no arguments, so it must describe this run and
        not the one that used the directory before."""
        report_dir = _report_dir(tmp_path)

        _plan(report_dir)

        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": _DATASET_SIZE
        }

    @pytest.mark.parametrize(("kwargs", "reason", "keeps_run_meta"), _SKIPS)
    def test_a_run_that_does_not_earn_collection_says_why(
        self, tmp_path, caplog, kwargs, reason, keeps_run_meta
    ):
        report_dir = _report_dir(tmp_path)

        with caplog.at_level(logging.INFO):
            assert _plan(report_dir, **kwargs) is None

        assert reason in caplog.text

    @pytest.mark.parametrize(("kwargs", "reason", "keeps_run_meta"), _SKIPS)
    def test_only_an_unprofiled_load_pattern_leaves_metadata_behind(
        self, tmp_path, kwargs, reason, keeps_run_meta
    ):
        """An unprofiled run is told to re-run the detector by hand, and that
        reads run_meta.json. Every other skip leaves nothing to read it: a
        run_meta.json with no verdict beside it is a file that only misleads.
        """
        report_dir = _report_dir(tmp_path)
        (report_dir / "run_meta.json").unlink()

        _plan(report_dir, **kwargs)

        assert (report_dir / "run_meta.json").exists() is keeps_run_meta

    def test_an_unprofiled_load_pattern_replaces_the_previous_runs_metadata(
        self, tmp_path
    ):
        """A stale size would send a by-hand re-run at the wrong super-pass
        boundary and produce a verdict for a window that never existed."""
        report_dir = _report_dir(tmp_path)

        assert _plan(report_dir, config=_agentic_config()) is None

        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": _DATASET_SIZE
        }

    def test_a_run_ruled_out_before_the_load_pattern_leaves_no_metadata(self, tmp_path):
        """The unprofiled-pattern message is the only thing that points at a
        hand re-run, so a run that never reaches that check has no reason to
        publish a size."""
        report_dir = _report_dir(tmp_path)
        (report_dir / "run_meta.json").unlink()

        assert _plan(report_dir, config=_agentic_config(enabled=False)) is None

        assert not (report_dir / "run_meta.json").exists()

    @pytest.mark.parametrize("dataset_size", [0, -1, -6396])
    def test_a_non_positive_dataset_size_is_no_size_at_all(
        self, tmp_path, caplog, dataset_size
    ):
        """A zero-length super-pass would divide the run into an unbounded
        number of them; a negative one is not a count."""
        report_dir = _report_dir(tmp_path)
        (report_dir / "run_meta.json").unlink()

        with caplog.at_level(logging.INFO):
            assert _plan(report_dir, dataset_size=dataset_size) is None

        assert "dataset size unknown" in caplog.text
        assert not (report_dir / "run_meta.json").exists()

    def test_an_unwritable_report_dir_does_not_fail_the_run(
        self, tmp_path, monkeypatch, caplog
    ):
        """A full disk or a read-only mount costs the run its diagnostics, not
        its results."""
        report_dir = _report_dir(tmp_path)
        real_write_text = Path.write_text

        def failing_write_text(self, *args, **kwargs):
            if self.name == "run_meta.json":
                raise OSError("no space left on device")
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failing_write_text)

        with caplog.at_level(logging.WARNING):
            assert _plan(report_dir) == steady_state.SteadyStatePlan(
                superpass_size=_DATASET_SIZE,
                verdict_path=report_dir / "steady_state.json",
                profile="concurrency",
            )

        assert "run_meta.json" in caplog.text
        assert not (
            report_dir / "run_meta.json"
        ).exists(), "a failed write must not leave the previous run's size in place"


class TestCollectedVerdict:
    """Whether the aggregator actually produced a verdict for this run."""

    def test_a_written_verdict_is_offered_to_the_report(self, tmp_path):
        (tmp_path / "steady_state.json").write_text("{}")

        assert steady_state.collected_verdict(tmp_path) == (
            tmp_path / "steady_state.json"
        )

    def test_no_verdict_is_reported_as_none(self, tmp_path, caplog):
        """The aggregator collects best-effort, so an absent verdict is a normal
        outcome the report has to survive rather than an error."""
        with caplog.at_level(logging.INFO):
            assert steady_state.collected_verdict(tmp_path) is None

        assert "steady_state.json" in caplog.text

    def test_a_directory_in_the_verdicts_place_is_not_a_verdict(self, tmp_path):
        """Everything downstream reads this path as a file; a directory here
        would raise out of a finished run instead."""
        (tmp_path / "steady_state.json").mkdir()

        assert steady_state.collected_verdict(tmp_path) is None


class TestDiscardVerdict:
    """Withdrawing a verdict the run turned out not to deserve."""

    def test_the_verdict_and_its_rendering_both_go(self, tmp_path):
        """steady_state.txt is the same verdict in prose: leaving it behind
        publishes the claim the JSON was withdrawn for."""
        report_dir = _report_dir(tmp_path)

        steady_state.discard_verdict(report_dir)

        assert not (report_dir / "steady_state.json").exists()
        assert not (report_dir / "steady_state.txt").exists()

    def test_the_super_pass_size_survives_a_withdrawn_verdict(self, tmp_path):
        """A run that aborted or drained incompletely is exactly the one worth
        re-running the standalone detector against by hand, and that reads
        run_meta.json."""
        report_dir = _report_dir(tmp_path)

        steady_state.discard_verdict(report_dir)

        assert json.loads((report_dir / "run_meta.json").read_text()) == {
            "dataset_size": 11
        }

    def test_withdrawing_a_verdict_that_was_never_written_is_not_an_error(
        self, tmp_path
    ):
        """Every finalize path that did not adopt a verdict calls this, and
        most runs never collected one, so absence is the common case."""
        (tmp_path / "run_meta.json").write_text('{"dataset_size": 8}')

        steady_state.discard_verdict(tmp_path)

        assert [p.name for p in tmp_path.iterdir()] == ["run_meta.json"]

    def test_the_event_log_is_not_this_steps_to_remove(self, tmp_path):
        report_dir = _report_dir(tmp_path)

        steady_state.discard_verdict(report_dir)

        assert (report_dir / "events.jsonl").exists()


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


def _as_dict(block):
    """A headline sub-struct as a plain dict of its set fields, or None.

    Lets the drop/keep cases below state one expected mapping per case rather
    than one assertion per field.
    """
    if block is None:
        return None
    if isinstance(block, tuple):
        return list(block)
    return {
        f.name: v
        for f in msgspec.structs.fields(block)
        if (v := getattr(block, f.name)) is not None
    }


_HEADLINE_CONTEXT_KEYS = ("superpass_size", "n_super_passes", "n_post_warmup")


class TestVerdictHeadline:
    """The headline lifted out of steady_state.json and onto the Report.

    This is the only place the detector's JSON is read, so it is the only place
    the values can be checked; everything downstream formats them into the run's
    primary artifacts.
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
        assert got.found is True
        assert got.superpass_size == 4388
        assert got.n_post_warmup == 11

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
        # A typed headline cannot carry them: the fields do not exist.
        assert not {"trajectories", "cov", "drift", "per_super_pass"} & {
            f.name for f in msgspec.structs.fields(got)
        }

    def test_missing_sizing_context_is_null_not_zero(self, tmp_path):
        """Absent context arrives as null rather than a zero that reads like a
        measurement. The headline is a struct, so the fields are always present
        -- what matters is that nothing invents a value for them."""
        path = self._write(tmp_path, {"steady_state": {"found": False}})

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got.found is False
        assert all(getattr(got, k) is None for k in _HEADLINE_CONTEXT_KEYS)

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
        ("field", "value", "expected"),
        [
            # Each case states the whole expected output for the field, so a
            # boundary that drops a GOOD value fails too. Re-deriving the rule
            # inside the assertion would accept that silently.
            ("tps", {"per_user": "fast", "system": 1.0}, {"system": 1.0}),
            ("tps", {"per_user": True, "system": 1.0}, {"system": 1.0}),
            ("tps", {"per_user": float("nan"), "system": float("inf")}, None),
            ("ttft", {"p50": None, "p90": "slow"}, None),
            ("ttft", {"p50": 1.5, "p90": "slow"}, {"p50_ns": 1.5}),
            ("window", {"sp_lo": "a", "sp_hi": 5, "n_samples": None}, {"sp_hi": 5}),
            ("drifting_up", "tpot", []),
            ("drifting_up", [None, 1, "tpot"], ["tpot"]),
            ("short_window", "yes", None),
            ("short_window", {"is_short": "truthy"}, {"is_short": True}),
        ],
    )
    def test_unusable_values_are_dropped_at_the_boundary(
        self, tmp_path, field, value, expected
    ):
        """This is the only place the detector's JSON is checked. Everything
        downstream formats these values into the run's primary artifacts, so a
        value that cannot be rendered must not get past here -- and one that can
        must survive."""
        path = tmp_path / "steady_state.json"
        path.write_text(json.dumps({"steady_state": {"found": True, field: value}}))

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert _as_dict(getattr(got, field)) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ({"superpass_size": 4388}, 4388),
            ({"superpass_size": "4388"}, None),
            ({"superpass_size": None}, None),
            ({"superpass_size": {"n": 1}}, None),
            ({"superpass_size": 4388.0}, 4388),
        ],
    )
    def test_the_sizing_numbers_are_checked_like_everything_else(
        self, tmp_path, raw, expected
    ):
        """These are copied out of the file and published in
        result_summary.json, so they get the same treatment as the rest."""
        path = self._write(tmp_path, {**raw, "steady_state": {"found": True}})

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got.superpass_size == expected

    @pytest.mark.parametrize(
        ("change_point", "expected"),
        [(7, 7), ("seven", None), ({"sp": 7}, None), (None, None), (7.0, 7)],
    )
    def test_the_change_point_is_checked_before_it_reaches_a_sentence(
        self, tmp_path, change_point, expected
    ):
        """This one is interpolated into a line of report.txt, so an unchecked
        value does not just sit in JSON -- it gets printed."""
        path = self._write(
            tmp_path,
            {
                "steady_state": {
                    "found": True,
                    "anomaly": {"detected": True, "change_point_sp": change_point},
                }
            },
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert (got.anomaly.change_point_sp if got.anomaly else None) == expected

    def test_the_confidence_intervals_and_tail_percentiles_survive(self, tmp_path):
        """A throughput figure without its interval is harder to judge, and p99
        is what a tail-latency reader looks for. Both are small scalars -- the
        reason for narrowing was the histograms, not these."""
        path = self._write(
            tmp_path,
            {
                "steady_state": {
                    "found": True,
                    "tps": {
                        "per_user": 27.4,
                        "system": 56030.1,
                        "per_user_ci": [27.2, 27.6],
                        "system_ci": [55769.0, 56291.1],
                    },
                    "ttft": {
                        "p50": 1.0,
                        "p90": 2.0,
                        "p99": 3.0,
                        "mean": 1.5,
                        "count": 47484,
                        "histogram": [{"lo": 0, "hi": 1}],
                    },
                }
            },
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got.tps is not None and got.ttft is not None
        assert got.tps.per_user_ci == (27.2, 27.6)
        assert got.tps.system_ci == (55769.0, 56291.1)
        assert got.ttft.p99_ns == 3.0
        assert got.ttft.mean_ns == 1.5
        assert got.ttft.count == 47484
        assert not hasattr(got.ttft, "histogram"), "the bulky part stays behind"

    def test_the_evidence_for_the_window_survives(self, tmp_path):
        """is_short alone says whether the window cleared the gate, not by how
        much, and skipped_short says whether earlier plateaus were rejected.
        Those are the grounds for trusting the numbers, and they cost bytes."""
        path = self._write(
            tmp_path,
            {
                "steady_state": {
                    "found": True,
                    "window": {
                        "sp_lo": 0,
                        "sp_hi": 11,
                        "n_samples": 47484,
                        "n_super_passes": 11,
                        "plateau_index": 0,
                        "n_plateaus": 1,
                        "skipped_short": 2,
                    },
                    "short_window": {
                        "is_short": False,
                        "window_duration_s": 3241.9,
                        "min_duration_s": 1742.5,
                        "dominant": "relaxation",
                        "kstar": 99,
                    },
                    "global_trend": {"tpot_p50": "steady", "ttft_p90": "up"},
                }
            },
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got.window is not None and got.short_window is not None
        assert got.window.skipped_short == 2
        assert got.window.n_plateaus == 1
        assert got.short_window.window_duration_s == 3241.9
        assert got.short_window.min_duration_s == 1742.5
        assert got.short_window.dominant == "relaxation"
        assert got.global_trend == {"tpot_p50": "steady", "ttft_p90": "up"}

    def test_the_bulky_evidence_stays_in_the_verdict_file(self, tmp_path):
        """Histograms were two thirds of the block on a real run, and plateaus
        is a full segmentation table. result_summary.json is not their home."""
        path = self._write(
            tmp_path,
            {
                "steady_state": {
                    "found": True,
                    "ttft": {"p50": 1.0, "p90": 2.0, "histogram": [{"lo": 0, "hi": 1}]},
                    "short_window": {"is_short": False, "kstar": 99},
                    "anomaly": {"detected": True, "plateaus": [[0, 5]], "pettitt": {}},
                }
            },
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        # The headline is typed, so the bulky fields have nowhere to land.
        for block in (got.ttft, got.short_window, got.anomaly):
            assert block is not None
            names = {f.name for f in msgspec.structs.fields(block)}
            assert not names & {"histogram", "kstar", "plateaus", "pettitt"}

    def test_the_level_shift_anomaly_survives_to_the_report(self, tmp_path):
        """The detector treats a level shift after the plateau as a first-class
        warning. Dropping it makes a degrading run print a clean headline."""
        path = self._write(
            tmp_path,
            {
                "steady_state": {
                    "found": True,
                    "anomaly": {
                        "detected": True,
                        "change_point_sp": 7,
                        "delta_pct": 12.5,
                        "plateaus": [[0, 5], [5, 9]],
                    },
                }
            },
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got.anomaly == LevelShift(
            detected=True, change_point_sp=7, delta_pct=12.5
        )

    @pytest.mark.parametrize(
        "anomaly", ["yes", None, {"detected": False}, {"detected": "maybe"}]
    )
    def test_an_unusable_anomaly_is_dropped(self, tmp_path, anomaly):
        path = self._write(
            tmp_path, {"steady_state": {"found": True, "anomaly": anomaly}}
        )

        got = steady_state.verdict_headline(path)

        assert got is not None
        assert got.anomaly is None

    def test_the_profile_caveat_rides_along(self, tmp_path):
        """poisson is supported but carries a reliability note. Scraping it from
        the child's stderr only reaches steady_state.txt; the report is the
        artifact a submitter actually reads."""
        path = self._write(tmp_path, {"steady_state": {"found": True}})

        got = steady_state.verdict_headline(path, load_pattern=LoadPatternType.POISSON)

        assert got is not None
        assert got.profile == "poisson"
        assert got.profile_caveat
        assert "\n" not in got.profile_caveat

    def test_a_profile_without_a_note_carries_none(self, tmp_path):
        path = self._write(tmp_path, {"steady_state": {"found": True}})

        got = steady_state.verdict_headline(
            path, load_pattern=LoadPatternType.CONCURRENCY
        )

        assert got is not None
        assert got.profile == "concurrency"
        assert got.profile_caveat is None

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
        assert got.superpass_size == 4388
        assert got.window is not None and got.window.n_samples == 17552
        assert got.tps is not None and got.tps.per_user == 302.3
        assert got.drifting_up == ("tpot",)
        assert got.short_window is not None
        assert got.short_window.is_short is False

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


@pytest.mark.unit
class TestStaleArtifactsBlockCollection:
    """A verdict the run cannot overwrite must not become the run's verdict.

    ``collected_verdict`` is an existence test, so a previous run's
    ``steady_state.json`` that survives the clear would be lifted into this
    run's ``result_summary.json`` and ``report.txt`` as if this run had
    produced it. That is a wrong number in a submission artifact, not a missing
    one, so the run declines to collect at all rather than risk publishing it.
    """

    def test_a_verdict_that_cannot_be_cleared_stops_collection(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "steady_state.json").write_text('{"found": true}')

        def refuse(self, missing_ok=False):
            raise PermissionError("read-only report directory")

        monkeypatch.setattr(Path, "unlink", refuse)

        plan = steady_state.collection_plan(
            _config(),
            tmp_path,
            tokenizer_name="tok",
            dataset_size=100,
            accuracy_only=False,
        )

        assert plan is None

    def test_a_clean_directory_collects(self, tmp_path):
        plan = steady_state.collection_plan(
            _config(),
            tmp_path,
            tokenizer_name="tok",
            dataset_size=100,
            accuracy_only=False,
        )

        assert plan == steady_state.SteadyStatePlan(
            superpass_size=100,
            verdict_path=tmp_path / "steady_state.json",
            profile="concurrency",
        )


@pytest.mark.unit
class TestPlanCarriesTheProfile:
    """The profile decides the CoV bounds and warmup driver the verdict is
    judged on. The parent resolves it because it knows the load pattern; the
    standalone detector resolves the same one from the run's config, which is
    what keeps a hand re-run comparable to the published verdict."""

    def test_the_profile_follows_the_load_pattern(self, tmp_path):
        report_dir = _report_dir(tmp_path)

        assert _plan(report_dir).profile == "concurrency"
