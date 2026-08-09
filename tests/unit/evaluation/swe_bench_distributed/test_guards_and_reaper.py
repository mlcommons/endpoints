# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource guards and the claim reaper."""

from __future__ import annotations

import os
import time

import pytest

from inference_endpoint.evaluation.swe_bench_distributed import guards as guards_mod
from inference_endpoint.evaluation.swe_bench_distributed.guards import (
    DEFAULT_KILL_BYTES,
    HealthTerm,
    HealthVerdict,
    MemoryGuard,
    ProcessSample,
    SelfKillRefused,
    combine_terms,
    kill_by_pid,
)
from inference_endpoint.evaluation.swe_bench_distributed.queue import (
    UnitOutcome,
    UnitResult,
    WorkQueue,
)
from inference_endpoint.evaluation.swe_bench_distributed.reaper import (
    Liveness,
    LivenessVerdict,
    reap,
)
from inference_endpoint.evaluation.swe_bench_distributed.units import plan_units

pytestmark = pytest.mark.unit

GIB = 1024**3


def executable_source(module) -> str:
    """Module source with comments and string literals removed."""
    import tokenize

    kept = []
    with open(module.__file__, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in {tokenize.COMMENT, tokenize.STRING}:
                continue
            kept.append(token.string)
    return " ".join(kept)


class FakeLiveness:
    def __init__(self, verdict: LivenessVerdict) -> None:
        self.verdict = verdict

    def probe(self, owner):
        return self.verdict


@pytest.fixture
def queue(tmp_path):
    plan = plan_units("run-a", [f"i-{i}" for i in range(20)], shard_size=10)
    return WorkQueue(tmp_path / "wq", plan)


class TestConjunction:
    def test_all_unhealthy_terms_fire(self):
        terms = [
            HealthTerm("a", HealthVerdict.UNHEALTHY, evidence=1),
            HealthTerm("b", HealthVerdict.UNHEALTHY, evidence=1),
        ]
        assert combine_terms(terms)[0] is HealthVerdict.UNHEALTHY

    def test_a_blind_term_makes_the_conjunction_indeterminate(self):
        # When an honest term permanently loses its data source, an AND-guard
        # collapses into its remaining, weaker clauses and starts firing on
        # healthy targets. That is how an idle watchdog killed a live bring-up.
        terms = [
            HealthTerm("loud", HealthVerdict.UNHEALTHY, evidence=1),
            HealthTerm("blind", HealthVerdict.UNHEALTHY, evidence=0),
        ]
        verdict, reason = combine_terms(terms)
        assert verdict is HealthVerdict.INDETERMINATE
        assert "blind" in reason

    def test_a_blind_term_never_yields_unhealthy(self):
        terms = [HealthTerm("blind", HealthVerdict.UNHEALTHY, evidence=0)]
        assert combine_terms(terms)[0] is not HealthVerdict.UNHEALTHY

    def test_one_healthy_term_spares_the_target(self):
        terms = [
            HealthTerm("a", HealthVerdict.UNHEALTHY, evidence=1),
            HealthTerm("b", HealthVerdict.HEALTHY, evidence=1),
        ]
        assert combine_terms(terms)[0] is HealthVerdict.HEALTHY

    def test_no_terms_is_indeterminate(self):
        assert combine_terms([])[0] is HealthVerdict.INDETERMINATE


class TestKillDiscipline:
    def test_there_is_no_pattern_kill_path_in_the_module(self):
        # A pattern such as "runtests.py" can appear in the guard's own command
        # line, and a long-lived daemon can carry a dead process's argv for days.
        # Executable code is inspected with comments and strings removed, so the
        # docstring explaining the rule cannot satisfy the test for it.
        code = executable_source(guards_mod)
        assert "pkill" not in code
        assert "pgrep" not in code

    def test_the_guard_never_shells_out(self):
        # There is no command line to match against in the first place.
        code = executable_source(guards_mod)
        assert "subprocess" not in code
        assert "os.system" not in code

    def test_killing_self_is_refused(self):
        with pytest.raises(SelfKillRefused, match="self"):
            kill_by_pid(os.getpid())

    def test_killing_an_ancestor_is_refused(self):
        with pytest.raises(SelfKillRefused, match="ancestor"):
            kill_by_pid(os.getppid())

    def test_a_nonsense_pid_is_refused(self):
        with pytest.raises(SelfKillRefused):
            kill_by_pid(0)


class TestMemoryGuard:
    def runaway(self, **overrides):
        payload = {
            "pid": 4242,
            "rss_bytes": 200 * GIB,
            "container_name": "sweb.eval.arm64.repo__proj-1",
            "ancestor_names": ("bash", "conmon"),
        }
        payload.update(overrides)
        return ProcessSample(**payload)

    def test_a_runaway_graded_test_is_unhealthy(self):
        action = MemoryGuard().evaluate(self.runaway())
        assert action.verdict is HealthVerdict.UNHEALTHY

    def test_a_runaway_outside_the_testbed_is_still_caught(self):
        # An earlier version required cwd inside /testbed. A runaway that had
        # grown to 667 GiB was skipped for 105 minutes because its cwd was /tmp,
        # so cwd is advisory detail and never a predicate.
        action = MemoryGuard().evaluate(self.runaway(cwd="/tmp"))
        assert action.verdict is HealthVerdict.UNHEALTHY
        assert {term.name for term in action.terms} == {"rss", "in_container"}

    def test_a_large_process_outside_a_container_is_spared(self):
        action = MemoryGuard().evaluate(self.runaway(ancestor_names=("bash", "sshd")))
        assert action.verdict is HealthVerdict.HEALTHY

    def test_a_normal_test_is_spared(self):
        action = MemoryGuard().evaluate(self.runaway(rss_bytes=3 * GIB))
        assert action.verdict is HealthVerdict.HEALTHY

    def test_unreadable_ancestry_is_indeterminate_not_a_kill(self):
        action = MemoryGuard().evaluate(self.runaway(ancestor_names=()))
        assert action.verdict is HealthVerdict.INDETERMINATE

    def test_the_default_threshold_leaves_wide_headroom(self):
        assert DEFAULT_KILL_BYTES >= 100 * GIB

    @pytest.mark.parametrize(
        ("container_name", "phase"),
        [
            ("sweb.eval.arm64.repo__proj-1", "eval"),
            ("minisweagent-abc123", "agent"),
            ("something-else", "unknown"),
            (None, "unknown"),
        ],
    )
    def test_phase_comes_from_the_container_name_and_fails_closed(
        self, container_name, phase
    ):
        # Only an eval kill turns an instance's error into a genuine failure, so
        # an unresolvable name must not be booked as one.
        guard = MemoryGuard()
        assert guard.phase_for(self.runaway(container_name=container_name)) == phase

    def test_the_marker_is_written_before_the_kill(self, tmp_path, monkeypatch):
        killed_dir = tmp_path / "killed"
        order: list[str] = []
        monkeypatch.setattr(
            guards_mod,
            "kill_by_pid",
            lambda pid, **kwargs: order.append("kill") or True,
        )
        guard = MemoryGuard(killed_dir=killed_dir)
        original = guard.record_kill

        def traced(*args, **kwargs):
            order.append("marker")
            return original(*args, **kwargs)

        monkeypatch.setattr(guard, "record_kill", traced)
        guard.act(self.runaway(), instance_id="repo__proj-1", apply=True)

        # A SIGKILLed test leaves an ambiguous log, so the record of having
        # killed it must survive even if the process dies first.
        assert order == ["marker", "kill"]
        assert list(killed_dir.glob("eval.repo__proj-1.*.json"))

    def test_dry_evaluation_does_not_kill(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            guards_mod, "kill_by_pid", lambda *a, **k: pytest.fail("killed")
        )
        action = MemoryGuard(killed_dir=tmp_path).act(
            self.runaway(), instance_id="x", apply=False
        )
        assert not action.killed


class TestReaper:
    def stale_claim(self, queue, unit_id="run-a.s00", age=7200.0):
        queue.claim(unit_id)
        heartbeat = queue.claims_dir / unit_id / "hb"
        past = time.time() - age
        os.utime(heartbeat, (past, past))

    def test_a_dead_owner_with_no_result_is_released(self, queue):
        self.stale_claim(queue)
        report = reap(queue, FakeLiveness(LivenessVerdict(Liveness.DEAD)), apply=True)
        assert report.released == ["run-a.s00"]
        assert "run-a.s00" in queue.available_unit_ids()

    def test_a_live_owner_is_never_released(self, queue):
        self.stale_claim(queue)
        report = reap(queue, FakeLiveness(LivenessVerdict(Liveness.ALIVE)), apply=True)
        # A false reap gives one unit two owners, duplicate results and a wrong
        # denominator, with no error anywhere.
        assert report.released == []

    def test_an_indeterminate_probe_releases_nothing(self, queue):
        self.stale_claim(queue)
        report = reap(
            queue,
            FakeLiveness(LivenessVerdict(Liveness.INDETERMINATE)),
            apply=True,
        )
        assert report.released == []
        assert "indeterminate" in report.kept["run-a.s00"]

    def test_a_fresh_heartbeat_is_never_released(self, queue):
        queue.claim("run-a.s00")
        report = reap(queue, FakeLiveness(LivenessVerdict(Liveness.DEAD)), apply=True)
        assert report.released == []

    def test_a_claim_with_a_result_is_never_released(self, queue):
        self.stale_claim(queue)
        unit = queue.plan.unit("run-a.s00")
        queue.results_dir.joinpath("run-a.s00.json").write_text(
            UnitResult(
                unit_id="run-a.s00",
                run_id="run-a",
                plan_digest=queue.plan.digest,
                outcome=UnitOutcome.SUCCEEDED,
                accounted_instance_ids=unit.instance_ids,
            ).to_dict()
            and '{"unit_id": "run-a.s00"}'
        )
        report = reap(queue, FakeLiveness(LivenessVerdict(Liveness.DEAD)), apply=True)
        assert report.released == []

    def test_a_dead_step_uses_the_shorter_threshold(self, queue):
        # A step that dies inside a live job takes its tasks with it at once, so
        # waiting an hour would block those units for the whole allocation.
        self.stale_claim(queue, age=1200.0)
        report = reap(
            queue,
            FakeLiveness(LivenessVerdict(Liveness.DEAD, scope="step")),
            stale_after_s=3600.0,
            step_stale_after_s=900.0,
            apply=True,
        )
        assert report.released == ["run-a.s00"]

    def test_dry_run_reports_without_releasing(self, queue):
        self.stale_claim(queue)
        report = reap(queue, FakeLiveness(LivenessVerdict(Liveness.DEAD)))
        assert report.released == ["run-a.s00"]
        assert queue.claimed_unit_ids() == {"run-a.s00"}

    def test_a_claim_with_unexpected_contents_is_left_alone(self, queue):
        self.stale_claim(queue)
        (queue.claims_dir / "run-a.s00" / "surprise").write_text("x")
        (queue.claims_dir / "run-a.s00" / "owner").unlink()
        report = reap(queue, FakeLiveness(LivenessVerdict(Liveness.DEAD)), apply=True)
        assert report.released == []
