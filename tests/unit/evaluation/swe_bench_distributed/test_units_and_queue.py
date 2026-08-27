# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit plan and work-queue semantics."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from inference_endpoint.evaluation.swe_bench_distributed.queue import (
    ClaimError,
    UnitOutcome,
    UnitResult,
    WorkQueue,
)
from inference_endpoint.evaluation.swe_bench_distributed.units import (
    PlanError,
    plan_units,
    read_plan,
)

pytestmark = pytest.mark.unit


def make_ids(n: int) -> list[str]:
    return [f"repo__proj-{i:03d}" for i in range(n)]


@pytest.fixture
def queue(tmp_path):
    plan = plan_units("run-a", make_ids(25), shard_size=10)
    return WorkQueue(tmp_path / "wq", plan)


def result_for(queue: WorkQueue, unit_id: str, **overrides) -> UnitResult:
    unit = queue.plan.unit(unit_id)
    payload: dict[str, Any] = {
        "unit_id": unit_id,
        "run_id": unit.run_id,
        "plan_digest": queue.plan.digest,
        "outcome": UnitOutcome.SUCCEEDED,
        "accounted_instance_ids": unit.instance_ids,
        "resolved_instance_ids": unit.instance_ids[:1],
    }
    payload.update(overrides)
    return UnitResult(**payload)


class TestPlan:
    def test_shards_in_order_with_a_short_tail(self):
        plan = plan_units("run-a", make_ids(25), shard_size=10)
        assert [len(unit.instance_ids) for unit in plan.units] == [10, 10, 5]
        assert plan.unit_ids == ("run-a.s00", "run-a.s01", "run-a.s02")
        # The short tail is never padded: a padded shard would claim ids it
        # never ran and the merge gate compares ids, not counts.
        assert plan.instance_ids == tuple(make_ids(25))

    def test_digest_depends_on_order(self):
        ids = make_ids(20)
        assert (
            plan_units("r", ids).digest != plan_units("r", list(reversed(ids))).digest
        )

    def test_digest_depends_on_run_id(self):
        ids = make_ids(20)
        assert plan_units("r1", ids).digest != plan_units("r2", ids).digest

    def test_duplicate_instance_ids_are_refused(self):
        with pytest.raises(PlanError, match="unique"):
            plan_units("r", ["a", "b", "a"])

    def test_empty_plan_is_refused(self):
        with pytest.raises(PlanError):
            plan_units("r", [])

    def test_plan_round_trips_and_self_verifies(self, tmp_path):
        plan = plan_units("run-a", make_ids(12), shard_size=5)
        path = plan.write(tmp_path)
        assert read_plan(path).digest == plan.digest

    def test_rewriting_a_different_plan_is_refused(self, tmp_path):
        plan_units("run-a", make_ids(10)).write(tmp_path)
        with pytest.raises(PlanError, match="refusing to overwrite"):
            plan_units("run-a", make_ids(11)).write(tmp_path)

    def test_tampered_plan_file_is_detected(self, tmp_path):
        plan = plan_units("run-a", make_ids(10))
        path = plan.write(tmp_path)
        raw = path.read_text().replace(plan.digest, "0" * 64)
        path.write_text(raw)
        with pytest.raises(PlanError, match="inconsistent"):
            read_plan(path)


class TestClaims:
    def test_a_second_claim_loses(self, queue):
        assert queue.claim("run-a.s00") is not None
        assert queue.claim("run-a.s00") is None

    def test_exactly_one_thread_wins_a_contested_claim(self, queue):
        winners: list[object] = []
        barrier = threading.Barrier(8)

        def contend():
            barrier.wait()
            if queue.claim("run-a.s01") is not None:
                winners.append(object())

        threads = [threading.Thread(target=contend) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(winners) == 1

    def test_claiming_a_unit_outside_the_plan_is_an_error(self, queue):
        with pytest.raises(ClaimError):
            queue.claim("other-run.s00")

    def test_release_removes_the_whole_directory(self, queue):
        queue.claim("run-a.s00")
        assert queue.release("run-a.s00")
        # Removing only `owner` would leave an ownerless directory that still
        # hides the unit -- a relabelled problem, not a fix.
        assert not (queue.claims_dir / "run-a.s00").exists()
        assert "run-a.s00" in queue.available_unit_ids()

    def test_owner_record_carries_identity(self, queue):
        record = queue.claim("run-a.s00")
        stored = queue.owner("run-a.s00")
        assert stored is not None
        assert stored.pid == record.pid
        assert stored.boot_id == record.boot_id
        assert stored.plan_digest == queue.plan.digest


class TestAvailability:
    def test_claims_and_results_both_hide_a_unit(self, queue):
        queue.claim("run-a.s00")
        queue.publish(result_for(queue, "run-a.s01"))
        assert queue.available_unit_ids() == ["run-a.s02"]

    def test_publish_releases_the_claim(self, queue):
        queue.claim("run-a.s00")
        queue.publish(result_for(queue, "run-a.s00"))
        # An abandoned or published unit that keeps its claim makes claims/ and
        # results/ disagree for the rest of the run.
        assert queue.claimed_unit_ids() == set()

    def test_abandon_publishes_and_releases(self, queue):
        queue.claim("run-a.s00")
        queue.abandon(result_for(queue, "run-a.s00", outcome=UnitOutcome.FAILED))
        stored = queue.result("run-a.s00")
        assert stored is not None and stored.abandoned
        assert queue.claimed_unit_ids() == set()

    def test_result_from_another_plan_is_refused(self, queue):
        bad = result_for(queue, "run-a.s00", plan_digest="0" * 64)
        with pytest.raises(ClaimError, match="plan digest"):
            queue.publish(bad)


class TestRequeue:
    def test_deleting_only_the_result_does_not_requeue(self, queue):
        queue.claim("run-a.s00")
        queue.publish(result_for(queue, "run-a.s00"))
        # Re-claim so a tombstone exists, mimicking an interrupted retry.
        queue.claim("run-a.s00")
        (queue.results_dir / "run-a.s00.json").unlink()
        assert "run-a.s00" not in queue.available_unit_ids()

    def test_deleting_only_the_claim_does_not_requeue(self, queue):
        queue.claim("run-a.s00")
        queue.publish(result_for(queue, "run-a.s00"))
        queue.release("run-a.s00")
        assert "run-a.s00" not in queue.available_unit_ids()

    def test_requeue_removes_result_claim_and_attempts(self, queue):
        queue.claim("run-a.s00")
        queue.record_attempt(result_for(queue, "run-a.s00", outcome=UnitOutcome.FAILED))
        queue.record_attempt(result_for(queue, "run-a.s00", outcome=UnitOutcome.INFRA))
        queue.publish(result_for(queue, "run-a.s00"))
        queue.claim("run-a.s00")

        removed = queue.requeue("run-a.s00")

        assert len(removed["results"]) == 1
        assert len(removed["claims"]) == 1
        assert len(removed["attempts"]) == 2
        assert "run-a.s00" in queue.available_unit_ids()
        assert queue.attempts("run-a.s00") == 0

    def test_requeue_outside_the_plan_is_an_error(self, queue):
        with pytest.raises(ClaimError):
            queue.requeue("other-run.s00")


class TestAttemptLedger:
    def test_environment_faults_do_not_consume_the_budget(self, queue):
        for _ in range(5):
            queue.record_attempt(
                result_for(queue, "run-a.s00", outcome=UnitOutcome.ENV_FAULT)
            )
        # A broken host is a property of the host, not of the unit. Charging it
        # to the unit abandons good units for landing in the wrong place.
        assert queue.attempts("run-a.s00") == 0

    def test_counted_failures_increment(self, queue):
        assert (
            queue.record_attempt(
                result_for(queue, "run-a.s00", outcome=UnitOutcome.FAILED)
            )
            == 1
        )
        assert (
            queue.record_attempt(
                result_for(queue, "run-a.s00", outcome=UnitOutcome.INFRA)
            )
            == 2
        )

    def test_evidence_is_snapshotted_before_a_retry_overwrites_it(
        self, queue, tmp_path
    ):
        source = tmp_path / "unit-run"
        source.mkdir()
        (source / "status.json").write_text('{"attempt": 1}')
        (source / "swe_bench_agent.log").write_text("first attempt log")

        target = queue.snapshot_evidence("run-a.s00", source, attempt=1)

        # The retry reuses the run directory, so a unit that fails then succeeds
        # would otherwise leave only the success's artifacts behind.
        (source / "status.json").write_text('{"attempt": 2}')
        assert (target / "status.json").read_text() == '{"attempt": 1}'
        assert (target / "swe_bench_agent.log.tail").read_text() == "first attempt log"


class TestReopen:
    def test_reopen_reads_the_plan_from_disk(self, queue):
        queue.publish(result_for(queue, "run-a.s00"))
        reopened = WorkQueue.open(queue.root)
        assert reopened.plan.digest == queue.plan.digest
        assert reopened.completed_unit_ids() == {"run-a.s00"}
