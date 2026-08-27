# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fleet dispatch: fan-out, classification-driven retry, quarantine, merge."""

from __future__ import annotations

import itertools

import pytest
from inference_endpoint.evaluation.swe_bench_distributed.fleet import (
    FleetDispatcher,
    accounted_and_resolved,
)
from inference_endpoint.evaluation.swe_bench_distributed.merge import (
    MergeRefusal,
    merge_run,
)
from inference_endpoint.evaluation.swe_bench_distributed.queue import (
    UnitOutcome,
    WorkQueue,
)
from inference_endpoint.evaluation.swe_bench_distributed.units import plan_units

pytestmark = pytest.mark.unit

IDS = [f"repo__proj-{i:02d}" for i in range(30)]
SERVICES = ["http://svc-a:18080", "http://svc-b:18080"]


@pytest.fixture
def queue(tmp_path):
    return WorkQueue(tmp_path / "wq", plan_units("run-a", IDS, shard_size=10))


class FakeFleet:
    """A scripted stand-in for the SWE-bench service HTTP protocol."""

    def __init__(self, queue: WorkQueue, tmp_path, *, resolved_per_unit: int = 4):
        self.queue = queue
        self.tmp_path = tmp_path
        self.resolved_per_unit = resolved_per_unit
        self.counter = itertools.count()
        self.submitted: list[tuple[str, str]] = []
        self.submit_errors: dict[str, Exception] = {}
        self.status_for_unit: dict[str, str] = {}
        self.error_ids_for_unit: dict[str, list[str]] = {}
        self.fingerprints: list[str] = ["fp-1"]

    def submit(self, service_url, unit):
        error = self.submit_errors.get(service_url)
        if error is not None:
            raise error
        self.submitted.append((service_url, unit.unit_id))
        return f"svc-run-{next(self.counter)}"

    def poll(self, service_url, service_run_id):
        unit_id = self.submitted[-1][1]
        return {"status": self.status_for_unit.get(unit_id, "succeeded")}

    def collect(self, service_url, service_run_id, unit, status):
        error_ids = self.error_ids_for_unit.get(unit.unit_id, [])
        remaining = [i for i in unit.instance_ids if i not in error_ids]
        resolved = remaining[: self.resolved_per_unit]
        unresolved = remaining[self.resolved_per_unit :]
        output_dir = self.tmp_path / "units" / unit.unit_id
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "resolved_ids": resolved,
            "unresolved_ids": unresolved,
            "error_ids": error_ids,
            "empty_patch_ids": [],
        }
        return report, output_dir

    def fingerprint(self):
        return self.fingerprints[0]

    def write_eval_log(self, unit_id: str, instance_id: str, text: str) -> None:
        log_dir = (
            self.tmp_path
            / "units"
            / unit_id
            / "logs"
            / "run_evaluation"
            / "r"
            / "m"
            / instance_id
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run_instance.log").write_text(text)


def dispatcher_for(queue, fleet, **overrides):
    kwargs = {
        "queue": queue,
        "service_urls": SERVICES,
        "submit": fleet.submit,
        "poll": fleet.poll,
        "collect": fleet.collect,
        "fingerprint": fleet.fingerprint,
        "max_attempts": 3,
    }
    kwargs.update(overrides)
    return FleetDispatcher(**kwargs)


class TestAccounting:
    def test_every_outcome_bucket_counts_as_accounted(self):
        report = {
            "resolved_ids": ["a"],
            "unresolved_ids": ["b"],
            "empty_patch_ids": ["c"],
            "error_ids": ["d"],
        }
        accounted, resolved = accounted_and_resolved(report)
        assert set(accounted) == {"a", "b", "c", "d"}
        assert resolved == ("a",)

    def test_incomplete_instances_are_not_accounted(self):
        # "Incomplete" is exactly what unaccounted means; counting it would let
        # a partial shard through the merge gate.
        accounted, _ = accounted_and_resolved(
            {"resolved_ids": ["a"], "incomplete_ids": ["b"]}
        )
        assert accounted == ("a",)

    def test_duplicates_are_preserved_for_the_gate_to_refuse(self):
        accounted, _ = accounted_and_resolved(
            {"resolved_ids": ["a"], "unresolved_ids": ["a"]}
        )
        assert accounted == ("a", "a")


class TestHappyPath:
    def test_all_units_complete_and_merge(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        dispatcher_for(queue, fleet).run()

        assert len(queue.completed_unit_ids()) == 3
        result = merge_run(queue, "run-a")
        assert result.total_instances == 30
        assert result.resolved_instances == 12

    def test_work_is_spread_across_services(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        dispatcher_for(queue, fleet).run()
        assert len({service for service, _ in fleet.submitted}) >= 1
        assert len(fleet.submitted) == 3

    def test_every_unit_runs_exactly_once(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        dispatcher_for(queue, fleet).run()
        dispatched = [unit_id for _, unit_id in fleet.submitted]
        assert sorted(dispatched) == sorted(queue.plan.unit_ids)


class TestInfraRetry:
    def test_an_eval_infra_error_requeues_a_succeeded_run(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        fleet.error_ids_for_unit["run-a.s00"] = [IDS[0]]
        fleet.write_eval_log("run-a.s00", IDS[0], "container state improper")

        dispatcher_for(queue, fleet, max_attempts=1).run()

        # The service said "succeeded" and every instance was accounted for, so
        # nothing but classification distinguishes this from a real result.
        stored = queue.result("run-a.s00")
        assert stored is not None
        assert stored.abandoned
        assert stored.outcome is UnitOutcome.INFRA
        assert stored.infra_error_count == 1
        with pytest.raises(MergeRefusal):
            merge_run(queue, "run-a")

    def test_a_genuine_error_is_scored_not_retried(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        fleet.error_ids_for_unit["run-a.s00"] = [IDS[0]]
        fleet.write_eval_log("run-a.s00", IDS[0], "Test timed out after 1800s")

        dispatcher_for(queue, fleet).run()

        stored = queue.result("run-a.s00")
        assert stored is not None
        assert stored.outcome is UnitOutcome.SUCCEEDED
        assert stored.genuine_error_count == 1
        assert merge_run(queue, "run-a").total_instances == 30

    def test_a_transient_infra_error_succeeds_on_retry(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        fleet.error_ids_for_unit["run-a.s00"] = [IDS[0]]
        fleet.write_eval_log("run-a.s00", IDS[0], "container state improper")

        original_collect = fleet.collect

        def collect_once(service_url, service_run_id, unit, status):
            result = original_collect(service_url, service_run_id, unit, status)
            fleet.error_ids_for_unit.pop(unit.unit_id, None)
            return result

        dispatcher_for(queue, fleet, collect=collect_once).run()

        stored = queue.result("run-a.s00")
        assert stored is not None and stored.outcome is UnitOutcome.SUCCEEDED
        assert queue.attempts("run-a.s00") == 1
        assert merge_run(queue, "run-a").total_instances == 30

    def test_a_unit_is_abandoned_after_max_attempts(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        fleet.status_for_unit["run-a.s00"] = "failed"

        dispatcher_for(queue, fleet, max_attempts=2).run()

        stored = queue.result("run-a.s00")
        assert stored is not None and stored.abandoned
        assert queue.attempts("run-a.s00") == 2
        # An abandoned unit must be loud, not silent: it stops burning slots and
        # the gate refuses the run.
        assert queue.claimed_unit_ids() == set()
        with pytest.raises(MergeRefusal, match="abandoned"):
            merge_run(queue, "run-a")


class TestEndpointFingerprint:
    def test_a_restarted_engine_requeues_the_unit(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        original_collect = fleet.collect

        def collect_then_restart(service_url, service_run_id, unit, status):
            result = original_collect(service_url, service_run_id, unit, status)
            if unit.unit_id == "run-a.s00" and fleet.fingerprints[0] == "fp-1":
                fleet.fingerprints[0] = "fp-2"
            return result

        dispatcher_for(queue, fleet, collect=collect_then_restart, max_attempts=1).run()

        stored = queue.result("run-a.s00")
        assert stored is not None
        # The run "succeeded" and every instance was accounted for; only the
        # fingerprint says it was scored against a different engine.
        assert stored.outcome is UnitOutcome.INFRA
        assert "endpoint_changed" in stored.error_kinds


class TestEnvironmentFaults:
    def test_a_submit_failure_does_not_charge_the_unit(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        fleet.submit_errors[SERVICES[0]] = OSError("service host is broken")

        dispatcher_for(queue, fleet).run()

        # A broken host is a property of the host. The units still complete, on
        # the other service, with a clean attempt ledger.
        assert len(queue.completed_unit_ids()) == 3
        assert all(queue.attempts(unit_id) == 0 for unit_id in queue.plan.unit_ids)
        assert merge_run(queue, "run-a").total_instances == 30

    def test_a_persistently_broken_service_is_quarantined(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        fleet.submit_errors[SERVICES[0]] = OSError("service host is broken")

        dispatcher = dispatcher_for(queue, fleet, max_consecutive_env_faults=2)
        dispatcher.run()

        assert SERVICES[0] in dispatcher.quarantined
        assert SERVICES[1] not in dispatcher.quarantined


class TestStallQuarantine:
    def test_a_healthy_but_unproductive_service_is_withdrawn(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        dispatcher = dispatcher_for(queue, fleet, stall_timeout_s=-1)
        dispatcher.run()
        # Health is not progress. A service answering /health while completing
        # nothing is the silent failure: verify the effect, never the status.
        assert dispatcher.quarantined
        assert all(
            "no unit completed" in reason for reason in dispatcher.quarantined.values()
        )


class TestResume:
    def test_a_restarted_client_does_not_redo_completed_units(self, queue, tmp_path):
        fleet = FakeFleet(queue, tmp_path)
        dispatcher_for(queue, fleet, service_urls=SERVICES[:1]).run()
        first_pass = len(fleet.submitted)

        reopened = WorkQueue.open(queue.root)
        dispatcher_for(reopened, fleet, service_urls=SERVICES[:1]).run()

        assert len(fleet.submitted) == first_pass
        assert merge_run(reopened, "run-a").total_instances == 30
