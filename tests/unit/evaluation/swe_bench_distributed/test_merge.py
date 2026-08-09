# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The merge gate: all-or-nothing, id-based, scoped to one run."""

from __future__ import annotations

import inspect

import pytest

from inference_endpoint.evaluation.swe_bench_distributed.merge import (
    MergeRefusal,
    merge_run,
    verify_inventory,
)
from inference_endpoint.evaluation.swe_bench_distributed.queue import (
    UnitOutcome,
    UnitResult,
    WorkQueue,
)
from inference_endpoint.evaluation.swe_bench_distributed.units import plan_units

pytestmark = pytest.mark.unit

IDS = [f"repo__proj-{i:02d}" for i in range(20)]


@pytest.fixture
def queue(tmp_path):
    return WorkQueue(tmp_path / "wq", plan_units("run-a", IDS, shard_size=10))


def publish(queue: WorkQueue, unit_id: str, **overrides) -> None:
    unit = queue.plan.unit(unit_id)
    payload = {
        "unit_id": unit_id,
        "run_id": unit.run_id,
        "plan_digest": queue.plan.digest,
        "outcome": UnitOutcome.SUCCEEDED,
        "accounted_instance_ids": unit.instance_ids,
        "resolved_instance_ids": unit.instance_ids[:3],
    }
    payload.update(overrides)
    queue.publish(UnitResult(**payload))


def publish_all(queue: WorkQueue) -> None:
    for unit_id in queue.plan.unit_ids:
        publish(queue, unit_id)


class TestHappyPath:
    def test_full_accounting_scores(self, queue):
        publish_all(queue)
        result = merge_run(queue, "run-a")
        assert result.total_instances == 20
        assert result.resolved_instances == 6
        assert result.resolved_rate == pytest.approx(0.3)
        assert result.unit_count == 2


class TestRefusals:
    def test_a_missing_unit_refuses(self, queue):
        publish(queue, "run-a.s00")
        # 10 results must never be divided by 20.
        with pytest.raises(MergeRefusal, match="have no result"):
            merge_run(queue, "run-a")

    def test_an_abandoned_unit_refuses(self, queue):
        publish(queue, "run-a.s00")
        publish(queue, "run-a.s01", abandoned=True, attempt=3)
        with pytest.raises(MergeRefusal, match="abandoned"):
            merge_run(queue, "run-a")

    def test_a_non_success_outcome_refuses(self, queue):
        publish(queue, "run-a.s00")
        publish(queue, "run-a.s01", outcome=UnitOutcome.FAILED)
        with pytest.raises(MergeRefusal, match="outcome failed"):
            merge_run(queue, "run-a")

    def test_infrastructure_damage_refuses(self, queue):
        publish(queue, "run-a.s00")
        publish(queue, "run-a.s01", infra_error_count=2)
        with pytest.raises(MergeRefusal, match="lost to infrastructure"):
            merge_run(queue, "run-a")

    def test_a_missing_id_refuses_even_though_the_count_is_wrong_by_one(self, queue):
        publish(queue, "run-a.s00")
        unit = queue.plan.unit("run-a.s01")
        publish(queue, "run-a.s01", accounted_instance_ids=unit.instance_ids[:-1])
        with pytest.raises(MergeRefusal, match="missing"):
            merge_run(queue, "run-a")

    def test_a_swapped_id_refuses_although_the_count_matches(self, queue):
        # The whole point of comparing ids rather than counts: this shard has
        # exactly ten entries and the wrong content.
        publish(queue, "run-a.s00")
        unit = queue.plan.unit("run-a.s01")
        swapped = unit.instance_ids[:-1] + ("some__other-99",)
        publish(queue, "run-a.s01", accounted_instance_ids=swapped)
        with pytest.raises(MergeRefusal, match="unplanned"):
            merge_run(queue, "run-a")

    def test_a_duplicated_id_within_one_unit_refuses(self, queue):
        publish(queue, "run-a.s00")
        unit = queue.plan.unit("run-a.s01")
        duped = unit.instance_ids[:-1] + (unit.instance_ids[0],)
        publish(queue, "run-a.s01", accounted_instance_ids=duped)
        with pytest.raises(MergeRefusal, match="duplicate"):
            merge_run(queue, "run-a")

    def test_resolved_ids_outside_the_shard_refuse(self, queue):
        publish(queue, "run-a.s00")
        unit = queue.plan.unit("run-a.s01")
        publish(
            queue,
            "run-a.s01",
            resolved_instance_ids=(*unit.instance_ids[:2], IDS[0]),
        )
        with pytest.raises(MergeRefusal, match="outside its shard"):
            merge_run(queue, "run-a")

    def test_a_foreign_plan_digest_refuses(self, queue):
        publish(queue, "run-a.s00")
        publish(queue, "run-a.s01")
        path = queue.results_dir / "run-a.s01.json"
        path.write_text(path.read_text().replace(queue.plan.digest, "f" * 64))
        with pytest.raises(MergeRefusal, match="belongs to plan"):
            merge_run(queue, "run-a")

    def test_a_result_outside_the_plan_refuses(self, queue):
        publish_all(queue)
        (queue.results_dir / "other-run.s00.json").write_text("{}")
        with pytest.raises(MergeRefusal, match="outside the plan"):
            merge_run(queue, "run-a")

    def test_an_unreadable_result_refuses(self, queue):
        publish_all(queue)
        (queue.results_dir / "run-a.s00.json").write_text("not json")
        with pytest.raises(MergeRefusal, match="unreadable"):
            merge_run(queue, "run-a")

    def test_every_reason_is_reported_at_once(self, queue):
        publish(queue, "run-a.s00", infra_error_count=1)
        with pytest.raises(MergeRefusal) as excinfo:
            merge_run(queue, "run-a")
        assert len(excinfo.value.reasons) >= 2


class TestScoping:
    def test_a_merge_is_always_scoped_to_one_run(self, queue):
        publish_all(queue)
        with pytest.raises(MergeRefusal, match="scoped to exactly one run"):
            merge_run(queue, "some-other-run")

    def test_there_is_no_merge_all(self):
        # "Merge everything that looks finished" once combined hundreds of
        # banked results from unrelated configurations into one number.
        signature = inspect.signature(merge_run)
        assert "run_id" in signature.parameters
        assert signature.parameters["run_id"].default is inspect.Parameter.empty
        assert not hasattr(
            __import__(
                "inference_endpoint.evaluation.swe_bench_distributed.merge",
                fromlist=["merge"],
            ),
            "merge_all",
        )


class TestInventory:
    def test_a_complete_run_is_consistent(self, queue):
        publish_all(queue)
        assert verify_inventory(queue).consistent

    def test_an_ownerless_claim_is_an_inventory_error(self, queue):
        publish_all(queue)
        claim_dir = queue.claims_dir / "run-a.s00"
        claim_dir.mkdir(parents=True)
        # Checking `owner` files with one tool and claim directories with
        # another is how a verification pass agrees with a broken system.
        report = verify_inventory(queue)
        assert report.ownerless_claims == ["run-a.s00"]
        assert not report.consistent

    def test_claims_without_results_are_reported(self, queue):
        queue.claim("run-a.s00")
        assert verify_inventory(queue).claims_without_results == ["run-a.s00"]
