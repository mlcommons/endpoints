# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reported and configured counts share operators while retaining rule semantics."""

import json
import shutil
from dataclasses import replace

import pytest
import yaml

from inference_endpoint.validation import (
    bundled_policy_path,
    load_policy,
    validate_submission,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def policy():
    return load_policy(bundled_policy_path())


def check_count(
    tmp_path,
    policy,
    rule_id,
    *,
    dataset="cnn_dailymail",
    completed=13368,
    trajectories=613,
    declared_instances=200,
    evaluated_instances=200,
):
    point = tmp_path / "results" / "system" / "kimi-k3" / "r16"
    point.mkdir(parents=True)
    config = {
        "concurrency": 16,
        "dataset": dataset,
        "runtime_settings": {
            "load_pattern": "agentic_inference",
            "runtime": {},
            "agentic_inference": {"num_trajectories_to_issue": trajectories},
        },
        "accuracy": {
            "swe_bench": {
                "extras": {"num_instances": declared_instances},
                "evaluated_instance_count": evaluated_instances,
            }
        },
    }
    (point / "point.yaml").write_text(json.dumps(config))
    (point / "result_summary.json").write_text(
        json.dumps(
            {
                "n_samples_completed": completed,
                "duration_ns": 1e9,
            }
        )
    )
    selected = replace(
        policy, checks=tuple(rule for rule in policy.checks if rule.id == rule_id)
    )
    assert len(selected.checks) == 1
    return validate_submission(tmp_path, policy=selected)


@pytest.mark.parametrize(
    ("completed", "passed"), [(13367, False), (13368, True), (20000, True)]
)
def test_min_completed_samples_accepts_extra_samples(
    tmp_path, policy, completed, passed
):
    report = check_count(tmp_path, policy, "min-completed-samples", completed=completed)
    assert len(report.results) == 1
    assert report.passed is passed


@pytest.mark.parametrize(
    ("trajectories", "passed"),
    [(0, False), (612, False), (613, True), (614, False), (1226, True)],
)
def test_trajectories_to_issue_requires_positive_dataset_multiple(
    tmp_path, policy, trajectories, passed
):
    report = check_count(
        tmp_path, policy, "trajectories-to-issue", trajectories=trajectories
    )
    assert len(report.results) == 1
    assert report.passed is passed


@pytest.mark.parametrize(
    ("declared", "evaluated", "passed"),
    [
        (200, 200, True),
        (199, 200, False),
        (200, 199, False),
        (400, 400, False),
        (None, 200, False),
        (200, None, False),
        (True, 200, False),
    ],
)
def test_swebench_instances_requires_both_exact_counts(
    tmp_path, policy, declared, evaluated, passed
):
    report = check_count(
        tmp_path,
        policy,
        "swebench-instance-count",
        declared_instances=declared,
        evaluated_instances=evaluated,
    )
    assert len(report.results) == 2
    assert report.passed is passed


@pytest.mark.parametrize("dataset", ["unknown", "agentic_combined", "swe_bench"])
def test_min_completed_samples_skips_unknown_datasets_and_other_units(
    tmp_path, policy, dataset
):
    report = check_count(
        tmp_path, policy, "min-completed-samples", dataset=dataset, completed=0
    )
    assert report.results == []


@pytest.mark.parametrize(
    "rule_id", ["trajectories-to-issue", "swebench-instance-count"]
)
def test_fixed_reference_count_missing_blocks_check(tmp_path, policy, rule_id):
    policy = replace(policy, catalogs={**policy.catalogs, "datasets": {}})
    report = check_count(tmp_path, policy, rule_id)
    assert report.errors
    assert all(result.key == "blocked" for result in report.results)


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": "cnn_dailymail", "dataset_source": "point.dataset"},
        {"dataset": None},
        {"operator": "unknown"},
        {
            "source": "result_summary.n_samples_completed",
            "sources": ["result_summary.n_samples_issued"],
        },
    ],
)
def test_count_requirements_reject_ambiguous_or_invalid_bindings(tmp_path, changes):
    target = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), target)
    path = target / "point_checks.yaml"
    document = yaml.safe_load(path.read_text())
    document["checks"]["trajectories-to-issue"].update(changes)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="trajectories-to-issue"):
        load_policy(target)
