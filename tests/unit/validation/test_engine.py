# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy edits change executed checks without modifying Python evaluators."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from inference_endpoint.validation import (
    Conditions,
    bundled_policy_path,
    load_policy,
    validate_submission,
)
from inference_endpoint.validation.artifacts import load_artifacts
from inference_endpoint.validation.engine import execute_submission
from inference_endpoint.validation.evaluator_base import Evaluator
from inference_endpoint.validation.models import Override, freeze
from inference_endpoint.validation.types import EvidenceKey, Severity
from inference_endpoint.validation.vocabulary import CheckKind

pytestmark = pytest.mark.unit
FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/validation/submissions/valid_standardized"
)


def policy_only(*ids):
    policy = load_policy(bundled_policy_path())
    return replace(
        policy, checks=tuple(rule for rule in policy.checks if rule.id in ids)
    )


def test_dataset_catalog_edit_changes_count_gate():
    policy = policy_only("min-query-count")
    original = execute_submission(FIXTURE, policy)
    catalog = {
        **policy.catalogs,
        "datasets": {"llm-perf-dataset-v1": {"sample_count": 10**12}},
    }
    stricter = execute_submission(FIXTURE, replace(policy, catalogs=freeze(catalog)))
    assert original.passed
    assert not stricter.passed
    assert all(f.rule == "min-query-count" for f in stricter.results)


def test_duration_catalog_edit_changes_runtime_gate():
    policy = policy_only("region-basis", "point-duration")
    original = execute_submission(FIXTURE, policy)
    catalog = {
        **policy.catalogs,
        "steady_state": {
            "minimum_duration_ms": dict.fromkeys(
                policy.catalogs["steady_state"]["minimum_duration_ms"], 10**12
            )
        },
    }
    stricter = execute_submission(FIXTURE, replace(policy, catalogs=freeze(catalog)))
    assert original.passed
    assert not stricter.passed


def test_new_yaml_rule_and_selection_override_execute(tmp_path):
    policy_dir = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), policy_dir)
    point_yaml = policy_dir / "point_checks.yaml"
    document = yaml.safe_load(point_yaml.read_text())
    document["checks"] = {
        "new-completed-floor": {
            "kind": "comparison",
            "left": {"kind": "field", "field": "result_summary.n_samples_completed"},
            "operator": "greater_than",
            "right": {"kind": "constant", "value": 10**12},
        }
    }
    point_yaml.write_text(yaml.safe_dump(document))
    policy = load_policy(policy_dir)
    rule = next(r for r in policy.checks if r.id == "new-completed-floor")
    assert not execute_submission(FIXTURE, replace(policy, checks=(rule,))).passed
    document["checks"]["new-completed-floor"]["applies_to"] = {
        "load_pattern": ["agentic_inference"]
    }
    point_yaml.write_text(yaml.safe_dump(document))
    policy = load_policy(policy_dir)
    rule = next(r for r in policy.checks if r.id == "new-completed-floor")
    assert execute_submission(FIXTURE, replace(policy, checks=(rule,))).results == []


def test_untrusted_nan_region_basis_blocks_without_crashing(tmp_path):
    submission = tmp_path / "submission"
    shutil.copytree(FIXTURE, submission)
    for path in submission.glob("results/*/*/r*/system_desc.json"):
        data = json.loads(path.read_text())
        data["max_supported_concurrency"] = float("nan")
        path.write_text(json.dumps(data))
    report = execute_submission(
        submission, policy_only("region-computation", "point-duration")
    )
    assert not report.passed


def test_registry_rejects_duplicate_kinds():
    with pytest.raises(ValueError, match="Duplicate evaluator"):

        class Duplicate(Evaluator, kind=CheckKind.COMPARISON):
            def __call__(self, check, artifacts):
                return []


def test_model_accuracy_catalog_changes_public_results():
    policy = policy_only("accuracy-gate")
    profile = {
        "accuracy": {
            "kind": "single_turn",
            "datasets": ["llm-perf-dataset-v1"],
            "metrics": {"rouge1": {"reference": 1, "minimum_multiplier": 1}},
        }
    }
    catalogs = {**policy.catalogs, "models": {"llama3_1-8b": profile}}
    policy = replace(policy, catalogs=freeze(catalogs))
    assert validate_submission(FIXTURE, policy=policy).passed
    profile["accuracy"]["metrics"]["rouge1"]["reference"] = 10**12
    stricter = replace(policy, catalogs=freeze(catalogs))
    assert not validate_submission(FIXTURE, policy=stricter).passed


def test_disabled_cohort_override_prevents_execution(tmp_path):
    policy_dir = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), policy_dir)
    catalog_file = policy_dir / "catalog.yaml"
    catalog = yaml.safe_load(catalog_file.read_text())
    catalog["enrollment"]["overrides"] = [
        {
            "id": "count-disabled",
            "rule": "min-query-count",
            "reason": "Cohort count exemption",
            "applies_to": {"models": ["llama3_1-8b"]},
            "enabled": False,
        }
    ]
    catalog_file.write_text(yaml.safe_dump(catalog))
    policy = load_policy(policy_dir)
    policy = replace(
        policy,
        checks=tuple(rule for rule in policy.checks if rule.id == "min-query-count"),
    )
    assert validate_submission(FIXTURE, policy=policy).results == []


def test_unregistered_evaluator_blocks(monkeypatch):
    monkeypatch.delitem(Evaluator.registry, CheckKind.COMPARISON)
    report = execute_submission(FIXTURE, policy_only("metric-consistency-duration"))
    assert report.errors
    assert all(f.key == "blocked" for f in report.errors)


def test_effective_override_changes_evaluator_threshold():
    policy = policy_only("point-cap")
    override = Override(
        id="smaller-cohort-cap",
        rule="point-cap",
        reason="Cohort point limit",
        applies_to=Conditions(models=("llama3_1-8b",)),
        enabled=True,
        requirements=freeze({"maximum": 1}),
    )
    assert validate_submission(FIXTURE, policy=policy).passed
    overridden = validate_submission(
        FIXTURE, policy=replace(policy, overrides=(override,))
    )
    assert not overridden.passed
    assert all("allowed [0, 1]" in finding.message for finding in overridden.results)


def test_disabled_region_basis_does_not_publish_dependencies():
    policy = policy_only("region-basis")
    exemption = Override(
        id="disabled-basis",
        rule="region-basis",
        reason="Disabled derivation",
        applies_to=Conditions(models=("llama3_1-8b",)),
        enabled=False,
        requirements=freeze({}),
    )
    artifacts = load_artifacts(FIXTURE, replace(policy, overrides=(exemption,)))
    assert not artifacts.derived.regions
    assert all(
        EvidenceKey.COMPUTED_REGIONS not in point.context.available
        for point in artifacts.index.points.values()
    )


def test_region_basis_uses_selected_collection_members(tmp_path):
    submission = tmp_path / "submission"
    shutil.copytree(FIXTURE, submission)
    path = next(submission.glob("results/*/*/r16/point.yaml"))
    data = yaml.safe_load(path.read_text())
    data["offline"] = "dedicated"
    path.write_text(yaml.safe_dump(data))
    policy = policy_only("region-basis")
    rule = replace(policy.checks[0], applies_to=Conditions(offline=("none",)))
    artifacts = load_artifacts(submission, replace(policy, checks=(rule,)))
    assert next(iter(artifacts.derived.regions.values()))["low_latency"] == (1, 20)


@pytest.mark.parametrize("rule_id", ["region-placement", "offline-point-present"])
def test_unknown_collection_operation_blocks(rule_id):
    policy = policy_only("region-basis", rule_id)
    rules = tuple(
        replace(
            rule, requirements=freeze({**rule.requirements, "operation": "unsupported"})
        )
        if rule.id == rule_id
        else rule
        for rule in policy.checks
    )
    report = execute_submission(FIXTURE, replace(policy, checks=rules))
    findings = [f for f in report.results if f.rule == rule_id]
    assert findings and all(f.key == "blocked" for f in findings)


def test_offline_uses_effective_numeric_concurrency_floor(tmp_path):
    submission = tmp_path / "submission"
    shutil.copytree(FIXTURE, submission)
    path = next(submission.glob("results/*/*/r1000/point.yaml"))
    data = yaml.safe_load(path.read_text())
    data["offline"] = "dedicated"
    path.write_text(yaml.safe_dump(data))
    policy = policy_only("offline-ordering")
    rule = replace(
        policy.checks[0],
        requirements=freeze(
            {**policy.checks[0].requirements, "concurrency_floor": 10**12}
        ),
    )
    report = execute_submission(submission, replace(policy, checks=(rule,)))
    assert report.results and report.results[0].severity.value == "warning"
    assert "1000000000000" in report.results[0].message


def test_elected_offline_uses_effective_required_concurrency():
    policy = policy_only("offline-point-present")
    rule = replace(
        policy.checks[0],
        requirements=freeze(
            {**policy.checks[0].requirements, "elected_must_equal": 10**12}
        ),
    )
    assert not execute_submission(FIXTURE, replace(policy, checks=(rule,))).passed


def test_schema_errors_use_effective_warning_severity(tmp_path):
    submission = tmp_path / "submission"
    shutil.copytree(FIXTURE, submission)
    path = next(submission.glob("results/*/*/r16/point.yaml"))
    data = yaml.safe_load(path.read_text())
    data["concurrency"] = "invalid"
    path.write_text(yaml.safe_dump(data))
    policy = policy_only("point-config-valid")
    rule = replace(policy.checks[0], severity=Severity.WARNING)
    report = execute_submission(submission, replace(policy, checks=(rule,)))
    assert report.passed and report.warnings
    assert all(f.severity.value == "warning" for f in report.results if f.path == path)


def test_conflicting_region_basis_partitions_block_explicitly():
    policy = policy_only("region-basis", "point-duration")
    partition = Override(
        id="different-clamp",
        rule="region-basis",
        reason="Partitioned region basis",
        applies_to=Conditions(offline=("elected",)),
        enabled=True,
        requirements=freeze({"upper_clamp": 4}),
    )
    report = execute_submission(FIXTURE, replace(policy, overrides=(partition,)))
    assert any(
        f.rule == "region-basis" and f.key == "blocked" and "Conflicting" in f.message
        for f in report.results
    )
    assert any(
        f.rule == "point-duration" and f.key == "blocked" for f in report.results
    )


@pytest.mark.parametrize(
    "option", ["seed_sets_path", "approved_sped_decode_heads_path"]
)
def test_api_rejects_catalog_overrides(option, tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        validate_submission(FIXTURE, **{option: tmp_path})


def test_catalog_environment_cannot_replace_validation_approvals(tmp_path, monkeypatch):
    monkeypatch.setenv("MLPERF_ENDPOINTS_SEED_SETS", str(tmp_path / "seeds.yaml"))
    monkeypatch.setenv(
        "MLPERF_ENDPOINTS_APPROVED_DRAFTERS", str(tmp_path / "heads.yaml")
    )
    policy = load_policy(bundled_policy_path())
    artifacts = load_artifacts(tmp_path, policy)
    assert "seed_sets" in artifacts.catalogs.values
    assert not artifacts.catalogs.errors
    assert (
        artifacts.catalogs.values["approved_sped_decode_heads"]
        == policy.catalogs["approved_sped_decode_heads"]
    )


@pytest.mark.parametrize(
    "submission", sorted(FIXTURE.parent.iterdir()), ids=lambda path: path.name
)
def test_native_submission_corpus_has_no_internal_evaluation_errors(submission):
    report = validate_submission(submission)
    assert report.results
    assert not any(
        "Evaluation could not complete" in finding.message for finding in report.results
    )
