# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral policy mutations for special evaluators."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from inference_endpoint.validation import (
    Conditions,
    Context,
    bundled_policy_path,
    checks,  # noqa: F401 - registers built-in evaluators
    load_policy,
    validate_submission,
)
from inference_endpoint.validation.artifacts import (
    ArtifactIndex,
    Artifacts,
    PointArtifacts,
    load_artifacts,
)
from inference_endpoint.validation.catalogs.seeds import SeedSet
from inference_endpoint.validation.checks.power import prepare
from inference_endpoint.validation.evaluator_base import Evaluator
from inference_endpoint.validation.evidence.accuracy import AccuracyResult
from inference_endpoint.validation.evidence.loaders import ParsedArtifact
from inference_endpoint.validation.models import Override, thaw
from inference_endpoint.validation.planner import PlannedCheck
from inference_endpoint.validation.power.calculation import PowerCalculator
from inference_endpoint.validation.power.models import SystemPower
from inference_endpoint.validation.types import Decision, EvidenceKey, Scope

pytestmark = pytest.mark.unit


def update_evidence(point, field, **updates):
    artifact = getattr(point.evidence, field)
    data = artifact.value.model_dump(
        mode="json", exclude_unset=True, exclude_computed_fields=True
    )
    data.update(updates)
    parsed = ParsedArtifact.from_json(data, type(artifact.value), point.path, field)
    assert not parsed.errors
    point.evidence = replace(point.evidence, **{field: parsed})


def update_accuracy(point, data):
    parsed = ParsedArtifact.from_json(
        data, AccuracyResult, point.path, "accuracy-valid"
    )
    assert not parsed.errors
    point.evidence = replace(point.evidence, accuracy=parsed)


def evaluate(check, artifacts):
    return Evaluator.lookup(check.rule.kind)()(check, artifacts)


@pytest.fixture
def artifacts():
    policy = load_policy(bundled_policy_path())
    point = PointArtifacts.from_json(
        Path("/submission/results/system/gpt-oss-120b/r16"),
        {
            "model_name": "gpt-oss-120b",
            "concurrency": 16,
            "offline": "elected",
            "runtime_settings": {"load_pattern": "concurrency", "runtime": {}},
        },
        {"n_samples_completed": 1, "duration_ns": 1e9},
        {},
    )
    point.context = Context(
        id=str(point.path), scope=Scope.POINT, model_id="gpt-oss-120b"
    )
    return Artifacts(
        Path("/submission"),
        policy,
        index=ArtifactIndex(
            points={str(point.path): point},
            curves={str(point.curve): [str(point.path)]},
        ),
    )


def check_for(artifacts, rule_id, **parameters):
    rule = next(rule for rule in artifacts.policy.checks if rule.id == rule_id)
    if parameters:
        rule = replace(rule, requirements={**rule.requirements, **parameters})
    point = next(iter(artifacts.index.points.values()))
    subject = (
        point.context
        if rule.scope is Scope.POINT
        else Context(
            id=str(point.curve), scope=rule.scope, model_id=point.context.model_id
        )
    )
    return PlannedCheck(subject, rule, Decision.READY, "Applicable")


def point(artifacts):
    return next(iter(artifacts.index.points.values()))


def test_accuracy_catalog_threshold_changes_acceptance(artifacts):
    p = point(artifacts)
    update_accuracy(
        p,
        {
            "mlperf_gpt_oss_accuracy": {
                "num_samples": 4395,
                "score": {"exact_match": 83},
            }
        },
    )
    check = check_for(artifacts, "accuracy-gate")
    assert all(result.passed for result in evaluate(check, artifacts))
    catalogs = thaw(artifacts.policy.catalogs)
    catalogs["models"]["gpt-oss-120b"]["accuracy"]["metrics"]["exact_match"][
        "reference"
    ] = 90
    artifacts.policy = replace(artifacts.policy, catalogs=catalogs)
    assert any(not result.passed for result in evaluate(check, artifacts))


def test_unrelated_accuracy_dataset_cannot_satisfy_required_gate(artifacts):
    update_accuracy(
        point(artifacts),
        {"other": {"num_samples": 999999, "score": {"exact_match": 100}}},
    )
    results = evaluate(check_for(artifacts, "accuracy-gate"), artifacts)
    assert len(results) == 1
    assert not results[0].passed
    assert "Required accuracy datasets missing" in results[0].message


def test_accuracy_count_uses_repeat_policy(artifacts):
    update_accuracy(
        point(artifacts), {"mlperf_gpt_oss_accuracy": {"num_samples": 3000, "score": 1}}
    )
    assert any(
        not r.passed
        for r in evaluate(check_for(artifacts, "accuracy-sample-count"), artifacts)
    )
    assert all(
        r.passed
        for r in evaluate(
            check_for(artifacts, "accuracy-sample-count", missing_repeats=2), artifacts
        )
    )


def test_steady_superpass_catalog_controls_official_window(artifacts):
    update_evidence(
        point(artifacts),
        "config",
        steady_state={
            "status": "windowable",
            "verdict": "steady_state",
            "state": {"tpot_p90": "plateau"},
            "window": {"super_pass_start": 1, "super_pass_end": 4, "n_super_passes": 4},
        },
    )
    check = check_for(artifacts, "steady-state-consistency")
    assert all(r.passed for r in evaluate(check, artifacts))
    catalogs = thaw(artifacts.policy.catalogs)
    catalogs["steady_state"]["minimum_super_passes"] = 5
    artifacts.policy = replace(artifacts.policy, catalogs=catalogs)
    assert any(not r.passed for r in evaluate(check, artifacts))


def test_seed_adoption_window_parameter_and_unpublished_cohort(artifacts):
    p = point(artifacts)
    update_evidence(p, "config", seed_set="A", target_cohort="2026-11-C0")
    artifacts.catalogs.values["seed_sets"] = {
        "A": SeedSet(
            "A", 1, 2, 3, ("2026-10-C0", "2026-10-C1", "2026-11-C0", "2026-11-C1")
        )
    }
    assert all(
        r.passed for r in evaluate(check_for(artifacts, "seed-set-adoption"), artifacts)
    )
    assert any(
        not r.passed
        for r in evaluate(
            check_for(artifacts, "seed-set-adoption", adoption_window_cohorts=2),
            artifacts,
        )
    )
    artifacts.catalogs.values["seed_sets"]["A"] = SeedSet("A", 1, 2, 3)
    results = evaluate(check_for(artifacts, "seed-set-adoption"), artifacts)
    assert results[0].key == "blocked"
    assert not results[0].passed


def test_accuracy_coverage_overlapping_band_and_elected_offline(artifacts):
    p = point(artifacts)
    update_accuracy(p, {"mlperf_gpt_oss_accuracy": {"score": 1}})
    artifacts.derived.regions[str(p.curve)] = {
        "low_concurrency": (2, 32),
        "med_concurrency": (33, 64),
        "high_concurrency": (65, 128),
    }
    catalogs = thaw(artifacts.policy.catalogs)
    catalogs["mandatory_accuracy_bands"] = ["ultra_low_concurrency", "low_concurrency"]
    artifacts.policy = replace(artifacts.policy, catalogs=catalogs)
    assert all(
        r.passed for r in evaluate(check_for(artifacts, "accuracy-coverage"), artifacts)
    )


def test_power_catalog_overhead_and_component_defaults(artifacts):
    descriptor = SystemPower.model_validate(
        {
            "system_desc_id": "system",
            "cooling": "air",
            "node_sets": [
                {
                    "node_set_id": 1,
                    "system_node_ensemble_id": 1,
                    "nodes_provisioned": 1,
                    "power_method": "component_sum",
                    "components": {
                        "cpu": {"model": "AMD EPYC 64-core", "count_per_node": 1},
                        "accelerator": {"model": "NVIDIA B200", "count_per_node": 1},
                        "scale_up_network": {"method": "none"},
                    },
                }
            ],
            "scale_out": {"present": False},
        }
    )
    catalog = thaw(artifacts.policy.catalogs["power"])
    default = PowerCalculator(descriptor, catalog).compute()
    assert default.provisioned_power_kw == 2.02
    catalog["defaults"]["accelerators"]["b200"] = 2000
    catalog["cooling_overhead"]["air"] = 0.1
    changed = PowerCalculator(descriptor, catalog).compute()
    assert changed.provisioned_power_kw == 2.58
    assert changed.estimated


def test_unknown_special_operation_is_blocked(artifacts):
    results = evaluate(
        check_for(artifacts, "warmup-salt", operation="unsupported"), artifacts
    )
    assert results[0].key == "blocked"
    assert not results[0].passed


def test_drafter_identity_requires_repository_and_revision(artifacts):
    p = point(artifacts)
    p.context = p.context.model_copy(update={"model_id": "kimi-k3"})
    update_evidence(
        p, "config", speculative_decoding={"weight_checksum": "same-checksum"}
    )
    results = evaluate(check_for(artifacts, "approved-drafter"), artifacts)
    assert any(not result.passed for result in results)
    artifacts.catalogs.values["approved_sped_decode_heads"] = artifacts.policy.catalogs[
        "approved_sped_decode_heads"
    ]
    approved = artifacts.policy.catalogs["approved_sped_decode_heads"][0]
    update_evidence(
        p,
        "config",
        speculative_decoding={
            "repository": approved["repository"],
            "revision": approved["revision"],
        },
    )
    assert all(
        result.passed
        for result in evaluate(check_for(artifacts, "approved-drafter"), artifacts)
    )


def test_invalid_reported_superpass_count_fails(artifacts):
    update_evidence(
        point(artifacts),
        "config",
        steady_state={
            "status": "windowable",
            "state": {"tpot_p90": "plateau"},
            "window": {"super_pass_start": 1, "super_pass_end": 4, "n_super_passes": 5},
        },
    )
    assert any(
        not result.passed
        for result in evaluate(
            check_for(artifacts, "steady-state-consistency"), artifacts
        )
    )
    assert all(
        result.passed
        for result in evaluate(
            check_for(
                artifacts,
                "steady-state-consistency",
                check_reported_super_pass_count=False,
            ),
            artifacts,
        )
    )


@pytest.fixture
def two_power_systems(tmp_path):
    policy = load_policy(bundled_policy_path())
    payload = {
        "system_desc_id": "system",
        "cooling": "air",
        "node_sets": [
            {
                "node_set_id": 1,
                "system_node_ensemble_id": 1,
                "nodes_provisioned": 1,
                "power_method": "component_sum",
                "components": {
                    "combined_cpu_accelerator": {
                        "value_w": 9800,
                        "source_type": "vendor_spec",
                        "source": "https://example.com/power",
                    },
                    "scale_up_network": {"method": "none"},
                },
            }
        ],
        "scale_out": {"present": False},
    }
    for name, division in [("system_a", "Standardized"), ("system_b", "Serviced")]:
        system = tmp_path / "results" / name
        p = system / "gpt-oss-120b" / "r16"
        p.mkdir(parents=True)
        (system / "system_power.json").write_text(json.dumps(payload))
        (p / "point.yaml").write_text(
            f"concurrency: 16\nmodel_name: gpt-oss-120b\ndivision: {division}\nruntime_settings:\n  load_pattern: concurrency\n  runtime: {{}}\n"
        )
    return load_artifacts(tmp_path, policy)


def test_power_descriptor_override_changes_only_selected_system(two_power_systems):
    artifacts = two_power_systems
    catalogs = thaw(artifacts.policy.catalogs)
    constants = catalogs["power"]
    constants["cooling_overhead"]["air"] = 1.0
    artifacts.policy = replace(
        artifacts.policy,
        overrides=(
            Override(
                "air-overhead",
                "power-descriptor",
                "System-specific overhead",
                Conditions(division=["Standardized"]),
                True,
                {"constants": constants},
            ),
        ),
    )
    prepare(artifacts)
    values = {
        Path(id).name: entry["computation"].provisioned_power_kw
        for id, entry in artifacts.derived.power.items()
    }
    assert values == {"system_a": 19.6, "system_b": 14.7}


def test_disabled_point_power_never_provides_denominator(two_power_systems):
    artifacts = two_power_systems
    artifacts.policy = replace(
        artifacts.policy,
        overrides=(
            Override(
                "no-denominator",
                "point-power",
                "Disabled",
                Conditions(division=["Standardized"]),
                False,
                {},
            ),
        ),
    )
    prepare(artifacts)
    for p in artifacts.index.points.values():
        if p.context.division.value == "Standardized":
            assert p.derived.power_kw is None
            assert EvidenceKey.POINT_POWER not in p.context.available
        else:
            assert p.derived.power_kw == 14.7
            assert EvidenceKey.POINT_POWER in p.context.available


def test_point_power_rounding_override_applies_during_preparation(two_power_systems):
    artifacts = two_power_systems
    artifacts.policy = replace(
        artifacts.policy,
        overrides=(
            Override(
                "whole-kw",
                "point-power",
                "Round selected point",
                Conditions(division=["Standardized"]),
                True,
                {"round_kw_decimals": 0},
            ),
        ),
    )
    prepare(artifacts)
    values = {
        p.path.parent.parent.name: p.derived.power_kw
        for p in artifacts.index.points.values()
    }
    assert values == {"system_a": 15.0, "system_b": 14.7}


@pytest.mark.parametrize(
    "enabled,parameters", [(False, {}), (True, {"validate_computed_values": False})]
)
def test_unevaluated_descriptor_provides_no_power_facts(
    two_power_systems, enabled, parameters
):
    artifacts = two_power_systems
    artifacts.policy = replace(
        artifacts.policy,
        overrides=(
            Override(
                "no-system-power",
                "power-descriptor",
                "Unavailable power",
                Conditions(division=["Standardized"]),
                enabled,
                parameters,
            ),
        ),
    )
    prepare(artifacts)
    selected = next(
        p
        for p in artifacts.index.points.values()
        if p.context.division.value == "Standardized"
    )
    assert selected.derived.power_kw is None
    assert EvidenceKey.POWER_COMPUTATION not in selected.context.available
    assert EvidenceKey.POINT_POWER not in selected.context.available
    other = next(
        p
        for p in artifacts.index.points.values()
        if p.context.division.value == "Serviced"
    )
    assert other.derived.power_kw == 14.7


def test_invalid_point_power_policy_returns_blocked_report(two_power_systems):
    artifacts = two_power_systems
    effective_checks = tuple(
        replace(
            rule, requirements={**rule.requirements, "round_kw_decimals": "invalid"}
        )
        if rule.id == "point-power"
        else rule
        for rule in artifacts.policy.checks
    )
    policy = replace(artifacts.policy, checks=effective_checks)
    report = validate_submission(artifacts.root, policy=policy)
    assert any(
        result.rule == "point-power" and result.key == "blocked"
        for result in report.errors
    )
    artifacts.policy = policy
    prepare(artifacts)
    assert all(
        point.derived.power_kw is None
        and EvidenceKey.POINT_POWER not in point.context.available
        for point in artifacts.index.points.values()
    )
