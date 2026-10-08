# SPDX-License-Identifier: Apache-2.0
import shutil

import pytest
import yaml
from pydantic import ValidationError

from inference_endpoint.validation import bundled_policy_path, load_policy
from inference_endpoint.validation.artifacts import PointArtifacts
from inference_endpoint.validation.evidence.accuracy import AccuracyResult
from inference_endpoint.validation.planner import plan_checks
from inference_endpoint.validation.power.calculation import PowerCalculator
from inference_endpoint.validation.power.models import SystemPower
from inference_endpoint.validation.types import Decision, EvidenceKey

pytestmark = pytest.mark.unit


def test_accuracy_nested_scores_are_owned():
    data = {"aime25": {"score": {"exact_match": 90}, "extras": {"source": "test"}}}
    parsed = AccuracyResult.model_validate(data)
    data["aime25"]["score"]["exact_match"] = 0
    data["aime25"]["extras"]["source"] = "changed"
    assert parsed.metric_scores() == {"aime25": {"exact_match": 90.0}}
    assert parsed.root["aime25"]["extras"]["source"] == "test"


@pytest.mark.parametrize("field", ["speculative_decoding", "drafter"])
@pytest.mark.parametrize("declaration", [{}, {"target_checksum": "declared-checksum"}])
def test_supplied_decode_head_keeps_approval_checks_selected(
    tmp_path, field, declaration
):
    point = PointArtifacts.from_json(
        tmp_path / "kimi-k3" / "r16",
        {
            "concurrency": 16,
            "runtime_settings": {"runtime": {}, "load_pattern": "agentic_inference"},
            field: declaration,
        },
        {"n_samples_completed": 1, "duration_ns": 1e9},
        {},
    )
    point = PointArtifacts.from_evidence(
        point.path, point.evidence, frozenset({EvidenceKey.APPROVED_SPED_DECODE_HEADS})
    )
    policy = load_policy(bundled_policy_path())
    checks = {
        check.rule.id: check for check in plan_checks(policy, [point.context]).checks
    }
    assert point.context.speculative_decoding is True
    assert checks["approved-drafter"].decision is Decision.READY
    assert checks["drafter-approval-lead-time"].decision is Decision.READY


@pytest.fixture
def policy_dir(tmp_path):
    directory = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), directory)
    return directory


@pytest.mark.parametrize("value", [[], "false", 0, None])
@pytest.mark.parametrize("override", [False, True])
def test_boolean_requirement_types_are_strict(policy_dir, value, override):
    path = policy_dir / ("catalog.yaml" if override else "point_checks.yaml")
    document = yaml.safe_load(path.read_text())
    if override:
        document["enrollment"]["overrides"].append(
            {
                "id": "bad-positivity",
                "rule": "metric-consistency-tpot-p90",
                "reason": "test",
                "applies_to": {"models": ["gpt-oss-120b"]},
                "requirements": {"strictly_positive": value},
            }
        )
    else:
        document["checks"]["metric-consistency-tpot-p90"]["strictly_positive"] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError, match="strictly_positive"):
        load_policy(policy_dir)


def test_missing_required_operand_is_rejected_when_loading(policy_dir):
    path = policy_dir / "point_checks.yaml"
    document = yaml.safe_load(path.read_text())
    del document["checks"]["metric-consistency-duration"]["right"]
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError, match="right"):
        load_policy(policy_dir)


@pytest.mark.parametrize("cooling", ["air", "liquid", "mixed"])
def test_cooling_modes_use_catalog_overhead(cooling):
    descriptor = SystemPower.model_validate(
        {
            "system_desc_id": "system",
            "cooling": cooling,
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
    catalog = load_policy(bundled_policy_path()).catalogs["power"]
    computed = PowerCalculator(descriptor, catalog).compute()
    assert computed.overhead_fraction == catalog["cooling_overhead"][cooling]
    assert computed.provisioned_power_kw == (
        2.02 if cooling in ("air", "mixed") else 1.75
    )


def test_unknown_cooling_stays_invalid():
    with pytest.raises(ValidationError, match="cooling"):
        SystemPower.model_validate({"cooling": "unknown"})


def test_finite_numeric_strings_in_accuracy_are_normalized():
    parsed = AccuracyResult.model_validate(
        {
            "rouge": {"score": {"rouge1": "45.12", "rouge2": 22.01}},
            "scalar": {"score": "83.13"},
        }
    )
    assert parsed.metric_scores() == {
        "rouge": {"rouge1": 45.12, "rouge2": 22.01},
        "scalar": {"score": 83.13},
    }
    assert parsed.root["rouge"]["score"]["rouge1"] == 45.12


@pytest.mark.parametrize("declaration", [None, "absent"])
def test_no_decode_head_keeps_approval_checks_excluded(tmp_path, declaration):
    config = {"concurrency": 16, "runtime_settings": {"runtime": {}}}
    if declaration is None:
        config["speculative_decoding"] = None
    point = PointArtifacts.from_json(
        tmp_path / "kimi-k3" / "r16",
        config,
        {"n_samples_completed": 1, "duration_ns": 1e9},
        {},
    )
    checks = {
        check.rule.id: check
        for check in plan_checks(
            load_policy(bundled_policy_path()), [point.context]
        ).checks
    }
    assert point.context.speculative_decoding is False
    assert checks["approved-drafter"].decision is Decision.EXCLUDED


def test_valid_override_retains_false_and_numeric_values(policy_dir):
    path = policy_dir / "catalog.yaml"
    document = yaml.safe_load(path.read_text())
    document["enrollment"]["overrides"].extend(
        [
            {
                "id": "no-positivity",
                "rule": "metric-consistency-tpot-p90",
                "reason": "test",
                "applies_to": {"models": ["gpt-oss-120b"]},
                "requirements": {"strictly_positive": False},
            },
            {
                "id": "numeric-floor",
                "rule": "offline-ordering",
                "reason": "test",
                "applies_to": {"models": ["gpt-oss-120b"]},
                "requirements": {"concurrency_floor": 128},
            },
        ]
    )
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    overrides = {
        override.id: override for override in load_policy(policy_dir).overrides
    }
    assert overrides["no-positivity"].requirements["strictly_positive"] is False
    assert overrides["numeric-floor"].requirements["concurrency_floor"] == 128


@pytest.mark.parametrize(
    ("rule", "changes"),
    [
        ("metric-consistency-duration", {"left": 12}),
        ("metric-consistency-duration", {"right": {"kind": "constant", "value": True}}),
        ("metric-consistency-duration", {"operator": "unknown"}),
        ("point-duration", {"basis_precedence": ["unknown"]}),
        ("seed-runtime-match", {"fields": []}),
        ("metric-consistency-tpot-p90", {"source": None}),
    ],
)
def test_malformed_nested_and_operation_contracts_are_rejected(
    policy_dir, rule, changes
):
    path = policy_dir / "point_checks.yaml"
    document = yaml.safe_load(path.read_text())
    document["checks"][rule].update(changes)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError, match=rule):
        load_policy(policy_dir)
