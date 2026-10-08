# SPDX-License-Identifier: Apache-2.0
import shutil

import pytest
import yaml

from inference_endpoint.validation import bundled_policy_path, load_policy

pytestmark = pytest.mark.unit


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
